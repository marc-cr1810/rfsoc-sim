//! DSP processing functions: FFT, Nyquist zone folding, mixing, and decimation.

#![allow(dead_code)]

use crate::rfdc::{AdcBlock, AdcTile, CoarseMixFreq, MixerType, MixerMode as RfdcMixerMode, FineMixerScale};
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Result of processing a signal through the full ADC + DDC chain.
#[derive(Debug, Clone)]
pub struct ProcessedSignal {
    /// Optional spectrum of raw source signal before RF chain filtering (dBFS).
    pub raw_source_spectrum_dbfs: Option<Vec<f64>>,
    /// Spectrum of the input signal (after RF chain filtering, before ADC), in dBFS.
    pub input_spectrum_dbfs: Vec<f64>,
    /// Frequency axis for the input spectrum, in MHz.
    pub input_freq_axis_mhz: Vec<f64>,
    /// Optional cumulative RF chain frequency response (dB).
    pub rf_chain_response_db: Option<Vec<f64>>,
    /// Optional frequency axis for the RF chain response (MHz).
    pub rf_chain_freq_axis_mhz: Option<Vec<f64>>,
    /// Spectrum after Nyquist zone folding (what the ADC sees), in dBFS.
    pub folded_spectrum_dbfs: Vec<f64>,
    /// Frequency axis for the folded spectrum (0..Fs/2), in MHz.
    pub folded_freq_axis_mhz: Vec<f64>,
    /// Spectrum after the DDC mixer stage, in dBFS.
    pub post_mixer_spectrum_dbfs: Vec<f64>,
    /// Frequency axis after mixer, in MHz.
    pub post_mixer_freq_axis_mhz: Vec<f64>,
    /// Composite decimation-filter response (dB) on `post_mixer_freq_axis_mhz`. This is the
    /// window through which the PL sees the post-mixer spectrum.
    pub decimation_response_db: Vec<f64>,
    /// True when the DDC delivers complex I/Q to the PL (spectrum spans ±Fout/2).
    pub complex_output: bool,
    /// NCO frequency the mixer actually ran at. Differs from the configured value only when
    /// that value sat outside ±Fs/2 and the wrap-and-sign convention applied.
    pub resolved_nco_freq_mhz: f64,
    /// Final spectrum after decimation, in dBFS.
    pub output_spectrum_dbfs: Vec<f64>,
    /// Frequency axis for the output, in MHz.
    pub output_freq_axis_mhz: Vec<f64>,
    /// Effective output sample rate in MHz.
    pub output_sample_rate_mhz: f64,
    /// Complex baseband output time-domain samples (for oscilloscope & constellation).
    pub output_time_samples: Vec<Complex<f64>>,
    /// True if the physical ADC waveform clipped at any point during this capture.
    pub overrange: bool,
}

// ...

use serde::{Deserialize, Serialize};
use std::cell::RefCell;

thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

/// FFT window functions for spectral analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FftWindow {
    Hanning,
    Hamming,
    BlackmanHarris,
    FlatTop,
    Rectangular,
}

impl FftWindow {
    pub const ALL: [FftWindow; 5] = [
        FftWindow::Hanning,
        FftWindow::Hamming,
        FftWindow::BlackmanHarris,
        FftWindow::FlatTop,
        FftWindow::Rectangular,
    ];

    pub fn apply(&self, buffer: &mut [Complex<f64>]) {
        let n = buffer.len();
        if n == 0 {
            return;
        }
        match self {
            FftWindow::Hanning => {
                for (i, sample) in buffer.iter_mut().enumerate() {
                    let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos());
                    *sample *= w;
                }
            }
            FftWindow::Hamming => {
                for (i, sample) in buffer.iter_mut().enumerate() {
                    let w = 0.54 - 0.46 * (2.0 * PI * i as f64 / n as f64).cos();
                    *sample *= w;
                }
            }
            FftWindow::BlackmanHarris => {
                for (i, sample) in buffer.iter_mut().enumerate() {
                    let a0 = 0.35875;
                    let a1 = 0.48829;
                    let a2 = 0.14128;
                    let a3 = 0.01168;
                    let phi = 2.0 * PI * i as f64 / n as f64;
                    let w = a0 - a1 * phi.cos() + a2 * (2.0 * phi).cos() - a3 * (3.0 * phi).cos();
                    *sample *= w;
                }
            }
            FftWindow::FlatTop => {
                for (i, sample) in buffer.iter_mut().enumerate() {
                    let a0 = 0.21557895;
                    let a1 = 0.41663158;
                    let a2 = 0.277263158;
                    let a3 = 0.083578947;
                    let a4 = 0.006947368;
                    let phi = 2.0 * PI * i as f64 / n as f64;
                    let w = a0 - a1 * phi.cos() + a2 * (2.0 * phi).cos() - a3 * (3.0 * phi).cos() + a4 * (4.0 * phi).cos();
                    *sample *= w;
                }
            }
            FftWindow::Rectangular => {}
        }
    }

    pub fn coherent_gain(&self) -> f64 {
        match self {
            FftWindow::Hanning => 0.5,
            FftWindow::Hamming => 0.54,
            FftWindow::BlackmanHarris => 0.35875,
            FftWindow::FlatTop => 0.21557895,
            FftWindow::Rectangular => 1.0,
        }
    }
}

impl std::fmt::Display for FftWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FftWindow::Hanning => write!(f, "Hanning"),
            FftWindow::Hamming => write!(f, "Hamming"),
            FftWindow::BlackmanHarris => write!(f, "Blackman-Harris"),
            FftWindow::FlatTop => write!(f, "Flat-Top"),
            FftWindow::Rectangular => write!(f, "Rectangular"),
        }
    }
}

/// The physical voltage present at the ADC pin.
///
/// The RF chain is modelled on complex samples, but the converter input is a single-ended
/// real voltage: only the real part is ever sampled. Collapsing to a real signal here keeps
/// the pre-ADC spectrum, the sampler and the folded spectrum on one consistent scale.
pub fn physical_voltage(samples: &[Complex<f64>]) -> Vec<Complex<f64>> {
    samples.iter().map(|s| Complex::new(s.re, 0.0)).collect()
}

/// Apply the converter's analog input bandwidth roll-off to the wideband voltage waveform.
///
/// Runs before sampling so that content in high Nyquist zones aliases down already
/// attenuated, the way a real track-and-hold behaves.
pub fn apply_analog_bandwidth(
    samples: &[Complex<f64>],
    sample_rate_mhz: f64,
    afe: &crate::rfdc::AnalogFrontEnd,
) -> Vec<Complex<f64>> {
    if !afe.enabled || samples.is_empty() || sample_rate_mhz <= 0.0 {
        return samples.to_vec();
    }

    let n = samples.len();
    let mut buffer: Vec<Complex<f64>> = samples.to_vec();
    FFT_PLANNER.with(|planner| {
        let fft = planner.borrow_mut().plan_fft_forward(n);
        fft.process(&mut buffer);
    });

    // Scale each bin by the analog response, treating bins above N/2 as negative frequencies.
    for (i, bin) in buffer.iter_mut().enumerate() {
        let k = if i <= n / 2 {
            i as f64
        } else {
            i as f64 - n as f64
        };
        let freq_mhz = k * sample_rate_mhz / n as f64;
        *bin *= afe.gain_linear(freq_mhz);
    }

    FFT_PLANNER.with(|planner| {
        let ifft = planner.borrow_mut().plan_fft_inverse(n);
        ifft.process(&mut buffer);
    });

    let norm = 1.0 / n as f64;
    buffer.iter().map(|c| Complex::new(c.re * norm, 0.0)).collect()
}

/// Apply analog hardware non-idealities (HD2/HD3 distortion) before sampling.
/// Distortion is applied to the real voltage waveform.
pub fn apply_analog_non_idealities(
    samples: &[Complex<f64>],
    non_idealities: &crate::rfdc::AdcNonIdealities,
) -> Vec<Complex<f64>> {
    if !non_idealities.enabled || samples.is_empty() {
        return samples.to_vec();
    }

    // For an input A·cos(θ), a2·v² produces a second harmonic of amplitude a2·A²/2 and
    // a3·v³ a third harmonic of amplitude a3·A³/4. The coefficients therefore carry factors
    // of 2 and 4 so that the configured dBc figures are the levels actually produced at
    // full scale (A = 1), rather than landing 6 dB and 12 dB low.
    let a2 = if non_idealities.hd2_dbc < 0.0 {
        2.0 * 10.0_f64.powf(non_idealities.hd2_dbc / 20.0)
    } else {
        0.0
    };
    let a3 = if non_idealities.hd3_dbc < 0.0 {
        4.0 * 10.0_f64.powf(non_idealities.hd3_dbc / 20.0)
    } else {
        0.0
    };

    // The squaring term also generates DC. The front end is AC-coupled through the balun,
    // so remove the mean rather than letting it appear as a DC offset.
    let mean_sq = if a2 > 0.0 {
        samples.iter().map(|s| s.re * s.re).sum::<f64>() / samples.len() as f64
    } else {
        0.0
    };

    samples
        .iter()
        .map(|&s| {
            let v = s.re;
            let mut out = v;
            if a2 > 0.0 {
                out += a2 * (v * v - mean_sq);
            }
            if a3 > 0.0 {
                out += a3 * v * v * v;
            }
            Complex::new(out, 0.0)
        })
        .collect()
}

/// Apply digital hardware non-idealities (Quantization, Clipping, Interleaving spurs) after sampling.
/// Also returns a boolean indicating if clipping occurred (overrange).
pub fn apply_digital_non_idealities(
    samples: &[Complex<f64>],
    non_idealities: &crate::rfdc::AdcNonIdealities,
) -> (Vec<Complex<f64>>, bool) {
    let mut overrange = false;

    if samples.is_empty() {
        return (Vec::new(), false);
    }

    let spur_amp = if non_idealities.enabled && non_idealities.interleaving_spur_dbc < 0.0 {
        10.0_f64.powf(non_idealities.interleaving_spur_dbc / 20.0)
    } else {
        0.0
    };

    let q_levels = if non_idealities.enabled && non_idealities.quantization_bits > 0 && non_idealities.quantization_bits <= 24 {
        (1u64 << non_idealities.quantization_bits) as f64
    } else {
        0.0
    };

    let max_val = 1.0; // Normalized +/- 1.0 full scale

    // Broadband noise to hit the configured ENOB. A full-scale sine has power 1/2, and
    // SNR = 6.02·ENOB + 1.76 dB, so the total in-band noise power is (1/2)·10^(−SNR/10).
    // The quantiser below already contributes Δ²/12, so only add the difference — otherwise
    // the two mechanisms would stack and the floor would sit below the specified ENOB.
    let mut noise_sigma = 0.0_f64;
    if non_idealities.enabled && non_idealities.enob > 0.0 {
        let snr_db = 6.02 * non_idealities.enob + 1.76;
        let total_noise_pwr = 0.5 * 10.0_f64.powf(-snr_db / 10.0);
        let quant_noise_pwr = if q_levels > 0.0 {
            let step = 2.0 / q_levels;
            step * step / 12.0
        } else {
            0.0
        };
        noise_sigma = (total_noise_pwr - quant_noise_pwr).max(0.0).sqrt();
    }
    // Deterministic per-call RNG so the floor is stable frame to frame rather than flickering.
    let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_gaussian = move || -> f64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let u1 = ((rng_state >> 11) as f64 / (1u64 << 53) as f64).max(1e-300);
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let u2 = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    };

    let processed: Vec<Complex<f64>> = samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let mut v = s.re;

            // 0. Thermal + aperture-jitter noise setting the ENOB-limited floor
            if noise_sigma > 0.0 {
                v += noise_sigma * next_gaussian();
            }

            // 1. Interleaving mismatch spur (typically Fs/2 and Fs/4)
            if spur_amp > 0.0 {
                let sign_fs2 = if i % 2 == 0 { 1.0 } else { -1.0 };
                let sign_fs4 = if i % 4 == 0 || i % 4 == 1 { 1.0 } else { -1.0 };
                // Mismatch scales with input
                v += v * spur_amp * (sign_fs2 + sign_fs4) * 0.5;
                
                // Offset spur (signal independent) at Fs/2
                v += spur_amp * sign_fs2 * 0.1; 
            }

            // 2. Clipping
            if v > max_val {
                v = max_val;
                overrange = true;
            } else if v < -max_val {
                v = -max_val;
                overrange = true;
            }

            // 3. Bit resolution quantization
            if q_levels > 0.0 {
                let half_q = q_levels / 2.0;
                let mut quant = (v * half_q).round();
                if quant >= half_q {
                    quant = half_q - 1.0;
                } else if quant < -half_q {
                    quant = -half_q;
                }
                v = quant / half_q;
            }

            Complex::new(v, 0.0)
        })
        .collect();

    (processed, overrange)
}

/// Resolve a requested NCO frequency to the value the mixer actually runs at.
///
/// The NCO is periodic in Fs, so a request outside ±Fs/2 wraps into that range. The sign then
/// follows the XRFdc convention: tuning to a frequency that lands in the zone *opposite* the
/// block's configured Nyquist zone yields a negative NCO, and one in the same zone parity
/// yields a positive one. With Fs = 4000 MHz, entering 2400 MHz gives −1600 on an odd-zone
/// block (2400 is in even zone 2, the opposite) and +1600 on an even-zone block.
///
/// A request already inside ±Fs/2 — which is everything `auto_tune` emits — passes through
/// untouched, sign included.
pub fn resolve_nco_freq(requested_mhz: f64, fs_mhz: f64, is_even_zone: bool) -> f64 {
    if fs_mhz <= 0.0 || requested_mhz.abs() <= fs_mhz / 2.0 {
        return requested_mhz;
    }
    let wrapped = (requested_mhz + fs_mhz / 2.0).rem_euclid(fs_mhz) - fs_mhz / 2.0;
    if is_even_zone && wrapped != 0.0 {
        -wrapped
    } else {
        wrapped
    }
}

/// Process a signal through the full ADC block pipeline.
///
/// Pipeline: input → fold spectrum → mix → decimate → output spectrum
pub fn process_adc_block(
    input_samples: &[Complex<f64>],
    input_sample_rate_mhz: f64,
    block: &AdcBlock,
    tile: &AdcTile,
    raw_source_samples: Option<&[Complex<f64>]>,
    rf_chain_response: Option<(Vec<f64>, Vec<f64>)>,
) -> ProcessedSignal {
    let fft_size = ANALYSIS_FFT_SIZE;
    let fs_mhz = tile.sample_rate_mhz();
    let ms = &block.mixer_settings;

    // 0. Apply DSA (Digital Step Attenuator) — reduces full-scale voltage before sampling
    let dsa_scale = if block.dsa_db > 0.0 {
        10.0_f64.powf(-block.dsa_db / 20.0)
    } else {
        1.0
    };
    // Collapse to the real voltage actually present at the converter pin before doing
    // anything else, so every downstream spectrum shares one amplitude reference.
    let dsa_samples: Vec<Complex<f64>> = if dsa_scale < 1.0 {
        input_samples.iter().map(|s| Complex::new(s.re * dsa_scale, 0.0)).collect()
    } else {
        physical_voltage(input_samples)
    };

    // 1. Analog input bandwidth roll-off of the track-and-hold, then HD2/HD3 (pre-sampling)
    let afe_samples =
        apply_analog_bandwidth(&dsa_samples, input_sample_rate_mhz, &block.analog_front_end);
    let analog_samples = apply_analog_non_idealities(&afe_samples, &block.non_idealities);

    // 2. Input spectrum (full wideband) — the real voltage at the ADC pin
    let (input_spectrum, input_freq) =
        compute_spectrum_positive(&analog_samples, fft_size, input_sample_rate_mhz);

    let raw_source_spectrum_dbfs = raw_source_samples.map(|samples| {
        let real_source = physical_voltage(samples);
        let (raw_spec, _) =
            compute_spectrum_positive(&real_source, fft_size, input_sample_rate_mhz);
        raw_spec
    });

    // 3. Sample wideband real physical voltage v(t) at the ADC tile sample rate Fs.
    // In hardware, track-and-hold ADC sampling folds ALL wideband Nyquist zones into 0..Fs/2.
    let tile_samples_analog = sample_adc_at_tile_rate(&analog_samples, input_sample_rate_mhz, fs_mhz);

    // 4. Apply digital non-idealities (Clipping, Quantization, Interleaving spurs)
    let (tile_samples, overrange) = apply_digital_non_idealities(&tile_samples_analog, &block.non_idealities);

    // Folded spectrum: actual ADC digital output spectrum (0..Fs/2)
    let (folded_spectrum, folded_freq) =
        compute_spectrum_positive(&tile_samples, fft_size, fs_mhz);

    let nco_freq = resolve_nco_freq(ms.freq, fs_mhz, block.nyquist_zone.is_even());

    // Determine FineMixerScale
    let scale = match ms.fine_mixer_scale {
        FineMixerScale::OnePointZero => 1.0,
        FineMixerScale::ZeroPointSeven => 0.7071067811865476, // 1/√2
        FineMixerScale::Auto => {
            // XRFdc driver: R2C uses 1.0, C2C uses 0.7071, R2R uses 1.0
            match ms.mixer_mode {
                RfdcMixerMode::IqToIq => 0.7071067811865476,
                _ => 1.0,
            }
        }
    };

    let mixed_samples = apply_mixer(
        &tile_samples,
        &block.mixer_settings,
        nco_freq,
        fs_mhz,
        fs_mhz,
        scale,
    );

    // 4. Compute post-mixer spectrum (at ADC tile rate Fs)
    let complex_output = block.produces_complex_output();
    let (post_mixer_spectrum, post_mixer_freq) = if complex_output {
        compute_spectrum(&mixed_samples, fft_size, fs_mhz)
    } else {
        compute_spectrum_positive(&mixed_samples, fft_size, fs_mhz)
    };

    // The decimation filter's window onto the post-mixer spectrum: everything outside it is
    // what the PL will *not* receive.
    let decim_factor = block.decimation.factor();
    let decimation_response_db: Vec<f64> = if decim_factor > 1 {
        let chain = decimation_chain(decim_factor);
        post_mixer_freq
            .iter()
            .map(|&f| chain.response_db(f / fs_mhz))
            .collect()
    } else {
        vec![0.0; post_mixer_freq.len()]
    };

    // 5. Apply QMC (Quadrature Modulation Correction) post-mixer, pre-decimation
    let qmc_samples = apply_qmc(&mixed_samples, &block.qmc_settings);

    // 6. Apply DDC decimation filter at the ADC tile rate Fs
    let decimated = apply_decimation(&qmc_samples, decim_factor);
    let actual_output_rate = block.output_rate_mhz(tile.sample_rate_gsps);

    // 6. Output spectrum
    let out_fft = output_fft_size(decim_factor);
    let (output_spectrum, output_freq) = if complex_output {
        compute_spectrum(&decimated, out_fft, actual_output_rate)
    } else {
        compute_spectrum_positive(&decimated, out_fft, actual_output_rate)
    };

    let (rf_chain_response_db, rf_chain_freq_axis_mhz) = match rf_chain_response {
        Some((resp, freq)) => (Some(resp), Some(freq)),
        None => (None, None),
    };

    ProcessedSignal {
        raw_source_spectrum_dbfs,
        input_spectrum_dbfs: input_spectrum,
        input_freq_axis_mhz: input_freq,
        rf_chain_response_db,
        rf_chain_freq_axis_mhz,
        folded_spectrum_dbfs: folded_spectrum,
        folded_freq_axis_mhz: folded_freq,
        post_mixer_spectrum_dbfs: post_mixer_spectrum,
        post_mixer_freq_axis_mhz: post_mixer_freq,
        decimation_response_db,
        complex_output,
        resolved_nco_freq_mhz: nco_freq,
        output_spectrum_dbfs: output_spectrum,
        output_freq_axis_mhz: output_freq,
        output_sample_rate_mhz: actual_output_rate,
        output_time_samples: decimated,
        overrange,
    }
}

/// Compute the power spectrum of complex samples using FFT with a specific window function.
pub fn compute_spectrum_with_window(
    samples: &[Complex<f64>],
    fft_size: usize,
    sample_rate_mhz: f64,
    window: FftWindow,
) -> (Vec<f64>, Vec<f64>) {
    let n = (fft_size.min(samples.len()) / 2) * 2;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut buffer: Vec<Complex<f64>> = samples[..n].to_vec();

    // Apply selected window function
    window.apply(&mut buffer);

    // Process FFT using thread-local planner cache
    FFT_PLANNER.with(|planner| {
        let fft = planner.borrow_mut().plan_fft_forward(n);
        fft.process(&mut buffer);
    });

    // Compute magnitude in dBFS (normalised to FFT size and window coherent gain)
    let norm = 1.0 / (n as f64 * window.coherent_gain());
    let spectrum_dbfs: Vec<f64> = buffer
        .iter()
        .map(|c| {
            let mag = c.norm() * norm;
            20.0 * mag.max(1e-15).log10()
        })
        .collect();

    // FFT-shift: move DC to centre
    let mut shifted = vec![0.0; n];
    let half = n / 2;
    shifted[..half].copy_from_slice(&spectrum_dbfs[half..]);
    shifted[half..].copy_from_slice(&spectrum_dbfs[..half]);

    // Frequency axis (centred)
    let freq_axis: Vec<f64> = (0..n)
        .map(|i| (i as f64 - half as f64) * sample_rate_mhz / n as f64)
        .collect();

    (shifted, freq_axis)
}

/// Compute the power spectrum of complex samples using FFT (default Hanning window).
pub fn compute_spectrum(
    samples: &[Complex<f64>],
    fft_size: usize,
    sample_rate_mhz: f64,
) -> (Vec<f64>, Vec<f64>) {
    compute_spectrum_with_window(samples, fft_size, sample_rate_mhz, FftWindow::Hanning)
}

/// Compute single-sided (positive frequency only) power spectrum with a specific window function.
pub fn compute_spectrum_positive_with_window(
    samples: &[Complex<f64>],
    fft_size: usize,
    sample_rate_mhz: f64,
    window: FftWindow,
) -> (Vec<f64>, Vec<f64>) {
    let n = (fft_size.min(samples.len()) / 2) * 2;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut buffer: Vec<Complex<f64>> = samples[..n].to_vec();

    // Apply selected window function
    window.apply(&mut buffer);

    FFT_PLANNER.with(|planner| {
        let fft = planner.borrow_mut().plan_fft_forward(n);
        fft.process(&mut buffer);
    });

    let norm = 1.0 / (n as f64 * window.coherent_gain());
    let half = n / 2;

    // Take only positive frequencies (0..Fs/2)
    let spectrum_dbfs: Vec<f64> = buffer[..=half]
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut mag = c.norm() * norm;
            // Scale positive frequencies by 2 (except DC and Nyquist) to account for folded negative energy
            if i > 0 && i < half {
                mag *= 2.0;
            }
            20.0 * mag.max(1e-15).log10()
        })
        .collect();

    let freq_axis: Vec<f64> = (0..=half)
        .map(|i| i as f64 * sample_rate_mhz / n as f64)
        .collect();

    (spectrum_dbfs, freq_axis)
}

/// Compute single-sided (positive frequency only) power spectrum (default Hanning window).
pub fn compute_spectrum_positive(
    samples: &[Complex<f64>],
    fft_size: usize,
    sample_rate_mhz: f64,
) -> (Vec<f64>, Vec<f64>) {
    compute_spectrum_positive_with_window(samples, fft_size, sample_rate_mhz, FftWindow::Hanning)
}

/// Sample wideband physical real voltage signal v(t) at the ADC tile sample rate Fs.
/// Uses a high-quality windowed sinc anti-aliasing interpolator to eliminate fractional
/// sample rate resampling artifacts (spurs).
pub fn sample_adc_at_tile_rate(
    wideband_samples: &[Complex<f64>],
    sim_fs_mhz: f64,
    tile_fs_mhz: f64,
) -> Vec<Complex<f64>> {
    if tile_fs_mhz <= 0.0 || wideband_samples.is_empty() {
        return Vec::new();
    }
    let ratio = sim_fs_mhz / tile_fs_mhz;
    let num_samples = (wideband_samples.len() as f64 / ratio).floor() as usize;
    let mut sampled = Vec::with_capacity(num_samples);

    let k_radius = 16isize; // 32-tap windowed sinc filter for >100 dB spur rejection
    let len = wideband_samples.len() as isize;

    for n in 0..num_samples {
        let sample_pos = n as f64 * ratio;
        let center_idx = sample_pos.floor() as isize;

        let mut val = 0.0;
        let mut weight_sum = 0.0;

        for k in (center_idx - k_radius)..=(center_idx + k_radius) {
            if k >= 0 && k < len {
                let dx = sample_pos - k as f64;
                let abs_dx = dx.abs();

                let sinc = if abs_dx < 1e-9 {
                    1.0
                } else {
                    (PI * dx).sin() / (PI * dx)
                };

                let norm_x = abs_dx / (k_radius as f64 + 1.0);
                if norm_x < 1.0 {
                    // Blackman-Harris window
                    let w = 0.35875
                        + 0.48829 * (PI * norm_x).cos()
                        + 0.14128 * (2.0 * PI * norm_x).cos()
                        + 0.01168 * (3.0 * PI * norm_x).cos();
                    let weight = sinc * w;
                    val += wideband_samples[k as usize].re * weight;
                    weight_sum += weight;
                }
            }
        }

        let final_v = if weight_sum.abs() > 1e-9 {
            val / weight_sum
        } else {
            0.0
        };

        sampled.push(Complex::new(final_v, 0.0));
    }

    sampled
}

/// Apply Quadrature Modulation Correction (QMC) to complex samples.
///
/// This models the XRFdc QMC block which corrects I/Q gain imbalance,
/// phase skew, and DC offset post-mixer.
pub fn apply_qmc(
    samples: &[Complex<f64>],
    qmc: &crate::rfdc::QmcSettings,
) -> Vec<Complex<f64>> {
    // No-op passthrough when settings are at defaults
    if (qmc.gain - 1.0).abs() < 1e-12 && qmc.phase.abs() < 1e-12 && qmc.offset.abs() < 1e-12 {
        return samples.to_vec();
    }

    let phase_rad = qmc.phase * PI / 180.0;
    let cos_p = phase_rad.cos();
    let sin_p = phase_rad.sin();
    let g = qmc.gain;

    samples
        .iter()
        .map(|&s| {
            let i_out = g * (s.re * cos_p - s.im * sin_p) + qmc.offset;
            let q_out = g * (s.re * sin_p + s.im * cos_p);
            Complex::new(i_out, q_out)
        })
        .collect()
}

/// Apply the DDC mixer to time-domain samples.
///
/// `samples`: complex input samples
/// `settings`: MixerSettings from the block configuration
/// `nco_freq_mhz`: resolved NCO frequency in MHz (after zone wrap/flip)
/// `sim_fs_mhz`: sampling rate of input samples in MHz (wideband simulation rate)
/// `tile_fs_mhz`: ADC tile sampling rate in MHz
/// `scale`: FineMixerScale factor (1.0 or 0.7071)
pub fn apply_mixer(
    samples: &[Complex<f64>],
    settings: &crate::rfdc::MixerSettings,
    nco_freq_mhz: f64,
    sim_fs_mhz: f64,
    tile_fs_mhz: f64,
    scale: f64,
) -> Vec<Complex<f64>> {
    // NCO phase offset, in degrees, as configured on the block.
    let phase0 = settings.phase_offset * PI / 180.0;

    match settings.mixer_type {
        MixerType::Off => samples.to_vec(),
        MixerType::Coarse => {
            let coarse_shift_mhz = match settings.coarse_mix_freq {
                CoarseMixFreq::FsOver4 => 0.25 * tile_fs_mhz,
                CoarseMixFreq::MinusFsOver4 => -0.25 * tile_fs_mhz,
                CoarseMixFreq::FsOver2 => 0.5 * tile_fs_mhz,
                CoarseMixFreq::Bypass | CoarseMixFreq::Off => 0.0,
            };
            if coarse_shift_mhz.abs() < 1e-12 {
                return samples.to_vec();
            }
            let omega = -2.0 * PI * coarse_shift_mhz / sim_fs_mhz;
            samples
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let angle = omega * i as f64 - phase0;
                    s * Complex::new(angle.cos(), angle.sin()) * scale
                })
                .collect()
        }
        MixerType::Fine => {
            let omega = -2.0 * PI * nco_freq_mhz / sim_fs_mhz;
            // Real R2C quadrature mixing: I = x[n]·cos(ωn), Q = -x[n]·sin(ωn) for a real input,
            // which maps perfectly to multiplying the complex sample (where im=0) by Complex::new(cos, sin).
            samples
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let angle = omega * i as f64 - phase0;
                    s * Complex::new(angle.cos(), angle.sin()) * scale
                })
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------
// DDC decimation filter chain
// ---------------------------------------------------------------------------

/// Passband edge of the DDC decimation response, as a fraction of the *output* sample rate.
///
/// The RFdc decimation filters are flat across the inner 80% of the output Nyquist band —
/// i.e. |f| ≤ 0.4·Fout for complex output — and reject everything that would alias into it.
/// 0.4–0.5·Fout is the filter transition band, which is why the usable DDC bandwidth is
/// quoted as 80% of the output rate rather than the full Nyquist span.
pub const DDC_PASSBAND_FRAC: f64 = 0.4;

/// Alias rejection of the decimation chain, in dB.
pub const DDC_STOPBAND_DB: f64 = 90.0;

/// Modified Bessel function of the first kind, order 0 — used by the Kaiser window.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let quarter_x_sq = x * x / 4.0;
    for k in 1..80 {
        term *= quarter_x_sq / (k as f64 * k as f64);
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

/// Design a linear-phase Kaiser-windowed-sinc low-pass FIR.
///
/// `fp` and `fst` are the passband and stopband edges in cycles/sample of the filter's input
/// rate, and `atten_db` is the required stopband attenuation. Taps are normalised to unity
/// DC gain and the length is forced odd so the group delay is an integer sample count.
fn design_kaiser_lowpass(fp: f64, fst: f64, atten_db: f64) -> Vec<f64> {
    let transition = (fst - fp).max(1e-6);
    let beta = if atten_db > 50.0 {
        0.1102 * (atten_db - 8.7)
    } else if atten_db > 21.0 {
        0.5842 * (atten_db - 21.0).powf(0.4) + 0.07886 * (atten_db - 21.0)
    } else {
        0.0
    };
    let mut num_taps =
        ((atten_db - 8.0) / (2.285 * 2.0 * PI * transition)).ceil() as usize + 1;
    num_taps = num_taps.clamp(3, 2049) | 1;

    let half = (num_taps / 2) as isize;
    let fc = 0.5 * (fp + fst); // -6 dB point, midway through the transition
    let i0_beta = bessel_i0(beta);

    let mut taps = Vec::with_capacity(num_taps);
    let mut sum = 0.0;
    for i in 0..num_taps as isize {
        let m = i - half;
        let sinc = if m == 0 {
            2.0 * fc
        } else {
            (2.0 * PI * fc * m as f64).sin() / (PI * m as f64)
        };
        let r = m as f64 / half as f64;
        let w = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0_beta;
        let tap = sinc * w;
        taps.push(tap);
        sum += tap;
    }
    if sum.abs() > 1e-12 {
        for tap in &mut taps {
            *tap /= sum;
        }
    }
    taps
}

/// Split an RFdc decimation factor into the per-stage factors of the hardware cascade.
///
/// The ×2 halfband stages run first, at the highest rates where their transition band is
/// widest and cheapest; the ×3/×5 stage runs last.
fn decimation_stage_factors(factor: u32) -> Vec<usize> {
    let mut remaining = factor;
    let mut stages = Vec::new();
    while remaining.is_multiple_of(2) {
        stages.push(2usize);
        remaining /= 2;
    }
    let mut odd_stages = Vec::new();
    for p in [3u32, 5, 7] {
        while remaining.is_multiple_of(p) {
            odd_stages.push(p as usize);
            remaining /= p;
        }
    }
    if remaining > 1 {
        odd_stages.push(remaining as usize);
    }
    stages.extend(odd_stages);
    stages
}

/// One decimate-by-`factor` stage of the chain.
struct DecimationStage {
    taps: Vec<f64>,
    factor: usize,
    /// Input rate of this stage, as a fraction of the chain input rate.
    input_rate: f64,
}

/// The full cascade for one decimation factor.
pub struct DecimationChain {
    stages: Vec<DecimationStage>,
    /// Output samples to discard so only fully settled samples are returned.
    pub warmup_out_samples: usize,
}

/// Build the decimation cascade for `factor`.
///
/// Every stage is a halfband-geometry low-pass: its −6 dB point sits at half the stage's
/// own output rate, with the transition band placed symmetrically so the composite passband
/// edge lands at `DDC_PASSBAND_FRAC` of the final output rate. Content between a stage's
/// output Nyquist and (output rate − passband edge) folds outside the final passband and is
/// removed by later stages, so each stage only has to reject what would land back in band —
/// which is how the hardware halfband cascade achieves its rejection cheaply.
fn build_decimation_chain(factor: u32) -> DecimationChain {
    let stage_factors = decimation_stage_factors(factor);
    let final_out_rate = 1.0 / factor as f64;
    let f_pass = DDC_PASSBAND_FRAC * final_out_rate;

    let mut stages: Vec<DecimationStage> = Vec::new();
    let mut rate = 1.0_f64;
    for &m in &stage_factors {
        let out_rate = rate / m as f64;
        let fp = f_pass / rate;
        let fst = ((out_rate - f_pass) / rate).max(fp + 1e-6);
        stages.push(DecimationStage {
            taps: design_kaiser_lowpass(fp, fst, DDC_STOPBAND_DB),
            factor: m,
            input_rate: rate,
        });
        rate = out_rate;
    }

    // Propagate the filter group delay through the cascade to get the settling time,
    // expressed in final output samples.
    let mut warmup = 0usize;
    for stage in &stages {
        let half = stage.taps.len() / 2;
        warmup = (warmup + half).div_ceil(stage.factor);
    }

    DecimationChain {
        stages,
        warmup_out_samples: warmup,
    }
}

thread_local! {
    static DECIM_CHAINS: RefCell<std::collections::HashMap<u32, std::rc::Rc<DecimationChain>>> =
        RefCell::new(std::collections::HashMap::new());
}

/// Fetch (or build and cache) the decimation cascade for `factor`.
pub fn decimation_chain(factor: u32) -> std::rc::Rc<DecimationChain> {
    DECIM_CHAINS.with(|cache| {
        cache
            .borrow_mut()
            .entry(factor)
            .or_insert_with(|| std::rc::Rc::new(build_decimation_chain(factor)))
            .clone()
    })
}

impl DecimationChain {
    /// Composite magnitude response in dB at `freq_norm` cycles/sample of the chain input.
    pub fn response_db(&self, freq_norm: f64) -> f64 {
        let mut mag = 1.0_f64;
        for stage in &self.stages {
            // Frequency normalised to this stage's own input rate.
            let w = 2.0 * PI * freq_norm / stage.input_rate;
            let half = (stage.taps.len() / 2) as isize;
            // Sum the symmetric linear-phase response about the centre tap.
            let mut acc = 0.0;
            for (i, &tap) in stage.taps.iter().enumerate() {
                let m = i as isize - half;
                acc += tap * (w * m as f64).cos();
            }
            mag *= acc.abs();
        }
        20.0 * mag.max(1e-12).log10()
    }

    fn run_stage(input: &[Complex<f64>], taps: &[f64], m: usize) -> Vec<Complex<f64>> {
        if input.is_empty() {
            return Vec::new();
        }
        let half = taps.len() / 2;
        let out_len = input.len() / m;
        let mut out = Vec::with_capacity(out_len);
        for idx in 0..out_len {
            let center = idx * m;
            let lo = center.saturating_sub(half);
            let hi = (center + half).min(input.len() - 1);
            // Taps are indexed relative to the centre sample, so the window is clipped at
            // the buffer edges rather than wrapping.
            let tap_offset = half - (center - lo);
            let mut acc = Complex::new(0.0, 0.0);
            for (x, &tap) in input[lo..=hi].iter().zip(&taps[tap_offset..]) {
                acc += x * tap;
            }
            out.push(acc);
        }
        out
    }

    /// Filter and downsample, returning only fully settled output samples.
    pub fn apply(&self, samples: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let mut current = samples.to_vec();
        for stage in &self.stages {
            current = Self::run_stage(&current, &stage.taps, stage.factor);
        }
        if self.warmup_out_samples < current.len() {
            current.drain(..self.warmup_out_samples);
        }
        current
    }
}

/// FFT length used for the ADC-rate spectra (input, folded, post-mixer).
pub const ANALYSIS_FFT_SIZE: usize = 2048;

/// FFT length used for the DDC output spectrum at a given decimation factor.
pub fn output_fft_size(decimation: u32) -> usize {
    (ANALYSIS_FFT_SIZE / decimation.max(1) as usize).max(64)
}

/// Round `n` up to the next 5-smooth number (only factors of 2, 3 and 5).
///
/// The wideband buffer feeds an FFT for the analog bandwidth roll-off, and an awkward length
/// like 31·571 pushes rustfft onto its slow general-radix path. Rounding up costs a couple of
/// percent in samples and buys back several milliseconds a frame.
pub fn next_smooth_size(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut best = usize::MAX;
    let mut p2 = 1usize;
    while p2 < n.saturating_mul(2) {
        let mut p23 = p2;
        while p23 < n.saturating_mul(2) {
            let mut cand = p23;
            while cand < n {
                cand = match cand.checked_mul(5) {
                    Some(v) => v,
                    None => break,
                };
            }
            if cand >= n {
                best = best.min(cand);
            }
            p23 = match p23.checked_mul(3) {
                Some(v) => v,
                None => break,
            };
        }
        p2 = match p2.checked_mul(2) {
            Some(v) => v,
            None => break,
        };
    }
    if best == usize::MAX { n } else { best }
}

/// Number of ADC-rate samples needed to fill both the ADC-rate spectra and the DDC output
/// spectrum at `decimation`, including the decimation chain's settling time.
///
/// Callers generating a wideband waveform must scale this by their oversampling ratio
/// (simulation rate / Fs), otherwise the FFTs silently run short and lose resolution.
pub fn required_tile_samples(decimation: u32) -> usize {
    let f = decimation.max(1) as usize;
    let for_output = output_fft_size(decimation) * f;
    let settling = if f > 1 {
        decimation_chain(decimation).warmup_out_samples * f
    } else {
        0
    };
    // Headroom absorbs the resampler's fractional-rate truncation.
    ANALYSIS_FFT_SIZE.max(for_output) + settling + 128
}

/// Apply the DDC decimation filter chain and downsample by `factor`.
///
/// Models the RFdc's cascaded halfband/×3/×5 decimation filters: flat across
/// |f| ≤ `DDC_PASSBAND_FRAC`·Fout with `DDC_STOPBAND_DB` rejection of everything that would
/// otherwise alias into that band. A factor of 1 bypasses the chain, as it does in hardware.
pub fn apply_decimation(samples: &[Complex<f64>], factor: u32) -> Vec<Complex<f64>> {
    if factor <= 1 || samples.is_empty() {
        return samples.to_vec();
    }
    decimation_chain(factor).apply(samples)
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_single_tone_peak_location() {
        // Generate a 100 MHz tone sampled at 1000 MHz
        let n = 1024;
        let fs = 1000.0;
        let f_tone = 100.0;
        let samples: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let angle = 2.0 * PI * f_tone * t;
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();

        let (spectrum, freq_axis) = compute_spectrum_positive(&samples, n, fs);

        // Find the peak
        let (peak_idx, _) = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let peak_freq = freq_axis[peak_idx];
        assert!(
            (peak_freq - f_tone).abs() < fs / n as f64,
            "Peak at {peak_freq} MHz, expected ~{f_tone} MHz"
        );
    }

    #[test]
    fn nyquist_folding_second_zone() {
        // Fold by actually sampling, which is what the pipeline does: a real 1300 MHz tone
        // sampled at Fs = 2000 MHz sits in zone 2 and must appear mirrored at 700 MHz.
        let sim_fs = 12000.0;
        let tile_fs = 2000.0;
        let n = 8192;

        let wideband: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let phi = 2.0 * PI * 1300.0 * i as f64 / sim_fs;
                Complex::new(phi.cos(), 0.0)
            })
            .collect();

        let sampled = sample_adc_at_tile_rate(&wideband, sim_fs, tile_fs);
        let (folded, folded_freq) = compute_spectrum_positive(&sampled, 2048, tile_fs);

        let (peak_idx, _) = folded
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let peak_freq = folded_freq[peak_idx];
        assert!(
            (peak_freq - 700.0).abs() < 2.0 * tile_fs / 2048.0,
            "Folded peak at {peak_freq} MHz, expected ~700 MHz"
        );
    }

    #[test]
    fn coarse_mixer_fs_over_4() {
        let n = 256;
        let fs = 1000.0;
        let samples: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let angle = 2.0 * PI * 250.0 * t; // 250 MHz = Fs/4
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();

        let coarse_ms = crate::rfdc::MixerSettings {
            mixer_type: crate::rfdc::MixerType::Coarse,
            mixer_mode: crate::rfdc::MixerMode::RealToIq,
            coarse_mix_freq: CoarseMixFreq::FsOver4,
            freq: 0.0,
            phase_offset: 0.0,
            fine_mixer_scale: crate::rfdc::FineMixerScale::Auto,
            event_source: crate::rfdc::EventSource::Tile,
        };
        let mixed = apply_mixer(&samples, &coarse_ms, 0.0, fs, fs, 1.0);
        assert_eq!(mixed.len(), n);

        // After mixing with -Fs/4, a tone at Fs/4 should move to ~DC
        let (spectrum, freq_axis) = compute_spectrum_positive(&mixed, n, fs);
        let (peak_idx, _) = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let peak_freq = freq_axis[peak_idx];
        assert!(
            peak_freq.abs() < fs / n as f64 * 2.0,
            "After Fs/4 mix, peak should be near DC, got {peak_freq} MHz"
        );
    }

    #[test]
    fn coarse_mixer_wideband_rate() {
        // Test coarse mixing when simulation rate (10,000 MHz) != ADC tile rate (4,000 MHz)
        let sim_fs = 10000.0;
        let tile_fs = 4000.0; // Fs/4 = 1000 MHz
        let n = 1024;

        // Generate a 1000 MHz tone sampled at 10,000 MHz
        let samples: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 / sim_fs;
                let angle = 2.0 * PI * 1000.0 * t; // 1000 MHz tone (= tile_fs / 4)
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();

        // Apply CoarseMix Fs/4 (should downshift by tile_fs/4 = 1000 MHz)
        let coarse_ms = crate::rfdc::MixerSettings {
            mixer_type: crate::rfdc::MixerType::Coarse,
            mixer_mode: crate::rfdc::MixerMode::RealToIq,
            coarse_mix_freq: CoarseMixFreq::FsOver4,
            freq: 0.0,
            phase_offset: 0.0,
            fine_mixer_scale: crate::rfdc::FineMixerScale::Auto,
            event_source: crate::rfdc::EventSource::Tile,
        };
        let mixed = apply_mixer(&samples, &coarse_ms, 0.0, sim_fs, tile_fs, 1.0);

        let (spectrum, freq_axis) = compute_spectrum_positive(&mixed, n, sim_fs);
        let (peak_idx, _) = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let peak_freq = freq_axis[peak_idx];
        assert!(
            peak_freq.abs() < sim_fs / n as f64 * 2.0,
            "1000 MHz tone mixed with Fs/4 coarse mixer (tile Fs=4000 MHz) should land near DC at sim rate 10000 MHz, got {peak_freq} MHz"
        );
    }

    #[test]
    fn decimation_reduces_sample_count() {
        let samples: Vec<Complex<f64>> = (0..1024)
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();

        // 1024/8 = 128 raw output samples, minus the chain's settling time, which is
        // discarded so callers only ever see fully settled samples.
        let warmup = 128 - apply_decimation(&samples, 8).len();
        assert!(
            warmup > 0 && warmup < 32,
            "×8 settling should cost a handful of output samples, got {warmup}"
        );
        assert_eq!(apply_decimation(&samples, 1).len(), 1024, "×1 must bypass");
    }

    #[test]
    fn decimation_rejects_out_of_band_aliases() {
        // A tone that would alias into the DDC passband must be rejected, not folded in.
        // This is what a real halfband decimation cascade guarantees.
        let n = 16384;
        let factor = 8u32;
        let f_out = 1.0 / factor as f64;

        let tone = |fnorm: f64| -> f64 {
            let s: Vec<Complex<f64>> = (0..n)
                .map(|i| {
                    let a = 2.0 * PI * fnorm * i as f64;
                    Complex::new(a.cos(), a.sin())
                })
                .collect();
            let out = apply_decimation(&s, factor);
            let mid = &out[out.len() / 8..7 * out.len() / 8];
            let p: f64 = mid.iter().map(|c| c.norm_sqr()).sum::<f64>() / mid.len() as f64;
            10.0 * p.max(1e-30).log10()
        };

        let reference = tone(1e-6);

        // In-band: must pass essentially untouched across the usable bandwidth.
        let in_band = tone(DDC_PASSBAND_FRAC * f_out) - reference;
        assert!(
            in_band.abs() < 0.5,
            "passband edge should be flat, got {in_band:.2} dB"
        );

        // Anything above (1 − 0.4)·Fout folds back inside the usable band, as does anything
        // near a multiple of Fout. All of it must be suppressed. Content between 0.5·Fout
        // and 0.6·Fout folds into the 0.4–0.5·Fout transition band instead, where neither
        // this model nor the hardware guarantees rejection.
        for &f in &[
            (1.0 - DDC_PASSBAND_FRAC) * f_out + 0.002,
            0.999 * f_out,
            f_out,
            2.0 * f_out,
            3.0 * f_out,
        ] {
            let leak = tone(f) - reference;
            assert!(
                leak < -80.0,
                "alias at {f:.4}·Fs leaked into the DDC band at {leak:.1} dB (need < -80)"
            );
        }
    }

    #[test]
    fn odd_sample_count_fft_shift_does_not_panic() {
        let samples: Vec<Complex<f64>> = (0..683)
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();

        let (spec, freq) = compute_spectrum(&samples, 2048, 1000.0);
        assert!(!spec.is_empty());
        assert_eq!(spec.len(), freq.len());
        assert_eq!(spec.len() % 2, 0);
    }

    #[test]
    fn window_functions_validity() {
        let samples = vec![Complex::new(1.0, 0.0); 128];
        for win in FftWindow::ALL {
            let mut buf = samples.clone();
            win.apply(&mut buf);
            assert_eq!(buf.len(), 128);
            if win == FftWindow::Hanning || win == FftWindow::BlackmanHarris {
                assert!(buf[0].norm() < 1e-3);
            }
        }
    }

    #[test]
    fn adc_non_idealities_hd2_hd3() {
        let samples: Vec<Complex<f64>> = (0..512)
            .map(|i| {
                let phi = 2.0 * PI * 0.1 * i as f64;
                Complex::new(phi.cos(), 0.0) // Real voltage
            })
            .collect();

        let mut non = crate::rfdc::AdcNonIdealities::default();
        non.enabled = true;
        non.hd2_dbc = -30.0;
        non.hd3_dbc = -40.0;

        let distorted = apply_analog_non_idealities(&samples, &non);
        assert_eq!(distorted.len(), samples.len());
        // Distorted samples should differ from pure sine
        let diff: f64 = samples.iter().zip(distorted.iter()).map(|(a, b)| (a - b).norm()).sum();
        assert!(diff > 1.0);
    }

    #[test]
    fn process_adc_block_auto_tuned_higher_zone() {
        use crate::rfdc::{AdcTile, MixerType, MixerMode};
        use crate::signal::{SignalGenerator, Tone, ToneModulation};

        // Recreate user scenario: Fs = 1.96608 GSPS (1966.08 MHz), Target = 2400 MHz (Zone 3)
        let mut tile = AdcTile::new(0);
        tile.sample_rate_gsps = 1.96608;

        let auto_res = tile.auto_tune(2400.0);
        tile.blocks[0].nyquist_zone = auto_res.nyquist_zone;
        tile.blocks[0].planner_zone = auto_res.zone_index;

        tile.blocks[0].mixer_settings.mixer_type = MixerType::Fine;
        tile.blocks[0].mixer_settings.mixer_mode = MixerMode::RealToIq;
        tile.blocks[0].mixer_settings.freq = auto_res.nco_freq_mhz;

        let block = tile.blocks[0].clone();

        // Generate 2400 MHz tone
        let mut sig_gen = SignalGenerator::default();
        sig_gen.tones = vec![Tone {
            frequency_mhz: 2400.0,
            amplitude_dbfs: -6.0,
            phase_deg: 0.0,
            modulation: ToneModulation::Cw,
            bandwidth_mhz: 0.0,
        }];
        sig_gen.noise_enabled = false;

        let sim_fs = 10000.0;
        let input_samples = sig_gen.generate(1024, sim_fs);
        let processed = process_adc_block(&input_samples, sim_fs, &block, &tile, Some(&input_samples), None);

        // Find peak in output spectrum
        let peaks = crate::ui::spectrum_view::find_spectral_peaks(
            &processed.output_spectrum_dbfs,
            &processed.output_freq_axis_mhz,
            -20.0,
        );

        assert!(!peaks.is_empty(), "Should detect peak near 0 Hz baseband");
        assert!(
            peaks[0].freq_mhz.abs() < 10.0,
            "Auto-tuned 2400 MHz tone in Zone 3 should land at 0 Hz baseband, got {:.1} MHz",
            peaks[0].freq_mhz
        );
        assert!(
            (peaks[0].mag_dbfs - (-12.0)).abs() < 3.0,
            "Peak magnitude should be close to -12 dBFS due to 6 dB drop from real-to-complex quadrature mixing, got {:.1} dBFS",
            peaks[0].mag_dbfs
        );
    }

    #[test]
    fn other_nyquist_zone_interferes_if_unfiltered() {
        use crate::rfdc::{AdcTile, MixerType, MixerMode};
        use crate::signal::{SignalGenerator, Tone, ToneModulation};

        let mut tile = AdcTile::new(0);
        tile.sample_rate_gsps = 1.96608;

        let auto_res = tile.auto_tune(2400.0);
        tile.blocks[0].nyquist_zone = auto_res.nyquist_zone;
        tile.blocks[0].planner_zone = auto_res.zone_index;
        tile.blocks[0].mixer_settings.mixer_type = MixerType::Fine;
        tile.blocks[0].mixer_settings.mixer_mode = MixerMode::RealToIq;
        tile.blocks[0].mixer_settings.freq = auto_res.nco_freq_mhz;

        let mut sig_gen = SignalGenerator::default();
        sig_gen.tones = vec![
            Tone {
                frequency_mhz: 2400.0,
                amplitude_dbfs: -6.0,
                phase_deg: 0.0,
                modulation: ToneModulation::Cw,
                bandwidth_mhz: 0.0,
            },
            Tone {
                frequency_mhz: 433.92,
                amplitude_dbfs: -6.0,
                phase_deg: 0.0,
                modulation: ToneModulation::Cw,
                bandwidth_mhz: 0.0,
            },
        ];
        sig_gen.noise_enabled = false;

        let sim_fs = 15000.0;
        let input_samples = sig_gen.generate(1024, sim_fs);
        let processed = process_adc_block(&input_samples, sim_fs, &tile.blocks[0], &tile, None, None);

        let peaks = crate::ui::spectrum_view::find_spectral_peaks(
            &processed.output_spectrum_dbfs,
            &processed.output_freq_axis_mhz,
            -20.0,
        );

        assert!(!peaks.is_empty());
        assert!(peaks[0].freq_mhz.abs() < 10.0, "Both signals construct peak at 0 Hz baseband");
    }

    #[test]
    fn out_of_band_interferer_does_not_alias_into_ddc() {
        use crate::rfdc::AdcTile;
        use crate::signal::{SignalGenerator, Tone, ToneModulation};

        // Fs = 4000 MHz, tuned to 2600 MHz (zone 2), ×8 decimation → ±250 MHz output band.
        // A second tone 300 MHz away is outside that band, so the PL must not see it.
        let mut tile = AdcTile::new(0);
        let mut block = tile.blocks[0].clone();
        block.auto_tune(tile.sample_rate_gsps, 2600.0);
        block.decimation = crate::rfdc::DecimationFactor::X8;
        tile.blocks[0] = block.clone();

        let mut sig_gen = SignalGenerator::default();
        sig_gen.tones = vec![
            Tone { frequency_mhz: 2600.0, amplitude_dbfs: -6.0, phase_deg: 0.0,
                   bandwidth_mhz: 0.0, modulation: ToneModulation::RealCosine },
            Tone { frequency_mhz: 2900.0, amplitude_dbfs: -6.0, phase_deg: 0.0,
                   bandwidth_mhz: 0.0, modulation: ToneModulation::RealCosine },
        ];
        sig_gen.noise_enabled = false;

        let sim_fs = 15000.0;
        let input = sig_gen.generate(16384, sim_fs);
        let processed = process_adc_block(&input, sim_fs, &block, &tile, None, None);

        let peaks = crate::ui::spectrum_view::find_spectral_peaks(
            &processed.output_spectrum_dbfs,
            &processed.output_freq_axis_mhz,
            -200.0,
        );
        assert!(!peaks.is_empty());

        let wanted = peaks[0];
        assert!(
            wanted.freq_mhz.abs() < 10.0,
            "tuned tone should sit at DC, got {:.1} MHz",
            wanted.freq_mhz
        );

        // The interferer folds to −200 MHz if the decimation filter leaks.
        let worst_alias = peaks
            .iter()
            .filter(|p| p.freq_mhz.abs() > 50.0)
            .map(|p| p.mag_dbfs)
            .fold(f64::NEG_INFINITY, f64::max);
        let rejection = worst_alias - wanted.mag_dbfs;
        assert!(
            rejection < -70.0,
            "out-of-band tone aliased into the DDC output at {rejection:.1} dBc (need < -70)"
        );
    }

    #[test]
    fn pre_adc_and_folded_levels_agree() {
        use crate::rfdc::{AdcTile, DecimationFactor, MixerType};
        use crate::signal::{SignalGenerator, Tone, ToneModulation};

        // The pre-ADC plot and the folded plot must report the same level for the same
        // signal, for a complex-exponential source as well as a real one — the ADC samples
        // the real voltage either way.
        for modulation in [ToneModulation::Cw, ToneModulation::RealCosine] {
            let tile = AdcTile::new(0);
            let mut block = tile.blocks[0].clone();
            block.mixer_settings.mixer_type = MixerType::Off;
            block.decimation = DecimationFactor::X1;

            let mut sig_gen = SignalGenerator::default();
            sig_gen.tones = vec![Tone {
                frequency_mhz: 1024.0,
                amplitude_dbfs: 0.0,
                phase_deg: 0.0,
                bandwidth_mhz: 0.0,
                modulation: modulation.clone(),
            }];
            sig_gen.noise_enabled = false;

            let input = sig_gen.generate(8192, 15000.0);
            let processed = process_adc_block(&input, 15000.0, &block, &tile, None, None);

            let peak_of = |s: &[f64]| s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let pre = peak_of(&processed.input_spectrum_dbfs);
            let folded = peak_of(&processed.folded_spectrum_dbfs);
            assert!(
                (pre - folded).abs() < 1.0,
                "{modulation:?}: pre-ADC {pre:.2} dBFS vs folded {folded:.2} dBFS"
            );
        }
    }

    #[test]
    fn analog_bandwidth_attenuates_high_nyquist_zones() {
        use crate::rfdc::{AdcTile, DecimationFactor, MixerType};
        use crate::signal::{SignalGenerator, Tone, ToneModulation};

        let tile = AdcTile::new(0); // Fs = 4000 MHz
        let mut block = tile.blocks[0].clone();
        block.mixer_settings.mixer_type = MixerType::Off;
        block.decimation = DecimationFactor::X1;
        block.analog_front_end.enabled = true;
        block.analog_front_end.bandwidth_ghz = 6.0;
        block.analog_front_end.order = 2;

        let folded_peak = |f_rf: f64| -> f64 {
            let mut sig_gen = SignalGenerator::default();
            sig_gen.tones = vec![Tone {
                frequency_mhz: f_rf,
                amplitude_dbfs: 0.0,
                phase_deg: 0.0,
                bandwidth_mhz: 0.0,
                modulation: ToneModulation::RealCosine,
            }];
            sig_gen.noise_enabled = false;
            let input = sig_gen.generate(8192, 15000.0);
            let processed = process_adc_block(&input, 15000.0, &block, &tile, None, None);
            processed
                .folded_spectrum_dbfs
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        };

        // Zone 1 sits well inside the analog passband; zone 4 is past the −3 dB corner and
        // must fold in measurably weaker rather than at full scale.
        let zone1 = folded_peak(500.0);
        let zone4 = folded_peak(7000.0);
        assert!(zone1 > -1.0, "in-band tone should be near full scale, got {zone1:.2}");
        assert!(
            zone4 < zone1 - 3.0,
            "zone 4 alias should be attenuated by the analog input BW: zone1 {zone1:.2} vs zone4 {zone4:.2}"
        );
    }

    #[test]
    fn harmonic_distortion_hits_requested_dbc() {
        // A -40 dBc HD2 request must actually produce a second harmonic 40 dB down.
        let n = 16384;
        let fs = 4000.0;
        let f0 = 300.0;
        let samples: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let phi = 2.0 * PI * f0 * i as f64 / fs;
                Complex::new(phi.cos(), 0.0)
            })
            .collect();

        let mut non = crate::rfdc::AdcNonIdealities::default();
        non.enabled = true;
        non.hd2_dbc = -40.0;
        non.hd3_dbc = -50.0;

        let distorted = apply_analog_non_idealities(&samples, &non);
        let (spec, freq) = compute_spectrum_positive(&distorted, n, fs);

        let level_at = |target: f64| -> f64 {
            let idx = freq
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (*a - target).abs().partial_cmp(&(*b - target).abs()).unwrap()
                })
                .unwrap()
                .0;
            spec[idx - 2..=idx + 2].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };

        let fundamental = level_at(f0);
        let hd2 = level_at(2.0 * f0) - fundamental;
        let hd3 = level_at(3.0 * f0) - fundamental;
        assert!((hd2 + 40.0).abs() < 1.0, "HD2 should be -40 dBc, got {hd2:.2}");
        assert!((hd3 + 50.0).abs() < 1.0, "HD3 should be -50 dBc, got {hd3:.2}");
    }

    #[test]
    fn enob_sets_the_noise_floor() {
        // The broadband floor must follow SNR = 6.02·ENOB + 1.76 dB.
        let n = 32768;
        let mut non = crate::rfdc::AdcNonIdealities::default();
        non.enabled = true;
        non.quantization_bits = 14;

        for enob in [8.0_f64, 11.5] {
            non.enob = enob;
            // Zero input: measure the noise power the converter contributes on its own.
            let quiet = vec![Complex::new(0.0, 0.0); n];
            let (noisy, _) = apply_digital_non_idealities(&quiet, &non);
            let pwr: f64 = noisy.iter().map(|c| c.re * c.re).sum::<f64>() / n as f64;
            // Full-scale sine power is 0.5, so SNR = 10·log10(0.5 / noise power).
            let snr = 10.0 * (0.5 / pwr.max(1e-30)).log10();
            let expected = 6.02 * enob + 1.76;
            assert!(
                (snr - expected).abs() < 1.5,
                "ENOB {enob}: measured SNR {snr:.1} dB, expected {expected:.1} dB"
            );
        }
    }

    #[test]
    fn nco_phase_offset_rotates_the_baseband() {
        use crate::rfdc::{CoarseMixFreq, EventSource, FineMixerScale, MixerSettings, MixerType};
        use crate::rfdc::MixerMode as MM;

        let n = 1024;
        let fs = 4000.0;
        let f0 = 500.0;
        let samples: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let phi = 2.0 * PI * f0 * i as f64 / fs;
                Complex::new(phi.cos(), 0.0)
            })
            .collect();

        let mixer = |phase_deg: f64| {
            let ms = MixerSettings {
                mixer_type: MixerType::Fine,
                mixer_mode: MM::RealToIq,
                coarse_mix_freq: CoarseMixFreq::Off,
                freq: f0,
                phase_offset: phase_deg,
                fine_mixer_scale: FineMixerScale::OnePointZero,
                event_source: EventSource::Tile,
            };
            let mixed = apply_mixer(&samples, &ms, f0, fs, fs, 1.0);
            // Mean of the settled baseband term gives its phase.
            let sum: Complex<f64> = mixed.iter().sum();
            sum.arg()
        };

        let delta = (mixer(90.0) - mixer(0.0)).to_degrees();
        let delta = ((delta + 540.0) % 360.0) - 180.0;
        assert!(
            (delta.abs() - 90.0).abs() < 2.0,
            "a 90° NCO phase offset should rotate the baseband by 90°, got {delta:.1}°"
        );
    }

    #[test]
    fn nco_sign_follows_configured_zone() {
        // XRFdc tuning convention: an NCO frequency landing in the zone *opposite* the
        // block's configured Nyquist zone resolves negative; one matching the configured
        // zone's parity resolves positive.
        let fs = 4000.0;
        let odd = false;
        let even = true;
        for (requested, is_even_zone, expected, note) in [
            (2400.0_f64, odd, -1600.0_f64, "zone 2 (even) requested on an odd-zone block"),
            (2400.0, even, 1600.0, "zone 2 (even) requested on an even-zone block"),
            (5300.0, even, -1300.0, "zone 3 (odd) requested on an even-zone block"),
            (5300.0, odd, 1300.0, "zone 3 (odd) requested on an odd-zone block"),
        ] {
            let got = resolve_nco_freq(requested, fs, is_even_zone);
            assert!(
                (got - expected).abs() < 1e-9,
                "{note}: got {got}, expected {expected}"
            );
            // Magnitude is always the alias frequency, only the sign is conventional.
            let zone = (requested / (fs / 2.0)).floor() as u32 + 1;
            let alias = if zone % 2 == 0 {
                zone as f64 * fs / 2.0 - requested
            } else {
                requested - (zone as f64 - 1.0) * fs / 2.0
            };
            assert!((got.abs() - alias).abs() < 1e-9, "{note}: |NCO| != alias {alias}");
        }

        // Requests already inside +/-Fs/2 pass straight through, sign included. This is the
        // whole auto_tune path, which is why it is unaffected by the convention above.
        for &is_even in &[odd, even] {
            assert!((resolve_nco_freq(1400.0, fs, is_even) - 1400.0).abs() < 1e-9);
            assert!((resolve_nco_freq(-1600.0, fs, is_even) - (-1600.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn auto_tune_nco_lands_signal_at_dc_with_correct_sense() {
        use crate::rfdc::{AdcTile, DecimationFactor};
        use crate::signal::{SignalGenerator, Tone, ToneModulation};

        // Auto-tune emits the alias already signed for the zone, and stays inside +/-Fs/2 so
        // the wrap convention never applies. An RF tone above the tuned centre lands above DC.
        let mut tile = AdcTile::new(0); // Fs = 4000 MHz
        let mut block = tile.blocks[0].clone();
        let at = block.auto_tune(tile.sample_rate_gsps, 2400.0); // zone 2, NCO -1600
        assert_eq!(at.zone_index, 2);
        assert!((at.nco_freq_mhz - (-1600.0)).abs() < 1e-9);
        block.decimation = DecimationFactor::X8;
        tile.blocks[0] = block.clone();

        let baseband_of = |rf_mhz: f64| -> f64 {
            let mut sig_gen = SignalGenerator::default();
            sig_gen.tones = vec![Tone {
                frequency_mhz: rf_mhz,
                amplitude_dbfs: -6.0,
                phase_deg: 0.0,
                bandwidth_mhz: 0.0,
                modulation: ToneModulation::RealCosine,
            }];
            sig_gen.noise_enabled = false;
            let input = sig_gen.generate(16384, 15000.0);
            let processed = process_adc_block(&input, 15000.0, &block, &tile, None, None);
            let peaks = crate::ui::spectrum_view::find_spectral_peaks(
                &processed.output_spectrum_dbfs,
                &processed.output_freq_axis_mhz,
                -60.0,
            );
            assert!(!peaks.is_empty(), "no peak found for {rf_mhz} MHz");
            peaks[0].freq_mhz
        };

        let below = baseband_of(2350.0);
        let above = baseband_of(2450.0);
        assert!(
            (below + 50.0).abs() < 5.0,
            "RF 50 MHz below centre should land at -50 MHz, got {below:.1}"
        );
        assert!(
            (above - 50.0).abs() < 5.0,
            "RF 50 MHz above centre should land at +50 MHz, got {above:.1}"
        );
    }

    #[test]
    fn smooth_size_rounds_up_to_2_3_5_factors() {
        for n in [4096usize, 8161, 12345, 17701, 100_001] {
            let s = next_smooth_size(n);
            assert!(s >= n, "{s} must be >= {n}");
            // Must factor into 2s, 3s and 5s only.
            let mut r = s;
            for p in [2usize, 3, 5] {
                while r % p == 0 {
                    r /= p;
                }
            }
            assert_eq!(r, 1, "{s} (from {n}) is not 5-smooth");
            // And it must not overshoot badly.
            assert!(
                (s as f64) < 1.25 * n as f64,
                "{s} overshoots {n} by more than 25%"
            );
        }
        assert_eq!(next_smooth_size(1024), 1024, "already-smooth sizes are kept");
    }

    #[test]
    fn analog_bandwidth_filter_is_spectrally_clean() {
        // The roll-off is applied by FFT, which is a circular operation. Confirm it does not
        // leave a raised broadband floor or wrap-around spurs behind the tone.
        let afe = crate::rfdc::AnalogFrontEnd {
            enabled: true,
            bandwidth_ghz: 6.0,
            order: 3,
        };
        let n = 8192;
        let fs = 15000.0;
        let f0 = 2600.0;
        let samples: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let phi = 2.0 * PI * f0 * i as f64 / fs;
                Complex::new(phi.cos(), 0.0)
            })
            .collect();

        let filtered = apply_analog_bandwidth(&samples, fs, &afe);
        assert_eq!(filtered.len(), n);
        // Output must stay real.
        assert!(filtered.iter().all(|c| c.im == 0.0));

        let (spec, freq) = compute_spectrum_positive(&filtered, n, fs);
        let peak = spec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Expected in-band attenuation at 2600 MHz.
        let expected = afe.response_db(f0);
        assert!(
            (peak - expected).abs() < 1.0,
            "peak {peak:.2} dBFS, expected ~{expected:.2} dBFS"
        );

        // Nothing else within 80 dB of the tone.
        let worst_spur = spec
            .iter()
            .zip(freq.iter())
            .filter(|&(_, &f)| (f - f0).abs() > 100.0)
            .map(|(&m, _)| m)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            worst_spur < peak - 80.0,
            "circular filtering left a spur at {worst_spur:.1} dBFS vs tone {peak:.1} dBFS"
        );
    }

    #[test]
    fn clipping_overrange_flag() {
        let samples: Vec<Complex<f64>> = vec![Complex::new(1.5, 0.0), Complex::new(-1.5, 0.0), Complex::new(0.5, 0.0)];
        let non = crate::rfdc::AdcNonIdealities::default(); // default has enabled=false for spur/quant, but clip still applies
        let (processed, overrange) = apply_digital_non_idealities(&samples, &non);
        
        assert!(overrange);
        assert_eq!(processed[0].re, 1.0);
        assert_eq!(processed[1].re, -1.0);
        assert_eq!(processed[2].re, 0.5);
    }

    #[test]
    fn r2c_mixer_image_generation() {
        let sim_fs = 1000.0;
        let tile_fs = 1000.0;
        let n = 256;
        let f_in = 100.0;
        
        let samples: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let phi = 2.0 * PI * f_in * i as f64 / sim_fs;
                Complex::new(phi.cos(), 0.0) // Real tone
            })
            .collect();
            
        // Mix with 100 MHz NCO (shifts signal down by 100 MHz)
        let ms = crate::rfdc::MixerSettings {
            mixer_type: crate::rfdc::MixerType::Fine,
            mixer_mode: crate::rfdc::MixerMode::RealToIq,
            coarse_mix_freq: crate::rfdc::CoarseMixFreq::Off,
            freq: 100.0,
            phase_offset: 0.0,
            fine_mixer_scale: crate::rfdc::FineMixerScale::Auto,
            event_source: crate::rfdc::EventSource::Tile,
        };
        let mixed = apply_mixer(&samples, &ms, 100.0, sim_fs, tile_fs, 1.0);
        
        // We should have energy at DC (100 - 100 = 0) AND at -200 MHz (-100 - 100 = -200)
        let (spectrum, _freq) = compute_spectrum(&mixed, n, sim_fs);
        
        // Find DC and -200 MHz bins
        let dc_idx = n / 2;
        let image_idx = n / 2 - (200.0 / sim_fs * n as f64) as usize;
        
        assert!(spectrum[dc_idx] > -20.0, "Missing DC component from mixing");
        assert!(spectrum[image_idx] > -20.0, "Missing -2w image from real-to-complex mixing");
    }

    #[test]
    fn qmc_gain_offset() {
        let qmc = crate::rfdc::QmcSettings { gain: 2.0, phase: 0.0, offset: 0.5 };
        let samples = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];
        let result = apply_qmc(&samples, &qmc);

        // s[0]: I=1, Q=0 → I_out = 2*1 + 0.5 = 2.5, Q_out = 2*0 = 0
        assert!((result[0].re - 2.5).abs() < 1e-9);
        assert!(result[0].im.abs() < 1e-9);

        // s[1]: I=0, Q=1 → I_out = 2*(-1·0) + 0.5 = 2*0 + 0.5 = -1.5, wait:
        // I_out = gain * (I*cos(0) - Q*sin(0)) + offset = 2*(0 - 0) + 0.5 = 0.5
        // Q_out = gain * (I*sin(0) + Q*cos(0)) = 2*(0 + 1) = 2.0
        assert!((result[1].re - 0.5).abs() < 1e-9);
        assert!((result[1].im - 2.0).abs() < 1e-9);
    }

    #[test]
    fn qmc_phase_rotation() {
        // 90° rotation: cos(90°)=0, sin(90°)=1
        let qmc = crate::rfdc::QmcSettings { gain: 1.0, phase: 90.0, offset: 0.0 };
        let samples = vec![Complex::new(1.0, 0.0)];
        let result = apply_qmc(&samples, &qmc);

        // I_out = 1*(1*cos(90°) - 0*sin(90°)) = 1*(0) = 0
        // Q_out = 1*(1*sin(90°) + 0*cos(90°)) = 1*(1) = 1
        assert!(result[0].re.abs() < 1e-9, "90° QMC should zero I, got {}", result[0].re);
        assert!((result[0].im - 1.0).abs() < 1e-9, "90° QMC should put full signal in Q, got {}", result[0].im);
    }

    #[test]
    fn dsa_attenuation() {
        use crate::rfdc::{AdcTile, MixerType};

        let mut tile = AdcTile::new(0);
        tile.sample_rate_gsps = 4.0;

        // Block with 6 dB DSA (should halve voltage → -6 dB)
        let mut block = tile.blocks[0].clone();
        block.dsa_db = 6.0;
        block.mixer_settings.mixer_type = MixerType::Off;

        let samples: Vec<Complex<f64>> = (0..512)
            .map(|i| {
                let phi = 2.0 * PI * 100.0 * i as f64 / 15000.0;
                Complex::new(phi.cos() * 0.5, 0.0) // 0.5 amplitude
            })
            .collect();

        let processed = process_adc_block(&samples, 15000.0, &block, &tile, None, None);

        // With DSA, the output peak should be ~6 dB lower than without DSA
        let mut block_no_dsa = block.clone();
        block_no_dsa.dsa_db = 0.0;
        let processed_no_dsa = process_adc_block(&samples, 15000.0, &block_no_dsa, &tile, None, None);

        let peak_with_dsa = processed.folded_spectrum_dbfs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let peak_no_dsa = processed_no_dsa.folded_spectrum_dbfs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let delta = peak_no_dsa - peak_with_dsa;
        assert!(
            (delta - 6.0).abs() < 1.5,
            "6 dB DSA should reduce signal by ~6 dB, got delta = {:.1} dB",
            delta
        );
    }

    #[test]
    fn fine_mixer_scale_auto_r2c_vs_c2c() {
        use crate::rfdc::{MixerSettings, MixerType, FineMixerScale, EventSource, CoarseMixFreq};
        use crate::rfdc::MixerMode as MM;

        let n = 256;
        let fs = 1000.0;
        let samples: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let phi = 2.0 * PI * 100.0 * i as f64 / fs;
                Complex::new(phi.cos(), 0.0)
            })
            .collect();

        let ms_r2c = MixerSettings {
            mixer_type: MixerType::Fine, mixer_mode: MM::RealToIq,
            coarse_mix_freq: CoarseMixFreq::Off, freq: 100.0, phase_offset: 0.0,
            fine_mixer_scale: FineMixerScale::Auto, event_source: EventSource::Tile,
        };
        let ms_c2c = MixerSettings {
            mixer_type: MixerType::Fine, mixer_mode: MM::IqToIq,
            coarse_mix_freq: CoarseMixFreq::Off, freq: 100.0, phase_offset: 0.0,
            fine_mixer_scale: FineMixerScale::Auto, event_source: EventSource::Tile,
        };

        // Auto scale for R2C should be 1.0
        let scale_r2c = match ms_r2c.fine_mixer_scale {
            FineMixerScale::Auto => match ms_r2c.mixer_mode { MM::IqToIq => 0.7071067811865476, _ => 1.0 },
            _ => 1.0,
        };
        // Auto scale for C2C should be 0.7071
        let scale_c2c = match ms_c2c.fine_mixer_scale {
            FineMixerScale::Auto => match ms_c2c.mixer_mode { MM::IqToIq => 0.7071067811865476, _ => 1.0 },
            _ => 1.0,
        };

        let mixed_r2c = apply_mixer(&samples, &ms_r2c, 100.0, fs, fs, scale_r2c);
        let mixed_c2c = apply_mixer(&samples, &ms_c2c, 100.0, fs, fs, scale_c2c);

        // C2C should have ~3 dB less power than R2C due to 0.7071 scaling
        let power_r2c: f64 = mixed_r2c.iter().map(|s| s.norm_sqr()).sum::<f64>() / n as f64;
        let power_c2c: f64 = mixed_c2c.iter().map(|s| s.norm_sqr()).sum::<f64>() / n as f64;
        let ratio_db = 10.0 * (power_r2c / power_c2c).log10();

        assert!(
            (ratio_db - 3.0).abs() < 0.5,
            "R2C/C2C power ratio should be ~3 dB, got {:.1} dB",
            ratio_db
        );
    }

    #[test]
    fn dbfs_calibration() {
        let n = 2048;
        let fs = 1024.0; // Use power of 2 so f=100 lands on exact bin 200 (avoids scalloping loss)
        
        // Full scale complex tone (amplitude 1.0)
        let samples: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let phi = 2.0 * PI * 100.0 * i as f64 / fs;
                Complex::new(phi.cos(), phi.sin())
            })
            .collect();
            
        let (spectrum, _) = compute_spectrum(&samples, n, fs);
        let peak_dbfs = spectrum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        // Peak should be 0.0 dBFS for a full scale complex tone
        assert!(
            peak_dbfs.abs() < 0.1,
            "Full scale complex tone should be 0 dBFS, got {:.2} dBFS",
            peak_dbfs
        );

        // Full scale real tone (amplitude 1.0)
        let samples_real: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let phi = 2.0 * PI * 100.0 * i as f64 / fs;
                Complex::new(phi.cos(), 0.0)
            })
            .collect();

        // One-sided positive spectrum
        let (spectrum_pos, _) = compute_spectrum_positive_with_window(&samples_real, n, fs, FftWindow::Hanning);
        let peak_pos_dbfs = spectrum_pos.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        // Peak should be 0.0 dBFS for a full scale real tone in a one-sided spectrum
        assert!(
            peak_pos_dbfs.abs() < 0.1,
            "Full scale real tone should be 0 dBFS in one-sided spectrum, got {:.2} dBFS",
            peak_pos_dbfs
        );
    }
}
