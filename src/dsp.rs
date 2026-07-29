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

/// Apply analog hardware non-idealities (HD2/HD3 distortion) before sampling.
/// Distortion is applied to the real voltage waveform.
pub fn apply_analog_non_idealities(
    samples: &[Complex<f64>],
    non_idealities: &crate::rfdc::AdcNonIdealities,
) -> Vec<Complex<f64>> {
    if !non_idealities.enabled || samples.is_empty() {
        return samples.to_vec();
    }

    let a2 = if non_idealities.hd2_dbc < 0.0 {
        10.0_f64.powf(non_idealities.hd2_dbc / 20.0)
    } else {
        0.0
    };
    let a3 = if non_idealities.hd3_dbc < 0.0 {
        10.0_f64.powf(non_idealities.hd3_dbc / 20.0)
    } else {
        0.0
    };

    samples
        .iter()
        .map(|&s| {
            let mut v = s.re;
            if a2 > 0.0 {
                v += a2 * v * v;
            }
            if a3 > 0.0 {
                v += a3 * v * v * v;
            }
            Complex::new(v, s.im) // keep im just in case, though it should be real
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

    let processed: Vec<Complex<f64>> = samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let mut v = s.re;

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
    let fft_size = 2048;
    let fs_mhz = tile.sample_rate_mhz();
    let ms = &block.mixer_settings;

    // 0. Apply DSA (Digital Step Attenuator) — reduces full-scale voltage before sampling
    let dsa_scale = if block.dsa_db > 0.0 {
        10.0_f64.powf(-block.dsa_db / 20.0)
    } else {
        1.0
    };
    let dsa_samples: Vec<Complex<f64>> = if dsa_scale < 1.0 {
        input_samples.iter().map(|&s| s * dsa_scale).collect()
    } else {
        input_samples.to_vec()
    };

    // 1. Apply analog non-idealities (HD2/HD3) to input samples (pre-sampling)
    let analog_samples = apply_analog_non_idealities(&dsa_samples, &block.non_idealities);

    // 2. Input spectrum (full wideband)
    let (input_spectrum, input_freq) =
        compute_spectrum_positive(&analog_samples, fft_size, input_sample_rate_mhz);

    let raw_source_spectrum_dbfs = raw_source_samples.map(|samples| {
        let (raw_spec, _) = compute_spectrum_positive(samples, fft_size, input_sample_rate_mhz);
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

    let is_even_zone = block.nyquist_zone.is_even();
    
    // NCO Frequency Negation Rule (XRFdc Driver behavior):
    // Only negate if |Freq| > Fs/2 AND we are in an EVEN zone.
    let mut nco_freq = ms.freq;
    if nco_freq.abs() > fs_mhz / 2.0 {
        // Wrap to [-Fs/2, Fs/2]
        nco_freq = (nco_freq + fs_mhz / 2.0).rem_euclid(fs_mhz) - fs_mhz / 2.0;
        if is_even_zone && nco_freq != 0.0 {
            nco_freq = -nco_freq;
        }
    }

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
    let (post_mixer_spectrum, post_mixer_freq) = if block.mixer_active() {
        compute_spectrum(&mixed_samples, fft_size, fs_mhz)
    } else {
        compute_spectrum_positive(&mixed_samples, fft_size, fs_mhz)
    };

    // 5. Apply QMC (Quadrature Modulation Correction) post-mixer, pre-decimation
    let qmc_samples = apply_qmc(&mixed_samples, &block.qmc_settings);

    // 6. Apply DDC decimation filter at the ADC tile rate Fs
    let decimated = apply_decimation(&qmc_samples, block.decimation.factor());
    let actual_output_rate = block.output_rate_mhz(tile.sample_rate_gsps);

    // 6. Output spectrum
    let output_fft_size = (fft_size / block.decimation.factor() as usize).max(64);
    let (output_spectrum, output_freq) = if block.mixer_active() {
        compute_spectrum(&decimated, output_fft_size, actual_output_rate)
    } else {
        compute_spectrum_positive(&decimated, output_fft_size, actual_output_rate)
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

/// Fold a wideband spectrum into a single Nyquist zone, simulating ADC sampling/aliasing.
///
/// This models the fundamental aliasing process of an ADC: all frequency content
/// folds down into the 0..Fs/2 band, with even Nyquist zones being spectrally inverted.
///
/// `input_spectrum`: magnitude in dBFS at each frequency bin
/// `input_freq_axis`: corresponding frequencies in MHz  
/// `fs_mhz`: ADC sampling rate in MHz
/// `zone`: which Nyquist zone is configured (affects calibration, not the folding itself)
/// Helper to linearly interpolate power spectrum (in linear power space) at an arbitrary frequency.
fn interpolate_spectrum_dbfs(freq_mhz: f64, spectrum: &[f64], freq_axis: &[f64]) -> f64 {
    if freq_axis.len() < 2 {
        return -200.0;
    }
    let f_min = freq_axis[0];
    let f_max = *freq_axis.last().unwrap();
    if freq_mhz < f_min || freq_mhz > f_max {
        return -200.0;
    }

    let df = freq_axis[1] - freq_axis[0];
    if df <= 0.0 {
        return -200.0;
    }

    let idx_f = (freq_mhz - f_min) / df;
    let idx0 = (idx_f.floor() as usize).min(spectrum.len() - 1);
    let idx1 = (idx0 + 1).min(spectrum.len() - 1);
    let t = idx_f - idx0 as f64;

    let p0 = 10.0_f64.powf(spectrum[idx0] / 10.0);
    let p1 = 10.0_f64.powf(spectrum[idx1] / 10.0);
    let p_interp = p0 + t * (p1 - p0);
    10.0 * p_interp.max(1e-20).log10()
}

/// Fold a wideband spectrum into a single Nyquist zone, simulating ADC sampling/aliasing.
///
/// Models realistic ADC aliasing by power-summing interpolated spectral energy from all
/// overlapping Nyquist zones into the 0..Fs/2 first Nyquist zone.
pub fn fold_spectrum(
    input_spectrum: &[f64],
    input_freq_axis: &[f64],
    fs_mhz: f64,
) -> (Vec<f64>, Vec<f64>) {
    let nyquist_bw = fs_mhz / 2.0;
    let num_output_bins = 512;
    let mut folded = vec![-200.0_f64; num_output_bins];

    let output_freq: Vec<f64> = (0..num_output_bins)
        .map(|i| i as f64 * nyquist_bw / num_output_bins as f64)
        .collect();

    if input_freq_axis.is_empty() || input_spectrum.is_empty() {
        return (folded, output_freq);
    }

    let max_input_freq = *input_freq_axis.last().unwrap();
    let max_zone = (max_input_freq / nyquist_bw).ceil() as usize;

    for (bin, &f_out) in output_freq.iter().enumerate() {
        let mut total_power = 0.0_f64;

        for zone in 1..=max_zone {
            let f_in = if zone % 2 == 1 {
                // Odd zone (1, 3, 5): direct mapping
                (zone as f64 - 1.0) * nyquist_bw + f_out
            } else {
                // Even zone (2, 4, 6): spectral inversion (mirrored)
                zone as f64 * nyquist_bw - f_out
            };

            let dbfs = interpolate_spectrum_dbfs(f_in, input_spectrum, input_freq_axis);
            if dbfs > -190.0 {
                total_power += 10.0_f64.powf(dbfs / 10.0);
            }
        }

        if total_power > 1e-20 {
            folded[bin] = 10.0 * total_power.log10();
        }
    }

    (folded, output_freq)
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
                    let angle = omega * i as f64;
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
                    let angle = omega * i as f64;
                    s * Complex::new(angle.cos(), angle.sin()) * scale
                })
                .collect()
        }
    }
}

/// Apply decimation using anti-aliasing FIR windowed-sinc filtering before downsampling.
///
/// Simulates multi-stage halfband/CIC decimation filters in Xilinx DDC IP blocks.
pub fn apply_decimation(samples: &[Complex<f64>], factor: u32) -> Vec<Complex<f64>> {
    if factor <= 1 || samples.is_empty() {
        return samples.to_vec();
    }

    let f = factor as usize;
    let cutoff = 0.45 / f as f64; // Normalized cutoff frequency
    let num_taps = (16 * f).min(64) | 1; // Odd tap count
    let half_taps = num_taps / 2;

    // Design a windowed Sinc low-pass FIR filter
    let mut fir = vec![0.0; num_taps];
    let mut sum = 0.0;
    for i in 0..num_taps {
        let n = i as f64 - half_taps as f64;
        let h = if n == 0.0 {
            2.0 * cutoff
        } else {
            (2.0 * PI * cutoff * n).sin() / (PI * n)
        };
        // Blackman window for high stopband attenuation
        let w = 0.42 - 0.5 * (2.0 * PI * i as f64 / (num_taps - 1) as f64).cos()
            + 0.08 * (4.0 * PI * i as f64 / (num_taps - 1) as f64).cos();
        fir[i] = h * w;
        sum += fir[i];
    }
    // Normalize gain to 0 dB in passband
    if sum > 0.0 {
        for tap in &mut fir {
            *tap /= sum;
        }
    }

    // Apply FIR filter then downsample
    let output_len = samples.len() / f;
    let mut output = Vec::with_capacity(output_len);

    for idx in 0..output_len {
        let center = idx * f;
        let mut filtered = Complex::new(0.0, 0.0);
        for (tap_idx, &coeff) in fir.iter().enumerate() {
            let sample_idx = center + tap_idx;
            if sample_idx >= half_taps && sample_idx - half_taps < samples.len() {
                filtered += samples[sample_idx - half_taps] * coeff;
            }
        }
        output.push(filtered);
    }

    output
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
        // A tone at 1300 MHz with Fs=2000 MHz should fold to 700 MHz (zone 2, mirrored)
        let nyquist_bw = 1000.0; // Fs/2
        let fs = 2000.0;

        // Create a simple spectrum with a single peak at 1300 MHz
        let num_bins = 1024;
        let max_freq = 3000.0; // span up to 3 GHz
        let input_freq: Vec<f64> = (0..num_bins)
            .map(|i| i as f64 * max_freq / num_bins as f64)
            .collect();
        let mut input_spectrum = vec![-100.0; num_bins];
        // Place a tone at the bin closest to 1300 MHz
        let tone_bin = (1300.0 / max_freq * num_bins as f64) as usize;
        input_spectrum[tone_bin] = 0.0; // 0 dBFS

        let (folded, folded_freq) = fold_spectrum(&input_spectrum, &input_freq, fs);

        // Find the peak in the folded spectrum
        let (peak_idx, _) = folded
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let peak_freq = folded_freq[peak_idx];
        // 1300 MHz is in zone 2 (1000-2000), mirrored: 2000-1300 = 700 MHz
        assert!(
            (peak_freq - 700.0).abs() < nyquist_bw / 512.0 * 2.0,
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

        let decimated = apply_decimation(&samples, 8);
        assert_eq!(decimated.len(), 128);
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
