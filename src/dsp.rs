//! DSP processing functions: FFT, Nyquist zone folding, mixing, and decimation.

#![allow(dead_code)]

use crate::rfdc::{AdcBlock, AdcTile, CoarseMixFreq, MixerMode};
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
}

// ...

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

    // 1. Input spectrum (full wideband)
    let (input_spectrum, input_freq) =
        compute_spectrum_positive(input_samples, fft_size, input_sample_rate_mhz);

    let raw_source_spectrum_dbfs = raw_source_samples.map(|samples| {
        let (raw_spec, _) = compute_spectrum_positive(samples, fft_size, input_sample_rate_mhz);
        raw_spec
    });

    // 2. Fold into Nyquist zone (simulates ADC sampling)
    let (folded_spectrum, folded_freq) =
        fold_spectrum(&input_spectrum, &input_freq, fs_mhz);

    // 3. Apply mixer to input time-domain samples at the input sample rate
    let mixed_samples = apply_mixer(input_samples, block.mixer_mode, block.nco_freq_mhz, input_sample_rate_mhz);

    // 4. Compute post-mixer spectrum
    let (post_mixer_spectrum, post_mixer_freq) = if block.mixer_active() {
        compute_spectrum(&mixed_samples, fft_size, input_sample_rate_mhz)
    } else {
        compute_spectrum_positive(&mixed_samples, fft_size, input_sample_rate_mhz)
    };

    // 5. Apply decimation down to the effective output rate
    let output_rate = block.output_rate_mhz(tile.sample_rate_gsps);
    let effective_decimation = ((input_sample_rate_mhz / output_rate).round() as u32).max(1);
    let decimated = apply_decimation(&mixed_samples, effective_decimation);

    // 6. Output spectrum
    let output_fft_size = (fft_size / block.decimation.factor() as usize).max(64);
    let (output_spectrum, output_freq) = if block.mixer_active() {
        compute_spectrum(&decimated, output_fft_size, output_rate)
    } else {
        compute_spectrum_positive(&decimated, output_fft_size, output_rate)
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
        output_sample_rate_mhz: output_rate,
    }
}

/// Compute the power spectrum of complex samples using FFT.
///
/// Returns (magnitude_dbfs, freq_axis_mhz) where frequencies are centred around 0.
pub fn compute_spectrum(
    samples: &[Complex<f64>],
    fft_size: usize,
    sample_rate_mhz: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = (fft_size.min(samples.len()) / 2) * 2;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut buffer: Vec<Complex<f64>> = samples[..n].to_vec();

    // Apply Hanning window
    for (i, sample) in buffer.iter_mut().enumerate() {
        let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos());
        *sample *= w;
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    // Compute magnitude in dBFS (normalised to FFT size)
    let norm = 1.0 / n as f64;
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

/// Compute single-sided (positive frequency only) power spectrum.
pub fn compute_spectrum_positive(
    samples: &[Complex<f64>],
    fft_size: usize,
    sample_rate_mhz: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = (fft_size.min(samples.len()) / 2) * 2;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut buffer: Vec<Complex<f64>> = samples[..n].to_vec();

    // Apply Hanning window
    for (i, sample) in buffer.iter_mut().enumerate() {
        let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos());
        *sample *= w;
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    let norm = 1.0 / n as f64;
    let half = n / 2;

    // Take only positive frequencies (0..Fs/2)
    let spectrum_dbfs: Vec<f64> = buffer[..=half]
        .iter()
        .map(|c| {
            let mag = c.norm() * norm;
            20.0 * mag.max(1e-15).log10()
        })
        .collect();

    let freq_axis: Vec<f64> = (0..=half)
        .map(|i| i as f64 * sample_rate_mhz / n as f64)
        .collect();

    (spectrum_dbfs, freq_axis)
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

/// Apply the DDC mixer to time-domain samples.
pub fn apply_mixer(
    samples: &[Complex<f64>],
    mode: MixerMode,
    nco_freq_mhz: f64,
    fs_mhz: f64,
) -> Vec<Complex<f64>> {
    match mode {
        MixerMode::Bypass => samples.to_vec(),
        MixerMode::CoarseMix(coarse) => {
            let freq_ratio = match coarse {
                CoarseMixFreq::FsOver4 => 0.25,
                CoarseMixFreq::MinusFsOver4 => -0.25,
                CoarseMixFreq::FsOver2 => 0.5,
            };
            let omega = -2.0 * PI * freq_ratio; // negative for downconversion
            samples
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let angle = omega * i as f64;
                    s * Complex::new(angle.cos(), angle.sin())
                })
                .collect()
        }
        MixerMode::FineMix => {
            let omega = -2.0 * PI * nco_freq_mhz / fs_mhz;
            samples
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let angle = omega * i as f64;
                    s * Complex::new(angle.cos(), angle.sin())
                })
                .collect()
        }
    }
}

/// Apply decimation (simple model: average + downsample).
///
/// In real hardware this is a multi-stage CIC+FIR filter chain.
/// We approximate it with a moving-average lowpass + downsample.
pub fn apply_decimation(samples: &[Complex<f64>], factor: u32) -> Vec<Complex<f64>> {
    if factor <= 1 {
        return samples.to_vec();
    }

    let f = factor as usize;
    let output_len = samples.len() / f;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let start = i * f;
        let end = (start + f).min(samples.len());
        let sum: Complex<f64> = samples[start..end].iter().sum();
        output.push(sum / f as f64);
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

        let mixed = apply_mixer(&samples, MixerMode::CoarseMix(CoarseMixFreq::FsOver4), 0.0, fs);
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
}
