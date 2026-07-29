//! RF component DSP models — the signal processing behind each node.

#![allow(dead_code)]

use num_complex::Complex;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};

/// Apply a frequency-domain gain response function H(|f_mhz|) -> gain_linear to complex IQ samples.
pub fn apply_frequency_response<F>(
    samples: &[Complex<f64>],
    sample_rate_mhz: f64,
    gain_fn: F,
) -> Vec<Complex<f64>>
where
    F: Fn(f64) -> f64,
{
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }

    let mut buffer = samples.to_vec();

    // Forward FFT
    let mut planner = FftPlanner::new();
    let fft_forward = planner.plan_fft_forward(n);
    fft_forward.process(&mut buffer);

    let df = sample_rate_mhz / n as f64;

    // Apply frequency response to FFT bins
    for i in 0..n {
        let freq_mhz = if i <= n / 2 {
            i as f64 * df
        } else {
            (n - i) as f64 * df
        };
        let g = gain_fn(freq_mhz);
        buffer[i] *= g;
    }

    // Inverse FFT
    let fft_inverse = planner.plan_fft_inverse(n);
    fft_inverse.process(&mut buffer);

    // Normalize FFT scale factor (1/N)
    let scale = 1.0 / n as f64;
    for sample in &mut buffer {
        *sample *= scale;
    }

    buffer
}

/// Spectrum representation flowing between nodes.
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// Magnitude values in dBFS.
    pub magnitude_dbfs: Vec<f64>,
    /// Corresponding frequency values in MHz.
    pub freq_axis_mhz: Vec<f64>,
    /// Sample rate of the data in MHz (for time-domain operations).
    pub sample_rate_mhz: f64,
}

impl Spectrum {
    pub fn new(num_bins: usize, max_freq_mhz: f64, sample_rate_mhz: f64) -> Self {
        let freq_axis: Vec<f64> = (0..num_bins)
            .map(|i| i as f64 * max_freq_mhz / num_bins as f64)
            .collect();
        Self {
            magnitude_dbfs: vec![-200.0; num_bins],
            freq_axis_mhz: freq_axis,
            sample_rate_mhz,
        }
    }

    pub fn num_bins(&self) -> usize {
        self.magnitude_dbfs.len()
    }
}

// ---------------------------------------------------------------------------
// Balun Model
// ---------------------------------------------------------------------------

/// Models a balun transformer (e.g., Mini-Circuits TCM2-33WX+).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalunModel {
    pub name: String,
    /// Insertion loss lookup table: (frequency_mhz, insertion_loss_db).
    pub il_table: Vec<(f64, f64)>,
    /// Minimum operating frequency in MHz.
    pub min_freq_mhz: f64,
    /// Maximum operating frequency in MHz.
    pub max_freq_mhz: f64,
}

impl Default for BalunModel {
    /// Default: Mini-Circuits TCM2-33WX+ (10 MHz – 3 GHz).
    fn default() -> Self {
        Self {
            name: "TCM2-33WX+".to_string(),
            il_table: vec![
                (10.0, 0.87),
                (100.0, 0.78),
                (400.0, 0.95),
                (700.0, 1.12),
                (1000.0, 1.30),
                (1600.0, 1.72),
                (2000.0, 2.01),
                (2500.0, 2.46),
                (2800.0, 2.74),
                (3000.0, 2.93),
            ],
            min_freq_mhz: 10.0,
            max_freq_mhz: 3000.0,
        }
    }
}

impl BalunModel {
    /// Get the interpolated insertion loss at a given frequency.
    pub fn insertion_loss_at(&self, freq_mhz: f64) -> f64 {
        if freq_mhz <= self.il_table[0].0 {
            return self.il_table[0].1;
        }
        if freq_mhz >= self.il_table.last().unwrap().0 {
            // Beyond operating range — very high loss
            return self.il_table.last().unwrap().1 + 10.0;
        }

        // Linear interpolation between table points
        for window in self.il_table.windows(2) {
            let (f0, il0) = window[0];
            let (f1, il1) = window[1];
            if freq_mhz >= f0 && freq_mhz <= f1 {
                let t = (freq_mhz - f0) / (f1 - f0);
                return il0 + t * (il1 - il0);
            }
        }

        // Fallback
        self.il_table.last().unwrap().1
    }

    /// Apply the balun's frequency response to a spectrum.
    pub fn apply(&self, spectrum: &mut Spectrum) {
        for (mag, freq) in spectrum
            .magnitude_dbfs
            .iter_mut()
            .zip(spectrum.freq_axis_mhz.iter())
        {
            let il = self.insertion_loss_at(freq.abs());
            *mag -= il;
        }
    }

    /// Apply balun frequency response to complex samples.
    pub fn process_samples(&self, samples: &[Complex<f64>], sample_rate_mhz: f64) -> Vec<Complex<f64>> {
        apply_frequency_response(samples, sample_rate_mhz, |freq| {
            let il_db = self.insertion_loss_at(freq);
            10.0_f64.powf(-il_db / 20.0)
        })
    }
}

// ---------------------------------------------------------------------------
// Filter Models
// ---------------------------------------------------------------------------

/// Filter type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
}

impl std::fmt::Display for FilterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterType::LowPass => write!(f, "Low-Pass"),
            FilterType::HighPass => write!(f, "High-Pass"),
            FilterType::BandPass => write!(f, "Band-Pass"),
        }
    }
}

/// Analog filter model using Butterworth response approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterModel {
    pub filter_type: FilterType,
    /// Cutoff frequency in MHz (for LP/HP) or centre frequency (for BP).
    pub cutoff_mhz: f64,
    /// Bandwidth in MHz (only used for BandPass).
    pub bandwidth_mhz: f64,
    /// Filter order (higher = steeper rolloff).
    pub order: u32,
}

impl Default for FilterModel {
    fn default() -> Self {
        Self {
            filter_type: FilterType::LowPass,
            cutoff_mhz: 1000.0,
            bandwidth_mhz: 200.0,
            order: 4,
        }
    }
}

impl FilterModel {
    /// Compute the filter attenuation at a given frequency (in dB, positive = loss).
    pub fn attenuation_at(&self, freq_mhz: f64) -> f64 {
        let n = self.order as f64;
        match self.filter_type {
            FilterType::LowPass => {
                let ratio = freq_mhz / self.cutoff_mhz;
                10.0 * n * (1.0 + ratio.powf(2.0 * n)).log10()
            }
            FilterType::HighPass => {
                if freq_mhz < 1e-6 {
                    return 200.0; // DC is fully rejected
                }
                let ratio = self.cutoff_mhz / freq_mhz;
                10.0 * n * (1.0 + ratio.powf(2.0 * n)).log10()
            }
            FilterType::BandPass => {
                let f0 = self.cutoff_mhz;
                let bw = self.bandwidth_mhz;
                if freq_mhz < 1e-6 {
                    return 200.0;
                }
                let ratio = (freq_mhz - f0) / (bw / 2.0);
                10.0 * n * (1.0 + ratio.powf(2.0 * n)).log10()
            }
        }
    }

    /// Apply this filter's frequency response to a spectrum.
    pub fn apply(&self, spectrum: &mut Spectrum) {
        for (mag, freq) in spectrum
            .magnitude_dbfs
            .iter_mut()
            .zip(spectrum.freq_axis_mhz.iter())
        {
            let atten = self.attenuation_at(freq.abs());
            *mag -= atten;
        }
    }

    /// Apply this filter's frequency response to complex samples.
    pub fn process_samples(&self, samples: &[Complex<f64>], sample_rate_mhz: f64) -> Vec<Complex<f64>> {
        apply_frequency_response(samples, sample_rate_mhz, |freq| {
            let atten_db = self.attenuation_at(freq);
            10.0_f64.powf(-atten_db / 20.0)
        })
    }
}

// ---------------------------------------------------------------------------
// Amplifier Model
// ---------------------------------------------------------------------------

/// Amplifier / LNA model with flat gain and optional noise figure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmplifierModel {
    /// Gain in dB.
    pub gain_db: f64,
    /// Noise figure in dB.
    pub noise_figure_db: f64,
    /// 1 dB compression point in dBm (for future use).
    pub p1db_dbm: f64,
}

impl Default for AmplifierModel {
    fn default() -> Self {
        Self {
            gain_db: 12.0,
            noise_figure_db: 2.0,
            p1db_dbm: 20.0,
        }
    }
}

impl AmplifierModel {
    pub fn apply(&self, spectrum: &mut Spectrum) {
        for mag in &mut spectrum.magnitude_dbfs {
            *mag += self.gain_db;
        }
    }

    pub fn process_samples(&self, samples: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let gain_lin = 10.0_f64.powf(self.gain_db / 20.0);
        samples.iter().map(|s| s * gain_lin).collect()
    }
}

// ---------------------------------------------------------------------------
// Attenuator Model
// ---------------------------------------------------------------------------

/// Simple attenuator with flat attenuation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttenuatorModel {
    /// Attenuation in dB (positive value).
    pub attenuation_db: f64,
}

impl Default for AttenuatorModel {
    fn default() -> Self {
        Self {
            attenuation_db: 6.0,
        }
    }
}

impl AttenuatorModel {
    pub fn apply(&self, spectrum: &mut Spectrum) {
        for mag in &mut spectrum.magnitude_dbfs {
            *mag -= self.attenuation_db;
        }
    }

    pub fn process_samples(&self, samples: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let atten_lin = 10.0_f64.powf(-self.attenuation_db / 20.0);
        samples.iter().map(|s| s * atten_lin).collect()
    }
}

// ---------------------------------------------------------------------------
// Splitter Model
// ---------------------------------------------------------------------------

/// Power splitter/combiner model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitterModel {
    /// Number of output ports.
    pub num_outputs: u32,
    /// Additional insertion loss in dB (beyond ideal splitting loss).
    pub excess_loss_db: f64,
}

impl Default for SplitterModel {
    fn default() -> Self {
        Self {
            num_outputs: 2,
            excess_loss_db: 0.5,
        }
    }
}

impl SplitterModel {
    /// Total loss per output port in dB.
    pub fn total_loss_db(&self) -> f64 {
        10.0 * (self.num_outputs as f64).log10() + self.excess_loss_db
    }

    pub fn apply(&self, spectrum: &mut Spectrum) {
        let loss = self.total_loss_db();
        for mag in &mut spectrum.magnitude_dbfs {
            *mag -= loss;
        }
    }

    pub fn process_samples(&self, samples: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let loss_lin = 10.0_f64.powf(-self.total_loss_db() / 20.0);
        samples.iter().map(|s| s * loss_lin).collect()
    }
}

// ---------------------------------------------------------------------------
// Mixer Model
// ---------------------------------------------------------------------------

/// Mixer model for frequency translation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixerModel {
    /// Local Oscillator (LO) frequency in MHz.
    pub lo_freq_mhz: f64,
    /// Conversion loss in dB (positive value).
    pub conversion_loss_db: f64,
}

impl Default for MixerModel {
    fn default() -> Self {
        Self {
            lo_freq_mhz: 100.0,
            conversion_loss_db: 7.0,
        }
    }
}

impl MixerModel {
    pub fn apply(&self, spectrum: &mut Spectrum) {
        // Idealized apply just adds the conversion loss to the signal path.
        for mag in &mut spectrum.magnitude_dbfs {
            *mag -= self.conversion_loss_db;
        }
    }

    pub fn process_samples(&self, samples: &[Complex<f64>], sample_rate_mhz: f64) -> Vec<Complex<f64>> {
        let n = samples.len();
        if n == 0 {
            return Vec::new();
        }
        let dt = 1.0 / sample_rate_mhz; // microseconds
        let loss_lin = 10.0_f64.powf(-self.conversion_loss_db / 20.0);
        let omega = 2.0 * std::f64::consts::PI * self.lo_freq_mhz;

        samples.iter().enumerate().map(|(i, &s)| {
            let t = i as f64 * dt;
            let lo = Complex::new((omega * t).cos(), (omega * t).sin());
            s * lo * loss_lin
        }).collect()
    }
}

// ---------------------------------------------------------------------------
// Phase Shifter Model
// ---------------------------------------------------------------------------

/// Phase shifter model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseShifterModel {
    /// Phase shift in degrees.
    pub phase_shift_deg: f64,
    /// Insertion loss in dB.
    pub insertion_loss_db: f64,
}

impl Default for PhaseShifterModel {
    fn default() -> Self {
        Self {
            phase_shift_deg: 90.0,
            insertion_loss_db: 1.5,
        }
    }
}

impl PhaseShifterModel {
    pub fn apply(&self, spectrum: &mut Spectrum) {
        for mag in &mut spectrum.magnitude_dbfs {
            *mag -= self.insertion_loss_db;
        }
    }

    pub fn process_samples(&self, samples: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let phase_rad = self.phase_shift_deg.to_radians();
        let phase_complex = Complex::new(phase_rad.cos(), phase_rad.sin());
        let loss_lin = 10.0_f64.powf(-self.insertion_loss_db / 20.0);
        let multiplier = phase_complex * loss_lin;
        
        samples.iter().map(|&s| s * multiplier).collect()
    }
}

// ---------------------------------------------------------------------------
// Directional Coupler Model
// ---------------------------------------------------------------------------

/// Directional coupler model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalCouplerModel {
    /// Coupling factor in dB.
    pub coupling_db: f64,
    /// Insertion loss of the main line in dB.
    pub insertion_loss_db: f64,
}

impl Default for DirectionalCouplerModel {
    fn default() -> Self {
        Self {
            coupling_db: 20.0,
            insertion_loss_db: 0.5,
        }
    }
}

impl DirectionalCouplerModel {
    pub fn apply(&self, spectrum: &mut Spectrum) {
        for mag in &mut spectrum.magnitude_dbfs {
            *mag -= self.insertion_loss_db;
        }
    }

    pub fn process_samples(&self, samples: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let loss_lin = 10.0_f64.powf(-self.insertion_loss_db / 20.0);
        samples.iter().map(|&s| s * loss_lin).collect()
    }
}

// ---------------------------------------------------------------------------
// Touchstone .s2p S-Parameter Component Model
// ---------------------------------------------------------------------------

/// Touchstone 2-port S-parameter component model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S2pModel {
    pub name: String,
    /// S21 lookup table: (frequency_mhz, gain_db).
    pub s21_table: Vec<(f64, f64)>,
    /// Noise Figure in dB.
    pub noise_figure_db: f64,
    /// OIP3 in dBm.
    pub oip3_dbm: f64,
}

impl Default for S2pModel {
    fn default() -> Self {
        Self {
            name: "Generic S2P Block".to_string(),
            s21_table: vec![
                (10.0, -0.5),
                (500.0, -0.8),
                (1000.0, -1.2),
                (2000.0, -2.0),
                (4000.0, -3.5),
                (6000.0, -6.0),
            ],
            noise_figure_db: 1.5,
            oip3_dbm: 35.0,
        }
    }
}

impl S2pModel {
    pub fn s21_gain_at(&self, freq_mhz: f64) -> f64 {
        if self.s21_table.is_empty() {
            return 0.0;
        }
        if freq_mhz <= self.s21_table[0].0 {
            return self.s21_table[0].1;
        }
        if freq_mhz >= self.s21_table.last().unwrap().0 {
            return self.s21_table.last().unwrap().1;
        }

        for window in self.s21_table.windows(2) {
            let (f0, g0) = window[0];
            let (f1, g1) = window[1];
            if freq_mhz >= f0 && freq_mhz <= f1 {
                let t = (freq_mhz - f0) / (f1 - f0);
                return g0 + t * (g1 - g0);
            }
        }
        self.s21_table.last().unwrap().1
    }

    pub fn apply(&self, spectrum: &mut Spectrum) {
        for (mag, freq) in spectrum
            .magnitude_dbfs
            .iter_mut()
            .zip(spectrum.freq_axis_mhz.iter())
        {
            let gain = self.s21_gain_at(freq.abs());
            *mag += gain;
        }
    }

    pub fn process_samples(&self, samples: &[Complex<f64>], sample_rate_mhz: f64) -> Vec<Complex<f64>> {
        apply_frequency_response(samples, sample_rate_mhz, |freq| {
            let gain_db = self.s21_gain_at(freq);
            10.0_f64.powf(gain_db / 20.0)
        })
    }
}

/// Parse a Touchstone .s2p file content into an `S2pModel`.
pub fn parse_touchstone_s2p(name: &str, content: &str) -> Result<S2pModel, String> {
    let mut freq_multiplier = 1.0;
    let mut is_db = true;
    let mut is_ri = false;
    let mut s21_table = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('!') {
            continue;
        }

        if line.starts_with('#') {
            let tokens: Vec<&str> = line[1..].split_whitespace().collect();
            for token in tokens {
                let tok_upper = token.to_uppercase();
                match tok_upper.as_str() {
                    "HZ" => freq_multiplier = 1e-6,
                    "KHZ" => freq_multiplier = 1e-3,
                    "MHZ" => freq_multiplier = 1.0,
                    "GHZ" => freq_multiplier = 1000.0,
                    "DB" => is_db = true,
                    "MA" => is_db = false,
                    "RI" => {
                        is_db = false;
                        is_ri = true;
                    }
                    _ => {}
                }
            }
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 9 {
            let freq = parts[0].parse::<f64>().map_err(|e| format!("Invalid freq: {e}"))? * freq_multiplier;
            let v1 = parts[3].parse::<f64>().map_err(|e| format!("Invalid S21 v1: {e}"))?;
            let v2 = parts[4].parse::<f64>().map_err(|e| format!("Invalid S21 v2: {e}"))?;

            let s21_db = if is_db {
                v1
            } else if is_ri {
                let mag = (v1 * v1 + v2 * v2).sqrt();
                20.0 * mag.max(1e-12).log10()
            } else {
                20.0 * v1.max(1e-12).log10()
            };

            s21_table.push((freq, s21_db));
        }
    }

    if s21_table.is_empty() {
        return Err("No valid S2P data points found".into());
    }

    s21_table.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    Ok(S2pModel {
        name: name.to_string(),
        s21_table,
        noise_figure_db: 2.0,
        oip3_dbm: 30.0,
    })
}

// ---------------------------------------------------------------------------
// Combiner Model
// ---------------------------------------------------------------------------

/// RF Combiner model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinerModel {
    /// Number of input ports (2-8).
    pub num_inputs: u32,
    /// Excess loss in dB beyond the theoretical split ratio.
    pub excess_loss_db: f64,
}

impl Default for CombinerModel {
    fn default() -> Self {
        Self {
            num_inputs: 2,
            excess_loss_db: 0.5,
        }
    }
}

impl CombinerModel {
    pub fn total_loss_db(&self) -> f64 {
        // Ideal combiner loss + excess insertion loss
        10.0 * (self.num_inputs as f64).log10() + self.excess_loss_db
    }

    pub fn apply(&self, spectrum: &mut Spectrum) {
        let loss = self.total_loss_db();
        for mag in &mut spectrum.magnitude_dbfs {
            *mag -= loss;
        }
    }

    pub fn process_samples(&self, combined_input: &mut [Complex<f64>]) {
        let scale = 10.0_f64.powf(-self.total_loss_db() / 20.0);
        for sample in combined_input.iter_mut() {
            *sample *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balun_interpolation_matches_datasheet() {
        let balun = BalunModel::default();
        // At 100 MHz, IL should be 0.78 dB
        assert!((balun.insertion_loss_at(100.0) - 0.78).abs() < 0.01);
        // At 1000 MHz, IL should be 1.30 dB
        assert!((balun.insertion_loss_at(1000.0) - 1.30).abs() < 0.01);
        // At 550 MHz (between 400 and 700), should interpolate
        let il_550 = balun.insertion_loss_at(550.0);
        assert!(il_550 > 0.95 && il_550 < 1.12);
    }

    #[test]
    fn lowpass_filter_passband() {
        let filter = FilterModel {
            filter_type: FilterType::LowPass,
            cutoff_mhz: 1000.0,
            order: 4,
            ..Default::default()
        };
        // Well below cutoff: very low attenuation
        let atten = filter.attenuation_at(100.0);
        assert!(atten < 1.0, "Passband attenuation should be < 1 dB, got {atten}");
    }

    #[test]
    fn lowpass_filter_stopband() {
        let filter = FilterModel {
            filter_type: FilterType::LowPass,
            cutoff_mhz: 1000.0,
            order: 4,
            ..Default::default()
        };
        // Well above cutoff: high attenuation
        let atten = filter.attenuation_at(5000.0);
        assert!(atten > 40.0, "Stopband attenuation should be high, got {atten}");
    }

    #[test]
    fn splitter_ideal_loss() {
        let splitter = SplitterModel {
            num_outputs: 2,
            excess_loss_db: 0.0,
        };
        // Ideal 2-way split = 3.01 dB
        assert!((splitter.total_loss_db() - 3.01).abs() < 0.1);
    }

    #[test]
    fn touchstone_s2p_parsing() {
        let sample_s2p = "! Sample Touchstone File\n# MHz S DB R 50\n100.0 -15.0 0.0 -0.5 0.0 -15.0 0.0 -15.0 0.0\n1000.0 -20.0 0.0 -2.5 0.0 -20.0 0.0 -20.0 0.0\n";
        let model = parse_touchstone_s2p("Test Filter", sample_s2p).unwrap();
        assert_eq!(model.name, "Test Filter");
        assert_eq!(model.s21_table.len(), 2);
        assert!((model.s21_gain_at(100.0) - (-0.5)).abs() < 1e-5);
        assert!((model.s21_gain_at(1000.0) - (-2.5)).abs() < 1e-5);
        assert!((model.s21_gain_at(550.0) - (-1.5)).abs() < 1e-5);
    }
}
