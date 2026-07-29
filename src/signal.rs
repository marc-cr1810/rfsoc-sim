//! Signal generation and IQ file loading.

#![allow(dead_code)]

use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::path::PathBuf;

/// A single tone component in the signal generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tone {
    /// Frequency in MHz.
    pub frequency_mhz: f64,
    /// Amplitude in dBFS (0 dBFS = full scale).
    pub amplitude_dbfs: f64,
    /// Phase offset in degrees.
    pub phase_deg: f64,
}

impl Default for Tone {
    fn default() -> Self {
        Self {
            frequency_mhz: 100.0,
            amplitude_dbfs: -6.0,
            phase_deg: 0.0,
        }
    }
}

impl Tone {
    /// Convert amplitude from dBFS to linear scale.
    pub fn linear_amplitude(&self) -> f64 {
        10.0_f64.powf(self.amplitude_dbfs / 20.0)
    }
}

/// Multi-tone signal generator with configurable noise floor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalGenerator {
    /// Individual tone components.
    pub tones: Vec<Tone>,
    /// Noise floor in dBFS.
    pub noise_floor_dbfs: f64,
    /// Whether noise is enabled.
    pub noise_enabled: bool,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self {
            tones: vec![Tone {
                frequency_mhz: 500.0,
                amplitude_dbfs: -6.0,
                phase_deg: 0.0,
            }],
            noise_floor_dbfs: -80.0,
            noise_enabled: true,
        }
    }
}

impl SignalGenerator {
    /// Generate complex IQ samples.
    ///
    /// - `num_samples`: number of complex samples to produce
    /// - `sample_rate_mhz`: sampling rate in MHz
    pub fn generate(&self, num_samples: usize, sample_rate_mhz: f64) -> Vec<Complex<f64>> {
        let mut samples = vec![Complex::new(0.0, 0.0); num_samples];
        let dt = 1.0 / sample_rate_mhz; // time step in µs (since freq is in MHz)

        // Add each tone
        for tone in &self.tones {
            let amp = tone.linear_amplitude();
            let phase_rad = tone.phase_deg * PI / 180.0;
            let omega = 2.0 * PI * tone.frequency_mhz; // rad/µs

            for (i, sample) in samples.iter_mut().enumerate() {
                let t = i as f64 * dt;
                let angle = omega * t + phase_rad;
                *sample += Complex::new(amp * angle.cos(), amp * angle.sin());
            }
        }

        // Add AWGN noise using a simple Box-Muller transform
        if self.noise_enabled {
            let noise_amp = 10.0_f64.powf(self.noise_floor_dbfs / 20.0);
            let mut seed: u64 = 0xDEAD_BEEF_CAFE_BABE;

            for sample in &mut samples {
                // Simple xorshift64 PRNG (good enough for visualization)
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let u1 = (seed as f64) / (u64::MAX as f64);
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let u2 = (seed as f64) / (u64::MAX as f64);

                // Box-Muller transform
                let r = (-2.0 * u1.max(1e-300).ln()).sqrt();
                let theta = 2.0 * PI * u2;
                *sample += Complex::new(noise_amp * r * theta.cos(), noise_amp * r * theta.sin());
            }
        }

        samples
    }
}

/// IQ file format specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IqFormat {
    /// Binary interleaved f32: I0, Q0, I1, Q1, ...
    BinaryF32,
    /// Binary interleaved f64: I0, Q0, I1, Q1, ...
    BinaryF64,
    /// CSV with I, Q columns
    Csv,
}

impl std::fmt::Display for IqFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IqFormat::BinaryF32 => write!(f, "Binary f32"),
            IqFormat::BinaryF64 => write!(f, "Binary f64"),
            IqFormat::Csv => write!(f, "CSV (I, Q)"),
        }
    }
}

/// IQ file loader for reading complex samples from disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IqFileLoader {
    /// Path to the IQ data file.
    pub path: Option<PathBuf>,
    /// Format of the IQ data.
    pub format: IqFormat,
    /// Sample rate of the loaded data in MHz.
    pub sample_rate_mhz: f64,
}

impl Default for IqFileLoader {
    fn default() -> Self {
        Self {
            path: None,
            format: IqFormat::BinaryF32,
            sample_rate_mhz: 1000.0,
        }
    }
}

impl IqFileLoader {
    /// Load IQ samples from the configured file.
    pub fn load(&self) -> Result<Vec<Complex<f64>>, String> {
        let path = self.path.as_ref().ok_or("No file path set")?;
        let data = std::fs::read(path).map_err(|e| format!("Failed to read file: {e}"))?;

        match self.format {
            IqFormat::BinaryF32 => {
                if data.len() % 8 != 0 {
                    return Err("File size not a multiple of 8 bytes (2 × f32)".into());
                }
                let samples: Vec<Complex<f64>> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        let i = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        let q = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                        Complex::new(i as f64, q as f64)
                    })
                    .collect();
                Ok(samples)
            }
            IqFormat::BinaryF64 => {
                if data.len() % 16 != 0 {
                    return Err("File size not a multiple of 16 bytes (2 × f64)".into());
                }
                let samples: Vec<Complex<f64>> = data
                    .chunks_exact(16)
                    .map(|chunk| {
                        let i = f64::from_le_bytes(chunk[0..8].try_into().unwrap());
                        let q = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
                        Complex::new(i, q)
                    })
                    .collect();
                Ok(samples)
            }
            IqFormat::Csv => {
                let text =
                    String::from_utf8(data).map_err(|e| format!("Invalid UTF-8: {e}"))?;
                let mut samples = Vec::new();
                for (line_num, line) in text.lines().enumerate() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() < 2 {
                        return Err(format!("Line {}: expected I,Q values", line_num + 1));
                    }
                    let i: f64 = parts[0]
                        .trim()
                        .parse()
                        .map_err(|e| format!("Line {}: invalid I value: {e}", line_num + 1))?;
                    let q: f64 = parts[1]
                        .trim()
                        .parse()
                        .map_err(|e| format!("Line {}: invalid Q value: {e}", line_num + 1))?;
                    samples.push(Complex::new(i, q));
                }
                Ok(samples)
            }
        }
    }
}

/// Unified signal source — either a generator or a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalSource {
    Generator(SignalGenerator),
    File(IqFileLoader),
}

impl Default for SignalSource {
    fn default() -> Self {
        SignalSource::Generator(SignalGenerator::default())
    }
}

impl SignalSource {
    pub fn sample_rate_mhz(&self) -> f64 {
        match self {
            SignalSource::Generator(_) => 10000.0, // 10 GHz default bandwidth for gen
            SignalSource::File(loader) => loader.sample_rate_mhz,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_correct_sample_count() {
        let sig_gen = SignalGenerator::default();
        let samples = sig_gen.generate(1024, 1000.0);
        assert_eq!(samples.len(), 1024);
    }

    #[test]
    fn single_tone_has_nonzero_energy() {
        let sig_gen = SignalGenerator {
            tones: vec![Tone {
                frequency_mhz: 100.0,
                amplitude_dbfs: 0.0,
                phase_deg: 0.0,
            }],
            noise_floor_dbfs: -120.0,
            noise_enabled: false,
        };
        let samples = sig_gen.generate(256, 1000.0);
        let energy: f64 = samples.iter().map(|s| s.norm_sqr()).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn tone_linear_amplitude() {
        let t = Tone {
            amplitude_dbfs: -6.0,
            ..Default::default()
        };
        // -6 dBFS ≈ 0.501
        assert!((t.linear_amplitude() - 0.501).abs() < 0.01);
    }
}
