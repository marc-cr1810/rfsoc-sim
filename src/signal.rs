//! Signal generation and IQ file loading.

#![allow(dead_code)]

use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::path::PathBuf;

/// Dynamic modulation modes for tone components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToneModulation {
    /// Continuous wave (CW) with continuous phase. (Complex Exponential)
    Cw,
    /// Real Cosine tone, energy at +f and -f
    RealCosine,
    /// Real Sine tone, energy at +f and -f (phase shifted)
    RealSine,
    /// Real Square wave
    Square,
    /// Real Sawtooth wave
    Sawtooth,
    /// Real Triangle wave
    Triangle,
    /// Linear Frequency Modulated (LFM) Chirp / FMCW Sweep.
    SweptChirp { sweep_period_ms: f64 },
    /// Frequency Modulation (FM).
    FmModulated { dev_mhz: f64, mod_freq_khz: f64 },
    /// Pulsed RF / Radar signal.
    PulsedRadar { pulse_width_us: f64, pri_us: f64 },
    /// Frequency Hopping Spread Spectrum.
    FreqHopping {
        hop_step_mhz: f64,
        num_channels: usize,
        hop_rate_hz: f64,
    },
    /// Digital QPSK modulation.
    DigitalQpsk { symbol_rate_ksps: f64 },
}

impl Default for ToneModulation {
    fn default() -> Self {
        ToneModulation::Cw
    }
}

impl std::fmt::Display for ToneModulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToneModulation::Cw => write!(f, "CW (Complex Tone)"),
            ToneModulation::RealCosine => write!(f, "Cosine"),
            ToneModulation::RealSine => write!(f, "Sine"),
            ToneModulation::Square => write!(f, "Square"),
            ToneModulation::Sawtooth => write!(f, "Sawtooth"),
            ToneModulation::Triangle => write!(f, "Triangle"),
            ToneModulation::SweptChirp { .. } => write!(f, "FMCW Chirp Sweep"),
            ToneModulation::FmModulated { .. } => write!(f, "FM Modulated"),
            ToneModulation::PulsedRadar { .. } => write!(f, "Pulsed Radar"),
            ToneModulation::FreqHopping { .. } => write!(f, "Frequency Hopping"),
            ToneModulation::DigitalQpsk { .. } => write!(f, "Digital QPSK"),
        }
    }
}

/// A single tone or modulated signal component in the signal generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tone {
    /// Frequency in MHz.
    pub frequency_mhz: f64,
    /// Amplitude in dBFS (0 dBFS = full scale).
    pub amplitude_dbfs: f64,
    /// Phase offset in degrees.
    pub phase_deg: f64,
    /// Signal bandwidth in MHz (0.0 = pure CW tone, >0.0 = modulated channel bandwidth).
    pub bandwidth_mhz: f64,
    /// Modulation type and dynamic time-domain parameters.
    pub modulation: ToneModulation,
}

impl Default for Tone {
    fn default() -> Self {
        Self {
            frequency_mhz: 300.0,
            amplitude_dbfs: -6.0,
            phase_deg: 0.0,
            bandwidth_mhz: 0.0,
            modulation: ToneModulation::Cw,
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
                frequency_mhz: 300.0,
                amplitude_dbfs: -6.0,
                phase_deg: 0.0,
                bandwidth_mhz: 0.0,
                modulation: ToneModulation::Cw,
            }],
            noise_floor_dbfs: -80.0,
            noise_enabled: true,
        }
    }
}

impl SignalGenerator {
    /// Generate complex IQ samples at start time t = 0.
    pub fn generate(&self, num_samples: usize, sample_rate_mhz: f64) -> Vec<Complex<f64>> {
        self.generate_at_time(num_samples, sample_rate_mhz, 0.0)
    }

    /// Generate complex IQ samples continuously starting at `start_time_us`.
    ///
    /// - `num_samples`: number of complex samples to produce
    /// - `sample_rate_mhz`: sampling rate in MHz
    /// - `start_time_us`: global simulation timestamp in microseconds
    pub fn generate_at_time(
        &self,
        num_samples: usize,
        sample_rate_mhz: f64,
        start_time_us: f64,
    ) -> Vec<Complex<f64>> {
        let mut samples = vec![Complex::new(0.0, 0.0); num_samples];
        let dt = 1.0 / sample_rate_mhz; // time step in µs (freq is in MHz)

        // Add each tone or modulated channel
        for tone in &self.tones {
            let amp = tone.linear_amplitude();
            let phase_rad = tone.phase_deg * PI / 180.0;

            match &tone.modulation {
                ToneModulation::SweptChirp { sweep_period_ms } => {
                    let sweep_period_us = (sweep_period_ms * 1000.0).max(1.0);
                    let bw = if tone.bandwidth_mhz > 0.0 {
                        tone.bandwidth_mhz
                    } else {
                        100.0
                    };
                    let half_bw = bw / 2.0;
                    let f_start = tone.frequency_mhz - half_bw;

                    for (i, sample) in samples.iter_mut().enumerate() {
                        let t_us = start_time_us + i as f64 * dt;
                        let t_rel = t_us % sweep_period_us;
                        let chirp_rate = bw / sweep_period_us;
                        let angle = 2.0 * PI * (f_start * t_rel + 0.5 * chirp_rate * t_rel * t_rel)
                            + phase_rad;
                        *sample += Complex::new(amp * angle.cos(), amp * angle.sin());
                    }
                }
                ToneModulation::FmModulated {
                    dev_mhz,
                    mod_freq_khz,
                } => {
                    let f_m_mhz = mod_freq_khz / 1000.0;
                    let beta = if f_m_mhz > 0.0 {
                        dev_mhz / f_m_mhz
                    } else {
                        0.0
                    };
                    let c_period = 1.0 / tone.frequency_mhz.max(1e-6);
                    let m_period = if f_m_mhz > 0.0 { 1.0 / f_m_mhz } else { 1000.0 };
                    let t_c_start = start_time_us % c_period;
                    let t_m_start = start_time_us % m_period;

                    for (i, sample) in samples.iter_mut().enumerate() {
                        let t_c = t_c_start + i as f64 * dt;
                        let t_m = t_m_start + i as f64 * dt;
                        let angle = 2.0 * PI * tone.frequency_mhz * t_c
                            + beta * (2.0 * PI * f_m_mhz * t_m).sin()
                            + phase_rad;
                        *sample += Complex::new(amp * angle.cos(), amp * angle.sin());
                    }
                }
                ToneModulation::PulsedRadar {
                    pulse_width_us,
                    pri_us,
                } => {
                    let pri = pri_us.max(1.0);
                    let pw = pulse_width_us.min(pri);
                    let c_period = 1.0 / tone.frequency_mhz.max(1e-6);
                    let t_c_start = start_time_us % c_period;
                    let phase_start = 2.0 * PI * tone.frequency_mhz * t_c_start + phase_rad;
                    let phase_step = 2.0 * PI * tone.frequency_mhz * dt;

                    for (i, sample) in samples.iter_mut().enumerate() {
                        let t_us = start_time_us + i as f64 * dt;
                        let t_rel = t_us % pri;
                        let pulse_env = if t_rel <= pw { 1.0 } else { 0.0 };
                        let angle = phase_start + i as f64 * phase_step;
                        *sample += Complex::new(amp * pulse_env * angle.cos(), amp * pulse_env * angle.sin());
                    }
                }
                ToneModulation::FreqHopping {
                    hop_step_mhz,
                    num_channels,
                    hop_rate_hz,
                } => {
                    let n_chan = (*num_channels).max(1);
                    let hop_dur_us = 1_000_000.0 / hop_rate_hz.max(1.0);

                    for (i, sample) in samples.iter_mut().enumerate() {
                        let t_us = start_time_us + i as f64 * dt;
                        let hop_idx = (t_us / hop_dur_us).floor() as u64;
                        let chan = ((hop_idx * 7 + 3) as usize) % n_chan;
                        let chan_offset = (chan as f64 - (n_chan as f64 - 1.0) / 2.0) * hop_step_mhz;
                        let inst_freq = tone.frequency_mhz + chan_offset;
                        let c_period = 1.0 / inst_freq.max(1e-6);
                        let t_c_start = t_us % c_period;
                        let angle = 2.0 * PI * inst_freq * t_c_start + phase_rad;
                        *sample += Complex::new(amp * angle.cos(), amp * angle.sin());
                    }
                }
                ToneModulation::DigitalQpsk { symbol_rate_ksps } => {
                    let sym_dur_us = 1_000.0 / symbol_rate_ksps.max(0.1);
                    let c_period = 1.0 / tone.frequency_mhz.max(1e-6);
                    let t_c_start = start_time_us % c_period;
                    let phase_start = 2.0 * PI * tone.frequency_mhz * t_c_start + phase_rad;
                    let phase_step = 2.0 * PI * tone.frequency_mhz * dt;

                    for (i, sample) in samples.iter_mut().enumerate() {
                        let t_us = start_time_us + i as f64 * dt;
                        let sym_idx = (t_us / sym_dur_us).floor() as u64;
                        let sym_phase = (sym_idx * 3 + 1) % 4;
                        let qpsk_angle = (sym_phase as f64) * PI / 2.0 + PI / 4.0;
                        let angle = phase_start + i as f64 * phase_step + qpsk_angle;
                        *sample += Complex::new(amp * angle.cos(), amp * angle.sin());
                    }
                }
                ToneModulation::Cw => {
                    let phase_start = 2.0 * PI * (tone.frequency_mhz * start_time_us).fract() + phase_rad;
                    let phase_step = 2.0 * PI * tone.frequency_mhz * dt;

                    for (i, sample) in samples.iter_mut().enumerate() {
                        let angle = phase_start + i as f64 * phase_step;
                        *sample += Complex::new(amp * angle.cos(), amp * angle.sin());
                    }
                }
                ToneModulation::RealCosine => {
                    let phase_start = 2.0 * PI * (tone.frequency_mhz * start_time_us).fract() + phase_rad;
                    let phase_step = 2.0 * PI * tone.frequency_mhz * dt;

                    for (i, sample) in samples.iter_mut().enumerate() {
                        let angle = phase_start + i as f64 * phase_step;
                        *sample += Complex::new(amp * angle.cos(), 0.0);
                    }
                }
                ToneModulation::RealSine => {
                    let phase_start = 2.0 * PI * (tone.frequency_mhz * start_time_us).fract() + phase_rad;
                    let phase_step = 2.0 * PI * tone.frequency_mhz * dt;

                    for (i, sample) in samples.iter_mut().enumerate() {
                        let angle = phase_start + i as f64 * phase_step;
                        *sample += Complex::new(amp * angle.sin(), 0.0);
                    }
                }
                ToneModulation::Square => {
                    let phase_start = 2.0 * PI * (tone.frequency_mhz * start_time_us).fract() + phase_rad;
                    let phase_step = 2.0 * PI * tone.frequency_mhz * dt;

                    for (i, sample) in samples.iter_mut().enumerate() {
                        let angle = (phase_start + i as f64 * phase_step) % (2.0 * PI);
                        let val = if angle < PI { 1.0 } else { -1.0 };
                        *sample += Complex::new(amp * val, 0.0);
                    }
                }
                ToneModulation::Sawtooth => {
                    let phase_start = 2.0 * PI * (tone.frequency_mhz * start_time_us).fract() + phase_rad;
                    let phase_step = 2.0 * PI * tone.frequency_mhz * dt;

                    for (i, sample) in samples.iter_mut().enumerate() {
                        let angle = (phase_start + i as f64 * phase_step) % (2.0 * PI);
                        let val = (angle / PI) - 1.0;
                        *sample += Complex::new(amp * val, 0.0);
                    }
                }
                ToneModulation::Triangle => {
                    let phase_start = 2.0 * PI * (tone.frequency_mhz * start_time_us).fract() + phase_rad;
                    let phase_step = 2.0 * PI * tone.frequency_mhz * dt;

                    for (i, sample) in samples.iter_mut().enumerate() {
                        let angle = (phase_start + i as f64 * phase_step) % (2.0 * PI);
                        let val = 2.0 * ((angle / PI) - 1.0).abs() - 1.0;
                        *sample += Complex::new(amp * val, 0.0);
                    }
                }
            }
        }

        // Add AWGN noise using Box-Muller transform
        if self.noise_enabled {
            let noise_db = -self.noise_floor_dbfs.abs();
            let noise_amp = 10.0_f64.powf(noise_db / 20.0);
            let mut seed: u64 = 0xDEAD_BEEF_CAFE_BABE ^ (start_time_us.to_bits());

            for sample in &mut samples {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let u1 = (seed as f64) / (u64::MAX as f64);
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let u2 = (seed as f64) / (u64::MAX as f64);

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
    /// Binary interleaved i16: I0, Q0, I1, Q1, ... (scaled to [-1, 1])
    Sc16,
    /// CSV with I, Q columns
    Csv,
}

impl std::fmt::Display for IqFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IqFormat::BinaryF32 => write!(f, "Binary f32 (fc32)"),
            IqFormat::BinaryF64 => write!(f, "Binary f64 (fc64)"),
            IqFormat::Sc16 => write!(f, "Binary i16 (sc16)"),
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
    /// Sample rate of the captured data in the file in MHz.
    pub sample_rate_mhz: f64,
    /// Whether to repeat the playback when the file ends.
    pub repeat: bool,
    /// Idle period in microseconds to wait between repeats.
    pub repeat_period_us: f64,
    
    #[serde(skip)]
    pub cached_samples: std::sync::Arc<std::sync::Mutex<Option<std::sync::Arc<Vec<Complex<f64>>>>>>,
    #[serde(skip)]
    pub last_path_loaded: std::sync::Arc<std::sync::Mutex<Option<PathBuf>>>,
}

impl Default for IqFileLoader {
    fn default() -> Self {
        Self {
            path: None,
            format: IqFormat::BinaryF32,
            sample_rate_mhz: 1000.0,
            repeat: true,
            repeat_period_us: 0.0,
            cached_samples: Default::default(),
            last_path_loaded: Default::default(),
        }
    }
}

impl IqFileLoader {
    /// Load IQ samples from the configured file.
    pub fn load(&self) -> Result<std::sync::Arc<Vec<Complex<f64>>>, String> {
        let path = self.path.as_ref().ok_or("No file path set")?;
        
        let mut last_path = self.last_path_loaded.lock().unwrap();
        let mut cache = self.cached_samples.lock().unwrap();
        
        if last_path.as_ref() == Some(path) && cache.is_some() {
            return Ok(cache.as_ref().unwrap().clone());
        }

        let data = std::fs::read(path).map_err(|e| format!("Failed to read file: {e}"))?;

        let samples_vec = match self.format {
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
                samples
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
                samples
            }
            IqFormat::Sc16 => {
                if data.len() % 4 != 0 {
                    return Err("File size not a multiple of 4 bytes (2 × i16)".into());
                }
                let samples: Vec<Complex<f64>> = data
                    .chunks_exact(4)
                    .map(|chunk| {
                        let i = i16::from_le_bytes([chunk[0], chunk[1]]);
                        let q = i16::from_le_bytes([chunk[2], chunk[3]]);
                        Complex::new(i as f64 / 32768.0, q as f64 / 32768.0)
                    })
                    .collect();
                samples
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
                samples
            }
        };
        
        let arc_samples = std::sync::Arc::new(samples_vec);
        *cache = Some(arc_samples.clone());
        *last_path = Some(path.clone());
        
        Ok(arc_samples)
    }

    /// Clears the cached file so it is forced to reload.
    pub fn clear_cache(&mut self) {
        *self.last_path_loaded.lock().unwrap() = None;
        *self.cached_samples.lock().unwrap() = None;
    }

    /// Generates exactly `num_samples` of IQ data simulating playback of the file at `out_sample_rate_mhz`.
    pub fn generate_at_time(
        &self,
        num_samples: usize,
        out_sample_rate_mhz: f64,
        start_time_us: f64,
    ) -> Result<Vec<Complex<f64>>, String> {
        let file_samples = self.load()?;
        let num_file_samples = file_samples.len();
        
        if num_file_samples == 0 {
            return Ok(vec![Complex::new(0.0, 0.0); num_samples]);
        }

        let mut output = vec![Complex::new(0.0, 0.0); num_samples];
        let file_fs_mhz = self.sample_rate_mhz.max(1e-6);
        let file_duration_us = num_file_samples as f64 / file_fs_mhz;
        let repeat_period_us = self.repeat_period_us.max(0.0);
        let cycle_duration_us = file_duration_us + repeat_period_us;
        
        let out_dt_us = 1.0 / out_sample_rate_mhz;

        for (i, sample) in output.iter_mut().enumerate() {
            let t_out_us = start_time_us + i as f64 * out_dt_us;
            
            let t_rel_us = if self.repeat {
                t_out_us % cycle_duration_us
            } else {
                t_out_us
            };
            
            if t_rel_us < file_duration_us {
                let idx_f = t_rel_us * file_fs_mhz;
                let idx = idx_f.round() as usize; // Nearest-neighbor interpolation
                if idx < num_file_samples {
                    *sample = file_samples[idx];
                }
            }
        }

        Ok(output)
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
                bandwidth_mhz: 0.0,
                modulation: ToneModulation::Cw,
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

    #[test]
    fn time_advancing_phase_continuity() {
        let sig_gen = SignalGenerator {
            tones: vec![Tone {
                frequency_mhz: 100.0,
                amplitude_dbfs: 0.0,
                phase_deg: 0.0,
                bandwidth_mhz: 0.0,
                modulation: ToneModulation::Cw,
            }],
            noise_floor_dbfs: -120.0,
            noise_enabled: false,
        };
        let sample_rate = 1000.0;
        let num_samples = 100;
        let dt = 1.0 / sample_rate; // 0.001 us per sample

        // Generate chunk 1 from t = 0..100 us
        let _chunk1 = sig_gen.generate_at_time(num_samples, sample_rate, 0.0);
        // Generate chunk 2 from t = 100 us
        let t_start_chunk2 = num_samples as f64 * dt;
        let chunk2 = sig_gen.generate_at_time(num_samples, sample_rate, t_start_chunk2);

        // Continuous generation of 200 samples
        let continuous = sig_gen.generate_at_time(200, sample_rate, 0.0);

        // First sample of chunk 2 should equal sample 100 of continuous generation
        let diff = (chunk2[0] - continuous[100]).norm();
        assert!(diff < 1e-9, "Phase discontinuity between consecutive time chunks!");
    }

    #[test]
    fn modulation_modes_energy() {
        let modes = vec![
            ToneModulation::Cw,
            ToneModulation::SweptChirp { sweep_period_ms: 5.0 },
            ToneModulation::FmModulated { dev_mhz: 10.0, mod_freq_khz: 5.0 },
            ToneModulation::PulsedRadar { pulse_width_us: 50.0, pri_us: 100.0 },
            ToneModulation::FreqHopping { hop_step_mhz: 20.0, num_channels: 4, hop_rate_hz: 100.0 },
            ToneModulation::DigitalQpsk { symbol_rate_ksps: 50.0 },
        ];

        for mod_mode in modes {
            let sig_generator = SignalGenerator {
                tones: vec![Tone {
                    frequency_mhz: 200.0,
                    amplitude_dbfs: -3.0,
                    phase_deg: 0.0,
                    bandwidth_mhz: 50.0,
                    modulation: mod_mode,
                }],
                noise_floor_dbfs: -100.0,
                noise_enabled: false,
            };

            let samples = sig_generator.generate_at_time(512, 1000.0, 10.0);
            let energy: f64 = samples.iter().map(|s| s.norm_sqr()).sum();
            assert!(energy > 0.0, "Modulated tone produced zero energy");
        }
    }

    #[test]
    fn iq_file_loader_sc16_parsing() {
        let dir = std::env::temp_dir();
        let file_path = dir.join("test_sc16.iq");
        
        // Write exactly two complex samples (4 i16 values = 8 bytes)
        // Sample 1: I=32767, Q=-32768
        // Sample 2: I=0, Q=16384
        let mut data = Vec::new();
        data.extend_from_slice(&32767i16.to_le_bytes());
        data.extend_from_slice(&(-32768i16).to_le_bytes());
        data.extend_from_slice(&0i16.to_le_bytes());
        data.extend_from_slice(&16384i16.to_le_bytes());
        std::fs::write(&file_path, &data).unwrap();

        let mut loader = IqFileLoader::default();
        loader.path = Some(file_path.clone());
        loader.format = IqFormat::Sc16;

        let samples = loader.load().unwrap();
        assert_eq!(samples.len(), 2);
        
        assert!((samples[0].re - (32767.0 / 32768.0)).abs() < 1e-6);
        assert!((samples[0].im - (-32768.0 / 32768.0)).abs() < 1e-6);
        assert!((samples[1].re - 0.0).abs() < 1e-6);
        assert!((samples[1].im - (16384.0 / 32768.0)).abs() < 1e-6);
        
        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn iq_file_loader_generate_at_time_repeating() {
        let dir = std::env::temp_dir();
        let file_path = dir.join("test_repeat.csv");
        
        // 4 samples at 1 MHz = 4 us file duration
        let csv_data = "1.0, 1.0\n2.0, 2.0\n3.0, 3.0\n4.0, 4.0\n";
        std::fs::write(&file_path, csv_data).unwrap();

        let mut loader = IqFileLoader::default();
        loader.path = Some(file_path.clone());
        loader.format = IqFormat::Csv;
        loader.sample_rate_mhz = 1.0;
        loader.repeat = true;
        loader.repeat_period_us = 2.0; // 2 us idle gap

        // File duration = 4 us. Idle gap = 2 us. Cycle = 6 us.
        // We will read 10 samples at 1 MHz starting at 0 us.
        let out_samples = loader.generate_at_time(10, 1.0, 0.0).unwrap();
        
        // 0..3: file samples
        // 4..5: idle gap (0)
        // 6..9: file samples (repeat)
        assert_eq!(out_samples[0], Complex::new(1.0, 1.0));
        assert_eq!(out_samples[3], Complex::new(4.0, 4.0));
        assert_eq!(out_samples[4], Complex::new(0.0, 0.0)); // gap
        assert_eq!(out_samples[5], Complex::new(0.0, 0.0)); // gap
        assert_eq!(out_samples[6], Complex::new(1.0, 1.0)); // repeat starts
        assert_eq!(out_samples[9], Complex::new(4.0, 4.0));

        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn basic_real_waveforms_energy() {
        let modes = vec![
            ToneModulation::RealSine,
            ToneModulation::RealCosine,
            ToneModulation::Square,
            ToneModulation::Sawtooth,
            ToneModulation::Triangle,
        ];

        for mod_mode in modes {
            let sig_generator = SignalGenerator {
                tones: vec![Tone {
                    frequency_mhz: 200.0,
                    amplitude_dbfs: -3.0,
                    phase_deg: 0.0,
                    bandwidth_mhz: 0.0,
                    modulation: mod_mode,
                }],
                noise_floor_dbfs: -100.0,
                noise_enabled: false,
            };

            let samples = sig_generator.generate_at_time(512, 1000.0, 10.0);
            
            // Should be strictly real-valued
            let real_energy: f64 = samples.iter().map(|s| s.re * s.re).sum();
            let imag_energy: f64 = samples.iter().map(|s| s.im * s.im).sum();
            
            assert!(real_energy > 0.0, "Real waveform produced zero energy");
            assert!(imag_energy < 1e-12, "Real waveform produced imaginary energy");
        }
    }
}
