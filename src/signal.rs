//! Signal generation and IQ file loading.

#![allow(dead_code)]

use num_complex::Complex;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::path::PathBuf;

/// Modulation applied to a tone.
///
/// Everything here is a **real** waveform. The analog domain carries one real voltage and the
/// converter samples only that, so a "complex tone" option would be indistinguishable from a
/// cosine — which is why there is one carrier variant rather than three.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum ToneModulation {
    /// Unmodulated carrier: a real sine wave at `phase_deg`.
    ///
    /// Give the tone a non-zero `bandwidth_mhz` and it becomes a modulated channel of that
    /// width instead of a single line.
    #[default]
    #[serde(alias = "RealCosine", alias = "RealSine")]
    Cw,
    /// Square wave, band-limited to a finite number of harmonics.
    Square,
    /// Sawtooth ramp, band-limited.
    Sawtooth,
    /// Triangle wave, band-limited.
    Triangle,
    /// Amplitude modulation at a given depth.
    AmModulated { depth_percent: f64, mod_freq_khz: f64 },
    /// Frequency modulation, `dev_mhz` peak deviation.
    FmModulated { dev_mhz: f64, mod_freq_khz: f64 },
    /// Linear FM chirp across `bandwidth_mhz`, sawtooth or triangular retrace.
    SweptChirp { sweep_period_us: f64, triangular: bool },
    /// Pulsed radar, optionally with a chirp inside each pulse for pulse compression.
    PulsedRadar {
        pulse_width_us: f64,
        pri_us: f64,
        /// Edge transition time in ns. Zero gives an ideal rectangle.
        rise_ns: f64,
        /// Intra-pulse LFM sweep width in MHz. Zero gives an unmodulated pulse.
        chirp_mhz: f64,
    },
    /// Frequency-hopping spread spectrum over a channel grid.
    FreqHopping {
        hop_step_mhz: f64,
        num_channels: usize,
        hop_rate_hz: f64,
    },
    /// QPSK with pseudo-random data and root-raised-cosine pulse shaping.
    DigitalQpsk { symbol_rate_msps: f64, rrc_alpha: f64 },
}

impl ToneModulation {
    /// Every variant, with sensible starting parameters, for building a UI selector.
    pub fn all_variants() -> Vec<ToneModulation> {
        vec![
            ToneModulation::Cw,
            ToneModulation::Square,
            ToneModulation::Sawtooth,
            ToneModulation::Triangle,
            ToneModulation::AmModulated { depth_percent: 50.0, mod_freq_khz: 1000.0 },
            ToneModulation::FmModulated { dev_mhz: 10.0, mod_freq_khz: 1000.0 },
            ToneModulation::SweptChirp { sweep_period_us: 100.0, triangular: false },
            ToneModulation::PulsedRadar {
                pulse_width_us: 2.0,
                pri_us: 20.0,
                rise_ns: 10.0,
                chirp_mhz: 0.0,
            },
            ToneModulation::FreqHopping {
                hop_step_mhz: 20.0,
                num_channels: 8,
                hop_rate_hz: 500_000.0,
            },
            ToneModulation::DigitalQpsk { symbol_rate_msps: 20.0, rrc_alpha: 0.35 },
        ]
    }

    /// Whether `bandwidth_mhz` means anything for this modulation, and what to call it.
    pub fn bandwidth_label(&self) -> Option<&'static str> {
        match self {
            ToneModulation::Cw => Some("Channel BW"),
            ToneModulation::SweptChirp { .. } => Some("Sweep BW"),
            _ => None,
        }
    }
}

impl std::fmt::Display for ToneModulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToneModulation::Cw => write!(f, "CW Carrier"),
            ToneModulation::Square => write!(f, "Square"),
            ToneModulation::Sawtooth => write!(f, "Sawtooth"),
            ToneModulation::Triangle => write!(f, "Triangle"),
            ToneModulation::AmModulated { .. } => write!(f, "AM Modulated"),
            ToneModulation::FmModulated { .. } => write!(f, "FM Modulated"),
            ToneModulation::SweptChirp { triangular, .. } => {
                if *triangular {
                    write!(f, "FMCW Chirp (triangular)")
                } else {
                    write!(f, "FMCW Chirp (sawtooth)")
                }
            }
            ToneModulation::PulsedRadar { chirp_mhz, .. } => {
                if *chirp_mhz > 0.0 {
                    write!(f, "Pulsed Radar (chirped)")
                } else {
                    write!(f, "Pulsed Radar")
                }
            }
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

    /// Whether this tone is a modulated channel rather than a bare carrier.
    fn is_channel(&self) -> bool {
        self.bandwidth_mhz > 0.0 && matches!(self.modulation, ToneModulation::Cw)
    }

    /// Synthesise a modulated channel `bandwidth_mhz` wide centred on the carrier.
    ///
    /// The field was documented as the modulated channel bandwidth but only the chirp ever
    /// read it, so a "20 MHz channel" came out as a single spectral line. A real modulated
    /// carrier — OFDM, QAM, anything with data on it — looks like band-limited noise, which is
    /// exactly what filling the in-band bins with random phase and transforming back gives:
    /// flat, precisely `bandwidth_mhz` wide, and with the same total power as the tone it
    /// replaces, so switching the bandwidth on does not move the level.
    ///
    /// The spectrum is filled conjugate-symmetrically, so the waveform is real.
    fn synthesise_channel(
        &self,
        samples: &mut [Complex<f64>],
        sample_rate_mhz: f64,
        start_time_us: f64,
    ) {
        let n = samples.len();
        if n < 4 || sample_rate_mhz <= 0.0 {
            return;
        }
        let amp = self.linear_amplitude();
        let df = sample_rate_mhz / n as f64;
        let half_bw = self.bandwidth_mhz / 2.0;

        // Data is not periodic, so the channel is redrawn each frame; seeding from the frame
        // time keeps it deterministic while letting it evolve in the waterfall.
        let mut seed = 0x2545_F491_4F6C_DD1D
            ^ self.frequency_mhz.to_bits()
            ^ start_time_us.to_bits().rotate_left(31);
        let mut next_gaussian = || -> f64 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u1 = (((seed >> 11) as f64 / (1u64 << 53) as f64).max(1e-300)).ln() * -2.0;
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u2 = (seed >> 11) as f64 / (1u64 << 53) as f64;
            u1.sqrt() * (2.0 * PI * u2).cos()
        };

        let mut spectrum = vec![Complex::new(0.0, 0.0); n];
        let mut filled = 0usize;
        for k in 1..n / 2 {
            let f = k as f64 * df;
            if (f - self.frequency_mhz).abs() > half_bw {
                continue;
            }
            let v = Complex::new(next_gaussian(), next_gaussian());
            spectrum[k] = v;
            spectrum[n - k] = v.conj();
            filled += 1;
        }
        if filled == 0 {
            return;
        }

        // Scale so the channel carries the same power the carrier would have: a real tone of
        // peak amplitude A has mean square A²/2.
        let target = amp * amp / 2.0;
        // rustfft's inverse transform is unnormalised, so by Parseval the mean power it will
        // produce is the plain sum of the bin powers.
        let raw: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum();
        let scale = (target / raw.max(1e-300)).sqrt();

        let mut buffer = spectrum;
        FftPlanner::new().plan_fft_inverse(n).process(&mut buffer);

        for (s, b) in samples.iter_mut().zip(buffer.iter()) {
            s.re += b.re * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Waveform helpers
// ---------------------------------------------------------------------------

/// Phase of a carrier at absolute time `t_us`, reduced to keep precision at large `t`.
///
/// Every waveform runs off the absolute simulation clock so consecutive frames join without a
/// seam. Taking the fractional part of the cycle count before scaling by 2π keeps the phase
/// accurate after seconds of simulated time, where `2π f t` on its own would have thrown away
/// most of its significant digits.
fn carrier_phase(freq_mhz: f64, t_us: f64, phase_rad: f64) -> f64 {
    2.0 * PI * (freq_mhz * t_us).fract() + phase_rad
}

/// Ceiling on how many harmonics a band-limited waveform is built from.
///
/// Also acts as the finite rise time every real generator has: a square wave from an
/// instrument is not made of infinitely many harmonics either.
const MAX_HARMONICS: usize = 64;

/// A band-limited harmonic series for the non-sinusoidal waveforms.
///
/// Evaluating an ideal square or ramp directly at the sample rate folds every harmonic above
/// Nyquist back into the band, which in a simulator whose whole point is teaching Nyquist
/// behaviour is indefensible — the aliases are indistinguishable from real signals. Summing
/// only the harmonics that fit below Nyquist gives the same waveform with none of the lies.
struct HarmonicSeries {
    /// (harmonic number, amplitude) pairs.
    terms: Vec<(f64, f64)>,
    /// Whether the series is built from sines (odd symmetry) or cosines.
    sine: bool,
    /// Normalisation so the waveform peaks at exactly 1.0.
    scale: f64,
}

impl HarmonicSeries {
    fn new(kind: &ToneModulation, freq_mhz: f64, nyquist_mhz: f64) -> Self {
        let max_n = if freq_mhz > 0.0 {
            ((nyquist_mhz / freq_mhz).floor() as usize).min(MAX_HARMONICS)
        } else {
            1
        }
        .max(1);

        let mut terms = Vec::new();
        match kind {
            // Square: odd harmonics falling as 1/n.
            ToneModulation::Square => {
                for n in (1..=max_n).step_by(2) {
                    terms.push((n as f64, 1.0 / n as f64));
                }
            }
            // Sawtooth: every harmonic, alternating sign, falling as 1/n.
            ToneModulation::Sawtooth => {
                for n in 1..=max_n {
                    let sign = if n % 2 == 1 { 1.0 } else { -1.0 };
                    terms.push((n as f64, sign / n as f64));
                }
            }
            // Triangle: odd harmonics falling as 1/n², sign alternating every other one.
            ToneModulation::Triangle => {
                for n in (1..=max_n).step_by(2) {
                    let sign = if (n - 1) / 2 % 2 == 0 { 1.0 } else { -1.0 };
                    terms.push((n as f64, sign / (n * n) as f64));
                }
            }
            _ => terms.push((1.0, 1.0)),
        }

        let mut series = Self { terms, sine: true, scale: 1.0 };
        // Normalise numerically: a truncated series overshoots (Gibbs), and the amount depends
        // on how many harmonics fitted, so the peak is measured rather than assumed.
        let mut peak = 0.0_f64;
        const PROBES: usize = 1024;
        for i in 0..PROBES {
            let theta = 2.0 * PI * i as f64 / PROBES as f64;
            peak = peak.max(series.raw(theta).abs());
        }
        series.scale = if peak > 0.0 { 1.0 / peak } else { 1.0 };
        series
    }

    fn raw(&self, theta: f64) -> f64 {
        self.terms
            .iter()
            .map(|&(n, a)| {
                let x = n * theta;
                a * if self.sine { x.sin() } else { x.cos() }
            })
            .sum()
    }

    /// Waveform value at fundamental phase `theta`, peaking at ±1.
    fn eval(&self, theta: f64) -> f64 {
        self.raw(theta) * self.scale
    }

    /// A stepper that advances each harmonic by a rotation instead of calling `sin` per sample.
    ///
    /// Sixty-odd trigonometric calls per sample is the difference between a waveform that is
    /// free to generate and one that dominates the frame. The phase step is constant, so each
    /// harmonic can be carried as a unit phasor and advanced by one complex multiply; f64 keeps
    /// the accumulated drift around 1e-13 over a whole frame.
    fn stepper(&self, phase_start: f64, phase_step: f64) -> HarmonicStepper {
        HarmonicStepper {
            terms: self
                .terms
                .iter()
                .map(|&(n, a)| {
                    let start = n * phase_start;
                    let step = n * phase_step;
                    (
                        a,
                        Complex::new(start.cos(), start.sin()),
                        Complex::new(step.cos(), step.sin()),
                    )
                })
                .collect(),
            scale: self.scale,
            sine: self.sine,
        }
    }

    /// Highest harmonic present, for reporting in the UI.
    fn top_harmonic(&self) -> f64 {
        self.terms.last().map(|&(n, _)| n).unwrap_or(1.0)
    }
}

/// Advances a harmonic series one sample at a time by rotating phasors.
struct HarmonicStepper {
    /// (amplitude, current phasor, per-sample rotation) for each harmonic.
    terms: Vec<(f64, Complex<f64>, Complex<f64>)>,
    scale: f64,
    sine: bool,
}

impl HarmonicStepper {
    fn next_value(&mut self) -> f64 {
        let mut acc = 0.0;
        for (a, z, rot) in self.terms.iter_mut() {
            acc += *a * if self.sine { z.im } else { z.re };
            *z *= *rot;
        }
        acc * self.scale
    }
}

/// Highest harmonic a band-limited waveform will contain at this frequency and sample rate.
pub fn top_harmonic_mhz(modulation: &ToneModulation, freq_mhz: f64, sample_rate_mhz: f64) -> f64 {
    match modulation {
        ToneModulation::Square | ToneModulation::Sawtooth | ToneModulation::Triangle => {
            let series = HarmonicSeries::new(modulation, freq_mhz, sample_rate_mhz / 2.0);
            series.top_harmonic() * freq_mhz
        }
        _ => freq_mhz,
    }
}

/// Accumulated phase of a linear FM sweep `bw` wide starting at `f_start`, at time `tau` into
/// a sweep of length `period`.
///
/// The triangular form sweeps up over the first half and back down over the second, with the
/// phase carried across the turn so the waveform stays continuous through it.
fn chirp_phase(f_start: f64, bw: f64, period: f64, tau: f64, triangular: bool) -> f64 {
    if !triangular {
        let rate = bw / period;
        return 2.0 * PI * (f_start * tau + 0.5 * rate * tau * tau);
    }
    let half = period / 2.0;
    let rate = bw / half;
    if tau < half {
        2.0 * PI * (f_start * tau + 0.5 * rate * tau * tau)
    } else {
        // Phase at the turning point, then sweeping back down from f_start + bw.
        let up = 2.0 * PI * (f_start * half + 0.5 * rate * half * half);
        let sigma = tau - half;
        up + 2.0 * PI * ((f_start + bw) * sigma - 0.5 * rate * sigma * sigma)
    }
}

/// Pulse envelope with raised-cosine edges.
///
/// A perfectly rectangular pulse has skirts reaching to infinity; every real transmitter has a
/// finite edge, and `rise` is what bounds the occupied spectrum.
fn pulse_envelope(tau: f64, width: f64, rise: f64) -> f64 {
    if tau < 0.0 || tau > width {
        return 0.0;
    }
    if rise <= 0.0 {
        return 1.0;
    }
    let ramp = |x: f64| 0.5 * (1.0 - (PI * x / rise).cos());
    if tau < rise {
        ramp(tau)
    } else if tau > width - rise {
        ramp(width - tau)
    } else {
        1.0
    }
}

/// Which channel the hop sequence visits at `idx`.
///
/// A stride-based sequence (`idx * 7 + 3`) shares a factor with any channel count that is a
/// multiple of the stride, and then never moves at all — seven channels stayed on channel 3
/// forever. Hashing the index instead visits the whole grid for any count.
fn hop_channel(idx: i64, num_channels: usize) -> usize {
    let mut h = (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h % num_channels as u64) as usize
}

/// Root-raised-cosine impulse response at `t` symbol periods from the centre.
///
/// `h(x) = [sin(πx(1−α)) + 4αx·cos(πx(1+α))] / [πx(1 − 16α²x²)]`, normalised so `h(0)` is
/// `1 − α + 4α/π`. The zero and the `x = ±1/(4α)` root of the denominator are both removable;
/// the latter is evaluated by averaging either side rather than by a closed form, which keeps
/// it right without depending on which sign convention a reference uses.
fn rrc(x: f64, alpha: f64) -> f64 {
    const EPS: f64 = 1e-7;
    if x.abs() < EPS {
        return 1.0 - alpha + 4.0 * alpha / PI;
    }
    let singular = 1.0 / (4.0 * alpha);
    if (x.abs() - singular).abs() < EPS {
        let s = x.signum() * singular;
        return 0.5 * (rrc_general(s - 10.0 * EPS, alpha) + rrc_general(s + 10.0 * EPS, alpha));
    }
    rrc_general(x, alpha)
}

fn rrc_general(x: f64, alpha: f64) -> f64 {
    let px = PI * x;
    let num = (px * (1.0 - alpha)).sin() + 4.0 * alpha * x * (px * (1.0 + alpha)).cos();
    let den = px * (1.0 - 16.0 * alpha * alpha * x * x);
    num / den
}

/// How many symbols either side of the current one contribute to the shaped waveform.
///
/// The RRC tail decays as 1/x, so a longer span is more faithful and more expensive. Six
/// symbols puts the truncation ripple far enough down not to widen the channel.
const RRC_SPAN: i64 = 6;

/// A sampled root-raised-cosine pulse, for looking up shaping weights cheaply.
///
/// The shaping needs one weight per contributing symbol per sample — thirteen evaluations of a
/// function full of transcendentals, which was costing more than the rest of the generator put
/// together. The pulse is smooth, so a fine table with linear interpolation is faithful to
/// about -110 dB and costs a couple of multiplies.
struct RrcTable {
    values: Vec<f64>,
    /// Table entries per symbol period.
    per_symbol: f64,
    span: f64,
}

impl RrcTable {
    fn new(alpha: f64) -> Self {
        const PER_SYMBOL: usize = 1024;
        let span = RRC_SPAN as f64;
        let n = (2.0 * span * PER_SYMBOL as f64) as usize + 1;
        let values = (0..n)
            .map(|i| rrc(-span + i as f64 / PER_SYMBOL as f64, alpha))
            .collect();
        Self { values, per_symbol: PER_SYMBOL as f64, span }
    }

    /// Interpolated pulse value `x` symbol periods from the centre.
    fn at(&self, x: f64) -> f64 {
        let pos = (x + self.span) * self.per_symbol;
        if pos < 0.0 {
            return 0.0;
        }
        let i = pos as usize;
        match (self.values.get(i), self.values.get(i + 1)) {
            (Some(a), Some(b)) => {
                let frac = pos - i as f64;
                a + (b - a) * frac
            }
            (Some(a), None) => *a,
            _ => 0.0,
        }
    }
}

/// Scaling that gives an RRC-shaped symbol stream unit mean power.
///
/// Mean power is `(1/T)∫h²`, evaluated once per call rather than per sample.
fn rrc_power_norm(alpha: f64) -> f64 {
    const STEPS: usize = 2048;
    let span = RRC_SPAN as f64;
    let dx = 2.0 * span / STEPS as f64;
    let energy: f64 = (0..STEPS)
        .map(|i| {
            let x = -span + (i as f64 + 0.5) * dx;
            rrc(x, alpha).powi(2) * dx
        })
        .sum();
    if energy > 0.0 { 1.0 / energy.sqrt() } else { 1.0 }
}

/// The QPSK symbol for index `k`, from a deterministic pseudo-random bit stream.
///
/// The previous sequence was `(k * 3 + 1) % 4`, which repeats every four symbols — a periodic
/// waveform whose spectrum is a handful of discrete lines rather than a modulated channel.
fn qpsk_symbol(k: i64) -> Complex<f64> {
    let mut h = (k as u64).wrapping_add(0x1234_5678).wrapping_mul(0x2545_F491_4F6C_DD1D);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    // Gray-coded quadrants at the standard 45° offsets.
    let angle = (h & 0b11) as f64 * PI / 2.0 + PI / 4.0;
    Complex::new(angle.cos(), angle.sin())
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
            // One source of truth for what a fresh tone looks like.
            tones: vec![Tone::default()],
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

            // A carrier with a stated channel bandwidth is a modulated signal, not a line.
            if tone.is_channel() {
                tone.synthesise_channel(&mut samples, sample_rate_mhz, start_time_us);
                continue;
            }

            match &tone.modulation {
                ToneModulation::Cw => {
                    for (i, sample) in samples.iter_mut().enumerate() {
                        let t = start_time_us + i as f64 * dt;
                        sample.re += amp * carrier_phase(tone.frequency_mhz, t, phase_rad).cos();
                    }
                }
                ToneModulation::Square | ToneModulation::Sawtooth | ToneModulation::Triangle => {
                    let series = HarmonicSeries::new(
                        &tone.modulation,
                        tone.frequency_mhz,
                        sample_rate_mhz / 2.0,
                    );
                    let mut stepper = series.stepper(
                        carrier_phase(tone.frequency_mhz, start_time_us, phase_rad),
                        2.0 * PI * tone.frequency_mhz * dt,
                    );
                    for sample in samples.iter_mut() {
                        sample.re += amp * stepper.next_value();
                    }
                }
                ToneModulation::AmModulated { depth_percent, mod_freq_khz } => {
                    // Carrier-referenced, as signal generators are: the envelope peaks at
                    // (1 + m) times the carrier, so deep modulation of a hot carrier really
                    // does overdrive whatever comes next.
                    let m = (depth_percent / 100.0).clamp(0.0, 1.0);
                    let f_m = mod_freq_khz / 1000.0;
                    for (i, sample) in samples.iter_mut().enumerate() {
                        let t = start_time_us + i as f64 * dt;
                        let env = 1.0 + m * carrier_phase(f_m, t, 0.0).cos();
                        sample.re += amp * env * carrier_phase(tone.frequency_mhz, t, phase_rad).cos();
                    }
                }
                ToneModulation::FmModulated { dev_mhz, mod_freq_khz } => {
                    let f_m = mod_freq_khz / 1000.0;
                    // Modulation index beta = deviation / modulating frequency; the sideband
                    // amplitudes come out as the Bessel functions J_n(beta).
                    let beta = if f_m > 0.0 { dev_mhz / f_m } else { 0.0 };
                    for (i, sample) in samples.iter_mut().enumerate() {
                        let t = start_time_us + i as f64 * dt;
                        let angle = carrier_phase(tone.frequency_mhz, t, phase_rad)
                            + beta * carrier_phase(f_m, t, 0.0).sin();
                        sample.re += amp * angle.cos();
                    }
                }
                ToneModulation::SweptChirp { sweep_period_us, triangular } => {
                    let period = sweep_period_us.max(1e-3);
                    let bw = if tone.bandwidth_mhz > 0.0 { tone.bandwidth_mhz } else { 100.0 };
                    let f_start = tone.frequency_mhz - bw / 2.0;
                    for (i, sample) in samples.iter_mut().enumerate() {
                        let t = start_time_us + i as f64 * dt;
                        let tau = t.rem_euclid(period);
                        let angle = chirp_phase(f_start, bw, period, tau, *triangular) + phase_rad;
                        sample.re += amp * angle.cos();
                    }
                }
                ToneModulation::PulsedRadar { pulse_width_us, pri_us, rise_ns, chirp_mhz } => {
                    let pri = pri_us.max(1e-3);
                    let pw = pulse_width_us.clamp(1e-4, pri);
                    let rise = (rise_ns / 1000.0).clamp(0.0, pw / 2.0);
                    for (i, sample) in samples.iter_mut().enumerate() {
                        let t = start_time_us + i as f64 * dt;
                        let tau = t.rem_euclid(pri);
                        let env = pulse_envelope(tau, pw, rise);
                        if env <= 0.0 {
                            continue;
                        }
                        // The carrier stays coherent pulse to pulse, as a real coherent radar
                        // does, and the intra-pulse chirp restarts with each pulse.
                        let mut angle = carrier_phase(tone.frequency_mhz, t, phase_rad);
                        if *chirp_mhz > 0.0 {
                            angle += chirp_phase(-chirp_mhz / 2.0, *chirp_mhz, pw, tau.min(pw), false);
                        }
                        sample.re += amp * env * angle.cos();
                    }
                }
                ToneModulation::FreqHopping { hop_step_mhz, num_channels, hop_rate_hz } => {
                    let n_chan = (*num_channels).max(1);
                    let hop_dur = 1e6 / hop_rate_hz.max(1e-3);
                    for (i, sample) in samples.iter_mut().enumerate() {
                        let t = start_time_us + i as f64 * dt;
                        let hop_idx = (t / hop_dur).floor() as i64;
                        let chan = hop_channel(hop_idx, n_chan);
                        let offset = (chan as f64 - (n_chan as f64 - 1.0) / 2.0) * hop_step_mhz;
                        let f_inst = (tone.frequency_mhz + offset).max(1e-6);
                        // Each hop lands wherever its own free-running phase would be, so hops
                        // are not phase-coherent with each other — as in a real hopping radio.
                        sample.re += amp * carrier_phase(f_inst, t, phase_rad).cos();
                    }
                }
                ToneModulation::DigitalQpsk { symbol_rate_msps, rrc_alpha } => {
                    let rs = symbol_rate_msps.max(1e-3);
                    let sym_dur = 1.0 / rs; // µs
                    let alpha = rrc_alpha.clamp(0.01, 1.0);
                    // Unit-power baseband, so the modulated carrier ends up with the same mean
                    // power as an unmodulated one at the same dBFS.
                    let shape = amp * rrc_power_norm(alpha);
                    let table = RrcTable::new(alpha);
                    // The symbols the block touches, worked out once: deriving each one costs a
                    // hash and two trigonometric calls, and every sample needs thirteen of them.
                    let first_sym = (start_time_us / sym_dur).floor() as i64 - RRC_SPAN;
                    let last_sym =
                        ((start_time_us + num_samples as f64 * dt) / sym_dur).floor() as i64
                            + RRC_SPAN;
                    let symbols: Vec<Complex<f64>> =
                        (first_sym..=last_sym).map(qpsk_symbol).collect();

                    for (i, sample) in samples.iter_mut().enumerate() {
                        let t = start_time_us + i as f64 * dt;
                        let x = t / sym_dur;
                        let centre = x.floor() as i64;
                        let mut acc = Complex::new(0.0, 0.0);
                        for k in (centre - RRC_SPAN)..=(centre + RRC_SPAN) {
                            let sym = symbols[(k - first_sym).clamp(0, symbols.len() as i64 - 1) as usize];
                            acc += sym * table.at(x - k as f64);
                        }
                        // Real passband signal: I on the cosine, Q on the sine.
                        let theta = carrier_phase(tone.frequency_mhz, t, phase_rad);
                        sample.re += shape * (acc.re * theta.cos() - acc.im * theta.sin());
                    }
                }
            }
        }

        // Additive white Gaussian noise, Box-Muller from a xorshift stream.
        //
        // The noise goes into the real component only, because that is the one physical
        // quantity a wire carries — and it is the only part the converter samples. Splitting
        // it across I and Q instead put half the configured power somewhere that gets
        // discarded, so the floor read 3 dB below its setting.
        if self.noise_enabled {
            let noise_amp = 10.0_f64.powf(self.noise_floor_dbfs / 20.0);
            let mut seed: u64 = 0xDEAD_BEEF_CAFE_BABE ^ (start_time_us.to_bits());
            let mut spare: Option<f64> = None;

            for sample in &mut samples {
                let g = match spare.take() {
                    Some(v) => v,
                    None => {
                        let mut next_unit = || {
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            (seed >> 11) as f64 / (1u64 << 53) as f64
                        };
                        let u1 = next_unit().max(1e-300);
                        let u2 = next_unit();
                        let r = (-2.0 * u1.ln()).sqrt();
                        let theta = 2.0 * PI * u2;
                        spare = Some(r * theta.sin());
                        r * theta.cos()
                    }
                };
                sample.re += noise_amp * g;
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
    use rustfft::FftPlanner;

    const FS: f64 = 15000.0;
    /// A record length where a 3000 MHz carrier lands exactly on a bin, so the DFT is coherent
    /// and leakage cannot be mistaken for occupied bandwidth. 15000 samples at 15 GHz is 1 µs.
    const N: usize = 15000;
    /// Ten microseconds, for anything whose repetition interval is measured in microseconds.
    const N_LONG: usize = 150_000;

    fn make_gen(modulation: ToneModulation, freq_mhz: f64, bandwidth_mhz: f64) -> SignalGenerator {
        SignalGenerator {
            tones: vec![Tone {
                frequency_mhz: freq_mhz,
                amplitude_dbfs: 0.0,
                phase_deg: 0.0,
                bandwidth_mhz,
                modulation,
            }],
            noise_floor_dbfs: -300.0,
            noise_enabled: false,
        }
    }

    /// Amplitude of the component at `f_mhz`, by coherent detection.
    fn probe(samples: &[Complex<f64>], f_mhz: f64) -> f64 {
        let mut acc = Complex::new(0.0, 0.0);
        for (i, v) in samples.iter().enumerate() {
            let th = -2.0 * PI * f_mhz * i as f64 / FS;
            acc += v.re * Complex::new(th.cos(), th.sin());
        }
        2.0 * acc.norm() / samples.len() as f64
    }

    fn probe_db(samples: &[Complex<f64>], f_mhz: f64) -> f64 {
        20.0 * probe(samples, f_mhz).max(1e-300).log10()
    }

    /// One-sided power spectrum, for occupancy measurements.
    ///
    /// Windowed with Blackman-Harris: anything that is not exactly periodic in the record —
    /// QPSK data, a pulse train at an unrelated rate — leaks badly through a rectangular
    /// window, and that leakage is indistinguishable from real occupied bandwidth.
    fn spectrum(samples: &[Complex<f64>]) -> Vec<f64> {
        let n = samples.len();
        let mut buf: Vec<Complex<f64>> = samples
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let x = 2.0 * PI * i as f64 / n as f64;
                let w = 0.35875 - 0.48829 * x.cos() + 0.14128 * (2.0 * x).cos()
                    - 0.01168 * (3.0 * x).cos();
                Complex::new(s.re * w, 0.0)
            })
            .collect();
        FftPlanner::new().plan_fft_forward(n).process(&mut buf);
        buf.iter().take(n / 2).map(|c| c.norm_sqr()).collect()
    }

    /// Span around `centre_mhz` holding `frac` of the total power, in MHz.
    fn occupied_bw(samples: &[Complex<f64>], centre_mhz: f64, frac: f64) -> f64 {
        let mag = spectrum(samples);
        let total: f64 = mag.iter().sum();
        let df = FS / samples.len() as f64;
        let centre = (centre_mhz / df).round() as usize;
        let mut acc = mag[centre];
        let mut half = 0usize;
        while acc < frac * total && half < mag.len() / 2 {
            half += 1;
            acc += mag.get(centre + half).copied().unwrap_or(0.0);
            acc += mag.get(centre.saturating_sub(half)).copied().unwrap_or(0.0);
        }
        2.0 * half as f64 * df
    }

    fn mean_power(samples: &[Complex<f64>]) -> f64 {
        samples.iter().map(|s| s.re * s.re).sum::<f64>() / samples.len() as f64
    }

    /// Bessel function of the first kind, by its power series. Only needed for the FM check.
    fn bessel_j(n: i32, x: f64) -> f64 {
        let mut sum = 0.0;
        for k in 0..40 {
            let mut term = (-1.0_f64).powi(k as i32) / (factorial(k) * factorial(k + n as usize));
            term *= (x / 2.0).powi(2 * k as i32 + n);
            sum += term;
        }
        sum
    }

    fn factorial(n: usize) -> f64 {
        (1..=n).map(|v| v as f64).product::<f64>().max(1.0)
    }

    // -----------------------------------------------------------------------
    // Housekeeping
    // -----------------------------------------------------------------------

    #[test]
    fn generator_produces_correct_sample_count() {
        assert_eq!(SignalGenerator::default().generate(1024, 1000.0).len(), 1024);
    }

    #[test]
    fn every_waveform_is_real_and_carries_power() {
        for modulation in ToneModulation::all_variants() {
            let label = modulation.to_string();
            // Long enough, and starting early enough, to catch a pulse from the pulsed modes.
            let s = make_gen(modulation, 3000.0, 100.0).generate_at_time(N_LONG, FS, 0.0);
            let imag: f64 = s.iter().map(|v| v.im.abs()).sum();
            assert!(imag < 1e-12, "{label} produced imaginary voltage: {imag}");
            assert!(mean_power(&s) > 1e-6, "{label} produced no power");
        }
    }

    #[test]
    fn every_waveform_joins_across_frames() {
        // The generator's contract: a frame starting at t is the same samples a longer run
        // through t would have produced. Anything that resets a phase accumulator breaks this,
        // and it shows up as a discontinuity in the waterfall.
        let n = 512;
        let dt = 1.0 / FS;
        for modulation in ToneModulation::all_variants() {
            let label = modulation.to_string();
            // The noise-like channel is redrawn per frame by design, so exclude that case.
            let g = make_gen(modulation, 3000.0, 0.0);
            let continuous = g.generate_at_time(2 * n, FS, 0.0);
            let second = g.generate_at_time(n, FS, n as f64 * dt);
            let worst = second
                .iter()
                .zip(continuous[n..].iter())
                .map(|(a, b)| (a.re - b.re).abs())
                .fold(0.0f64, f64::max);
            assert!(worst < 1e-9, "{label} jumped by {worst} at the frame boundary");
        }
    }

    #[test]
    fn amplitude_is_the_peak_of_the_waveform() {
        // dBFS is a peak reference, so every carrier-like waveform should just touch it.
        for modulation in [
            ToneModulation::Cw,
            ToneModulation::Square,
            ToneModulation::Sawtooth,
            ToneModulation::Triangle,
        ] {
            let label = modulation.to_string();
            let g = make_gen(modulation, 100.0, 0.0);
            let s = g.generate_at_time(N, FS, 0.0);
            let peak = s.iter().map(|v| v.re.abs()).fold(0.0f64, f64::max);
            assert!(
                (peak - 1.0).abs() < 0.02,
                "{label} peaked at {peak}, expected full scale"
            );
        }
    }

    #[test]
    fn tone_linear_amplitude() {
        let t = Tone { amplitude_dbfs: -6.0, ..Default::default() };
        assert!((t.linear_amplitude() - 0.501).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Carrier and periodic waveforms
    // -----------------------------------------------------------------------

    #[test]
    fn cw_is_a_single_line_at_the_right_level() {
        for dbfs in [0.0, -6.0, -40.0] {
            let mut g = make_gen(ToneModulation::Cw, 3000.0, 0.0);
            g.tones[0].amplitude_dbfs = dbfs;
            let s = g.generate_at_time(N, FS, 0.0);
            assert!(
                (probe_db(&s, 3000.0) - dbfs).abs() < 0.01,
                "{dbfs} dBFS carrier measured {}",
                probe_db(&s, 3000.0)
            );
            // All the power in one place: the analysis window spreads a line over a few bins,
            // and essentially nothing should sit outside them.
            let mag = spectrum(&s);
            let total: f64 = mag.iter().sum();
            let bin = (3000.0 / (FS / N as f64)).round() as usize;
            let in_line: f64 = mag[bin - 3..=bin + 3].iter().sum();
            assert!(in_line > 0.999 * total, "line holds {} of the power", in_line / total);
        }
    }

    #[test]
    fn phase_control_turns_a_cosine_into_a_sine() {
        let mut g = make_gen(ToneModulation::Cw, 3000.0, 0.0);
        g.tones[0].phase_deg = -90.0;
        let s = g.generate_at_time(N, FS, 0.0);
        // A -90 degree cosine starts at zero and rises.
        assert!(s[0].re.abs() < 1e-9, "starts at {}", s[0].re);
        assert!(s[1].re > 0.0);
        // The level is unchanged.
        assert!(probe_db(&s, 3000.0).abs() < 0.01);
    }

    #[test]
    fn square_wave_has_the_right_harmonics_and_no_aliases() {
        let f0 = 100.0;
        let s = make_gen(ToneModulation::Square, f0, 0.0).generate_at_time(N, FS, 0.0);
        let fundamental = probe(&s, f0);

        // Odd harmonics fall as 1/n; even ones are absent.
        for n in [3.0, 5.0, 7.0, 9.0, 11.0] {
            let ratio = probe(&s, n * f0) / fundamental;
            assert!(
                (ratio - 1.0 / n).abs() < 0.02,
                "harmonic {n}: ratio {ratio}, expected {}",
                1.0 / n
            );
        }
        for n in [2.0, 4.0, 6.0] {
            assert!(
                probe(&s, n * f0) / fundamental < 1e-6,
                "even harmonic {n} should be absent"
            );
        }

        // Nothing above the harmonic ceiling, and in particular nothing folded back down.
        let top = top_harmonic_mhz(&ToneModulation::Square, f0, FS);
        assert!(top <= FS / 2.0, "top harmonic {top} is above Nyquist");
        let mag = spectrum(&s);
        let df = FS / N as f64;
        let above: f64 = mag
            .iter()
            .enumerate()
            .filter(|(i, _)| *i as f64 * df > top + f0)
            .map(|(_, m)| *m)
            .sum();
        let total: f64 = mag.iter().sum();
        assert!(
            above / total < 1e-9,
            "energy above the band limit: {}",
            above / total
        );
    }

    #[test]
    fn sawtooth_and_triangle_follow_their_series() {
        let f0 = 100.0;
        let saw = make_gen(ToneModulation::Sawtooth, f0, 0.0).generate_at_time(N, FS, 0.0);
        let fund = probe(&saw, f0);
        // Sawtooth: every harmonic, falling as 1/n.
        for n in [2.0, 3.0, 4.0, 5.0] {
            let ratio = probe(&saw, n * f0) / fund;
            assert!(
                (ratio - 1.0 / n).abs() < 0.02,
                "sawtooth harmonic {n}: {ratio} vs {}",
                1.0 / n
            );
        }

        let tri = make_gen(ToneModulation::Triangle, f0, 0.0).generate_at_time(N, FS, 0.0);
        let fund = probe(&tri, f0);
        // Triangle: odd harmonics only, falling as 1/n².
        for n in [3.0, 5.0, 7.0] {
            let ratio = probe(&tri, n * f0) / fund;
            assert!(
                (ratio - 1.0 / (n * n)).abs() < 0.02,
                "triangle harmonic {n}: {ratio} vs {}",
                1.0 / (n * n)
            );
        }
        assert!(probe(&tri, 2.0 * f0) / fund < 1e-6);
    }

    #[test]
    fn a_low_frequency_square_stays_below_nyquist() {
        // 10 MHz has room for 750 harmonics before Nyquist; the cap stands in for the finite
        // rise time a real generator has, and must not be exceeded either way.
        let top = top_harmonic_mhz(&ToneModulation::Square, 10.0, FS);
        assert!(top <= FS / 2.0);
        assert!(top <= 10.0 * MAX_HARMONICS as f64);
        // And a high frequency keeps at least the fundamental.
        assert!(top_harmonic_mhz(&ToneModulation::Square, 7000.0, FS) >= 7000.0);
    }

    // -----------------------------------------------------------------------
    // Analog modulation
    // -----------------------------------------------------------------------

    #[test]
    fn am_sidebands_sit_at_half_the_depth() {
        let f_c = 3000.0;
        let f_m_khz = 10_000.0; // 10 MHz, well resolved by this record
        for depth in [25.0, 50.0, 100.0] {
            let g = make_gen(
                ToneModulation::AmModulated { depth_percent: depth, mod_freq_khz: f_m_khz },
                f_c,
                0.0,
            );
            let s = g.generate_at_time(N, FS, 0.0);
            let carrier = probe(&s, f_c);
            let upper = probe(&s, f_c + f_m_khz / 1000.0);
            let lower = probe(&s, f_c - f_m_khz / 1000.0);
            let m = depth / 100.0;
            // Each sideband is m/2 of the carrier.
            assert!(
                (upper / carrier - m / 2.0).abs() < 0.01,
                "depth {depth}%: upper sideband ratio {}",
                upper / carrier
            );
            assert!((lower - upper).abs() < 1e-6, "sidebands should be symmetric");
        }
    }

    #[test]
    fn am_envelope_peaks_above_the_carrier() {
        // Carrier-referenced, as instruments are: 100% modulation doubles the peak.
        let mut g = make_gen(
            ToneModulation::AmModulated { depth_percent: 100.0, mod_freq_khz: 10_000.0 },
            3000.0,
            0.0,
        );
        g.tones[0].amplitude_dbfs = -6.0;
        let s = g.generate_at_time(N, FS, 0.0);
        let peak = s.iter().map(|v| v.re.abs()).fold(0.0f64, f64::max);
        let carrier_amp = 10.0_f64.powf(-6.0 / 20.0);
        assert!(
            (peak - 2.0 * carrier_amp).abs() < 0.02,
            "peak {peak}, expected {}",
            2.0 * carrier_amp
        );
    }

    #[test]
    fn fm_sidebands_follow_the_bessel_functions() {
        // The textbook signature of FM: sideband n sits at J_n(beta) of the unmodulated
        // carrier. Nothing else produces that pattern, so it pins the implementation down.
        let f_c = 3000.0;
        let f_m = 20.0; // MHz
        let beta = 2.0;
        let g = make_gen(
            ToneModulation::FmModulated { dev_mhz: beta * f_m, mod_freq_khz: f_m * 1000.0 },
            f_c,
            0.0,
        );
        let s = g.generate_at_time(N, FS, 0.0);

        for n in 0..=4 {
            let expected = bessel_j(n, beta).abs();
            let measured = probe(&s, f_c + n as f64 * f_m);
            assert!(
                (measured - expected).abs() < 0.02,
                "sideband {n}: measured {measured}, J_{n}({beta}) = {expected}"
            );
        }
    }

    #[test]
    fn fm_carrier_nulls_at_the_first_bessel_zero() {
        // J_0(2.4048) = 0, so the carrier disappears entirely at that modulation index.
        let f_c = 3000.0;
        let f_m = 20.0;
        let g = make_gen(
            ToneModulation::FmModulated {
                dev_mhz: 2.404_825_557_695_773 * f_m,
                mod_freq_khz: f_m * 1000.0,
            },
            f_c,
            0.0,
        );
        let s = g.generate_at_time(N, FS, 0.0);
        let carrier = probe_db(&s, f_c);
        let sideband = probe_db(&s, f_c + f_m);
        assert!(
            carrier < sideband - 40.0,
            "carrier {carrier} dB should have nulled well below the sideband {sideband} dB"
        );
    }

    // -----------------------------------------------------------------------
    // Chirps
    // -----------------------------------------------------------------------

    #[test]
    fn chirp_sweeps_the_requested_band() {
        let f_c = 3000.0;
        for bw in [50.0, 200.0, 800.0] {
            let g = make_gen(
                ToneModulation::SweptChirp { sweep_period_us: 1.0, triangular: false },
                f_c,
                bw,
            );
            let s = g.generate_at_time(N_LONG, FS, 0.0);
            let occupied = occupied_bw(&s, f_c, 0.99);
            assert!(
                (occupied - bw).abs() < 0.25 * bw,
                "asked for {bw} MHz, occupied {occupied:.1} MHz"
            );
        }
    }

    #[test]
    fn triangular_chirp_is_continuous_through_the_turn() {
        // The whole point of triangular FMCW is that it does not jump at the top of the sweep.
        let period = 1.0;
        let bw = 200.0;
        let g = make_gen(
            ToneModulation::SweptChirp { sweep_period_us: period, triangular: true },
            3000.0,
            bw,
        );
        let s = g.generate_at_time(N_LONG, FS, 0.0);
        // Sample either side of the turning point and check the waveform has no step in it.
        let turn = (period / 2.0 * FS) as usize;
        let step = |i: usize| (s[i + 1].re - s[i].re).abs();
        let typical: f64 = (100..200).map(step).sum::<f64>() / 100.0;
        assert!(
            step(turn) < 5.0 * typical.max(1e-6),
            "step of {} at the turn against a typical {typical}",
            step(turn)
        );
        // And it still covers the band.
        assert!((occupied_bw(&s, 3000.0, 0.99) - bw).abs() < 0.3 * bw);
    }

    // -----------------------------------------------------------------------
    // Pulsed radar
    // -----------------------------------------------------------------------

    #[test]
    fn pulse_duty_cycle_sets_the_average_power() {
        let pw = 0.1;
        let pri = 1.0;
        let g = make_gen(
            ToneModulation::PulsedRadar {
                pulse_width_us: pw,
                pri_us: pri,
                rise_ns: 0.0,
                chirp_mhz: 0.0,
            },
            3000.0,
            0.0,
        );
        let s = g.generate_at_time(N, FS, 0.0);
        // A full-scale sine at 10% duty averages 0.5 x 0.1.
        let expected = 0.5 * pw / pri;
        let measured = mean_power(&s);
        assert!(
            (measured / expected - 1.0).abs() < 0.05,
            "measured {measured}, expected {expected}"
        );
    }

    #[test]
    fn pulse_spectrum_has_lines_at_the_prf() {
        let pri = 1.0; // µs -> 1 MHz PRF
        let g = make_gen(
            ToneModulation::PulsedRadar {
                pulse_width_us: 0.1,
                pri_us: pri,
                rise_ns: 0.0,
                chirp_mhz: 0.0,
            },
            3000.0,
            0.0,
        );
        let s = g.generate_at_time(N_LONG, FS, 0.0);
        let prf = 1.0 / pri; // MHz
        // Lines at the carrier and at multiples of the PRF; nothing halfway between.
        let on_line = probe(&s, 3000.0 + prf);
        let off_line = probe(&s, 3000.0 + prf / 2.0);
        assert!(
            on_line > 100.0 * off_line,
            "line {on_line} against the gap {off_line}"
        );
    }

    #[test]
    fn pulse_rise_time_softens_the_skirts() {
        let make = |rise_ns: f64| {
            make_gen(
                ToneModulation::PulsedRadar {
                    pulse_width_us: 0.2,
                    pri_us: 1.0,
                    rise_ns,
                    chirp_mhz: 0.0,
                },
                3000.0,
                0.0,
            )
            .generate_at_time(N_LONG, FS, 0.0)
        };
        let sharp = make(0.0);
        let soft = make(50.0);
        // Far out in the skirts, a finite edge is worth a lot of suppression.
        let far = 3000.0 + 60.0;
        assert!(
            probe_db(&soft, far) < probe_db(&sharp, far) - 15.0,
            "soft {} vs sharp {} dB at {far} MHz",
            probe_db(&soft, far),
            probe_db(&sharp, far)
        );
        // The edges cost a predictable amount of energy: the mean square of a raised-cosine
        // ramp is 3/8, so a pulse of width w with ramps of r carries w - 1.25r.
        let pw = 0.2;
        let rise = 0.05;
        let expected = (pw - 1.25 * rise) / pw;
        let ratio = mean_power(&soft) / mean_power(&sharp);
        assert!(
            (ratio - expected).abs() < 0.03,
            "edge energy ratio {ratio}, expected {expected}"
        );
    }

    #[test]
    fn intra_pulse_chirp_widens_the_pulse_spectrum() {
        let make = |chirp_mhz: f64| {
            make_gen(
                ToneModulation::PulsedRadar {
                    pulse_width_us: 0.5,
                    pri_us: 2.0,
                    rise_ns: 10.0,
                    chirp_mhz,
                },
                3000.0,
                0.0,
            )
            .generate_at_time(N_LONG, FS, 0.0)
        };
        let plain = make(0.0);
        let chirped = make(100.0);
        // An unmodulated 0.5 µs pulse is a few MHz wide; sweeping 100 MHz across it is not.
        let plain_bw = occupied_bw(&plain, 3000.0, 0.9);
        let chirped_bw = occupied_bw(&chirped, 3000.0, 0.9);
        assert!(plain_bw < 20.0, "plain pulse occupied {plain_bw} MHz");
        assert!(
            (chirped_bw - 100.0).abs() < 30.0,
            "chirped pulse occupied {chirped_bw} MHz, expected about 100"
        );
    }

    // -----------------------------------------------------------------------
    // Frequency hopping
    // -----------------------------------------------------------------------

    #[test]
    fn hop_sequence_visits_every_channel() {
        // A stride-based sequence shares a factor with some channel counts and then sits still.
        // Seven channels with a stride of seven was the case that never moved at all.
        for n_chan in 2..=32usize {
            let mut seen = vec![false; n_chan];
            for idx in 0..(n_chan * 60) as i64 {
                seen[hop_channel(idx, n_chan)] = true;
            }
            assert!(
                seen.iter().all(|v| *v),
                "{n_chan} channels: only visited {} of them",
                seen.iter().filter(|v| **v).count()
            );
        }
    }

    #[test]
    fn hopping_lands_on_the_channel_grid() {
        let step = 20.0;
        let n_chan = 8usize;
        let f_c = 3000.0;
        let hop_rate = 1e6; // 1 µs dwell
        let g = make_gen(
            ToneModulation::FreqHopping {
                hop_step_mhz: step,
                num_channels: n_chan,
                hop_rate_hz: hop_rate,
            },
            f_c,
            0.0,
        );

        // Look at one dwell at a time and confirm the tone is on a grid point.
        let dwell_samples = (1e6 / hop_rate * FS) as usize;
        for hop in 0..8i64 {
            let t0 = hop as f64 * 1e6 / hop_rate;
            let s = g.generate_at_time(dwell_samples, FS, t0);
            let expected_chan = hop_channel(hop, n_chan);
            let expected_f =
                f_c + (expected_chan as f64 - (n_chan as f64 - 1.0) / 2.0) * step;
            // The dwell is short, so compare against the neighbouring grid points rather than
            // resolving the line precisely.
            let mut best = (0.0, 0.0);
            for c in 0..n_chan {
                let f = f_c + (c as f64 - (n_chan as f64 - 1.0) / 2.0) * step;
                let p = probe(&s, f);
                if p > best.0 {
                    best = (p, f);
                }
            }
            assert!(
                (best.1 - expected_f).abs() < 1e-6,
                "hop {hop}: strongest at {} MHz, expected {expected_f}",
                best.1
            );
        }
    }

    // -----------------------------------------------------------------------
    // QPSK
    // -----------------------------------------------------------------------

    #[test]
    fn qpsk_occupies_one_plus_alpha_symbol_rates() {
        let f_c = 3000.0;
        for (rs, alpha) in [(20.0, 0.35), (50.0, 0.2), (100.0, 0.5)] {
            let g = make_gen(
                ToneModulation::DigitalQpsk { symbol_rate_msps: rs, rrc_alpha: alpha },
                f_c,
                0.0,
            );
            let s = g.generate_at_time(N, FS, 0.0);
            let occupied = occupied_bw(&s, f_c, 0.99);
            let expected = (1.0 + alpha) * rs;
            assert!(
                (occupied - expected).abs() < 0.3 * expected,
                "Rs {rs} alpha {alpha}: occupied {occupied:.1} MHz, expected about {expected:.1}"
            );
        }
    }

    #[test]
    fn qpsk_data_is_not_periodic() {
        // The old sequence repeated every four symbols, which makes a line spectrum rather than
        // a modulated channel. Check the symbols do not repeat on any short period.
        for period in 1..=16i64 {
            let mut identical = true;
            for k in 0..200i64 {
                if (qpsk_symbol(k) - qpsk_symbol(k + period)).norm() > 1e-9 {
                    identical = false;
                    break;
                }
            }
            assert!(!identical, "symbol stream repeats every {period}");
        }
    }

    #[test]
    fn qpsk_symbols_are_the_four_constellation_points() {
        let mut quadrants = [0usize; 4];
        for k in 0..4000i64 {
            let s = qpsk_symbol(k);
            assert!((s.norm() - 1.0).abs() < 1e-12, "symbol off the unit circle");
            let q = match (s.re > 0.0, s.im > 0.0) {
                (true, true) => 0,
                (false, true) => 1,
                (false, false) => 2,
                (true, false) => 3,
            };
            quadrants[q] += 1;
        }
        // Roughly uniform over the four points.
        for (q, count) in quadrants.iter().enumerate() {
            assert!(
                (*count as f64 - 1000.0).abs() < 200.0,
                "quadrant {q} got {count} of 4000 symbols"
            );
        }
    }

    #[test]
    fn rrc_is_a_nyquist_pulse() {
        // A root-raised-cosine impulse response peaks at the symbol instant. Its square-root
        // pairing means it is not itself zero at every other symbol, but it must be small.
        for alpha in [0.1, 0.35, 1.0] {
            assert!(rrc(0.0, alpha) > 0.5, "peak too low for alpha {alpha}");
            for k in 1..=4 {
                assert!(
                    rrc(k as f64, alpha).abs() < 0.3,
                    "alpha {alpha}: tail at {k} symbols is {}",
                    rrc(k as f64, alpha)
                );
            }
            // And the singular points evaluate to something finite.
            let sing = rrc(1.0 / (4.0 * alpha), alpha);
            assert!(sing.is_finite(), "singularity not handled for alpha {alpha}");
        }
    }

    // -----------------------------------------------------------------------
    // Channels and noise
    // -----------------------------------------------------------------------

    #[test]
    fn channel_bandwidth_actually_occupies_the_band() {
        for bw in [20.0, 100.0, 500.0] {
            let s = make_gen(ToneModulation::Cw, 3000.0, bw).generate(N, FS);
            let occupied = occupied_bw(&s, 3000.0, 0.99);
            assert!(
                occupied > 0.5 * bw && occupied < 2.0 * bw,
                "asked for {bw} MHz, occupied {occupied:.1} MHz"
            );
        }
    }

    #[test]
    fn channel_carries_the_same_power_as_the_carrier() {
        // Switching the bandwidth on should not move the level.
        let line = make_gen(ToneModulation::Cw, 3000.0, 0.0).generate(N, FS);
        let channel = make_gen(ToneModulation::Cw, 3000.0, 200.0).generate(N, FS);
        let db = 10.0 * (mean_power(&channel) / mean_power(&line)).log10();
        assert!(db.abs() < 0.1, "channel is {db} dB off the carrier it replaced");
    }

    #[test]
    fn noise_floor_lands_on_its_setting() {
        for floor in [-40.0, -80.0, -120.0] {
            let g = SignalGenerator { tones: vec![], noise_floor_dbfs: floor, noise_enabled: true };
            let s = g.generate_at_time(1 << 15, FS, 7.5);
            let db = 10.0 * mean_power(&s).log10();
            assert!((db - floor).abs() < 0.5, "asked for {floor} dBFS, measured {db}");
            let q: f64 = s.iter().map(|v| v.im.abs()).sum();
            assert!(q < 1e-12, "the analog domain is real, but Q carried {q}");
        }
    }

    #[test]
    fn tones_add_independently() {
        // Two tones at the same level should each read at that level, and together carry 3 dB
        // more power than one.
        let g = SignalGenerator {
            tones: vec![
                Tone {
                    frequency_mhz: 2000.0,
                    amplitude_dbfs: -6.0,
                    phase_deg: 0.0,
                    bandwidth_mhz: 0.0,
                    modulation: ToneModulation::Cw,
                },
                Tone {
                    frequency_mhz: 2020.0,
                    amplitude_dbfs: -6.0,
                    phase_deg: 0.0,
                    bandwidth_mhz: 0.0,
                    modulation: ToneModulation::Cw,
                },
            ],
            noise_floor_dbfs: -300.0,
            noise_enabled: false,
        };
        let s = g.generate_at_time(N, FS, 0.0);
        assert!((probe_db(&s, 2000.0) + 6.0).abs() < 0.01);
        assert!((probe_db(&s, 2020.0) + 6.0).abs() < 0.01);
        let single = 0.5 * 10.0_f64.powf(-6.0 / 10.0);
        assert!((10.0 * (mean_power(&s) / single).log10() - 3.01).abs() < 0.05);
    }

    // -----------------------------------------------------------------------
    // IQ file loading
    // -----------------------------------------------------------------------

    #[test]
    fn iq_file_loader_sc16_parsing() {
        let file_path = std::env::temp_dir().join("test_sc16.iq");
        let mut data = Vec::new();
        data.extend_from_slice(&32767i16.to_le_bytes());
        data.extend_from_slice(&(-32768i16).to_le_bytes());
        data.extend_from_slice(&0i16.to_le_bytes());
        data.extend_from_slice(&16384i16.to_le_bytes());
        std::fs::write(&file_path, &data).unwrap();

        let loader = IqFileLoader {
            path: Some(file_path.clone()),
            format: IqFormat::Sc16,
            ..Default::default()
        };
        let samples = loader.load().unwrap();
        assert_eq!(samples.len(), 2);
        assert!((samples[0].re - (32767.0 / 32768.0)).abs() < 1e-6);
        assert!((samples[0].im - (-1.0)).abs() < 1e-6);
        assert!((samples[1].im - 0.5).abs() < 1e-6);

        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn iq_file_loader_generate_at_time_repeating() {
        let file_path = std::env::temp_dir().join("test_repeat.csv");
        std::fs::write(&file_path, "1.0, 1.0\n2.0, 2.0\n3.0, 3.0\n4.0, 4.0\n").unwrap();

        let loader = IqFileLoader {
            path: Some(file_path.clone()),
            format: IqFormat::Csv,
            sample_rate_mhz: 1.0,
            repeat: true,
            repeat_period_us: 2.0,
            ..Default::default()
        };

        // File is 4 µs long with a 2 µs gap, so the cycle is 6 µs.
        let out = loader.generate_at_time(10, 1.0, 0.0).unwrap();
        assert_eq!(out[0], Complex::new(1.0, 1.0));
        assert_eq!(out[3], Complex::new(4.0, 4.0));
        assert_eq!(out[4], Complex::new(0.0, 0.0));
        assert_eq!(out[5], Complex::new(0.0, 0.0));
        assert_eq!(out[6], Complex::new(1.0, 1.0));
        assert_eq!(out[9], Complex::new(4.0, 4.0));

        std::fs::remove_file(file_path).unwrap();
    }
}
