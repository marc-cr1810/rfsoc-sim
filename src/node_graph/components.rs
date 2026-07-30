//! RF component models — the analog physics behind each node in the front-end graph.
//!
//! # Conventions
//!
//! **The analog domain is real.** Every component here takes and returns a real-valued
//! voltage waveform, carried in a `Complex<f64>` buffer with a zero imaginary part purely so
//! the same containers work either side of the ADC. A physical two-port has a conjugate-
//! symmetric transfer function — `H(-f) = H*(f)` — so a real input can only ever produce a
//! real output. [`apply_transfer_function`] enforces that, which is what keeps a mixer
//! producing *both* sidebands and a phase shifter from turning a cosine into something that
//! vanishes when the converter samples it.
//!
//! **Amplitude reference.** `1.0` is the ADC's full-scale input voltage, so a sine of
//! amplitude `a` sits at `20·log10(a)` dBFS. Absolute power figures (P1dB, OIP3, kTB) need a
//! physical anchor for that, which is [`FULL_SCALE_DBM`].

#![allow(dead_code)]

use num_complex::Complex;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::f64::consts::PI;

thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

// ---------------------------------------------------------------------------
// Reference plane: normalised voltage <-> absolute power
// ---------------------------------------------------------------------------

/// Absolute power of a full-scale sine at the converter input, in dBm.
///
/// The ZU48DR RF-ADC full-scale input is 1 V peak-to-peak differential into 100 Ω, so
/// `Vp = 0.5 V` and `P = Vp²/2R = 0.25/200 = 1.25 mW`, i.e. +0.97 dBm. Every dBm figure in
/// the RF chain (P1dB, OIP3, thermal noise) is referred to this one anchor, which is what
/// lets a normalised waveform be compared against a datasheet number at all.
pub const FULL_SCALE_DBM: f64 = 0.97;

/// Thermal noise power spectral density at 290 K, in dBm/Hz (`10·log10(kT₀)`).
pub const KT0_DBM_PER_HZ: f64 = -173.98;

/// Reference temperature for noise-figure definitions, in kelvin (IEEE T₀ = 290 K).
pub const T0_KELVIN: f64 = 290.0;

/// Peak amplitude of a sine at `dbm`, in units of ADC full scale.
pub fn dbm_to_amplitude(dbm: f64) -> f64 {
    10.0_f64.powf((dbm - FULL_SCALE_DBM) / 20.0)
}

/// Absolute power in dBm of a sine of peak amplitude `amp` (full scale = 1.0).
pub fn amplitude_to_dbm(amp: f64) -> f64 {
    20.0 * amp.max(1e-300).log10() + FULL_SCALE_DBM
}

/// Available thermal noise power in a bandwidth, as a normalised power (`1.0` = full scale).
///
/// `N = kTB`, converted through [`FULL_SCALE_DBM`]. At 290 K across a 15 GHz simulation band
/// this is −73.2 dBFS of total power, which spread over a 64k-point FFT puts the per-bin
/// floor near −121 dBFS — the same place a real wideband receiver sits.
pub fn thermal_noise_power(bandwidth_mhz: f64, temperature_k: f64) -> f64 {
    if bandwidth_mhz <= 0.0 || temperature_k <= 0.0 {
        return 0.0;
    }
    let bw_hz = bandwidth_mhz * 1e6;
    let dbm = KT0_DBM_PER_HZ + 10.0 * (temperature_k / T0_KELVIN).log10() + 10.0 * bw_hz.log10();
    10.0_f64.powf((dbm - FULL_SCALE_DBM) / 10.0)
}

// ---------------------------------------------------------------------------
// Per-evaluation context
// ---------------------------------------------------------------------------

/// Per-frame state that every component needs but none of them own.
#[derive(Debug, Clone, Copy)]
pub struct ChainCtx {
    /// Wideband simulation rate in MHz.
    pub sample_rate_mhz: f64,
    /// Absolute simulation timestamp of the first sample, in µs.
    ///
    /// Local oscillators run off this rather than off a per-block index, so a mixer stays
    /// phase-continuous from frame to frame exactly as the signal generator does.
    pub time_us: f64,
    /// Physical temperature of the chain in kelvin, or `None` to disable thermal noise.
    pub temperature_k: Option<f64>,
    /// Per-node seed, so each stage's noise is independent but reproducible.
    pub seed: u64,
}

impl ChainCtx {
    pub fn new(sample_rate_mhz: f64, time_us: f64, temperature_k: Option<f64>) -> Self {
        Self { sample_rate_mhz, time_us, temperature_k, seed: 0 }
    }

    /// Derive the context seen by one node.
    pub fn for_node(&self, node_seed: u64) -> Self {
        Self { seed: node_seed, ..*self }
    }

    /// An RNG unique to this node, frame and purpose.
    fn rng(&self, salt: u64) -> Rng {
        Rng::new(
            self.seed
                ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ self.time_us.to_bits().rotate_left(17),
        )
    }
}

/// xorshift64* with Box-Muller on top. Deterministic, and fast enough to run per sample.
struct Rng {
    state: u64,
    spare: Option<f64>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1, spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal deviate.
    fn gaussian(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.next_unit().max(1e-300);
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

// ---------------------------------------------------------------------------
// Frequency-domain machinery
// ---------------------------------------------------------------------------

/// Map an FFT bin index to its signed frequency in MHz.
fn bin_freq_mhz(i: usize, n: usize, sample_rate_mhz: f64) -> f64 {
    let k = if i <= n / 2 { i as f64 } else { i as f64 - n as f64 };
    k * sample_rate_mhz / n as f64
}

/// Apply a complex transfer function `H` to a real-valued waveform.
///
/// `transfer` is only ever asked about non-negative frequencies; the negative half of the
/// spectrum is filled with `H*(f)`, which is what makes the result real. Magnitude *and*
/// phase are honoured, so a filter contributes its true group delay and a Butterworth skirt
/// rings the way the analogue part does.
pub fn apply_transfer_function<F>(
    samples: &[Complex<f64>],
    sample_rate_mhz: f64,
    transfer: F,
) -> Vec<Complex<f64>>
where
    F: Fn(f64) -> Complex<f64>,
{
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }

    let mut buffer = samples.to_vec();
    FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(n).process(&mut buffer));

    for (i, bin) in buffer.iter_mut().enumerate() {
        let f = bin_freq_mhz(i, n, sample_rate_mhz);
        let h = transfer(f.abs());
        *bin *= if f < 0.0 { h.conj() } else { h };
    }

    FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(n).process(&mut buffer));

    let scale = 1.0 / n as f64;
    buffer.iter().map(|c| Complex::new(c.re * scale, 0.0)).collect()
}

// ---------------------------------------------------------------------------
// Stage noise
// ---------------------------------------------------------------------------

/// Noise a stage of power gain `g` and noise factor `f` adds at its own output.
///
/// From the definition of noise figure, a stage driven by a matched load at T₀ delivers
/// `G·kTB` of source noise plus `(F−1)·G·kTB` of its own. Only the second term belongs to
/// the component — the first is already in the waveform arriving at its input. A passive at
/// ambient has `F = 1/G`, so this collapses to `(1−G)·kTB` and a lossy part ends up handing
/// on exactly `kTB`, as it must.
fn added_noise_power(gain_pow: f64, noise_factor: f64, kt_b: f64) -> f64 {
    ((noise_factor - 1.0).max(0.0)) * gain_pow * kt_b
}

/// Add white noise for a stage whose gain and noise figure are flat across the band.
pub fn add_flat_noise(
    samples: &mut [Complex<f64>],
    ctx: &ChainCtx,
    gain_pow: f64,
    noise_factor: f64,
) {
    let Some(temp) = ctx.temperature_k else { return };
    let kt_b = thermal_noise_power(ctx.sample_rate_mhz, temp);
    let power = added_noise_power(gain_pow, noise_factor, kt_b);
    if power <= 0.0 {
        return;
    }
    let sigma = power.sqrt();
    let mut rng = ctx.rng(0x004E_4F49_5345);
    for s in samples.iter_mut() {
        s.re += sigma * rng.gaussian();
    }
}

/// Add the available thermal noise a matched source delivers into the chain.
///
/// An antenna or a signal generator looking into 50 Ω hands over `kTB` whether or not it is
/// transmitting anything. Noise figure is *defined* against that reference, so leaving it out
/// understates the first stage's contribution and makes a measured SNR disagree with the
/// cascaded budget by however much the first stage's own noise falls short of `F·G·kTB`.
pub fn add_source_noise(samples: &mut [Complex<f64>], ctx: &ChainCtx) {
    let Some(temp) = ctx.temperature_k else { return };
    let power = thermal_noise_power(ctx.sample_rate_mhz, temp);
    if power <= 0.0 {
        return;
    }
    let sigma = power.sqrt();
    let mut rng = ctx.rng(0x5000_0000_0001_u64);
    for s in samples.iter_mut() {
        s.re += sigma * rng.gaussian();
    }
}

/// Apply a stage's transfer function and add its own noise, in a single FFT pair.
///
/// `resp` returns the complex transfer and the noise figure in dB at a non-negative
/// frequency. Both the filtering and the noise happen in the frequency domain, so the noise
/// is synthesised there directly rather than being generated white and transformed — which
/// keeps a frequency-shaped stage down to one forward and one inverse transform.
///
/// The spectrum is filled conjugate-symmetrically, so the result is real to rounding.
pub fn apply_stage<F>(samples: &[Complex<f64>], ctx: &ChainCtx, resp: F) -> Vec<Complex<f64>>
where
    F: Fn(f64) -> (Complex<f64>, f64),
{
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }

    let mut buffer = samples.to_vec();
    FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(n).process(&mut buffer));

    let kt_b = ctx
        .temperature_k
        .map(|t| thermal_noise_power(ctx.sample_rate_mhz, t))
        .unwrap_or(0.0);
    let mut rng = ctx.rng(0x5EED);
    // White noise of variance kTB has E|X_k|² = N·kTB in every bin; the shaping factor then
    // scales that to the stage's own contribution.
    let bin_sigma = (n as f64 * kt_b).sqrt();

    // Only the lower half is visited; the upper half mirrors it.
    for i in 0..=n / 2 {
        let f = i as f64 * ctx.sample_rate_mhz / n as f64;
        let (h, nf_db) = resp(f);
        let mut lower = buffer[i] * h;
        let mirror = n - i;

        if kt_b > 0.0 {
            let g = h.norm_sqr();
            let amp = added_noise_power(g, 10.0_f64.powf(nf_db / 10.0), 1.0).sqrt() * bin_sigma;
            if amp > 0.0 {
                // Real and imaginary parts each carry half the bin's power.
                let noise = Complex::new(rng.gaussian(), rng.gaussian()) * (amp * std::f64::consts::FRAC_1_SQRT_2);
                lower += noise;
            }
        }

        buffer[i] = lower;
        // The input is a real voltage, so its spectrum is already conjugate-symmetric; writing
        // the mirror bin keeps it that way once the noise has been added.
        if mirror != i && mirror < n {
            buffer[mirror] = lower.conj();
        }
    }
    // DC and Nyquist have no partner to be conjugate with, so they must be real.
    buffer[0].im = 0.0;
    if n.is_multiple_of(2) {
        buffer[n / 2].im = 0.0;
    }

    FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(n).process(&mut buffer));

    let scale = 1.0 / n as f64;
    buffer.iter().map(|c| Complex::new(c.re * scale, 0.0)).collect()
}

// ---------------------------------------------------------------------------
// Memoryless nonlinearity
// ---------------------------------------------------------------------------

/// AM/AM nonlinearity fitted to a 1 dB compression point and an output IP3.
///
/// `y = v + a₃v³ + a₅v⁵`, with `a₃` set by IIP3 (two tones of amplitude `A` put IM3 at
/// `¾|a₃|A³`, equal to the fundamental when `A = A_iip3`) and `a₅` then set so the
/// fundamental is down exactly 1 dB at the compression point.
///
/// A pure cubic ties the two specs together at `P1dB = IIP3 − 9.6 dB`; the fifth-order term
/// is what lets a part with a wider gap between them be represented. When the specs imply
/// *more* compression than the cubic already gives, `a₅` would have to expand the
/// characteristic, so it is clamped to zero and IM3 accuracy wins.
#[derive(Debug, Clone, Copy)]
pub struct NonlinearFit {
    a3: f64,
    a5: f64,
    /// Input amplitude at which the polynomial stops being monotone.
    v_max: f64,
    /// Output at `v_max`; the characteristic saturates here.
    y_max: f64,
}

impl NonlinearFit {
    /// Build a fit from input-referred amplitudes. `None` if the stage is ideally linear.
    pub fn new(iip3_amp: Option<f64>, p1db_in_amp: Option<f64>) -> Option<Self> {
        let a3 = match iip3_amp {
            Some(a) if a > 0.0 && a.is_finite() => -4.0 / (3.0 * a * a),
            _ => 0.0,
        };

        let mut a5 = 0.0;
        if let Some(v1) = p1db_in_amp.filter(|v| *v > 0.0 && v.is_finite()) {
            // Fundamental scaling of y at drive v1, target −1 dB.
            let cubic = 0.75 * a3 * v1 * v1;
            let want = 10.0_f64.powf(-1.0 / 20.0) - 1.0;
            let quintic = (want - cubic) / (0.625 * v1 * v1 * v1 * v1);
            a5 = quintic.min(0.0);
        }

        if a3 == 0.0 && a5 == 0.0 {
            return None;
        }

        // First stationary point of y' = 1 + 3a₃v² + 5a₅v⁴, in terms of x = v².
        let x = if a5 == 0.0 {
            -1.0 / (3.0 * a3)
        } else {
            let disc = 9.0 * a3 * a3 - 20.0 * a5;
            (-3.0 * a3 - disc.sqrt()) / (10.0 * a5)
        };
        if x <= 0.0 || !x.is_finite() {
            return None;
        }
        let v_max = x.sqrt();
        let y_max = v_max + a3 * v_max.powi(3) + a5 * v_max.powi(5);

        Some(Self { a3, a5, v_max, y_max })
    }

    /// Apply the characteristic to one instantaneous voltage.
    pub fn apply(&self, v: f64) -> f64 {
        if v.abs() >= self.v_max {
            return v.signum() * self.y_max;
        }
        v + self.a3 * v.powi(3) + self.a5 * v.powi(5)
    }

    /// Small-signal gain compression at input amplitude `amp`, in dB (negative = compressed).
    pub fn compression_db(&self, amp: f64) -> f64 {
        if amp <= 0.0 {
            return 0.0;
        }
        let a = amp.min(self.v_max);
        let fundamental = 1.0 + 0.75 * self.a3 * a * a + 0.625 * self.a5 * a.powi(4);
        20.0 * fundamental.max(1e-12).log10()
    }
}

// ---------------------------------------------------------------------------
// Analog filter prototypes
// ---------------------------------------------------------------------------

/// Which polynomial an analytically generated filter follows.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum FilterResponse {
    /// Maximally flat passband, −3 dB exactly at the corner.
    #[default]
    Butterworth,
    /// Equiripple passband, steeper skirt for the same order.
    Chebyshev { ripple_db: f64 },
}

impl std::fmt::Display for FilterResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterResponse::Butterworth => write!(f, "Butterworth"),
            FilterResponse::Chebyshev { ripple_db } => write!(f, "Chebyshev {ripple_db:.2} dB"),
        }
    }
}

/// Largest filter order the pole arrays are sized for.
const MAX_FILTER_ORDER: usize = 16;

/// A normalised low-pass prototype: its left-half-plane poles and numerator gain.
///
/// Held in a fixed-size array and cached, because `transfer_at` is called once per FFT bin —
/// tens of thousands of times per component per frame — and recomputing the poles from
/// trigonometry each time dominated the whole evaluation.
#[derive(Clone, Copy)]
struct Prototype {
    poles: [Complex<f64>; MAX_FILTER_ORDER],
    order: usize,
    gain: f64,
}

thread_local! {
    static PROTOTYPE_CACHE: RefCell<Vec<((u64, u32), Prototype)>> = const { RefCell::new(Vec::new()) };
}

/// Left-half-plane poles of the normalised low-pass prototype, plus its numerator gain.
fn prototype(response: FilterResponse, order: u32) -> Prototype {
    let order = order.clamp(1, MAX_FILTER_ORDER as u32);
    let key = match response {
        FilterResponse::Butterworth => (0, order),
        FilterResponse::Chebyshev { ripple_db } => (ripple_db.to_bits(), order),
    };
    if let Some(found) = PROTOTYPE_CACHE.with(|c| {
        c.borrow()
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, p)| *p)
    }) {
        return found;
    }
    let computed = compute_prototype(response, order);
    PROTOTYPE_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        // A handful of distinct filters per graph; drop the lot rather than grow unbounded.
        if cache.len() >= 16 {
            cache.clear();
        }
        cache.push((key, computed));
    });
    computed
}

fn compute_prototype(response: FilterResponse, order: u32) -> Prototype {
    let n = order.max(1) as usize;
    let poles: Vec<Complex<f64>> = match response {
        FilterResponse::Butterworth => (0..n)
            .map(|k| {
                let theta = PI * (2.0 * k as f64 + 1.0) / (2.0 * n as f64) + PI / 2.0;
                Complex::new(theta.cos(), theta.sin())
            })
            .collect(),
        FilterResponse::Chebyshev { ripple_db } => {
            let eps = (10.0_f64.powf(ripple_db.max(0.001) / 10.0) - 1.0).sqrt();
            let a = (1.0 / eps).asinh() / n as f64;
            (0..n)
                .map(|k| {
                    let theta = PI * (2.0 * k as f64 + 1.0) / (2.0 * n as f64);
                    Complex::new(-a.sinh() * theta.sin(), a.cosh() * theta.cos())
                })
                .collect()
        }
    };

    // Numerator chosen so the passband peaks at 0 dB: unity at DC for Butterworth and for
    // odd-order Chebyshev, and one ripple down at DC for even-order Chebyshev.
    let mut num = Complex::new(1.0, 0.0);
    for p in &poles {
        num *= -p;
    }
    let mut gain = num.norm();
    if let FilterResponse::Chebyshev { ripple_db } = response
        && n.is_multiple_of(2)
    {
        let eps_sq = 10.0_f64.powf(ripple_db.max(0.001) / 10.0) - 1.0;
        gain /= (1.0 + eps_sq).sqrt();
    }

    let mut fixed = [Complex::new(0.0, 0.0); MAX_FILTER_ORDER];
    fixed[..n].copy_from_slice(&poles);
    Prototype { poles: fixed, order: n, gain }
}

/// Decay time of the slowest pole in a prototype, normalised to the corner frequency.
///
/// The pole nearest the jω axis is the one that rings longest; its distance from the axis is
/// the decay rate in units of the corner's angular frequency.
fn prototype_slowest_decay(response: FilterResponse, order: u32) -> f64 {
    let proto = prototype(response, order);
    proto.poles[..proto.order]
        .iter()
        .map(|p| p.re.abs())
        .fold(f64::INFINITY, f64::min)
        .max(1e-6)
}

/// Ring-down time of a filter whose slowest pole sits at `decay` × the corner, in ns.
///
/// Twenty-five time constants puts the residual near −220 dB, far below anything the rest of
/// the pipeline can resolve.
fn settling_from_corner(corner_mhz: f64, decay: f64) -> f64 {
    if corner_mhz <= 0.0 {
        return 0.0;
    }
    let tau_us = 1.0 / (2.0 * PI * corner_mhz * decay);
    25.0 * tau_us * 1000.0
}

/// Evaluate a prototype at normalised complex frequency `s`.
fn eval_prototype(proto: &Prototype, s: Complex<f64>) -> Complex<f64> {
    let mut den = Complex::new(1.0, 0.0);
    for p in &proto.poles[..proto.order] {
        den *= s - p;
    }
    if den.norm() < 1e-300 {
        return Complex::new(0.0, 0.0);
    }
    Complex::new(proto.gain, 0.0) / den
}

/// Complex response of an n-pole low-pass at `freq_mhz` with corner `cutoff_mhz`.
pub fn lowpass_transfer(
    response: FilterResponse,
    order: u32,
    cutoff_mhz: f64,
    freq_mhz: f64,
) -> Complex<f64> {
    if cutoff_mhz <= 0.0 {
        return Complex::new(1.0, 0.0);
    }
    eval_prototype(
        &prototype(response, order),
        Complex::new(0.0, freq_mhz / cutoff_mhz),
    )
}

/// Complex response of an n-pole high-pass, via the `s → ω_c/s` transform.
pub fn highpass_transfer(
    response: FilterResponse,
    order: u32,
    cutoff_mhz: f64,
    freq_mhz: f64,
) -> Complex<f64> {
    if cutoff_mhz <= 0.0 {
        return Complex::new(1.0, 0.0);
    }
    if freq_mhz.abs() < 1e-12 {
        return Complex::new(0.0, 0.0);
    }
    eval_prototype(
        &prototype(response, order),
        Complex::new(0.0, -cutoff_mhz / freq_mhz),
    )
}

/// Complex response of a band-pass, via the `s → Q(s/ω₀ + ω₀/s)` transform.
///
/// This is the transform real band-pass filters follow, so the skirts are symmetric about
/// `f₀` on a log axis: an octave below the centre is attenuated the same as an octave above.
pub fn bandpass_transfer(
    response: FilterResponse,
    order: u32,
    center_mhz: f64,
    bandwidth_mhz: f64,
    freq_mhz: f64,
) -> Complex<f64> {
    if center_mhz <= 0.0 || bandwidth_mhz <= 0.0 {
        return Complex::new(1.0, 0.0);
    }
    if freq_mhz.abs() < 1e-12 {
        return Complex::new(0.0, 0.0);
    }
    let q = center_mhz / bandwidth_mhz;
    let detune = q * (freq_mhz / center_mhz - center_mhz / freq_mhz);
    eval_prototype(&prototype(response, order), Complex::new(0.0, detune))
}

// ---------------------------------------------------------------------------
// The component interface
// ---------------------------------------------------------------------------

/// One physical two-port (or N-port) in the front-end chain.
pub trait RfComponent {
    /// Small-signal complex transfer function at a non-negative frequency, for output `port`.
    fn transfer_at(&self, freq_mhz: f64, port: usize) -> Complex<f64>;

    /// Noise figure in dB at a non-negative frequency.
    ///
    /// Defaults to the passive-at-ambient result, `F = 1/G`: a part that loses 3 dB has a
    /// 3 dB noise figure. Active stages override this.
    fn noise_figure_db_at(&self, freq_mhz: f64) -> f64 {
        -self.response_db(freq_mhz, 0)
    }

    /// Output-referred third-order intercept in dBm, or `None` if ideally linear.
    fn oip3_dbm(&self) -> Option<f64> {
        None
    }

    /// True when `transfer_at` does not depend on frequency, which skips two FFTs.
    fn is_flat(&self) -> bool {
        false
    }

    /// How long this component's impulse response takes to die away, in nanoseconds.
    ///
    /// Frequency-domain multiplication is circular convolution, so a block has to be given
    /// this much run-up ahead of the samples that will be kept — otherwise the tail folds
    /// back onto the head and the answer depends on the block length. Narrow stages ring for
    /// a long time, so the requirement comes from the component rather than a fixed guess.
    fn settling_ns(&self) -> f64 {
        0.0
    }

    /// Magnitude response in dB (positive = gain).
    fn response_db(&self, freq_mhz: f64, port: usize) -> f64 {
        20.0 * self.transfer_at(freq_mhz, port).norm().max(1e-15).log10()
    }

    /// Push a block of real analog voltage through this component.
    fn process(&self, samples: &[Complex<f64>], ctx: &ChainCtx, port: usize) -> Vec<Complex<f64>> {
        linear_stage(self, samples, ctx, port)
    }
}

/// Run a block through a component's transfer function and add the noise it contributes.
///
/// Flat components take a scalar path that skips the transforms entirely, which is most of
/// them — pads, splitters, couplers and any amplifier without a bandwidth limit.
pub fn linear_stage<C: RfComponent + ?Sized>(
    comp: &C,
    samples: &[Complex<f64>],
    ctx: &ChainCtx,
    port: usize,
) -> Vec<Complex<f64>> {
    if comp.is_flat() {
        let g = comp.transfer_at(0.0, port).norm();
        let mut out: Vec<Complex<f64>> = samples
            .iter()
            .map(|s| Complex::new(s.re * g, 0.0))
            .collect();
        let nf = 10.0_f64.powf(comp.noise_figure_db_at(0.0) / 10.0);
        add_flat_noise(&mut out, ctx, g * g, nf);
        out
    } else {
        apply_stage(samples, ctx, |f| {
            (comp.transfer_at(f, port), comp.noise_figure_db_at(f))
        })
    }
}

// ---------------------------------------------------------------------------
// Balun
// ---------------------------------------------------------------------------

/// Models a balun transformer (e.g. Mini-Circuits TCM2-33WX+).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalunModel {
    pub name: String,
    /// Insertion loss lookup table: (frequency_mhz, insertion_loss_db).
    pub il_table: Vec<(f64, f64)>,
    /// Minimum operating frequency in MHz.
    pub min_freq_mhz: f64,
    /// Maximum operating frequency in MHz.
    pub max_freq_mhz: f64,
    /// Amplitude imbalance between the two differential arms, in dB.
    #[serde(default)]
    pub amplitude_imbalance_db: f64,
    /// Phase imbalance from the ideal 180°, in degrees.
    #[serde(default)]
    pub phase_imbalance_deg: f64,
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
            amplitude_imbalance_db: 0.3,
            phase_imbalance_deg: 3.0,
        }
    }
}

/// Linear interpolation over a sorted (frequency, value) table, clamped at both ends.
fn interp_table(table: &[(f64, f64)], freq_mhz: f64) -> f64 {
    match table {
        [] => 0.0,
        [only] => only.1,
        _ => {
            if freq_mhz <= table[0].0 {
                return table[0].1;
            }
            let last = *table.last().unwrap();
            if freq_mhz >= last.0 {
                return last.1;
            }
            for w in table.windows(2) {
                let (f0, v0) = w[0];
                let (f1, v1) = w[1];
                if freq_mhz >= f0 && freq_mhz <= f1 && f1 > f0 {
                    let t = (freq_mhz - f0) / (f1 - f0);
                    return v0 + t * (v1 - v0);
                }
            }
            last.1
        }
    }
}

impl BalunModel {
    /// In-band insertion loss from the datasheet table, without the band-edge roll-off.
    pub fn table_loss_at(&self, freq_mhz: f64) -> f64 {
        interp_table(&self.il_table, freq_mhz)
    }

    /// Total insertion loss at a frequency, including both band edges.
    pub fn insertion_loss_at(&self, freq_mhz: f64) -> f64 {
        -self.response_db(freq_mhz, 0)
    }

    /// Differential transfer function of the imbalance alone.
    ///
    /// The two arms carry `g₊·s/2` and `−g₋·s/2·e^{jθ}`, so the differential output the
    /// converter sees is `(g₊ + g₋e^{jθ})/2` — near unity for realistic imbalance. What
    /// imbalance actually costs is common-mode rejection, reported by [`Self::cmrr_db`].
    fn imbalance_transfer(&self) -> Complex<f64> {
        let g_p = 10.0_f64.powf(self.amplitude_imbalance_db / 40.0);
        let g_m = 10.0_f64.powf(-self.amplitude_imbalance_db / 40.0);
        let theta = self.phase_imbalance_deg.to_radians();
        (Complex::new(g_p, 0.0) + Complex::new(theta.cos(), theta.sin()) * g_m) / 2.0
    }

    /// Common-mode rejection ratio implied by the imbalance, in dB.
    ///
    /// The residual common-mode is what converts to even-order distortion in the converter's
    /// differential front end; the ADC block's HD2 setting is where that lands.
    pub fn cmrr_db(&self) -> f64 {
        let g_p = 10.0_f64.powf(self.amplitude_imbalance_db / 40.0);
        let g_m = 10.0_f64.powf(-self.amplitude_imbalance_db / 40.0);
        let theta = self.phase_imbalance_deg.to_radians();
        let arm = Complex::new(theta.cos(), theta.sin()) * g_m;
        let diff = (Complex::new(g_p, 0.0) + arm).norm();
        let common = (Complex::new(g_p, 0.0) - arm).norm();
        if common < 1e-12 {
            return 200.0;
        }
        20.0 * (diff / common).log10()
    }
}

impl RfComponent for BalunModel {
    fn transfer_at(&self, freq_mhz: f64, _port: usize) -> Complex<f64> {
        let il = 10.0_f64.powf(-self.table_loss_at(freq_mhz) / 20.0);

        // A transformer is a band-pass: the low end rolls off as the core stops coupling, the
        // high end as parasitics take over. Corners sit just outside the specified band so the
        // in-band loss still matches the datasheet table.
        let mut h = Complex::new(il, 0.0) * self.imbalance_transfer();
        if self.min_freq_mhz > 0.0 {
            h *= highpass_transfer(
                FilterResponse::Butterworth,
                2,
                self.min_freq_mhz * 0.8,
                freq_mhz,
            );
        }
        if self.max_freq_mhz > 0.0 {
            h *= lowpass_transfer(
                FilterResponse::Butterworth,
                3,
                self.max_freq_mhz * 1.25,
                freq_mhz,
            );
        }
        h
    }

    fn settling_ns(&self) -> f64 {
        // The low corner is the slow one: a transformer's low-frequency roll-off rings for
        // far longer than its high-frequency parasitics do.
        if self.min_freq_mhz > 0.0 {
            let decay = prototype_slowest_decay(FilterResponse::Butterworth, 2);
            settling_from_corner(self.min_freq_mhz * 0.8, decay)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Filters
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

/// Analog filter built from a real pole prototype, so magnitude *and* phase are physical.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterModel {
    pub filter_type: FilterType,
    /// Cutoff frequency in MHz (for LP/HP) or centre frequency (for BP).
    pub cutoff_mhz: f64,
    /// Bandwidth in MHz (only used for BandPass).
    pub bandwidth_mhz: f64,
    /// Filter order (higher = steeper rolloff).
    pub order: u32,
    /// Which polynomial the filter follows.
    #[serde(default)]
    pub response: FilterResponse,
    /// Flat insertion loss of the passband in dB, as a real part always has.
    #[serde(default)]
    pub insertion_loss_db: f64,
}

impl Default for FilterModel {
    fn default() -> Self {
        Self {
            filter_type: FilterType::LowPass,
            cutoff_mhz: 1000.0,
            bandwidth_mhz: 200.0,
            order: 4,
            response: FilterResponse::Butterworth,
            insertion_loss_db: 0.5,
        }
    }
}

impl FilterModel {
    /// Attenuation at a frequency in dB (positive = loss).
    pub fn attenuation_at(&self, freq_mhz: f64) -> f64 {
        -self.response_db(freq_mhz, 0)
    }

    /// Group delay in nanoseconds, from the slope of the phase response.
    pub fn group_delay_ns(&self, freq_mhz: f64) -> f64 {
        let df = (self.cutoff_mhz * 1e-4).max(1e-6);
        let p0 = self.transfer_at((freq_mhz - df).max(1e-9), 0).arg();
        let p1 = self.transfer_at(freq_mhz + df, 0).arg();
        let mut dphi = p1 - p0;
        while dphi > PI {
            dphi -= 2.0 * PI;
        }
        while dphi < -PI {
            dphi += 2.0 * PI;
        }
        // -dφ/dω, with frequency in MHz giving a delay in µs; scale to ns.
        -dphi / (2.0 * PI * 2.0 * df) * 1000.0
    }
}

impl RfComponent for FilterModel {
    fn transfer_at(&self, freq_mhz: f64, _port: usize) -> Complex<f64> {
        let shape = match self.filter_type {
            FilterType::LowPass => {
                lowpass_transfer(self.response, self.order, self.cutoff_mhz, freq_mhz)
            }
            FilterType::HighPass => {
                highpass_transfer(self.response, self.order, self.cutoff_mhz, freq_mhz)
            }
            FilterType::BandPass => bandpass_transfer(
                self.response,
                self.order,
                self.cutoff_mhz,
                self.bandwidth_mhz,
                freq_mhz,
            ),
        };
        shape * 10.0_f64.powf(-self.insertion_loss_db / 20.0)
    }

    fn settling_ns(&self) -> f64 {
        let decay = prototype_slowest_decay(self.response, self.order);
        match self.filter_type {
            // A band-pass rings for as long as its half-bandwidth allows, not its centre.
            FilterType::BandPass => settling_from_corner(self.bandwidth_mhz / 2.0, decay),
            FilterType::LowPass | FilterType::HighPass => {
                settling_from_corner(self.cutoff_mhz, decay)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Amplifier
// ---------------------------------------------------------------------------

/// Amplifier / LNA with a finite gain bandwidth, a noise figure and real compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmplifierModel {
    /// Small-signal gain in dB.
    pub gain_db: f64,
    /// Noise figure in dB.
    pub noise_figure_db: f64,
    /// Output-referred 1 dB compression point in dBm.
    pub p1db_dbm: f64,
    /// Output-referred third-order intercept in dBm.
    #[serde(default = "default_amp_oip3")]
    pub oip3_dbm: f64,
    /// −3 dB gain bandwidth in MHz; 0 disables the roll-off.
    #[serde(default = "default_amp_bandwidth")]
    pub bandwidth_mhz: f64,
}

fn default_amp_oip3() -> f64 {
    30.0
}

fn default_amp_bandwidth() -> f64 {
    6000.0
}

impl Default for AmplifierModel {
    fn default() -> Self {
        Self {
            gain_db: 12.0,
            noise_figure_db: 2.0,
            p1db_dbm: 20.0,
            oip3_dbm: 30.0,
            bandwidth_mhz: 6000.0,
        }
    }
}

impl AmplifierModel {
    fn gain_linear(&self) -> f64 {
        10.0_f64.powf(self.gain_db / 20.0)
    }

    /// Normalised gain shape (0 dB in band), a single pole at the gain bandwidth.
    fn shape_at(&self, freq_mhz: f64) -> Complex<f64> {
        if self.bandwidth_mhz <= 0.0 {
            return Complex::new(1.0, 0.0);
        }
        lowpass_transfer(FilterResponse::Butterworth, 1, self.bandwidth_mhz, freq_mhz)
    }

    /// The fitted compression characteristic, if the part is not ideally linear.
    pub fn nonlinearity(&self) -> Option<NonlinearFit> {
        let g = self.gain_linear();
        if g <= 0.0 {
            return None;
        }
        // Datasheet figures are output-referred; the polynomial acts on the input.
        let iip3 = dbm_to_amplitude(self.oip3_dbm) / g;
        // At the compression point the output is already 1 dB down on the ideal gain.
        let p1db_in = dbm_to_amplitude(self.p1db_dbm) * 10.0_f64.powf(1.0 / 20.0) / g;
        NonlinearFit::new(Some(iip3), Some(p1db_in))
    }

    /// Input-referred 1 dB compression point in dBm.
    pub fn input_p1db_dbm(&self) -> f64 {
        self.p1db_dbm + 1.0 - self.gain_db
    }
}

impl RfComponent for AmplifierModel {
    fn transfer_at(&self, freq_mhz: f64, _port: usize) -> Complex<f64> {
        self.shape_at(freq_mhz) * self.gain_linear()
    }

    fn noise_figure_db_at(&self, _freq_mhz: f64) -> f64 {
        self.noise_figure_db
    }

    fn oip3_dbm(&self) -> Option<f64> {
        Some(self.oip3_dbm)
    }

    fn settling_ns(&self) -> f64 {
        settling_from_corner(self.bandwidth_mhz, 1.0)
    }

    fn is_flat(&self) -> bool {
        self.bandwidth_mhz <= 0.0
    }

    fn process(&self, samples: &[Complex<f64>], ctx: &ChainCtx, port: usize) -> Vec<Complex<f64>> {
        // Compression acts on the instantaneous input voltage; the gain and its roll-off then
        // follow, so the harmonics the device just made are attenuated by the amplifier's own
        // bandwidth — the order a real part does it in.
        let compressed: Vec<Complex<f64>> = match self.nonlinearity() {
            Some(fit) => samples
                .iter()
                .map(|s| Complex::new(fit.apply(s.re), 0.0))
                .collect(),
            None => samples.to_vec(),
        };
        linear_stage(self, &compressed, ctx, port)
    }
}

// ---------------------------------------------------------------------------
// Attenuator
// ---------------------------------------------------------------------------

/// Resistive pad. Flat, and — being passive at ambient — its noise figure is its loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttenuatorModel {
    /// Attenuation in dB (positive value).
    pub attenuation_db: f64,
}

impl Default for AttenuatorModel {
    fn default() -> Self {
        Self { attenuation_db: 6.0 }
    }
}

impl RfComponent for AttenuatorModel {
    fn transfer_at(&self, _freq_mhz: f64, _port: usize) -> Complex<f64> {
        Complex::new(10.0_f64.powf(-self.attenuation_db / 20.0), 0.0)
    }

    fn is_flat(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Splitter / Combiner
// ---------------------------------------------------------------------------

/// Power splitter. Each output carries `1/N` of the input power plus the excess loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitterModel {
    /// Number of output ports.
    pub num_outputs: u32,
    /// Additional insertion loss in dB (beyond ideal splitting loss).
    pub excess_loss_db: f64,
}

impl Default for SplitterModel {
    fn default() -> Self {
        Self { num_outputs: 2, excess_loss_db: 0.5 }
    }
}

impl SplitterModel {
    /// Total loss per output port in dB.
    pub fn total_loss_db(&self) -> f64 {
        10.0 * (self.num_outputs.max(1) as f64).log10() + self.excess_loss_db
    }
}

impl RfComponent for SplitterModel {
    fn transfer_at(&self, _freq_mhz: f64, _port: usize) -> Complex<f64> {
        Complex::new(10.0_f64.powf(-self.total_loss_db() / 20.0), 0.0)
    }

    fn is_flat(&self) -> bool {
        true
    }
}

/// Power combiner. Voltages sum, then the split ratio applies — so two coherent inputs gain
/// 3 dB over one, and two uncorrelated ones gain nothing, exactly as a real hybrid behaves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinerModel {
    /// Number of input ports (2-8).
    pub num_inputs: u32,
    /// Excess loss in dB beyond the theoretical split ratio.
    pub excess_loss_db: f64,
}

impl Default for CombinerModel {
    fn default() -> Self {
        Self { num_inputs: 2, excess_loss_db: 0.5 }
    }
}

impl CombinerModel {
    pub fn total_loss_db(&self) -> f64 {
        10.0 * (self.num_inputs.max(1) as f64).log10() + self.excess_loss_db
    }
}

impl RfComponent for CombinerModel {
    fn transfer_at(&self, _freq_mhz: f64, _port: usize) -> Complex<f64> {
        Complex::new(10.0_f64.powf(-self.total_loss_db() / 20.0), 0.0)
    }

    fn is_flat(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Mixer
// ---------------------------------------------------------------------------

/// Which output port index of a directional coupler is the coupled arm.
pub const COUPLER_COUPLED_PORT: usize = 1;

/// Frequency-translating mixer driven by a real local oscillator.
///
/// A real LO is a *real* waveform, so it produces both the sum and the difference product —
/// there is no way to build an analog mixer that emits only one. Modelling it as a complex
/// exponential silently deleted one sideband, which is the whole reason image-reject
/// filtering exists.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixerModel {
    /// Local Oscillator (LO) frequency in MHz.
    pub lo_freq_mhz: f64,
    /// Conversion loss in dB (positive value), referred to the wanted product.
    pub conversion_loss_db: f64,
    /// Noise figure in dB.
    #[serde(default = "default_mixer_nf")]
    pub noise_figure_db: f64,
    /// Output-referred third-order intercept in dBm.
    #[serde(default = "default_mixer_oip3")]
    pub oip3_dbm: f64,
    /// LO feedthrough at the output port, in dBFS. Signal-independent.
    #[serde(default = "default_mixer_lo_leak")]
    pub lo_leakage_dbfs: f64,
    /// Third-harmonic content of the LO drive in dBc, which sets the 3×LO±RF spurs.
    #[serde(default = "default_mixer_lo_h3")]
    pub lo_harmonic3_dbc: f64,
}

fn default_mixer_nf() -> f64 {
    7.0
}

fn default_mixer_oip3() -> f64 {
    20.0
}

fn default_mixer_lo_leak() -> f64 {
    -60.0
}

fn default_mixer_lo_h3() -> f64 {
    -30.0
}

impl Default for MixerModel {
    fn default() -> Self {
        Self {
            lo_freq_mhz: 100.0,
            conversion_loss_db: 7.0,
            noise_figure_db: 7.0,
            oip3_dbm: 20.0,
            lo_leakage_dbfs: -60.0,
            lo_harmonic3_dbc: -30.0,
        }
    }
}

impl MixerModel {
    /// Conversion gain of the wanted product as a voltage ratio.
    fn conversion_linear(&self) -> f64 {
        10.0_f64.powf(-self.conversion_loss_db / 20.0)
    }

    /// The two products a real input tone at `freq_mhz` lands on, in MHz.
    pub fn product_freqs(&self, freq_mhz: f64) -> (f64, f64) {
        (
            (freq_mhz - self.lo_freq_mhz).abs(),
            freq_mhz + self.lo_freq_mhz,
        )
    }
}

impl RfComponent for MixerModel {
    fn transfer_at(&self, _freq_mhz: f64, _port: usize) -> Complex<f64> {
        Complex::new(self.conversion_linear(), 0.0)
    }

    fn noise_figure_db_at(&self, _freq_mhz: f64) -> f64 {
        self.noise_figure_db
    }

    fn oip3_dbm(&self) -> Option<f64> {
        Some(self.oip3_dbm)
    }

    fn process(&self, samples: &[Complex<f64>], ctx: &ChainCtx, _port: usize) -> Vec<Complex<f64>> {
        let n = samples.len();
        if n == 0 {
            return Vec::new();
        }

        let dt = 1.0 / ctx.sample_rate_mhz; // µs, against MHz frequencies
        let omega = 2.0 * PI * self.lo_freq_mhz;
        // cos splits an input tone into two half-amplitude products; the factor of two puts
        // the wanted one at exactly the specified conversion loss, as datasheets define it.
        let conv = 2.0 * self.conversion_linear();
        let h3 = if self.lo_harmonic3_dbc < 0.0 {
            10.0_f64.powf(self.lo_harmonic3_dbc / 20.0)
        } else {
            0.0
        };
        let leak = if self.lo_leakage_dbfs < 0.0 {
            10.0_f64.powf(self.lo_leakage_dbfs / 20.0)
        } else {
            0.0
        };

        // The nonlinearity acts on the input, and each output product ends up at
        // `input × conversion_linear`, so that — not the LO's factor of two — is what refers
        // the output intercept back to the input.
        let nl = NonlinearFit::new(
            Some(dbm_to_amplitude(self.oip3_dbm) / self.conversion_linear().max(1e-12)),
            None,
        );

        let mut out: Vec<Complex<f64>> = samples
            .iter()
            .enumerate()
            .map(|(i, s)| {
                // Absolute time keeps the LO coherent from one frame to the next.
                let t = ctx.time_us + i as f64 * dt;
                let phase = omega * t;
                let lo = phase.cos() + h3 * (3.0 * phase).cos();
                let v = match &nl {
                    Some(fit) => fit.apply(s.re),
                    None => s.re,
                };
                Complex::new(v * lo * conv + leak * phase.cos(), 0.0)
            })
            .collect();

        let g = self.conversion_linear();
        add_flat_noise(
            &mut out,
            ctx,
            g * g,
            10.0_f64.powf(self.noise_figure_db / 10.0),
        );
        out
    }
}

// ---------------------------------------------------------------------------
// Phase shifter
// ---------------------------------------------------------------------------

/// How a phase shifter achieves its shift, which decides how it behaves off-centre.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhaseShiftKind {
    /// Constant phase across frequency, as a vector modulator gives.
    ConstantPhase,
    /// A length of line: the phase is set at a reference frequency and scales with it.
    TrueDelay,
}

impl std::fmt::Display for PhaseShiftKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhaseShiftKind::ConstantPhase => write!(f, "Constant Phase"),
            PhaseShiftKind::TrueDelay => write!(f, "True Time Delay"),
        }
    }
}

/// Phase shifter.
///
/// Both kinds are conjugate-symmetric, so a real input stays real. Multiplying every sample
/// by `e^{jφ}` instead — which is not a realisable two-port — scales a real waveform by
/// `cos φ` once the converter takes its real part, and vanishes entirely at 90°.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseShifterModel {
    /// Phase shift in degrees.
    pub phase_shift_deg: f64,
    /// Insertion loss in dB.
    pub insertion_loss_db: f64,
    /// Constant-phase or true-delay behaviour.
    #[serde(default = "default_ps_kind")]
    pub kind: PhaseShiftKind,
    /// Frequency at which `phase_shift_deg` is specified, for the true-delay kind.
    #[serde(default = "default_ps_ref")]
    pub ref_freq_mhz: f64,
}

fn default_ps_kind() -> PhaseShiftKind {
    PhaseShiftKind::ConstantPhase
}

fn default_ps_ref() -> f64 {
    1000.0
}

impl Default for PhaseShifterModel {
    fn default() -> Self {
        Self {
            phase_shift_deg: 90.0,
            insertion_loss_db: 1.5,
            kind: PhaseShiftKind::ConstantPhase,
            ref_freq_mhz: 1000.0,
        }
    }
}

impl PhaseShifterModel {
    /// Time delay in nanoseconds, for the true-delay kind.
    pub fn delay_ns(&self) -> f64 {
        if self.ref_freq_mhz <= 0.0 {
            return 0.0;
        }
        self.phase_shift_deg / 360.0 / self.ref_freq_mhz * 1000.0
    }
}

impl RfComponent for PhaseShifterModel {
    fn transfer_at(&self, freq_mhz: f64, _port: usize) -> Complex<f64> {
        let loss = 10.0_f64.powf(-self.insertion_loss_db / 20.0);
        let phi = match self.kind {
            PhaseShiftKind::ConstantPhase => -self.phase_shift_deg.to_radians(),
            PhaseShiftKind::TrueDelay => {
                if self.ref_freq_mhz <= 0.0 {
                    0.0
                } else {
                    -self.phase_shift_deg.to_radians() * freq_mhz / self.ref_freq_mhz
                }
            }
        };
        Complex::from_polar(loss, phi)
    }

    fn settling_ns(&self) -> f64 {
        match self.kind {
            PhaseShiftKind::TrueDelay => self.delay_ns().abs(),
            // A constant-phase shift is a Hilbert-type response, whose tail is long in
            // principle but concentrated within a few cycles of the lowest frequency present.
            PhaseShiftKind::ConstantPhase => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Directional coupler
// ---------------------------------------------------------------------------

/// Directional coupler with a through path and a coupled port.
///
/// Port 0 is the main line, port 1 the coupled arm at `coupling_db` below the input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalCouplerModel {
    /// Coupling factor in dB.
    pub coupling_db: f64,
    /// Insertion loss of the main line in dB, beyond the power tapped off.
    pub insertion_loss_db: f64,
    /// Directivity in dB, reported for reference; reverse waves are not simulated.
    #[serde(default = "default_coupler_directivity")]
    pub directivity_db: f64,
}

fn default_coupler_directivity() -> f64 {
    25.0
}

impl Default for DirectionalCouplerModel {
    fn default() -> Self {
        Self { coupling_db: 20.0, insertion_loss_db: 0.5, directivity_db: 25.0 }
    }
}

impl DirectionalCouplerModel {
    /// Main-line loss in dB, including the power diverted to the coupled port.
    ///
    /// A 20 dB coupler taps off 1% of the power, so the through path loses 0.044 dB before
    /// its own dissipative loss is counted.
    pub fn through_loss_db(&self) -> f64 {
        let coupled_frac = 10.0_f64.powf(-self.coupling_db / 10.0);
        self.insertion_loss_db - 10.0 * (1.0 - coupled_frac).max(1e-6).log10()
    }
}

impl RfComponent for DirectionalCouplerModel {
    fn transfer_at(&self, _freq_mhz: f64, port: usize) -> Complex<f64> {
        let db = if port == COUPLER_COUPLED_PORT {
            -(self.coupling_db + self.insertion_loss_db)
        } else {
            -self.through_loss_db()
        };
        Complex::new(10.0_f64.powf(db / 20.0), 0.0)
    }

    fn is_flat(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Touchstone .s2p component
// ---------------------------------------------------------------------------

/// Touchstone 2-port S-parameter component model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S2pModel {
    pub name: String,
    /// S21 lookup table: (frequency_mhz, gain_db, phase_deg).
    pub s21_table: Vec<(f64, f64, f64)>,
    /// S11 magnitude lookup table: (frequency_mhz, return_loss_db).
    #[serde(default)]
    pub s11_table: Vec<(f64, f64)>,
    /// S22 magnitude lookup table: (frequency_mhz, return_loss_db).
    #[serde(default)]
    pub s22_table: Vec<(f64, f64)>,
    /// Noise Figure in dB.
    pub noise_figure_db: f64,
    /// OIP3 in dBm.
    pub oip3_dbm: f64,
    /// Whether the measured S21 phase is applied. Off makes the block zero-phase.
    #[serde(default = "default_true")]
    pub use_measured_phase: bool,
}

fn default_true() -> bool {
    true
}

impl Default for S2pModel {
    fn default() -> Self {
        Self {
            name: "Generic S2P Block".to_string(),
            s21_table: vec![
                (10.0, -0.5, 0.0),
                (500.0, -0.8, -12.0),
                (1000.0, -1.2, -25.0),
                (2000.0, -2.0, -51.0),
                (4000.0, -3.5, -104.0),
                (6000.0, -6.0, -158.0),
            ],
            s11_table: vec![(10.0, -22.0), (1000.0, -20.0), (6000.0, -12.0)],
            s22_table: vec![(10.0, -22.0), (1000.0, -19.0), (6000.0, -11.0)],
            noise_figure_db: 1.5,
            oip3_dbm: 35.0,
            use_measured_phase: true,
        }
    }
}

impl S2pModel {
    pub fn s21_gain_at(&self, freq_mhz: f64) -> f64 {
        let mags: Vec<(f64, f64)> = self.s21_table.iter().map(|&(f, m, _)| (f, m)).collect();
        interp_table(&mags, freq_mhz)
    }

    pub fn s21_phase_at(&self, freq_mhz: f64) -> f64 {
        // Unwrapped before interpolating, or a wrap between two points reads as a huge jump.
        let mut unwrapped: Vec<(f64, f64)> = Vec::with_capacity(self.s21_table.len());
        let mut offset = 0.0;
        let mut prev: Option<f64> = None;
        for &(f, _, p) in &self.s21_table {
            if let Some(pp) = prev {
                let d = p + offset - pp;
                if d > 180.0 {
                    offset -= 360.0;
                } else if d < -180.0 {
                    offset += 360.0;
                }
            }
            let val = p + offset;
            unwrapped.push((f, val));
            prev = Some(val);
        }
        interp_table(&unwrapped, freq_mhz)
    }

    /// Input return loss in dB at a frequency (negative = matched), or `None` if unmeasured.
    pub fn return_loss_db(&self, freq_mhz: f64) -> Option<f64> {
        if self.s11_table.is_empty() {
            None
        } else {
            Some(interp_table(&self.s11_table, freq_mhz))
        }
    }

    /// Input VSWR at a frequency, or `None` if S11 was not measured.
    pub fn vswr(&self, freq_mhz: f64) -> Option<f64> {
        let rl = self.return_loss_db(freq_mhz)?;
        let gamma = 10.0_f64.powf(rl / 20.0).min(0.999_999);
        Some((1.0 + gamma) / (1.0 - gamma))
    }
}

impl RfComponent for S2pModel {
    fn transfer_at(&self, freq_mhz: f64, _port: usize) -> Complex<f64> {
        let mag = 10.0_f64.powf(self.s21_gain_at(freq_mhz) / 20.0);
        if self.use_measured_phase {
            Complex::from_polar(mag, self.s21_phase_at(freq_mhz).to_radians())
        } else {
            Complex::new(mag, 0.0)
        }
    }

    fn noise_figure_db_at(&self, freq_mhz: f64) -> f64 {
        // A measured block can be either active or lossy; the noise figure can never be
        // better than the insertion loss of a passive part, so take the worse of the two.
        let passive = -self.s21_gain_at(freq_mhz);
        self.noise_figure_db.max(passive)
    }

    fn oip3_dbm(&self) -> Option<f64> {
        Some(self.oip3_dbm)
    }

    fn settling_ns(&self) -> f64 {
        // A measured response resolved to a spacing of Δf can only describe structure lasting
        // up to 1/Δf, so that is the longest impulse response the data can imply.
        let mut min_step = f64::INFINITY;
        for w in self.s21_table.windows(2) {
            let step = w[1].0 - w[0].0;
            if step > 0.0 {
                min_step = min_step.min(step);
            }
        }
        if min_step.is_finite() { 1000.0 / min_step } else { 0.0 }
    }
}

/// Parse a Touchstone `.s2p` file into an [`S2pModel`].
pub fn parse_touchstone_s2p(name: &str, content: &str) -> Result<S2pModel, String> {
    #[derive(PartialEq)]
    enum Fmt {
        Db,
        Ma,
        Ri,
    }

    let mut freq_multiplier = 1.0;
    // Touchstone's default when the option line omits the format is magnitude-angle.
    let mut fmt = Fmt::Ma;
    let mut s21_table: Vec<(f64, f64, f64)> = Vec::new();
    let mut s11_table: Vec<(f64, f64)> = Vec::new();
    let mut s22_table: Vec<(f64, f64)> = Vec::new();
    let mut noise_figure_db: Option<f64> = None;

    let to_db_deg = |v1: f64, v2: f64, fmt: &Fmt| -> (f64, f64) {
        match fmt {
            Fmt::Db => (v1, v2),
            Fmt::Ma => (20.0 * v1.abs().max(1e-12).log10(), v2),
            Fmt::Ri => {
                let c = Complex::new(v1, v2);
                (
                    20.0 * c.norm().max(1e-12).log10(),
                    c.arg().to_degrees(),
                )
            }
        }
    };

    for line in content.lines() {
        // Trailing comments are legal on data lines.
        let line = match line.find('!') {
            Some(0) => continue,
            Some(i) => &line[..i],
            None => line,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix('#') {
            for token in rest.split_whitespace() {
                match token.to_uppercase().as_str() {
                    "HZ" => freq_multiplier = 1e-6,
                    "KHZ" => freq_multiplier = 1e-3,
                    "MHZ" => freq_multiplier = 1.0,
                    "GHZ" => freq_multiplier = 1000.0,
                    "DB" => fmt = Fmt::Db,
                    "MA" => fmt = Fmt::Ma,
                    "RI" => fmt = Fmt::Ri,
                    _ => {}
                }
            }
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        // A noise-parameter block has five columns; S-parameter data for a 2-port has nine.
        if parts.len() == 5 && !s21_table.is_empty() {
            if let Ok(nf) = parts[1].parse::<f64>() {
                noise_figure_db = Some(noise_figure_db.map_or(nf, |v: f64| v.min(nf)));
            }
            continue;
        }
        if parts.len() < 9 {
            continue;
        }

        let freq = parts[0]
            .parse::<f64>()
            .map_err(|e| format!("Invalid frequency: {e}"))?
            * freq_multiplier;
        let mut vals = [0.0f64; 8];
        for (i, v) in vals.iter_mut().enumerate() {
            *v = parts[i + 1]
                .parse::<f64>()
                .map_err(|e| format!("Invalid S-parameter at {freq} MHz: {e}"))?;
        }

        let (s11_db, _) = to_db_deg(vals[0], vals[1], &fmt);
        let (s21_db, s21_deg) = to_db_deg(vals[2], vals[3], &fmt);
        let (s22_db, _) = to_db_deg(vals[6], vals[7], &fmt);

        s11_table.push((freq, s11_db));
        s21_table.push((freq, s21_db, s21_deg));
        s22_table.push((freq, s22_db));
    }

    if s21_table.is_empty() {
        return Err("No valid S2P data points found".into());
    }

    s21_table.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    s11_table.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    s22_table.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Without measured noise data, fall back to the insertion loss — right for a passive,
    // and an honest lower bound for anything else.
    let fallback_nf = -s21_table
        .iter()
        .map(|&(_, m, _)| m)
        .fold(f64::INFINITY, f64::min)
        .min(0.0);

    Ok(S2pModel {
        name: name.to_string(),
        s21_table,
        s11_table,
        s22_table,
        noise_figure_db: noise_figure_db.unwrap_or(fallback_nf),
        oip3_dbm: 30.0,
        use_measured_phase: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balun_interpolation_matches_datasheet() {
        let balun = BalunModel {
            amplitude_imbalance_db: 0.0,
            phase_imbalance_deg: 0.0,
            ..Default::default()
        };
        // In-band the datasheet table is what the response follows.
        assert!((balun.table_loss_at(100.0) - 0.78).abs() < 0.01);
        assert!((balun.table_loss_at(1000.0) - 1.30).abs() < 0.01);
        let il_550 = balun.table_loss_at(550.0);
        assert!(il_550 > 0.95 && il_550 < 1.12);
        // Mid-band, the band-edge roll-offs contribute almost nothing.
        assert!(
            (balun.insertion_loss_at(1000.0) - 1.30).abs() < 0.3,
            "mid-band IL should track the table, got {}",
            balun.insertion_loss_at(1000.0)
        );
    }

    #[test]
    fn balun_rolls_off_like_a_transformer() {
        let balun = BalunModel::default();
        // Out of band the roll-off has to have a *slope* — a real transformer keeps falling.
        // The high end is a 3-pole, so 18 dB per octave once it is clear of the corner.
        let a = balun.insertion_loss_at(8000.0);
        let b = balun.insertion_loss_at(16000.0);
        assert!(
            (b - a - 18.0).abs() < 1.5,
            "high-end slope {:.1} dB/octave, expected 18",
            b - a
        );
        // The low end is the 2-pole high-pass a transformer really is: 12 dB per octave.
        let c = balun.insertion_loss_at(2.0);
        let d = balun.insertion_loss_at(1.0);
        assert!(
            (d - c - 12.0).abs() < 1.5,
            "low-end slope {:.1} dB/octave, expected 12",
            d - c
        );
        // And in band it stays out of the way.
        assert!(balun.insertion_loss_at(1000.0) < 2.0);
    }

    #[test]
    fn balun_cmrr_degrades_with_imbalance() {
        let good = BalunModel { amplitude_imbalance_db: 0.05, phase_imbalance_deg: 0.5, ..Default::default() };
        let bad = BalunModel { amplitude_imbalance_db: 1.0, phase_imbalance_deg: 10.0, ..Default::default() };
        assert!(good.cmrr_db() > bad.cmrr_db() + 10.0);
        // Even a poor balun barely touches the differential amplitude.
        assert!(bad.insertion_loss_at(1000.0) - good.insertion_loss_at(1000.0) < 0.5);
    }

    #[test]
    fn butterworth_cutoff_is_3db_down() {
        for order in [1u32, 2, 4, 8] {
            let f = FilterModel {
                filter_type: FilterType::LowPass,
                cutoff_mhz: 1000.0,
                order,
                insertion_loss_db: 0.0,
                ..Default::default()
            };
            let at_fc = f.attenuation_at(1000.0);
            assert!(
                (at_fc - 3.01).abs() < 0.05,
                "order {order}: A(fc) = {at_fc}, expected 3.01 dB"
            );
            // Butterworth rolls off 20n dB/decade, no more and no less.
            let decade = f.attenuation_at(10000.0);
            assert!(
                (decade - 20.0 * order as f64).abs() < 0.1,
                "order {order}: A(10 fc) = {decade}, expected {}",
                20.0 * order as f64
            );
        }
    }

    #[test]
    fn lowpass_filter_passband_and_stopband() {
        let filter = FilterModel {
            filter_type: FilterType::LowPass,
            cutoff_mhz: 1000.0,
            order: 4,
            insertion_loss_db: 0.0,
            ..Default::default()
        };
        assert!(filter.attenuation_at(100.0) < 1.0);
        assert!(filter.attenuation_at(5000.0) > 40.0);
    }

    #[test]
    fn highpass_mirrors_lowpass() {
        let hp = FilterModel {
            filter_type: FilterType::HighPass,
            cutoff_mhz: 1000.0,
            order: 3,
            insertion_loss_db: 0.0,
            ..Default::default()
        };
        assert!((hp.attenuation_at(1000.0) - 3.01).abs() < 0.05);
        assert!(hp.attenuation_at(100.0) > 55.0);
        assert!(hp.attenuation_at(10000.0) < 0.01);
    }

    #[test]
    fn bandpass_is_geometrically_symmetric() {
        let bp = FilterModel {
            filter_type: FilterType::BandPass,
            cutoff_mhz: 1000.0,
            bandwidth_mhz: 200.0,
            order: 4,
            insertion_loss_db: 0.0,
            ..Default::default()
        };
        // 500 and 2000 MHz are the same distance from 1000 MHz on a log axis.
        let lo = bp.attenuation_at(500.0);
        let hi = bp.attenuation_at(2000.0);
        assert!(
            (lo - hi).abs() < 0.5,
            "geometrically symmetric points should match: {lo} vs {hi}"
        );
        // Band edges are the 3 dB points.
        assert!((bp.attenuation_at(1105.0) - 3.01).abs() < 0.6, "{}", bp.attenuation_at(1105.0));
        assert!(bp.attenuation_at(1000.0) < 0.01);
    }

    #[test]
    fn chebyshev_is_steeper_and_ripples() {
        let butter = FilterModel {
            filter_type: FilterType::LowPass,
            cutoff_mhz: 1000.0,
            order: 5,
            response: FilterResponse::Butterworth,
            insertion_loss_db: 0.0,
            ..Default::default()
        };
        let cheby = FilterModel {
            response: FilterResponse::Chebyshev { ripple_db: 0.5 },
            ..butter.clone()
        };
        assert!(
            cheby.attenuation_at(2000.0) > butter.attenuation_at(2000.0) + 10.0,
            "Chebyshev should be steeper: {} vs {}",
            cheby.attenuation_at(2000.0),
            butter.attenuation_at(2000.0)
        );
        // Passband stays inside the ripple spec.
        for i in 1..20 {
            let f = i as f64 * 50.0;
            let a = cheby.attenuation_at(f);
            assert!((-0.01..=0.55).contains(&a), "ripple out of spec at {f} MHz: {a}");
        }
    }

    #[test]
    fn filter_has_physical_group_delay() {
        let f = FilterModel {
            filter_type: FilterType::LowPass,
            cutoff_mhz: 1000.0,
            order: 4,
            insertion_loss_db: 0.0,
            ..Default::default()
        };
        // A 1 GHz 4-pole Butterworth has ~0.4 ns of in-band group delay, and more near the
        // corner than at DC. Zero-phase magnitude-only filtering gave exactly nothing.
        let dc = f.group_delay_ns(10.0);
        let knee = f.group_delay_ns(900.0);
        assert!(dc > 0.1 && dc < 1.0, "DC group delay {dc} ns");
        assert!(knee > dc, "group delay should peak near the corner: {dc} -> {knee}");
    }

    #[test]
    fn real_input_stays_real_through_every_component() {
        let ctx = ChainCtx::new(15000.0, 0.0, None);
        let n = 512;
        let sig: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 / 15000.0;
                Complex::new((2.0 * PI * 1000.0 * t).cos() * 0.5, 0.0)
            })
            .collect();

        let comps: Vec<(&str, Box<dyn RfComponent>)> = vec![
            ("balun", Box::new(BalunModel::default())),
            ("filter", Box::new(FilterModel::default())),
            ("amp", Box::new(AmplifierModel::default())),
            ("atten", Box::new(AttenuatorModel::default())),
            ("splitter", Box::new(SplitterModel::default())),
            ("combiner", Box::new(CombinerModel::default())),
            ("mixer", Box::new(MixerModel::default())),
            ("phase", Box::new(PhaseShifterModel::default())),
            ("coupler", Box::new(DirectionalCouplerModel::default())),
            ("s2p", Box::new(S2pModel::default())),
        ];

        for (name, c) in comps {
            let out = c.process(&sig, &ctx, 0);
            let imag: f64 = out.iter().map(|s| s.im.abs()).sum();
            assert!(imag < 1e-9, "{name} produced imaginary voltage: {imag}");
            let power: f64 = out.iter().map(|s| s.re * s.re).sum();
            assert!(power > 0.0, "{name} produced no output");
        }
    }

    #[test]
    fn splitter_ideal_loss() {
        let splitter = SplitterModel { num_outputs: 2, excess_loss_db: 0.0 };
        assert!((splitter.total_loss_db() - 3.01).abs() < 0.1);
    }

    #[test]
    fn coupler_ports_differ() {
        let dc = DirectionalCouplerModel { coupling_db: 20.0, insertion_loss_db: 0.5, directivity_db: 25.0 };
        let main = dc.response_db(1000.0, 0);
        let coupled = dc.response_db(1000.0, COUPLER_COUPLED_PORT);
        assert!((coupled - -20.5).abs() < 0.01, "coupled port at {coupled} dB");
        // Tapping 1% of the power costs the through path 0.044 dB on top of its own loss.
        assert!((main - -0.544).abs() < 0.01, "through port at {main} dB");
    }

    #[test]
    fn phase_shifter_preserves_amplitude_at_any_angle() {
        let ctx = ChainCtx::new(15000.0, 0.0, None);
        let n = 4096;
        let sig: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((2.0 * PI * 1000.0 * i as f64 / 15000.0).cos(), 0.0))
            .collect();
        let p_in: f64 = sig.iter().map(|s| s.re * s.re).sum::<f64>() / n as f64;

        for deg in [0.0, 30.0, 60.0, 90.0, 135.0, 180.0] {
            let ps = PhaseShifterModel {
                phase_shift_deg: deg,
                insertion_loss_db: 0.0,
                kind: PhaseShiftKind::ConstantPhase,
                ref_freq_mhz: 1000.0,
            };
            let out = ps.process(&sig, &ctx, 0);
            let p_out: f64 = out.iter().map(|s| s.re * s.re).sum::<f64>() / n as f64;
            let db = 10.0 * (p_out / p_in).log10();
            assert!(db.abs() < 0.2, "{deg}° changed the level by {db} dB");
        }
    }

    #[test]
    fn phase_shifter_actually_shifts_phase() {
        let ctx = ChainCtx::new(15000.0, 0.0, None);
        let n = 3000; // 200 whole cycles of 1 GHz, so the DFT probe is coherent
        let sig: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((2.0 * PI * 1000.0 * i as f64 / 15000.0).cos(), 0.0))
            .collect();
        let probe = |s: &[Complex<f64>]| -> f64 {
            let mut acc = Complex::new(0.0, 0.0);
            for (i, v) in s.iter().enumerate() {
                let th = -2.0 * PI * 1000.0 * i as f64 / 15000.0;
                acc += v.re * Complex::new(th.cos(), th.sin());
            }
            acc.arg().to_degrees()
        };
        let ps = PhaseShifterModel {
            phase_shift_deg: 90.0,
            insertion_loss_db: 0.0,
            kind: PhaseShiftKind::ConstantPhase,
            ref_freq_mhz: 1000.0,
        };
        let out = ps.process(&sig, &ctx, 0);
        let shift = probe(&out) - probe(&sig);
        assert!((shift + 90.0).abs() < 1.0, "expected -90°, got {shift}");
    }

    #[test]
    fn true_delay_scales_phase_with_frequency() {
        let ps = PhaseShifterModel {
            phase_shift_deg: 90.0,
            insertion_loss_db: 0.0,
            kind: PhaseShiftKind::TrueDelay,
            ref_freq_mhz: 1000.0,
        };
        let at_ref = ps.transfer_at(1000.0, 0).arg().to_degrees();
        let at_double = ps.transfer_at(2000.0, 0).arg().to_degrees();
        assert!((at_ref + 90.0).abs() < 1e-6);
        assert!((at_double + 180.0).abs() < 1e-6);
        assert!((ps.delay_ns() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn mixer_produces_both_sidebands() {
        let fs = 15000.0;
        let n = 15000;
        let ctx = ChainCtx::new(fs, 0.0, None);
        let sig: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((2.0 * PI * 1000.0 * i as f64 / fs).cos(), 0.0))
            .collect();
        let mixer = MixerModel {
            lo_freq_mhz: 300.0,
            conversion_loss_db: 0.0,
            lo_leakage_dbfs: -200.0,
            lo_harmonic3_dbc: 0.0,
            oip3_dbm: 60.0, // out of the way, so only the conversion is under test
            ..Default::default()
        };
        let out = mixer.process(&sig, &ctx, 0);

        let probe = |f_mhz: f64| -> f64 {
            let mut acc = Complex::new(0.0, 0.0);
            for (i, v) in out.iter().enumerate() {
                let th = -2.0 * PI * f_mhz * i as f64 / fs;
                acc += v.re * Complex::new(th.cos(), th.sin());
            }
            20.0 * (2.0 * acc.norm() / n as f64).max(1e-300).log10()
        };
        // Both products, each at the specified conversion loss relative to the input.
        assert!((probe(700.0) - 0.0).abs() < 0.05, "difference product at {}", probe(700.0));
        assert!((probe(1300.0) - 0.0).abs() < 0.05, "sum product at {}", probe(1300.0));
    }

    #[test]
    fn mixer_conversion_loss_lands_on_the_wanted_product() {
        let fs = 15000.0;
        let n = 15000;
        let ctx = ChainCtx::new(fs, 0.0, None);
        let sig: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(0.1 * (2.0 * PI * 1000.0 * i as f64 / fs).cos(), 0.0))
            .collect();
        let mixer = MixerModel {
            lo_freq_mhz: 300.0,
            conversion_loss_db: 7.0,
            lo_leakage_dbfs: -200.0,
            lo_harmonic3_dbc: 0.0,
            oip3_dbm: 60.0,
            ..Default::default()
        };
        let out = mixer.process(&sig, &ctx, 0);
        let mut acc = Complex::new(0.0, 0.0);
        for (i, v) in out.iter().enumerate() {
            let th = -2.0 * PI * 700.0 * i as f64 / fs;
            acc += v.re * Complex::new(th.cos(), th.sin());
        }
        let amp = 2.0 * acc.norm() / n as f64;
        // Datasheet conversion loss is the loss to the wanted product, and the inherent 6 dB
        // sideband split is part of that figure rather than on top of it.
        let db = 20.0 * (amp / 0.1).log10();
        assert!((db + 7.0).abs() < 0.05, "wanted product came out at {db} dB");
    }

    #[test]
    fn mixer_im3_matches_its_oip3() {
        let fs = 15000.0;
        let n = 15000;
        let ctx = ChainCtx::new(fs, 0.0, None);
        let mixer = MixerModel {
            lo_freq_mhz: 300.0,
            conversion_loss_db: 7.0,
            lo_leakage_dbfs: -200.0,
            lo_harmonic3_dbc: 0.0,
            oip3_dbm: 15.0,
            ..Default::default()
        };
        let tone = dbm_to_amplitude(-20.0);
        let sig: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                Complex::new(
                    tone * (2.0 * PI * 1000.0 * t).cos() + tone * (2.0 * PI * 1020.0 * t).cos(),
                    0.0,
                )
            })
            .collect();
        let out = mixer.process(&sig, &ctx, 0);

        let probe = |f_mhz: f64| -> f64 {
            let mut acc = Complex::new(0.0, 0.0);
            for (i, v) in out.iter().enumerate() {
                let th = -2.0 * PI * f_mhz * i as f64 / fs;
                acc += v.re * Complex::new(th.cos(), th.sin());
            }
            amplitude_to_dbm(2.0 * acc.norm() / n as f64)
        };
        // Downconverted tones at 700/720 MHz, IM3 at 680 MHz.
        let fund = probe(700.0);
        let im3 = probe(680.0);
        let implied = fund + (fund - im3) / 2.0;
        assert!(
            (implied - 15.0).abs() < 1.0,
            "implied OIP3 {implied} dBm, expected 15 (fund {fund}, IM3 {im3})"
        );
    }

    #[test]
    fn mixer_lo_is_phase_continuous_across_frames() {
        let fs = 15000.0;
        let n = 512;
        let dt = 1.0 / fs;
        let mixer = MixerModel { lo_freq_mhz: 300.0, ..Default::default() };
        let tone = |count: usize, t0: f64| -> Vec<Complex<f64>> {
            (0..count)
                .map(|i| Complex::new((2.0 * PI * 1000.0 * (t0 + i as f64 * dt)).cos(), 0.0))
                .collect()
        };

        let cont = mixer.process(&tone(2 * n, 0.0), &ChainCtx::new(fs, 0.0, None), 0);
        let second = mixer.process(
            &tone(n, n as f64 * dt),
            &ChainCtx::new(fs, n as f64 * dt, None),
            0,
        );
        let err = (second[0] - cont[n]).norm();
        assert!(err < 1e-9, "LO phase jumped between frames: {err}");
    }

    #[test]
    fn mixer_lo_leakage_appears_at_the_lo() {
        let fs = 15000.0;
        let n = 15000;
        let ctx = ChainCtx::new(fs, 0.0, None);
        let sig = vec![Complex::new(0.0, 0.0); n];
        let mixer = MixerModel {
            lo_freq_mhz: 300.0,
            lo_leakage_dbfs: -40.0,
            lo_harmonic3_dbc: 0.0,
            ..Default::default()
        };
        let out = mixer.process(&sig, &ctx, 0);
        let mut acc = Complex::new(0.0, 0.0);
        for (i, v) in out.iter().enumerate() {
            let th = -2.0 * PI * 300.0 * i as f64 / fs;
            acc += v.re * Complex::new(th.cos(), th.sin());
        }
        let level = 20.0 * (2.0 * acc.norm() / n as f64).log10();
        assert!((level + 40.0).abs() < 0.2, "LO leakage at {level} dBFS");
    }

    #[test]
    fn amplifier_compresses_at_its_p1db() {
        // 20 dB of gain, +10 dBm output P1dB: drive it to exactly that point and the gain
        // should be down 1 dB, not the textbook-perfect 20 dB a linear model gives.
        let amp = AmplifierModel {
            gain_db: 20.0,
            noise_figure_db: 2.0,
            p1db_dbm: 10.0,
            oip3_dbm: 22.0,
            bandwidth_mhz: 0.0,
        };
        let fit = amp.nonlinearity().expect("nonlinear");
        let in_amp = dbm_to_amplitude(amp.input_p1db_dbm());
        let comp = fit.compression_db(in_amp);
        assert!(
            (comp + 1.0).abs() < 0.15,
            "expected -1 dB at P1dB, got {comp} dB"
        );
        // Well below compression it stays linear.
        assert!(fit.compression_db(in_amp / 100.0).abs() < 0.01);
    }

    #[test]
    fn amplifier_output_saturates() {
        let ctx = ChainCtx::new(15000.0, 0.0, None);
        let amp = AmplifierModel {
            gain_db: 30.0,
            noise_figure_db: 2.0,
            p1db_dbm: 10.0,
            oip3_dbm: 20.0,
            bandwidth_mhz: 0.0,
        };
        let n = 2048;
        let sig: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((2.0 * PI * 1000.0 * i as f64 / 15000.0).cos(), 0.0))
            .collect();
        let out = amp.process(&sig, &ctx, 0);
        let peak = out.iter().map(|s| s.re.abs()).fold(0.0f64, f64::max);
        let linear_peak = 10.0_f64.powf(30.0 / 20.0);
        assert!(
            peak < linear_peak * 0.5,
            "a full-scale drive should be deep in compression: peak {peak} vs linear {linear_peak}"
        );
        // Saturated output must still be bounded, not blowing up through the quintic term.
        assert!(peak.is_finite() && peak < linear_peak);
    }

    #[test]
    fn amplifier_generates_im3_at_the_specified_oip3() {
        let fs = 15000.0;
        let n = 15000;
        let ctx = ChainCtx::new(fs, 0.0, None);
        let amp = AmplifierModel {
            gain_db: 10.0,
            noise_figure_db: 2.0,
            p1db_dbm: 5.0,
            oip3_dbm: 25.0,
            bandwidth_mhz: 0.0,
        };
        // Two tones well below compression, 20 MHz apart.
        let tone_amp = dbm_to_amplitude(-30.0);
        let sig: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let v = tone_amp * (2.0 * PI * 1000.0 * t).cos()
                    + tone_amp * (2.0 * PI * 1020.0 * t).cos();
                Complex::new(v, 0.0)
            })
            .collect();
        let out = amp.process(&sig, &ctx, 0);

        let probe = |f_mhz: f64| -> f64 {
            let mut acc = Complex::new(0.0, 0.0);
            for (i, v) in out.iter().enumerate() {
                let th = -2.0 * PI * f_mhz * i as f64 / fs;
                acc += v.re * Complex::new(th.cos(), th.sin());
            }
            amplitude_to_dbm(2.0 * acc.norm() / n as f64)
        };
        let fund = probe(1000.0);
        let im3 = probe(980.0);
        // OIP3 = Pout + delta/2 for third-order products.
        let implied_oip3 = fund + (fund - im3) / 2.0;
        assert!(
            (implied_oip3 - 25.0).abs() < 1.0,
            "implied OIP3 {implied_oip3} dBm, expected 25 (fund {fund}, IM3 {im3})"
        );
    }

    #[test]
    fn passive_stage_noise_lands_at_ktb() {
        // A 10 dB pad at 290 K must hand on kTB regardless of what it is fed, because the
        // loss it applies to the incoming noise is exactly made up by its own contribution.
        let fs = 15000.0;
        let n = 1 << 15;
        let ctx = ChainCtx::new(fs, 0.0, Some(290.0)).for_node(1);
        let kt_b = thermal_noise_power(fs, 290.0);

        let mut rng = Rng::new(12345);
        let sigma = kt_b.sqrt();
        let input: Vec<Complex<f64>> = (0..n)
            .map(|_| Complex::new(sigma * rng.gaussian(), 0.0))
            .collect();

        let pad = AttenuatorModel { attenuation_db: 10.0 };
        let out = pad.process(&input, &ctx, 0);
        let p_out: f64 = out.iter().map(|s| s.re * s.re).sum::<f64>() / n as f64;
        let db = 10.0 * (p_out / kt_b).log10();
        assert!(db.abs() < 0.5, "a lossy passive should still output kTB, got {db} dB");
    }

    #[test]
    fn amplifier_noise_figure_sets_its_output_noise() {
        let fs = 15000.0;
        let n = 1 << 15;
        let kt_b = thermal_noise_power(fs, 290.0);
        let amp = AmplifierModel {
            gain_db: 20.0,
            noise_figure_db: 6.0,
            p1db_dbm: 30.0,
            oip3_dbm: 40.0,
            bandwidth_mhz: 0.0,
        };
        let ctx = ChainCtx::new(fs, 0.0, Some(290.0)).for_node(7);

        // Driven by a kTB source, the output noise must be F·G·kTB.
        let mut rng = Rng::new(999);
        let sigma = kt_b.sqrt();
        let input: Vec<Complex<f64>> = (0..n)
            .map(|_| Complex::new(sigma * rng.gaussian(), 0.0))
            .collect();
        let out = amp.process(&input, &ctx, 0);
        let p_out: f64 = out.iter().map(|s| s.re * s.re).sum::<f64>() / n as f64;
        let g = 10.0_f64.powf(20.0 / 10.0);
        let expected = 10.0_f64.powf(6.0 / 10.0) * g * kt_b;
        let db = 10.0 * (p_out / expected).log10();
        assert!(db.abs() < 0.5, "output noise off by {db} dB from F·G·kTB");
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
        // S11 is kept now, so return loss and VSWR are available.
        assert!((model.return_loss_db(100.0).unwrap() - (-15.0)).abs() < 1e-5);
        let vswr = model.vswr(100.0).unwrap();
        assert!((vswr - 1.433).abs() < 0.01, "VSWR {vswr}");
        // A passive block cannot have a noise figure better than its loss.
        assert!(model.noise_figure_db_at(1000.0) >= 2.5 - 1e-9);
    }

    #[test]
    fn touchstone_ri_and_ma_formats() {
        // Magnitude-angle: S21 magnitude 0.5 = -6.02 dB at -45 degrees.
        let ma = "# GHz S MA R 50\n1.0 0.1 0 0.5 -45 0.5 -45 0.1 0\n2.0 0.1 0 0.5 -90 0.5 -90 0.1 0\n";
        let m = parse_touchstone_s2p("ma", ma).unwrap();
        assert!((m.s21_gain_at(1000.0) - (-6.02)).abs() < 0.01);
        assert!((m.s21_phase_at(1000.0) - (-45.0)).abs() < 0.01);
        // Frequencies were in GHz.
        assert!((m.s21_table[1].0 - 2000.0).abs() < 1e-6);

        // Real-imaginary: 0.5 + 0j is also -6.02 dB, at 0 degrees.
        let ri = "# MHz S RI R 50\n100 0 0 0.5 0 0.5 0 0 0\n";
        let r = parse_touchstone_s2p("ri", ri).unwrap();
        assert!((r.s21_gain_at(100.0) - (-6.02)).abs() < 0.01);
        assert!(r.s21_phase_at(100.0).abs() < 0.01);
    }

    #[test]
    fn touchstone_phase_unwraps_across_points() {
        // Phase running past -180 wraps in the file; interpolation must not see a jump.
        let s2p = "# MHz S DB R 50\n1000 -20 0 -1 -170 -1 -170 -20 0\n2000 -20 0 -1 170 -1 170 -20 0\n";
        let m = parse_touchstone_s2p("wrap", s2p).unwrap();
        // Halfway between -170 and -190 is -180, not 0.
        let mid = m.s21_phase_at(1500.0);
        assert!((mid + 180.0).abs() < 1.0, "unwrapped midpoint {mid}");
    }

    #[test]
    fn dbm_reference_round_trips() {
        assert!((amplitude_to_dbm(1.0) - FULL_SCALE_DBM).abs() < 1e-9);
        assert!((dbm_to_amplitude(FULL_SCALE_DBM) - 1.0).abs() < 1e-9);
        // Half amplitude is 6 dB down.
        assert!((amplitude_to_dbm(0.5) - (FULL_SCALE_DBM - 6.0206)).abs() < 1e-3);
    }

    #[test]
    fn thermal_noise_matches_ktb() {
        // -174 dBm/Hz over 15 GHz is -72.2 dBm, i.e. -73.2 dBFS on this reference.
        let p = thermal_noise_power(15000.0, 290.0);
        let dbfs = 10.0 * p.log10();
        assert!((dbfs + 73.2).abs() < 0.2, "kTB came out at {dbfs} dBFS");
        // Doubling the temperature adds 3 dB.
        let hot = 10.0 * thermal_noise_power(15000.0, 580.0).log10();
        assert!((hot - dbfs - 3.01).abs() < 0.02);
    }
}
