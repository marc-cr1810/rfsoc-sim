//! DSP processing functions: FFT, Nyquist zone folding, mixing, and decimation.

#![allow(dead_code)]

use crate::rfdc::{AdcBlock, AdcTile, CoarseMixFreq, MixerType, MixerMode as RfdcMixerMode, FineMixerScale};
use num_complex::Complex;
use rayon::prelude::*;
use realfft::RealFftPlanner;
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
    /// Window used for every spectrum here. Peak picking needs it to know what a tone's own
    /// leakage skirt looks like, so it travels with the data rather than being assumed.
    pub display_window: FftWindow,
    /// Resolution bandwidth of the output spectrum, in MHz: the span divided by the number of
    /// samples actually transformed. The trace carries more points than this after display
    /// padding, so anything that reasons about how far apart two features really are — peak
    /// picking, marker readouts — has to use this rather than the point spacing.
    pub output_rbw_mhz: f64,
    /// Samples the output transform actually consumed. Below the requested
    /// [`SpectrumDetail::output_bins`] whenever the wideband sample budget could not supply a
    /// long enough record, which is how the UI knows to say so rather than quietly under-deliver.
    pub output_bins_analysed: usize,
    /// Bin count the output transform asked for, before any budget clipping.
    pub output_bins_requested: usize,
    /// Complex baseband output time-domain samples (for oscilloscope & constellation).
    pub output_time_samples: Vec<Complex<f64>>,
    /// True if the physical ADC waveform clipped at any point during this capture.
    pub overrange: bool,
    /// Monotonic counter identifying this capture, assigned by the caller.
    ///
    /// The spectrum is recomputed on its own schedule rather than once per frame, so anything
    /// accumulating history — the waterfall — needs to tell a genuinely new capture from a
    /// repaint of the previous one. Sequence numbers do that; comparing the data would not,
    /// since two consecutive captures of a static signal are legitimately identical.
    pub sequence: u64,
}

// ...

use serde::{Deserialize, Serialize};
use std::cell::RefCell;

thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
    /// Separate planner for the real-input transforms. Everything from the ADC pin to the
    /// mixer is a real voltage, and a real-to-complex transform of it costs half what running
    /// the same data through a complex transform with a zero imaginary part costs.
    static REAL_FFT_PLANNER: RefCell<RealFftPlanner<f64>> = RefCell::new(RealFftPlanner::new());
}

/// Sample count above which a per-sample stage is spread across the thread pool.
///
/// Every stage below is a pure function of the sample index, so splitting one into chunks is
/// always *correct*; it is only worth doing once the chunk carries more work than the hand-off
/// costs. A `Fast`/×1 capture is 4096 samples and finishes in tens of microseconds, so it stays
/// on the calling thread.
pub(crate) const PAR_MIN_LEN: usize = 32_768;

/// Samples one worker takes at a time from a parallel stage.
///
/// Sized so a chunk is a few tens of microseconds of work — long enough to amortise the
/// hand-off, short enough that the last chunk cannot leave the other workers idle for long.
pub(crate) const PAR_CHUNK: usize = 8_192;

/// Reusable working buffers for the capture pipeline.
///
/// The wideband stages allocate megabytes per capture and free them again at the end of it. The
/// allocator hands blocks that size straight back to the kernel, so the next capture faults
/// every page back in one at a time — about a third of a `Max` capture's cost, spent on nothing
/// but zeroing memory that was already ours. Parking the allocations in a per-thread pool
/// removes that without changing a single arithmetic operation.
///
/// A thread holds at most [`POOL_LIMIT`] buffers of each type, so what it retains is roughly one
/// capture's working set rather than a growing hoard.
mod scratch {
    use super::Complex;
    use std::cell::RefCell;

    /// Buffers of each type a thread keeps parked. The pipeline holds a handful alive at once.
    const POOL_LIMIT: usize = 8;

    thread_local! {
        static REALS: RefCell<Vec<Vec<f64>>> = const { RefCell::new(Vec::new()) };
        static COMPLEX: RefCell<Vec<Vec<Complex<f64>>>> = const { RefCell::new(Vec::new()) };
    }

    /// Take a buffer with room for `len`, emptied — for a caller that will `extend` into it.
    macro_rules! take_empty {
        ($pool:expr, $len:expr) => {
            $pool.with(|p| {
                let mut p = p.borrow_mut();
                let idx = p.iter().position(|b: &Vec<_>| b.capacity() >= $len);
                let mut v = match idx {
                    Some(i) => p.swap_remove(i),
                    None => p.pop().unwrap_or_default(),
                };
                v.clear();
                v.reserve($len);
                v
            })
        };
    }

    /// Hand a buffer back, unless this thread is already holding enough of them.
    macro_rules! give {
        ($pool:expr, $v:expr) => {
            $pool.with(|p| {
                let mut p = p.borrow_mut();
                if p.len() < POOL_LIMIT {
                    p.push($v);
                }
            })
        };
    }

    /// An empty `f64` buffer with room for `len`.
    pub fn reals(len: usize) -> Vec<f64> {
        take_empty!(REALS, len)
    }

    /// `len` zeroed `f64`s, for a stage that writes its output by index.
    pub fn reals_zeroed(len: usize) -> Vec<f64> {
        let mut v = reals(len);
        v.resize(len, 0.0);
        v
    }

    /// An empty complex buffer with room for `len`.
    pub fn complex(len: usize) -> Vec<Complex<f64>> {
        take_empty!(COMPLEX, len)
    }

    /// `len` zeroed complex samples, for a stage that writes its output by index.
    pub fn complex_zeroed(len: usize) -> Vec<Complex<f64>> {
        let mut v = complex(len);
        v.resize(len, Complex::new(0.0, 0.0));
        v
    }

    /// `len` complex samples of *whatever the last user left there*.
    ///
    /// Only for a caller that writes every element before reading any of it — a transpose, say.
    /// Skipping the zero fill is the point: on a repeat of the same capture size nothing is
    /// written at all, where `complex_zeroed` would memset megabytes each time.
    pub fn complex_overwritten(len: usize) -> Vec<Complex<f64>> {
        COMPLEX.with(|p| {
            let mut p = p.borrow_mut();
            let idx = p.iter().position(|b: &Vec<_>| b.capacity() >= len);
            let mut v = match idx {
                Some(i) => p.swap_remove(i),
                None => p.pop().unwrap_or_default(),
            };
            v.resize(len, Complex::new(0.0, 0.0));
            v.truncate(len);
            v
        })
    }

    pub fn recycle_reals(v: Vec<f64>) {
        give!(REALS, v);
    }

    pub fn recycle_complex(v: Vec<Complex<f64>>) {
        give!(COMPLEX, v);
    }
}

/// Split `out` into [`PAR_CHUNK`]-sized pieces and run `fill(first_index, piece)` over each.
///
/// The split is by index and never by thread, so the same buffer is always divided the same
/// way — a stage that seeds anything off `first_index` produces bit-identical output whether it
/// ran on one thread or all of them. Short buffers take the same route without the pool, where
/// the hand-off would cost more than the work.
pub(crate) fn for_each_chunk<T: Send>(
    out: &mut [T],
    fill: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    if out.len() >= PAR_MIN_LEN {
        out.par_chunks_mut(PAR_CHUNK)
            .enumerate()
            .for_each(|(c, chunk)| fill(c * PAR_CHUNK, chunk));
    } else {
        out.chunks_mut(PAR_CHUNK)
            .enumerate()
            .for_each(|(c, chunk)| fill(c * PAR_CHUNK, chunk));
    }
}

/// [`for_each_chunk`], for a stage that also reports whether something happened somewhere.
fn any_chunk<T: Send>(
    out: &mut [T],
    fill: impl Fn(usize, &mut [T]) -> bool + Sync + Send,
) -> bool {
    if out.len() >= PAR_MIN_LEN {
        out.par_chunks_mut(PAR_CHUNK)
            .enumerate()
            .map(|(c, chunk)| fill(c * PAR_CHUNK, chunk))
            .reduce(|| false, |a, b| a || b)
    } else {
        out.chunks_mut(PAR_CHUNK)
            .enumerate()
            .map(|(c, chunk)| fill(c * PAR_CHUNK, chunk))
            .fold(false, |a, b| a || b)
    }
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

    /// The window's coefficient at sample `i` of `n`.
    fn coefficient(&self, i: usize, n: usize) -> f64 {
        let phi = 2.0 * PI * i as f64 / n as f64;
        match self {
            FftWindow::Hanning => 0.5 * (1.0 - phi.cos()),
            FftWindow::Hamming => 0.54 - 0.46 * phi.cos(),
            FftWindow::BlackmanHarris => {
                let a0 = 0.35875;
                let a1 = 0.48829;
                let a2 = 0.14128;
                let a3 = 0.01168;
                a0 - a1 * phi.cos() + a2 * (2.0 * phi).cos() - a3 * (3.0 * phi).cos()
            }
            FftWindow::FlatTop => {
                let a0 = 0.21557895;
                let a1 = 0.41663158;
                let a2 = 0.277263158;
                let a3 = 0.083578947;
                let a4 = 0.006947368;
                a0 - a1 * phi.cos() + a2 * (2.0 * phi).cos() - a3 * (3.0 * phi).cos()
                    + a4 * (4.0 * phi).cos()
            }
            FftWindow::Rectangular => 1.0,
        }
    }

    pub fn apply(&self, buffer: &mut [Complex<f64>]) {
        let n = buffer.len();
        if n == 0 || *self == FftWindow::Rectangular {
            return;
        }
        for (i, sample) in buffer.iter_mut().enumerate() {
            *sample *= self.coefficient(i, n);
        }
    }

    /// Window a real buffer in place. Same coefficients as [`Self::apply`].
    pub fn apply_real(&self, buffer: &mut [f64]) {
        let n = buffer.len();
        if n == 0 || *self == FftWindow::Rectangular {
            return;
        }
        for (i, sample) in buffer.iter_mut().enumerate() {
            *sample *= self.coefficient(i, n);
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

    /// Main-lobe half-width in RBW bins, the sidelobe plateau level (dB below the peak) and
    /// where it ends, and the roll-off beyond it in dB per octave of distance.
    ///
    /// Fitted to each window's own transform so the envelope stays *under* the real sidelobes
    /// — the published first-sidelobe-and-slope figures don't bound them, because the first
    /// few sidelobes sit well above a line drawn from the first one. Checked by
    /// `window_leakage_envelope_matches_the_real_transform`.
    fn leakage_shape(&self) -> (f64, f64, f64, f64) {
        match self {
            //                          lobe  plateau  ends  roll-off
            FftWindow::Hanning => (2.0, 31.5, 3.0, 17.3),
            FftWindow::Hamming => (2.0, 42.7, 5.0, 3.5),
            FftWindow::BlackmanHarris => (5.0, 92.0, 6.0, 3.0),
            FftWindow::FlatTop => (6.0, 93.5, 7.0, 0.0),
            FftWindow::Rectangular => (1.0, 13.3, 2.0, 6.0),
        }
    }

    /// How far below a tone's peak this window's leakage sits, `bins` RBW bins away from it.
    ///
    /// Returns 0 inside the main lobe — nothing there is separable from the tone itself. Used
    /// to tell a real second signal from the first one's own sidelobes, which zero-padded
    /// display traces resolve as genuine local maxima.
    pub fn leakage_envelope_db(&self, bins: f64) -> f64 {
        let (lobe, plateau_db, plateau_to, rolloff) = self.leakage_shape();
        let bins = bins.abs();
        if bins <= lobe {
            return 0.0;
        }
        if bins <= plateau_to {
            return plateau_db;
        }
        plateau_db + rolloff * (bins / plateau_to).log2()
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
/// the pre-ADC spectrum, the sampler and the folded spectrum on one consistent scale — and
/// everything from here to the mixer then runs on `f64` rather than on a `Complex<f64>` whose
/// imaginary half is known to be zero, which halves the memory traffic of the widest buffers
/// in the program and lets the per-sample loops vectorise.
pub fn physical_voltage(samples: &[Complex<f64>]) -> Vec<f64> {
    let mut out = scratch::reals(samples.len());
    out.extend(samples.iter().map(|s| s.re));
    out
}

/// Apply the converter's analog input bandwidth roll-off to the wideband voltage waveform.
///
/// Runs before sampling so that content in high Nyquist zones aliases down already
/// attenuated, the way a real track-and-hold behaves.
pub fn apply_analog_bandwidth(
    samples: &[f64],
    sample_rate_mhz: f64,
    afe: &crate::rfdc::AnalogFrontEnd,
) -> Vec<f64> {
    if !afe.enabled || samples.len() < 2 || sample_rate_mhz <= 0.0 {
        let mut out = scratch::reals(samples.len());
        out.extend_from_slice(samples);
        return out;
    }

    let n = samples.len();

    // The signal is real, so the transform is conjugate-symmetric and only the non-negative
    // half of the bins exists. That is also why the gain table is half length.
    let mut spectrum = real_fft_forward(samples);

    // Scale each bin by the analog response. The curve is real and even, and depends only on
    // the bin grid and the AFE settings, so it is built once and reused rather than evaluated
    // per bin per capture.
    let gains = analog_bandwidth_gains(n, sample_rate_mhz, afe);
    for (bin, &g) in spectrum.iter_mut().zip(gains.iter()) {
        *bin *= g;
    }

    let mut out = real_fft_inverse(&mut spectrum, n);
    scratch::recycle_complex(spectrum);
    let norm = 1.0 / n as f64;
    for v in out.iter_mut() {
        *v *= norm;
    }
    out
}

/// Forward transform of a real signal, giving its `n/2 + 1` non-negative-frequency bins.
fn real_fft_forward(samples: &[f64]) -> Vec<Complex<f64>> {
    let n = samples.len();
    if let Some(bins) = parallel_real_fft_forward(samples) {
        return bins;
    }
    let fft = REAL_FFT_PLANNER.with(|planner| planner.borrow_mut().plan_fft_forward(n));
    let mut input = scratch::reals(n);
    input.extend_from_slice(samples);
    let mut spectrum = scratch::complex_zeroed(n / 2 + 1);
    fft.process(&mut input, &mut spectrum)
        .expect("real FFT length mismatch");
    scratch::recycle_reals(input);
    spectrum
}

/// Inverse of [`real_fft_forward`], unnormalised — the caller divides by `n`.
///
/// Consumes the spectrum: the transform is free to scribble on its input.
fn real_fft_inverse(spectrum: &mut [Complex<f64>], n: usize) -> Vec<f64> {
    if let Some(out) = parallel_real_fft_inverse(spectrum, n) {
        return out;
    }
    let ifft = REAL_FFT_PLANNER.with(|planner| planner.borrow_mut().plan_fft_inverse(n));
    let mut out = scratch::reals_zeroed(n);
    ifft.process(spectrum, &mut out)
        .expect("real inverse FFT length mismatch");
    out
}

// ---------------------------------------------------------------------------
// Multi-threaded transforms for the wideband record
// ---------------------------------------------------------------------------

/// Shortest transform worth splitting across the pool.
///
/// Below this the sub-transforms and the transposes between them cost more than the single
/// long pass they replace, and rustfft's own AVX kernels are already running flat out.
const PARALLEL_FFT_MIN: usize = 1 << 16;

/// Rows one task takes at a time, so it can amortise a single FFT scratch allocation.
const FFT_ROWS_PER_TASK: usize = 8;

/// Factor `n` into the two roughly equal halves the four-step decomposition needs.
///
/// Powers of two only. Anything else falls back to the single-threaded path rather than
/// growing a general mixed-radix case for lengths this program never uses.
fn fft_split(n: usize) -> Option<(usize, usize)> {
    if !n.is_power_of_two() || n < PARALLEL_FFT_MIN {
        return None;
    }
    let n1 = 1usize << (n.trailing_zeros() / 2);
    Some((n1, n / n1))
}

/// `dst[j·rows + i] = src[i·cols + j]`, in cache-sized blocks and across the pool.
///
/// Each task owns a stripe of destination rows, so no two tasks write the same memory.
fn transpose(src: &[Complex<f64>], dst: &mut [Complex<f64>], rows: usize, cols: usize) {
    /// Square block that keeps both the read and the write side in cache.
    const BLOCK: usize = 32;
    dst.par_chunks_mut(BLOCK * rows).enumerate().for_each(|(stripe, out)| {
        let j0 = stripe * BLOCK;
        for i0 in (0..rows).step_by(BLOCK) {
            for j in j0..(j0 + BLOCK).min(cols) {
                for i in i0..(i0 + BLOCK).min(rows) {
                    out[(j - j0) * rows + i] = src[i * cols + j];
                }
            }
        }
    });
}

/// Transform `buf` in place as `n2` transforms of length `n1` and then `n1` of length `n2`.
///
/// This is the textbook `N = N1·N2` factorisation of the DFT — the same one every mixed-radix
/// FFT is built from — so it computes the same transform as a single long pass, differing only
/// in the order the identical butterflies are evaluated. What it buys is that both halves are
/// hundreds of independent short transforms, which the thread pool can take in parallel, where
/// one long transform is stuck on a single core.
///
/// Returns `false` if `n` is not a length it handles, leaving `buf` untouched.
fn parallel_fft(buf: &mut [Complex<f64>], inverse: bool) -> bool {
    let n = buf.len();
    let Some((n1, n2)) = fft_split(n) else {
        return false;
    };

    let (inner1, inner2) = FFT_PLANNER.with(|planner| {
        let mut planner = planner.borrow_mut();
        if inverse {
            (planner.plan_fft_inverse(n1), planner.plan_fft_inverse(n2))
        } else {
            (planner.plan_fft_forward(n1), planner.plan_fft_forward(n2))
        }
    });

    // Rows of the transform run along the *other* axis each pass, so each pass is preceded by
    // a transpose that makes its vectors contiguous.
    let mut a = scratch::complex_overwritten(n);
    let mut b = scratch::complex_overwritten(n);

    // Pass 1: the length-`n1` transforms, over what were columns of the input.
    transpose(buf, &mut a, n1, n2);
    let sign = if inverse { 1.0 } else { -1.0 };
    a.par_chunks_mut(n1 * FFT_ROWS_PER_TASK)
        .enumerate()
        .for_each(|(task, block)| {
            let mut fft_scratch = vec![Complex::new(0.0, 0.0); inner1.get_inplace_scratch_len()];
            for (r, row) in block.chunks_mut(n1).enumerate() {
                inner1.process_with_scratch(row, &mut fft_scratch);
                // Then the twiddle factor W_N^{row·col}, stepped along the row rather than
                // evaluated per element — the same recurrence the NCO uses.
                let row_idx = task * FFT_ROWS_PER_TASK + r;
                let step = sign * 2.0 * PI * row_idx as f64 / n as f64;
                let rot = Complex::new(step.cos(), step.sin());
                let mut z = Complex::new(1.0, 0.0);
                for (col, v) in row.iter_mut().enumerate() {
                    *v *= z;
                    z *= rot;
                    if col % PHASOR_RENORM_INTERVAL == 0 {
                        z /= z.norm();
                    }
                }
            }
        });

    // Pass 2: the length-`n2` transforms, over what are now columns of `a`.
    transpose(&a, &mut b, n2, n1);
    b.par_chunks_mut(n2 * FFT_ROWS_PER_TASK).for_each(|block| {
        let mut fft_scratch = vec![Complex::new(0.0, 0.0); inner2.get_inplace_scratch_len()];
        for row in block.chunks_mut(n2) {
            inner2.process_with_scratch(row, &mut fft_scratch);
        }
    });

    // And back into the natural bin order.
    transpose(&b, buf, n1, n2);

    scratch::recycle_complex(a);
    scratch::recycle_complex(b);
    true
}

/// [`real_fft_forward`] via [`parallel_fft`], or `None` if the length does not suit it.
///
/// A real signal of length `n` is packed into `n/2` complex samples — the even samples on the
/// real axis, the odd ones on the imaginary — transformed at half length, and unpacked. That
/// is what `realfft` does too; doing it here is what lets the transform underneath be the
/// parallel one.
fn parallel_real_fft_forward(samples: &[f64]) -> Option<Vec<Complex<f64>>> {
    let n = samples.len();
    if n % 2 != 0 {
        return None;
    }
    let half = n / 2;
    fft_split(half)?;

    let mut packed = scratch::complex(half);
    packed.extend(samples.chunks_exact(2).map(|p| Complex::new(p[0], p[1])));
    parallel_fft(&mut packed, false);

    let mut bins = scratch::complex_overwritten(half + 1);
    // DC and Nyquist are the two real bins, and both come out of Z[0] alone.
    let z0 = packed[0];
    bins[0] = Complex::new(z0.re + z0.im, 0.0);
    bins[half] = Complex::new(z0.re - z0.im, 0.0);

    // X[k] = E[k] + W_n^k · O[k], where E and O are the transforms of the even and odd samples,
    // recovered from the packed transform's conjugate-symmetric halves.
    let step = -2.0 * PI / n as f64;
    let rot = Complex::new(step.cos(), step.sin());
    for_each_chunk(&mut bins[1..half], |base, out| {
        let k0 = base + 1;
        // W_n^k over the chunk, stepped rather than evaluated per bin.
        let start = step * k0 as f64;
        let mut w = Complex::new(start.cos(), start.sin());
        for (i, slot) in out.iter_mut().enumerate() {
            let k = k0 + i;
            let (a, b) = (packed[k], packed[half - k].conj());
            let even = (a + b) * 0.5;
            let odd = (a - b) * Complex::new(0.0, -0.5);
            *slot = even + w * odd;
            w *= rot;
            if i % PHASOR_RENORM_INTERVAL == 0 {
                w /= w.norm();
            }
        }
    });

    scratch::recycle_complex(packed);
    Some(bins)
}

/// [`real_fft_inverse`] via [`parallel_fft`], or `None` if the length does not suit it.
///
/// The exact inverse of [`parallel_real_fft_forward`]'s packing, and unnormalised to match the
/// single-threaded path it stands in for.
fn parallel_real_fft_inverse(bins: &[Complex<f64>], n: usize) -> Option<Vec<f64>> {
    if n % 2 != 0 || bins.len() != n / 2 + 1 {
        return None;
    }
    let half = n / 2;
    fft_split(half)?;

    let mut packed = scratch::complex_overwritten(half);
    let step = 2.0 * PI / n as f64;
    let rot = Complex::new(step.cos(), step.sin());
    for_each_chunk(&mut packed, |base, out| {
        let start = step * base as f64;
        let mut w = Complex::new(start.cos(), start.sin());
        for (i, slot) in out.iter_mut().enumerate() {
            let k = base + i;
            // X[k] and its mirror give back the even and odd transforms; the mirror of bin 0
            // is the Nyquist bin.
            let (a, b) = (bins[k], bins[half - k].conj());
            let even = (a + b) * 0.5;
            let odd = (a - b) * 0.5 * w;
            *slot = even + Complex::new(0.0, 1.0) * odd;
            w *= rot;
            if i % PHASOR_RENORM_INTERVAL == 0 {
                w /= w.norm();
            }
        }
    });

    parallel_fft(&mut packed, true);

    // The half-length inverse leaves the record scaled by n/2 where the caller expects n, and
    // interleaved as it was packed.
    let mut out = scratch::reals_zeroed(n);
    for_each_chunk(&mut out, |base, dst| {
        for (i, slot) in dst.iter_mut().enumerate() {
            let j = base + i;
            let z = packed[j / 2];
            *slot = 2.0 * if j % 2 == 0 { z.re } else { z.im };
        }
    });

    scratch::recycle_complex(packed);
    Some(out)
}

/// Identifies an analog-bandwidth gain curve: the bin grid, and the response over it.
#[derive(PartialEq, Eq, Clone, Copy)]
struct AfeGainKey {
    n: usize,
    rate_bits: u64,
    bandwidth_bits: u64,
    order: u32,
}

thread_local! {
    static AFE_GAINS: RefCell<Option<(AfeGainKey, std::rc::Rc<Vec<f64>>)>> =
        const { RefCell::new(None) };
}

/// Analog roll-off gain per FFT bin, memoised on the grid it was built for.
///
/// `n` is the transform length; the table covers the `n/2 + 1` non-negative-frequency bins a
/// real-input transform produces.
fn analog_bandwidth_gains(
    n: usize,
    sample_rate_mhz: f64,
    afe: &crate::rfdc::AnalogFrontEnd,
) -> std::rc::Rc<Vec<f64>> {
    let key = AfeGainKey {
        n,
        rate_bits: sample_rate_mhz.to_bits(),
        bandwidth_bits: afe.bandwidth_ghz.to_bits(),
        order: afe.order,
    };

    AFE_GAINS.with(|cache| {
        let mut slot = cache.borrow_mut();
        if let Some((cached_key, values)) = slot.as_ref()
            && *cached_key == key
        {
            return values.clone();
        }

        let values = std::rc::Rc::new(
            (0..=n / 2)
                .map(|i| afe.gain_linear(i as f64 * sample_rate_mhz / n as f64))
                .collect::<Vec<f64>>(),
        );
        *slot = Some((key, values.clone()));
        values
    })
}

/// Apply analog hardware non-idealities (HD2/HD3 distortion) before sampling.
/// Distortion is applied to the real voltage waveform.
pub fn apply_analog_non_idealities(
    samples: &[f64],
    non_idealities: &crate::rfdc::AdcNonIdealities,
) -> Vec<f64> {
    let mut out = scratch::reals(samples.len());
    out.extend_from_slice(samples);
    apply_analog_non_idealities_in_place(&mut out, non_idealities);
    out
}

/// [`apply_analog_non_idealities`] for a caller that already owns the buffer.
///
/// The distortion is pointwise, so there is nothing to gain from a second copy of a record this
/// wide — and the AFE stage before it hands over a buffer it no longer needs.
pub fn apply_analog_non_idealities_in_place(
    samples: &mut [f64],
    non_idealities: &crate::rfdc::AdcNonIdealities,
) {
    if !non_idealities.enabled || samples.is_empty() {
        return;
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
        samples.iter().map(|v| v * v).sum::<f64>() / samples.len() as f64
    } else {
        0.0
    };

    // Both terms are driven by the undistorted sample: they are separate products of the same
    // input, not a chain.
    let distort = |v: &mut f64| {
        let x = *v;
        if a2 > 0.0 {
            *v += a2 * (x * x - mean_sq);
        }
        if a3 > 0.0 {
            *v += a3 * x * x * x;
        }
    };

    if samples.len() >= PAR_MIN_LEN {
        samples.par_iter_mut().with_min_len(PAR_CHUNK).for_each(distort);
    } else {
        samples.iter_mut().for_each(distort);
    }
}

/// Apply digital hardware non-idealities (Quantization, Clipping, Interleaving spurs) after sampling.
/// Also returns a boolean indicating if clipping occurred (overrange).
pub fn apply_digital_non_idealities(
    samples: &[f64],
    non_idealities: &crate::rfdc::AdcNonIdealities,
) -> (Vec<f64>, bool) {
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
    // One run of the chain over `src`, writing `dst`, for samples whose absolute indices start
    // at `base`. Nothing here reads the previous sample, so a chunk only needs to know where it
    // sits: the interleaving spur pattern comes from the absolute index, and the noise stream is
    // seeded from it. That keeps the floor deterministic — the same capture gives the same
    // realisation whether or not it was split — rather than flickering frame to frame.
    let run = |base: usize, dst: &mut [f64]| -> bool {
        let src = &samples[base..base + dst.len()];
        let mut overrange = false;
        let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15 ^ splitmix64(base as u64 + 1);
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

        for (k, (&s, out)) in src.iter().zip(dst.iter_mut()).enumerate() {
            let i = base + k;
            let mut v = s;

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

            *out = v;
        }
        overrange
    };

    let mut processed = scratch::reals_zeroed(samples.len());
    let overrange = any_chunk(&mut processed, run);

    (processed, overrange)
}

/// Mix a counter into a well-distributed seed, so chunk `n` and chunk `n+1` start their noise
/// streams somewhere unrelated rather than a few xorshift steps apart.
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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
    analysis: SpectrumAnalysis,
) -> ProcessedSignal {
    let display_window = analysis.window;
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
    let dsa_samples: Vec<f64> = if dsa_scale < 1.0 {
        let mut v = scratch::reals(input_samples.len());
        v.extend(input_samples.iter().map(|s| s.re * dsa_scale));
        v
    } else {
        physical_voltage(input_samples)
    };

    // 1. Analog input bandwidth roll-off of the track-and-hold, then HD2/HD3 (pre-sampling).
    // The distortion runs in place on the bandwidth stage's output: it is pointwise, and a
    // second copy of a record this wide is pure memory traffic.
    let mut analog_samples =
        apply_analog_bandwidth(&dsa_samples, input_sample_rate_mhz, &block.analog_front_end);
    scratch::recycle_reals(dsa_samples);
    apply_analog_non_idealities_in_place(&mut analog_samples, &block.non_idealities);

    // A high `detail` lengthens the capture for the DDC output's benefit; these panes look at
    // the same samples, so they take whatever length is going. See `analysis_fft_size`.
    let wide_fft = analysis_fft_size(analog_samples.len());
    // Every spectrum below is drawn, not measured, so each is interpolated up to
    // DISPLAY_FFT_SIZE points. See `display_pad_factor` for why the raw bins are not enough.
    let wide_pad = display_pad_factor(wide_fft);

    // 2. Input spectrum (full wideband) — the real voltage at the ADC pin
    let (input_spectrum, input_freq) = compute_spectrum_positive_padded_real(
        &analog_samples,
        wide_fft,
        input_sample_rate_mhz,
        display_window,
        wide_pad,
    );

    let raw_source_spectrum_dbfs = raw_source_samples.map(|samples| {
        // Only the samples the transform will actually read: this pane is the one consumer of
        // the source record, and it looks at `wide_fft` of them however long the capture is.
        let analysed = wide_fft.min(samples.len());
        let real_source = physical_voltage(&samples[..analysed]);
        // Matched to the input pane's length: the two are drawn overlaid, and transforms of
        // different lengths would put their noise floors at different levels.
        let (raw_spec, _) = compute_spectrum_positive_padded_real(
            &real_source,
            wide_fft,
            input_sample_rate_mhz,
            display_window,
            wide_pad,
        );
        scratch::recycle_reals(real_source);
        raw_spec
    });

    // 3. Sample wideband real physical voltage v(t) at the ADC tile sample rate Fs.
    // In hardware, track-and-hold ADC sampling folds ALL wideband Nyquist zones into 0..Fs/2.
    let tile_samples_analog = sample_adc_at_tile_rate(&analog_samples, input_sample_rate_mhz, fs_mhz);

    // 4. Apply digital non-idealities (Clipping, Quantization, Interleaving spurs)
    let (tile_samples, overrange) = apply_digital_non_idealities(&tile_samples_analog, &block.non_idealities);

    // Folded spectrum: actual ADC digital output spectrum (0..Fs/2)
    let tile_fft = analysis_fft_size(tile_samples.len());
    let tile_pad = display_pad_factor(tile_fft);
    let (folded_spectrum, folded_freq) = compute_spectrum_positive_padded_real(
        &tile_samples,
        tile_fft,
        fs_mhz,
        display_window,
        tile_pad,
    );

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
        compute_spectrum_padded(&mixed_samples, tile_fft, fs_mhz, display_window, tile_pad)
    } else {
        compute_spectrum_positive_padded(&mixed_samples, tile_fft, fs_mhz, display_window, tile_pad)
    };

    // The decimation filter's window onto the post-mixer spectrum: everything outside it is
    // what the PL will *not* receive.
    let decim_factor = block.decimation.factor();
    let decimation_response_db =
        decimation_response_on_axis(decim_factor, fs_mhz, &post_mixer_freq);

    // 5. Apply QMC (Quadrature Modulation Correction) post-mixer, pre-decimation
    let qmc_samples = apply_qmc(&mixed_samples, &block.qmc_settings);

    // 6. Apply DDC decimation filter at the ADC tile rate Fs
    let decimated = apply_decimation(&qmc_samples, decim_factor);
    let actual_output_rate = block.output_rate_mhz(tile.sample_rate_gsps);

    // 6. Output spectrum. The requested bin count is independent of `decim_factor` — the span
    // shrank with decimation, so holding the bins fixed is what turns decimation into
    // resolution. `required_tile_samples` is what makes the record long enough to honour it.
    let out_fft = analysis.detail.output_bins();
    // Samples the output transform actually sees, matching what compute_spectrum_* will use.
    // Short of `out_fft` when the wideband budget capped the capture; everything downstream
    // reads the resolution off this rather than off the request.
    let out_analysed = (out_fft.min(decimated.len()) / 2) * 2;
    let out_pad = display_pad_factor(out_analysed);
    let output_rbw_mhz = if out_analysed > 0 {
        actual_output_rate / out_analysed as f64
    } else {
        0.0
    };
    let (output_spectrum, output_freq) = if complex_output {
        compute_spectrum_padded(&decimated, out_fft, actual_output_rate, display_window, out_pad)
    } else {
        compute_spectrum_positive_padded(&decimated, out_fft, actual_output_rate, display_window, out_pad)
    };

    let (rf_chain_response_db, rf_chain_freq_axis_mhz) = match rf_chain_response {
        Some((resp, freq)) => (Some(resp), Some(freq)),
        None => (None, None),
    };

    // Park the working buffers for the next capture. Everything here has been read for the
    // last time; what leaves in `ProcessedSignal` is not recycled.
    scratch::recycle_reals(analog_samples);
    scratch::recycle_reals(tile_samples_analog);
    scratch::recycle_reals(tile_samples);
    scratch::recycle_complex(mixed_samples);
    scratch::recycle_complex(qmc_samples);

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
        display_window,
        output_rbw_mhz,
        output_bins_analysed: out_analysed,
        output_bins_requested: out_fft,
        output_time_samples: decimated,
        overrange,
        sequence: 0,
    }
}

/// Window `n` samples and zero-pad the result out to `n * pad` before transforming.
///
/// Padding buys no resolution — the resolution bandwidth is still set by the `n` samples that
/// actually carry signal — but it samples the window's transfer function densely enough to
/// draw. Without it a bin-centred tone lands on exactly the three non-zero bins of a Hanning
/// main lobe and every trace shows it as a hairline spike with vertical sides, while the same
/// tone half a bin away smears into a broad skirt. See `display_pad_factor`.
fn windowed_padded_fft(
    samples: &[Complex<f64>],
    n: usize,
    window: FftWindow,
    pad: usize,
) -> Vec<Complex<f64>> {
    let mut buffer = scratch::complex(n * pad.max(1));
    buffer.extend_from_slice(&samples[..n]);
    window.apply(&mut buffer);
    // Pad after windowing: windowing the zeros would taper them into the real samples.
    buffer.resize(n * pad.max(1), Complex::new(0.0, 0.0));
    let len = buffer.len();
    FFT_PLANNER.with(|planner| {
        let fft = planner.borrow_mut().plan_fft_forward(len);
        fft.process(&mut buffer);
    });
    buffer
}

/// [`windowed_padded_fft`] for a real signal: same transform, but only the `len/2 + 1` bins a
/// real input can produce, at half the cost of running the same data through a complex FFT.
fn windowed_padded_rfft(
    samples: &[f64],
    n: usize,
    window: FftWindow,
    pad: usize,
) -> Vec<Complex<f64>> {
    let mut buffer = scratch::reals(n * pad.max(1));
    buffer.extend_from_slice(&samples[..n]);
    window.apply_real(&mut buffer);
    buffer.resize(n * pad.max(1), 0.0);
    let len = buffer.len();
    let fft = REAL_FFT_PLANNER.with(|planner| planner.borrow_mut().plan_fft_forward(len));
    let mut spectrum = scratch::complex_zeroed(len / 2 + 1);
    fft.process(&mut buffer, &mut spectrum)
        .expect("real FFT length mismatch");
    scratch::recycle_reals(buffer);
    spectrum
}

/// Magnitudes in dBFS, from bins that have not been normalised yet.
///
/// Works in power rather than amplitude — `10·log10(|c|²)` instead of `20·log10(|c|)` — which
/// drops a square root per bin. A display trace is hundreds of thousands of bins after padding,
/// so this is worth doing and worth spreading across the pool.
fn bins_to_dbfs(bins: &[Complex<f64>], norm: f64) -> Vec<f64> {
    let norm_sq = norm * norm;
    let to_db = |c: &Complex<f64>| 10.0 * (c.norm_sqr() * norm_sq).max(1e-30).log10();
    if bins.len() >= PAR_MIN_LEN {
        bins.par_iter().with_min_len(PAR_CHUNK).map(to_db).collect()
    } else {
        bins.iter().map(to_db).collect()
    }
}

/// Compute the power spectrum of complex samples using FFT with a specific window function.
///
/// `pad` zero-pads the transform to `pad`× the analysed length, interpolating the trace for
/// display without changing the resolution bandwidth. Use 1 for a plain analysis FFT.
pub fn compute_spectrum_padded(
    samples: &[Complex<f64>],
    fft_size: usize,
    sample_rate_mhz: f64,
    window: FftWindow,
    pad: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = (fft_size.min(samples.len()) / 2) * 2;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let buffer = windowed_padded_fft(samples, n, window, pad);
    let len = buffer.len();

    // Compute magnitude in dBFS. Normalisation uses the *analysed* length `n`, not the padded
    // length — the padding contributes no energy, so scaling by it would under-read every bin.
    let norm = 1.0 / (n as f64 * window.coherent_gain());
    let spectrum_dbfs = bins_to_dbfs(&buffer, norm);
    scratch::recycle_complex(buffer);

    // FFT-shift: move DC to centre
    let mut shifted = vec![0.0; len];
    let half = len / 2;
    shifted[..half].copy_from_slice(&spectrum_dbfs[half..]);
    shifted[half..].copy_from_slice(&spectrum_dbfs[..half]);

    // Frequency axis (centred)
    let freq_axis: Vec<f64> = (0..len)
        .map(|i| (i as f64 - half as f64) * sample_rate_mhz / len as f64)
        .collect();

    (shifted, freq_axis)
}

/// Compute the power spectrum of complex samples using FFT with a specific window function.
pub fn compute_spectrum_with_window(
    samples: &[Complex<f64>],
    fft_size: usize,
    sample_rate_mhz: f64,
    window: FftWindow,
) -> (Vec<f64>, Vec<f64>) {
    compute_spectrum_padded(samples, fft_size, sample_rate_mhz, window, 1)
}

/// Compute the power spectrum of complex samples using FFT (default Hanning window).
pub fn compute_spectrum(
    samples: &[Complex<f64>],
    fft_size: usize,
    sample_rate_mhz: f64,
) -> (Vec<f64>, Vec<f64>) {
    compute_spectrum_with_window(samples, fft_size, sample_rate_mhz, FftWindow::Hanning)
}

/// Single-sided (positive frequency only) power spectrum, zero-padded by `pad`.
///
/// See [`compute_spectrum_padded`] for what padding does and does not buy.
pub fn compute_spectrum_positive_padded(
    samples: &[Complex<f64>],
    fft_size: usize,
    sample_rate_mhz: f64,
    window: FftWindow,
    pad: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = (fft_size.min(samples.len()) / 2) * 2;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let buffer = windowed_padded_fft(samples, n, window, pad);
    let len = buffer.len();
    let half = len / 2;
    let norm = 1.0 / (n as f64 * window.coherent_gain());
    let dbfs = positive_bins_to_dbfs(&buffer[..=half], norm);
    scratch::recycle_complex(buffer);

    (dbfs, positive_freq_axis(len, sample_rate_mhz))
}

/// Single-sided power spectrum of a *real* signal, zero-padded by `pad`.
///
/// Identical output to running the same samples through [`compute_spectrum_positive_padded`]
/// with a zero imaginary part, for half the transform cost: a real input has no independent
/// negative-frequency bins to compute in the first place.
pub fn compute_spectrum_positive_padded_real(
    samples: &[f64],
    fft_size: usize,
    sample_rate_mhz: f64,
    window: FftWindow,
    pad: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = (fft_size.min(samples.len()) / 2) * 2;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let bins = windowed_padded_rfft(samples, n, window, pad);
    let len = n * pad.max(1);
    let norm = 1.0 / (n as f64 * window.coherent_gain());
    let dbfs = positive_bins_to_dbfs(&bins, norm);
    scratch::recycle_complex(bins);

    (dbfs, positive_freq_axis(len, sample_rate_mhz))
}

/// dBFS for the non-negative half of a transform, `bins[0]` being DC and `bins[half]` Nyquist.
///
/// Everything between them stands in for its mirror image as well, so it carries twice the
/// amplitude — 6.02 dB, added here rather than doubling each magnitude before the logarithm.
fn positive_bins_to_dbfs(bins: &[Complex<f64>], norm: f64) -> Vec<f64> {
    const FOLD_DB: f64 = 6.020_599_913_279_624; // 20·log10(2)
    let mut dbfs = bins_to_dbfs(bins, norm);
    if dbfs.len() > 2 {
        let last = dbfs.len() - 1;
        for v in &mut dbfs[1..last] {
            *v += FOLD_DB;
        }
    }
    dbfs
}

/// Bin centres of the non-negative half of a `len`-point transform, in MHz.
fn positive_freq_axis(len: usize, sample_rate_mhz: f64) -> Vec<f64> {
    (0..=len / 2)
        .map(|i| i as f64 * sample_rate_mhz / len as f64)
        .collect()
}

/// Compute single-sided (positive frequency only) power spectrum with a specific window function.
pub fn compute_spectrum_positive_with_window(
    samples: &[Complex<f64>],
    fft_size: usize,
    sample_rate_mhz: f64,
    window: FftWindow,
) -> (Vec<f64>, Vec<f64>) {
    compute_spectrum_positive_padded(samples, fft_size, sample_rate_mhz, window, 1)
}

/// Compute single-sided (positive frequency only) power spectrum (default Hanning window).
pub fn compute_spectrum_positive(
    samples: &[Complex<f64>],
    fft_size: usize,
    sample_rate_mhz: f64,
) -> (Vec<f64>, Vec<f64>) {
    compute_spectrum_positive_with_window(samples, fft_size, sample_rate_mhz, FftWindow::Hanning)
}

/// [`compute_spectrum_positive_with_window`] for a real signal.
pub fn compute_spectrum_positive_real_with_window(
    samples: &[f64],
    fft_size: usize,
    sample_rate_mhz: f64,
    window: FftWindow,
) -> (Vec<f64>, Vec<f64>) {
    compute_spectrum_positive_padded_real(samples, fft_size, sample_rate_mhz, window, 1)
}

/// [`compute_spectrum_positive`] for a real signal.
pub fn compute_spectrum_positive_real(
    samples: &[f64],
    fft_size: usize,
    sample_rate_mhz: f64,
) -> (Vec<f64>, Vec<f64>) {
    compute_spectrum_positive_real_with_window(samples, fft_size, sample_rate_mhz, FftWindow::Hanning)
}

/// Half-width of the resampling kernel: 32-tap windowed sinc for >100 dB spur rejection.
const RESAMPLER_RADIUS: isize = 16;

/// Taps the kernel spans, centred on the sample below the requested position.
const RESAMPLER_TAPS: usize = 2 * RESAMPLER_RADIUS as usize + 1;

/// Fractional positions the kernel is tabulated at.
///
/// The kernel depends only on the fractional part of the sample position, so it can be built
/// once instead of per output sample — which is worth doing: evaluated directly it costs four
/// transcendentals per tap, and a capture at high decimation needs millions of taps.
/// Intermediate phases are linearly interpolated between neighbours, which keeps the error far
/// below the kernel's own spur floor rather than at the phase quantisation.
const RESAMPLER_PHASES: usize = 4096;

/// One windowed-sinc kernel per quantised fractional phase.
fn resampler_kernels() -> &'static [[f64; RESAMPLER_TAPS]] {
    static KERNELS: std::sync::OnceLock<Vec<[f64; RESAMPLER_TAPS]>> = std::sync::OnceLock::new();
    KERNELS.get_or_init(|| {
        // One phase past the end so the interpolation always has a right-hand neighbour.
        (0..=RESAMPLER_PHASES)
            .map(|p| {
                let frac = p as f64 / RESAMPLER_PHASES as f64;
                let mut taps = [0.0; RESAMPLER_TAPS];
                for (j, tap) in taps.iter_mut().enumerate() {
                    // Tap j sits at index `centre - RADIUS + j`, so it is this far from the
                    // requested position.
                    let dx = frac + RESAMPLER_RADIUS as f64 - j as f64;
                    let abs_dx = dx.abs();

                    let norm_x = abs_dx / (RESAMPLER_RADIUS as f64 + 1.0);
                    if norm_x >= 1.0 {
                        continue;
                    }

                    let sinc = if abs_dx < 1e-9 {
                        1.0
                    } else {
                        (PI * dx).sin() / (PI * dx)
                    };
                    // Blackman-Harris window
                    let w = 0.35875
                        + 0.48829 * (PI * norm_x).cos()
                        + 0.14128 * (2.0 * PI * norm_x).cos()
                        + 0.01168 * (3.0 * PI * norm_x).cos();
                    *tap = sinc * w;
                }
                taps
            })
            .collect()
    })
}

/// Sample wideband physical real voltage signal v(t) at the ADC tile sample rate Fs.
/// Uses a high-quality windowed sinc anti-aliasing interpolator to eliminate fractional
/// sample rate resampling artifacts (spurs).
pub fn sample_adc_at_tile_rate(
    wideband_samples: &[f64],
    sim_fs_mhz: f64,
    tile_fs_mhz: f64,
) -> Vec<f64> {
    if tile_fs_mhz <= 0.0 || wideband_samples.is_empty() {
        return Vec::new();
    }
    let ratio = sim_fs_mhz / tile_fs_mhz;
    let num_samples = (wideband_samples.len() as f64 / ratio).floor() as usize;
    let kernels = resampler_kernels();
    let gains = resampler_kernel_gains();
    let len = wideband_samples.len() as isize;

    // Output `n` reads a fixed window around `n · ratio` and nothing else, so a chunk needs
    // only its own starting index.
    let run = |base: usize, dst: &mut [f64]| {
        for (k, out) in dst.iter_mut().enumerate() {
            let sample_pos = (base + k) as f64 * ratio;
            let center_idx = sample_pos.floor() as isize;
            let frac = sample_pos - center_idx as f64;

            // Blend the two nearest tabulated phases onto the exact fractional position.
            let phase = frac * RESAMPLER_PHASES as f64;
            let phase_idx = (phase.floor() as usize).min(RESAMPLER_PHASES - 1);
            let t = phase - phase_idx as f64;
            let (lo_k, hi_k) = (&kernels[phase_idx], &kernels[phase_idx + 1]);

            let first = center_idx - RESAMPLER_RADIUS;
            let mut val = 0.0;

            *out = if first >= 0 && first + RESAMPLER_TAPS as isize <= len {
                // Interior: every tap lands on a real sample, so the kernel's gain is the
                // whole kernel's — a property of the phase, not of the data, and tabulated
                // alongside it. Hoisting it out leaves one multiply-add per tap in here.
                let window = &wideband_samples[first as usize..first as usize + RESAMPLER_TAPS];
                for j in 0..RESAMPLER_TAPS {
                    val += window[j] * (lo_k[j] + t * (hi_k[j] - lo_k[j]));
                }
                let gain = gains[phase_idx] + t * (gains[phase_idx + 1] - gains[phase_idx]);
                if gain.abs() > 1e-9 { val / gain } else { 0.0 }
            } else {
                // Near an edge, renormalise over the taps that landed inside the buffer, so
                // the first and last few outputs keep unit gain rather than rolling off.
                let mut weight_sum = 0.0;
                for j in 0..RESAMPLER_TAPS {
                    let idx = first + j as isize;
                    if idx >= 0 && idx < len {
                        let tap = lo_k[j] + t * (hi_k[j] - lo_k[j]);
                        val += wideband_samples[idx as usize] * tap;
                        weight_sum += tap;
                    }
                }
                if weight_sum.abs() > 1e-9 { val / weight_sum } else { 0.0 }
            };
        }
    };

    let mut sampled = scratch::reals_zeroed(num_samples);
    for_each_chunk(&mut sampled, run);

    sampled
}

/// Total gain of each tabulated kernel, i.e. the sum of its taps.
///
/// The interior branch of the resampler divides by this to hold unit gain. Summing 33 taps per
/// output sample is the same arithmetic every time the same phase comes round, so it is done
/// once per phase instead.
fn resampler_kernel_gains() -> &'static [f64] {
    static GAINS: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
    GAINS.get_or_init(|| {
        resampler_kernels()
            .iter()
            .map(|k| k.iter().sum::<f64>())
            .collect()
    })
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
        let mut out = scratch::complex(samples.len());
        out.extend_from_slice(samples);
        return out;
    }

    let phase_rad = qmc.phase * PI / 180.0;
    let cos_p = phase_rad.cos();
    let sin_p = phase_rad.sin();
    let g = qmc.gain;

    let correct = |&s: &Complex<f64>| {
        let i_out = g * (s.re * cos_p - s.im * sin_p) + qmc.offset;
        let q_out = g * (s.re * sin_p + s.im * cos_p);
        Complex::new(i_out, q_out)
    };

    let mut out = scratch::complex(samples.len());
    if samples.len() >= PAR_MIN_LEN {
        out.par_extend(samples.par_iter().with_min_len(PAR_CHUNK).map(correct));
    } else {
        out.extend(samples.iter().map(correct));
    }
    out
}

/// Apply the DDC mixer to the real samples coming off the converter.
///
/// `samples`: real input samples, as the ADC delivers them
/// `settings`: MixerSettings from the block configuration
/// `nco_freq_mhz`: resolved NCO frequency in MHz (after zone wrap/flip)
/// `sim_fs_mhz`: sampling rate of input samples in MHz (wideband simulation rate)
/// `tile_fs_mhz`: ADC tile sampling rate in MHz
/// `scale`: FineMixerScale factor (1.0 or 0.7071)
pub fn apply_mixer(
    samples: &[f64],
    settings: &crate::rfdc::MixerSettings,
    nco_freq_mhz: f64,
    sim_fs_mhz: f64,
    tile_fs_mhz: f64,
    scale: f64,
) -> Vec<Complex<f64>> {
    // NCO phase offset, in degrees, as configured on the block.
    let phase0 = settings.phase_offset * PI / 180.0;

    let bypass = || -> Vec<Complex<f64>> {
        let mut out = scratch::complex(samples.len());
        out.extend(samples.iter().map(|&v| Complex::new(v, 0.0)));
        out
    };

    let omega = match settings.mixer_type {
        MixerType::Off => return bypass(),
        MixerType::Coarse => {
            let coarse_shift_mhz = match settings.coarse_mix_freq {
                CoarseMixFreq::FsOver4 => 0.25 * tile_fs_mhz,
                CoarseMixFreq::MinusFsOver4 => -0.25 * tile_fs_mhz,
                CoarseMixFreq::FsOver2 => 0.5 * tile_fs_mhz,
                CoarseMixFreq::Bypass | CoarseMixFreq::Off => 0.0,
            };
            if coarse_shift_mhz.abs() < 1e-12 {
                return bypass();
            }
            -2.0 * PI * coarse_shift_mhz / sim_fs_mhz
        }
        // Real R2C quadrature mixing: I = x[n]·cos(ωn), Q = -x[n]·sin(ωn) for a real input,
        // which is the real sample times the NCO phasor.
        MixerType::Fine => -2.0 * PI * nco_freq_mhz / sim_fs_mhz,
    };

    mix_with_nco(samples, omega, phase0, scale)
}

/// Multiply real samples by `scale · e^{j(ω·n − φ₀)}`.
///
/// The phasor is stepped rather than evaluated: `e^{jω(n+1)} = e^{jωn} · e^{jω}` costs one
/// complex multiply where `cos`/`sin` per sample cost two transcendentals. Rounding creeps into
/// the magnitude over a long record, so the phasor is renormalised periodically — see
/// [`PHASOR_RENORM_INTERVAL`]. Chunks re-derive their own starting phasor from the absolute
/// index, so splitting the work changes nothing about the result.
fn mix_with_nco(samples: &[f64], omega: f64, phase0: f64, scale: f64) -> Vec<Complex<f64>> {
    let rot = Complex::new(omega.cos(), omega.sin());

    let run = |base: usize, dst: &mut [Complex<f64>]| {
        let src = &samples[base..base + dst.len()];
        let start = omega * base as f64 - phase0;
        let mut z = Complex::new(start.cos(), start.sin()) * scale;
        for (k, (&s, out)) in src.iter().zip(dst.iter_mut()).enumerate() {
            *out = z * s;
            z *= rot;
            if k % PHASOR_RENORM_INTERVAL == 0 {
                z *= scale / z.norm();
            }
        }
    };

    let mut mixed = scratch::complex_zeroed(samples.len());
    for_each_chunk(&mut mixed, run);
    mixed
}

/// How often a stepped phasor is pulled back onto its circle.
///
/// Each step multiplies in one rounding error, so the magnitude drifts as roughly `√n · ε`.
/// Renormalising this often keeps the drift near the `f64` floor for any record length, and
/// costs one square root per thousand samples.
pub(crate) const PHASOR_RENORM_INTERVAL: usize = 1024;

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

/// Identifies a decimation response curve: the cascade, and the axis it was sampled on.
///
/// The axis is uniform, so its ends and length pin it exactly.
#[derive(PartialEq, Eq, Clone, Copy)]
struct DecimResponseKey {
    factor: u32,
    fs_bits: u64,
    len: usize,
    first_bits: u64,
    last_bits: u64,
}

thread_local! {
    static DECIM_RESPONSE: RefCell<Option<(DecimResponseKey, std::rc::Rc<Vec<f64>>)>> =
        const { RefCell::new(None) };
}

/// Composite decimation-filter response over `freq_axis`, memoised on the axis it was built for.
///
/// `response_db` costs a cosine per tap per point, and the post-mixer axis is thousands of
/// points wide — tens of milliseconds per capture for a curve that only moves when the
/// decimation factor, the sample rate, or the axis does. None of those change between the
/// captures of a running simulation, so in practice this is computed once.
fn decimation_response_on_axis(factor: u32, fs_mhz: f64, freq_axis: &[f64]) -> Vec<f64> {
    if factor <= 1 {
        return vec![0.0; freq_axis.len()];
    }

    let key = DecimResponseKey {
        factor,
        fs_bits: fs_mhz.to_bits(),
        len: freq_axis.len(),
        first_bits: freq_axis.first().copied().unwrap_or(0.0).to_bits(),
        last_bits: freq_axis.last().copied().unwrap_or(0.0).to_bits(),
    };

    DECIM_RESPONSE.with(|cache| {
        let mut slot = cache.borrow_mut();
        if let Some((cached_key, values)) = slot.as_ref()
            && *cached_key == key
        {
            return values.as_ref().clone();
        }

        let chain = decimation_chain(factor);
        let values: std::rc::Rc<Vec<f64>> = std::rc::Rc::new(
            freq_axis
                .iter()
                .map(|&f| chain.response_db(f / fs_mhz))
                .collect(),
        );
        *slot = Some((key, values.clone()));
        values.as_ref().clone()
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

        // Output `idx` depends only on the input window around `idx · m`, so the stage splits
        // by output index with no state carried between chunks.
        let run = |base: usize, dst: &mut [Complex<f64>]| {
            for (k, out) in dst.iter_mut().enumerate() {
                let center = (base + k) * m;
                let lo = center.saturating_sub(half);
                let hi = (center + half).min(input.len() - 1);
                // Taps are indexed relative to the centre sample, so the window is clipped at
                // the buffer edges rather than wrapping.
                let tap_offset = half - (center - lo);
                let mut acc = Complex::new(0.0, 0.0);
                for (x, &tap) in input[lo..=hi].iter().zip(&taps[tap_offset..]) {
                    acc += x * tap;
                }
                *out = acc;
            }
        };

        let mut out = scratch::complex_zeroed(out_len);
        // Each output costs one pass over the taps, so it is the product that decides whether
        // the stage is worth spreading — a short buffer through a long filter still is.
        if out_len * taps.len() >= PAR_MIN_LEN {
            out.par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .for_each(|(c, dst)| run(c * PAR_CHUNK, dst));
        } else {
            run(0, &mut out);
        }
        out
    }

    /// Filter and downsample, returning only fully settled output samples.
    pub fn apply(&self, samples: &[Complex<f64>]) -> Vec<Complex<f64>> {
        if self.stages.is_empty() {
            let mut out = scratch::complex(samples.len());
            out.extend_from_slice(samples);
            return out;
        }
        // The first stage reads the caller's buffer directly. Copying it in only to filter out
        // of it is megabytes of traffic for nothing, and every later stage is already writing
        // into a buffer of its own.
        let mut current = Vec::new();
        for (i, stage) in self.stages.iter().enumerate() {
            let input = if i == 0 { samples } else { &current };
            let next = Self::run_stage(input, &stage.taps, stage.factor);
            // The stage just read its input for the last time.
            scratch::recycle_complex(std::mem::replace(&mut current, next));
        }
        if self.warmup_out_samples < current.len() {
            current.drain(..self.warmup_out_samples);
        }
        current
    }
}

/// Shortest analysis FFT the ADC-rate spectra (input, folded, post-mixer) will use.
pub const ANALYSIS_FFT_SIZE: usize = 2048;

/// Longest analysis FFT the ADC-rate spectra will stretch to when the record allows it.
///
/// A high [`SpectrumDetail`] lengthens the capture for the DDC output's sake, and the ADC-rate
/// panes are looking at the same samples. Using them costs one larger transform and no extra
/// generation, so the wideband and folded views sharpen for free. Capped because past this the
/// transform starts to cost more than it returns on a 160-pixel-tall plot.
pub const ANALYSIS_FFT_MAX: usize = 16384;

/// Largest transform the available record supports, within the analysis FFT bounds.
///
/// Powers of two only, to stay on rustfft's radix-2 path.
fn analysis_fft_size(available: usize) -> usize {
    if available < ANALYSIS_FFT_SIZE {
        return ANALYSIS_FFT_SIZE;
    }
    let mut n = ANALYSIS_FFT_SIZE;
    while n * 2 <= available.min(ANALYSIS_FFT_MAX) {
        n *= 2;
    }
    n
}

/// Ceiling on the wideband samples generated per capture, whatever [`SpectrumDetail`] asks for.
///
/// The wideband buffer runs at the simulation rate, so it is `SIM_SAMPLE_RATE / Fs` times longer
/// than the record the DDC actually needs — a 3.75× multiplier at 4 GSPS, on top of the
/// decimation factor. `Max` detail at ×40 would want ~4.9 M samples; this stops there and the
/// output pane reports the coarser resolution it really achieved via `output_bins_analysed`.
pub const MAX_WIDEBAND_SAMPLES: usize = 524_288;

/// How finely the post-DDC output spectrum is resolved.
///
/// This is the real bin count of the output transform, held constant across decimation factors:
/// the output span shrinks with `D` while the bins do not, so the resolution bandwidth
/// `(Fs/D) / bins` *improves* with decimation, which is the point of a DDC. It costs
/// proportionally more wideband input (see [`required_tile_samples`]), which is why it is the
/// user's choice rather than a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SpectrumDetail {
    Fast,
    #[default]
    Balanced,
    Fine,
    Max,
}

impl SpectrumDetail {
    pub const ALL: [SpectrumDetail; 4] = [
        SpectrumDetail::Fast,
        SpectrumDetail::Balanced,
        SpectrumDetail::Fine,
        SpectrumDetail::Max,
    ];

    /// Real (unpadded) bins the output transform is asked for.
    pub fn output_bins(self) -> usize {
        match self {
            SpectrumDetail::Fast => 512,
            SpectrumDetail::Balanced => 2048,
            SpectrumDetail::Fine => 8192,
            SpectrumDetail::Max => 32768,
        }
    }
}

impl std::fmt::Display for SpectrumDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            SpectrumDetail::Fast => "Fast",
            SpectrumDetail::Balanced => "Balanced",
            SpectrumDetail::Fine => "Fine",
            SpectrumDetail::Max => "Max",
        };
        write!(f, "{name} ({} bins)", self.output_bins())
    }
}

/// How every spectrum in one capture is analysed. Both fields are display choices, not hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpectrumAnalysis {
    pub window: FftWindow,
    pub detail: SpectrumDetail,
}

impl Default for SpectrumAnalysis {
    fn default() -> Self {
        Self {
            window: DEFAULT_DISPLAY_WINDOW,
            detail: SpectrumDetail::default(),
        }
    }
}

/// Point count every spectrum is interpolated up to before it reaches a plot.
///
/// A windowed FFT of `N` samples resolves a CW tone into a main lobe only a few bins wide —
/// three, for Hanning on a bin-centred tone. Drawn straight, that is a hairline spike with
/// vertical sides, and it moves under the tone by a full bin at a time. Zero-padding to a
/// fixed point count samples the same main lobe finely enough that the trace shows its real
/// shape and the peak reads its true amplitude regardless of where the tone falls.
pub const DISPLAY_FFT_SIZE: usize = 4096;

/// Window the display starts on. Blackman-Harris rather than the usual Hanning: its sidelobes
/// are 60 dB lower, which keeps a CW tone's leakage skirt off the trace entirely, and it costs
/// only ~36% in -3 dB main-lobe width. Selectable per-view; see `FftWindow::ALL`.
pub const DEFAULT_DISPLAY_WINDOW: FftWindow = FftWindow::BlackmanHarris;

/// Zero-pad factor that lifts an `n`-point transform to at least [`DISPLAY_FFT_SIZE`] points.
///
/// Powers of two only, so the padded length stays on rustfft's radix-2 path.
pub fn display_pad_factor(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut pad = 1;
    while n * pad < DISPLAY_FFT_SIZE {
        pad *= 2;
    }
    pad
}

/// How much dearer a radix-3 or radix-5 stage is than a radix-2 one, per sample.
///
/// Measured on rustfft's f64 path over every 5-smooth length between 100k and 160k: per-sample
/// cost runs from ~7.9 ns for a pure power of two to ~12.4 ns for `2·5⁷`, and tracks the
/// exponents of 3 and 5 closely enough for a ranking heuristic. Normalised against radix-2.
const RADIX3_PENALTY: f64 = 0.032;
const RADIX5_PENALTY: f64 = 0.044;

/// Share of the per-capture cost that scales with the transform's radix mix rather than just
/// with the sample count.
///
/// The wideband buffer feeds two FFTs (forward and inverse, for the analog roll-off), but also
/// signal generation, resampling and the non-idealities, which are all plainly linear in `n`.
/// Roughly 30/70 at the sizes this runs at — so a length has to be *properly* faster to be
/// worth extra samples, not merely a nicer factorisation.
const FFT_COST_SHARE: f64 = 0.3;

/// Candidate lengths considered above `n` before settling for the smallest.
const SIZE_SEARCH_HEADROOM: f64 = 1.3;

/// Relative cost of a capture of length `n`, or `None` if `n` is not 5-smooth.
fn smooth_size_cost(n: usize) -> Option<f64> {
    let mut m = n;
    let mut e3 = 0u32;
    let mut e5 = 0u32;
    while m.is_multiple_of(2) {
        m /= 2;
    }
    while m.is_multiple_of(3) {
        m /= 3;
        e3 += 1;
    }
    while m.is_multiple_of(5) {
        m /= 5;
        e5 += 1;
    }
    if m != 1 {
        return None;
    }
    let radix_penalty = RADIX3_PENALTY * e3 as f64 + RADIX5_PENALTY * e5 as f64;
    Some(n as f64 * (1.0 + FFT_COST_SHARE * radix_penalty))
}

/// Round `n` up to a 5-smooth length (only factors of 2, 3 and 5), preferring cheap ones.
///
/// The wideband buffer feeds an FFT for the analog bandwidth roll-off, and an awkward length
/// like 31·571 pushes rustfft onto its slow general-radix path. But not every smooth length is
/// equally good either: the smallest one above `n` is often heavy in radix 5 — 125000 is
/// `2³·5⁶` — and a slightly larger, more radix-2-friendly length transforms faster despite
/// carrying more samples. This weighs both effects (see [`smooth_size_cost`]) across the
/// candidates within [`SIZE_SEARCH_HEADROOM`] and takes the cheapest, falling back to the
/// smallest smooth length when nothing better turns up.
pub fn next_smooth_size(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }

    let ceiling = ((n as f64) * SIZE_SEARCH_HEADROOM) as usize;
    let mut smallest = usize::MAX;
    let mut best: Option<(f64, usize)> = None;

    let mut p2 = 1usize;
    while p2 <= ceiling {
        let mut p23 = p2;
        while p23 <= ceiling {
            let mut cand = p23;
            while cand < n {
                cand = match cand.checked_mul(5) {
                    Some(v) => v,
                    None => break,
                };
            }
            if cand >= n {
                // Tracked even when it overshoots the band, so there is always a fallback.
                smallest = smallest.min(cand);
            }
            // Walk the radix-5 ladder through the whole search band, not just its first rung.
            while cand >= n && cand <= ceiling {
                if let Some(cost) = smooth_size_cost(cand)
                    && best.is_none_or(|(best_cost, _)| cost < best_cost)
                {
                    best = Some((cost, cand));
                }
                cand = match cand.checked_mul(5) {
                    Some(v) => v,
                    None => break,
                };
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

    match best {
        Some((_, size)) => size,
        // Nothing smooth inside the headroom: fall back to the next smooth length at any size.
        None if smallest != usize::MAX => smallest,
        None => n.next_power_of_two(),
    }
}

/// Number of ADC-rate samples needed to fill both the ADC-rate spectra and the DDC output
/// spectrum at `decimation` and `detail`, including the decimation chain's settling time.
///
/// Callers generating a wideband waveform must scale this by their oversampling ratio
/// (simulation rate / Fs), otherwise the FFTs silently run short and lose resolution.
///
/// The decimator emits one sample per `decimation` inputs, so holding the output bin count fixed
/// makes this term grow linearly with the decimation factor — that is the whole cost of
/// [`SpectrumDetail`], and why [`MAX_WIDEBAND_SAMPLES`] exists to bound it.
pub fn required_tile_samples(decimation: u32, detail: SpectrumDetail) -> usize {
    let f = decimation.max(1) as usize;
    let for_output = detail.output_bins() * f;
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

        let wideband: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 1300.0 * i as f64 / sim_fs).cos())
            .collect();

        let sampled = sample_adc_at_tile_rate(&wideband, sim_fs, tile_fs);
        let (folded, folded_freq) = compute_spectrum_positive_real(&sampled, 2048, tile_fs);

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
        // The mixer sees the converter's real output, so this is a cosine, not a phasor.
        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 250.0 * t).cos() // 250 MHz = Fs/4
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

        // After mixing with −Fs/4, a real tone at Fs/4 puts its positive-frequency half at DC.
        // Its negative half lands on −Fs/2, i.e. the Nyquist bin, and is just as strong — so
        // this checks the level at DC rather than where the strongest bin happens to be.
        let (spectrum, freq_axis) = compute_spectrum(&mixed, n, fs);
        let peak = spectrum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let dc_idx = freq_axis
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap()
            .0;
        assert!(
            spectrum[dc_idx] > peak - 0.5,
            "After Fs/4 mix, DC should carry the tone: DC {:.1} dBFS vs peak {peak:.1} dBFS",
            spectrum[dc_idx]
        );
    }

    #[test]
    fn coarse_mixer_wideband_rate() {
        // Test coarse mixing when simulation rate (10,000 MHz) != ADC tile rate (4,000 MHz)
        let sim_fs = 10000.0;
        let tile_fs = 4000.0; // Fs/4 = 1000 MHz
        let n = 1024;

        // Generate a 1000 MHz tone sampled at 10,000 MHz
        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sim_fs;
                (2.0 * PI * 1000.0 * t).cos() // 1000 MHz tone (= tile_fs / 4)
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
        // Real voltage
        let samples: Vec<f64> = (0..512).map(|i| (2.0 * PI * 0.1 * i as f64).cos()).collect();

        let mut non = crate::rfdc::AdcNonIdealities::default();
        non.enabled = true;
        non.hd2_dbc = -30.0;
        non.hd3_dbc = -40.0;

        let distorted = apply_analog_non_idealities(&samples, &non);
        assert_eq!(distorted.len(), samples.len());
        // Distorted samples should differ from pure sine
        let diff: f64 = samples.iter().zip(distorted.iter()).map(|(a, b)| (a - b).abs()).sum();
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
        let processed = process_adc_block(&input_samples, sim_fs, &block, &tile, Some(&input_samples), None, SpectrumAnalysis::default());

        // Find peak in output spectrum
        let peaks = crate::ui::spectrum_view::find_spectral_peaks(
            &processed.output_spectrum_dbfs,
            &processed.output_freq_axis_mhz,
            -20.0,
            processed.output_rbw_mhz,
            processed.display_window,
        );

        assert!(!peaks.is_empty(), "Should detect peak near 0 Hz baseband");
        // At ×1 there is no decimation filter, so the R2C mixer's negative image survives at
        // -2·f_baseband at the same level as the wanted tone. Which of the two tops the
        // magnitude-sorted list is a coin flip, so ask for the one at DC by frequency.
        let dc_peak = peaks
            .iter()
            .find(|pk| pk.freq_mhz.abs() < 10.0)
            .unwrap_or_else(|| {
                panic!(
                    "Auto-tuned 2400 MHz tone in Zone 3 should land at 0 Hz baseband, strongest peak was {:.1} MHz",
                    peaks[0].freq_mhz
                )
            });
        assert!(
            (dc_peak.mag_dbfs - (-12.0)).abs() < 3.0,
            "Peak magnitude should be close to -12 dBFS due to 6 dB drop from real-to-complex quadrature mixing, got {:.1} dBFS",
            dc_peak.mag_dbfs
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
        let processed = process_adc_block(&input_samples, sim_fs, &tile.blocks[0], &tile, None, None, SpectrumAnalysis::default());

        let peaks = crate::ui::spectrum_view::find_spectral_peaks(
            &processed.output_spectrum_dbfs,
            &processed.output_freq_axis_mhz,
            -20.0,
            processed.output_rbw_mhz,
            processed.display_window,
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
                   bandwidth_mhz: 0.0, modulation: ToneModulation::Cw },
            Tone { frequency_mhz: 2900.0, amplitude_dbfs: -6.0, phase_deg: 0.0,
                   bandwidth_mhz: 0.0, modulation: ToneModulation::Cw },
        ];
        sig_gen.noise_enabled = false;

        let sim_fs = 15000.0;
        let input = sig_gen.generate(16384, sim_fs);
        let processed = process_adc_block(&input, sim_fs, &block, &tile, None, None, SpectrumAnalysis::default());

        let peaks = crate::ui::spectrum_view::find_spectral_peaks(
            &processed.output_spectrum_dbfs,
            &processed.output_freq_axis_mhz,
            -200.0,
            processed.output_rbw_mhz,
            processed.display_window,
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
        for modulation in [ToneModulation::Cw, ToneModulation::Cw] {
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
            let processed = process_adc_block(&input, 15000.0, &block, &tile, None, None, SpectrumAnalysis::default());

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
                modulation: ToneModulation::Cw,
            }];
            sig_gen.noise_enabled = false;
            let input = sig_gen.generate(8192, 15000.0);
            let processed = process_adc_block(&input, 15000.0, &block, &tile, None, None, SpectrumAnalysis::default());
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
        let samples: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).cos())
            .collect();

        let mut non = crate::rfdc::AdcNonIdealities::default();
        non.enabled = true;
        non.hd2_dbc = -40.0;
        non.hd3_dbc = -50.0;

        let distorted = apply_analog_non_idealities(&samples, &non);
        let (spec, freq) = compute_spectrum_positive_real(&distorted, n, fs);

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
            let quiet = vec![0.0; n];
            let (noisy, _) = apply_digital_non_idealities(&quiet, &non);
            let pwr: f64 = noisy.iter().map(|v| v * v).sum::<f64>() / n as f64;
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
        let samples: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).cos())
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
                modulation: ToneModulation::Cw,
            }];
            sig_gen.noise_enabled = false;
            let input = sig_gen.generate(16384, 15000.0);
            let processed = process_adc_block(&input, 15000.0, &block, &tile, None, None, SpectrumAnalysis::default());
            let peaks = crate::ui::spectrum_view::find_spectral_peaks(
                &processed.output_spectrum_dbfs,
                &processed.output_freq_axis_mhz,
                -60.0,
                processed.output_rbw_mhz,
                processed.display_window,
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
        let samples: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).cos())
            .collect();

        let filtered = apply_analog_bandwidth(&samples, fs, &afe);
        assert_eq!(filtered.len(), n);
        assert!(filtered.iter().all(|v| v.is_finite()));

        let (spec, freq) = compute_spectrum_positive_real(&filtered, n, fs);
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
        let samples: Vec<f64> = vec![1.5, -1.5, 0.5];
        let non = crate::rfdc::AdcNonIdealities::default(); // default has enabled=false for spur/quant, but clip still applies
        let (processed, overrange) = apply_digital_non_idealities(&samples, &non);

        assert!(overrange);
        assert_eq!(processed[0], 1.0);
        assert_eq!(processed[1], -1.0);
        assert_eq!(processed[2], 0.5);
    }

    #[test]
    fn r2c_mixer_image_generation() {
        let sim_fs = 1000.0;
        let tile_fs = 1000.0;
        let n = 256;
        let f_in = 100.0;
        
        // Real tone
        let samples: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f_in * i as f64 / sim_fs).cos())
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

        let processed = process_adc_block(&samples, 15000.0, &block, &tile, None, None, SpectrumAnalysis::default());

        // With DSA, the output peak should be ~6 dB lower than without DSA
        let mut block_no_dsa = block.clone();
        block_no_dsa.dsa_db = 0.0;
        let processed_no_dsa = process_adc_block(&samples, 15000.0, &block_no_dsa, &tile, None, None, SpectrumAnalysis::default());

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
        let samples: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).cos())
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

    /// A bin-centred CW tone puts all its energy in the three non-zero bins of a Hanning main
    /// lobe. Drawn straight, that is a hairline spike with vertical sides — what the display
    /// looked like before the trace was interpolated. Padding has to sample the same lobe
    /// densely enough to draw its actual shape.
    #[test]
    fn display_padding_resolves_the_main_lobe_of_a_bin_centred_tone() {
        let n = 256;
        let fs = 500.0;
        // Exactly on bin 0 of the shifted spectrum: the worst case for a raw FFT trace.
        let samples: Vec<Complex<f64>> = (0..n).map(|_| Complex::new(0.5, 0.0)).collect();

        let (raw, _) = compute_spectrum_padded(&samples, n, fs, FftWindow::Hanning, 1);
        let (padded, _) = compute_spectrum_padded(&samples, n, fs, FftWindow::Hanning, 16);

        let peak = |s: &[f64]| s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // Count points within 20 dB of the peak — the visible width of the drawn lobe.
        let lobe = |s: &[f64]| {
            let p = peak(s);
            s.iter().filter(|&&v| v > p - 20.0).count()
        };

        assert_eq!(
            lobe(&raw),
            3,
            "raw trace should show the classic 3-bin Hanning lobe"
        );
        assert!(
            lobe(&padded) >= 24,
            "padded trace should sample the same lobe with many points, got {}",
            lobe(&padded)
        );
        // Padding must not change the level: it adds no energy.
        assert!(
            (peak(&padded) - peak(&raw)).abs() < 0.01,
            "padding changed the peak level: {:.3} vs {:.3} dBFS",
            peak(&padded),
            peak(&raw)
        );
    }

    /// The flip side of the hairline spike: a tone landing between bins reads low and smears.
    /// Padding removes the scalloping error, so the peak marker reports the same level
    /// wherever the tone falls.
    #[test]
    fn display_padding_removes_scalloping_loss() {
        let n = 256;
        let fs = 500.0;
        let bin = fs / n as f64;

        let tone = |f_mhz: f64| -> Vec<Complex<f64>> {
            (0..n)
                .map(|i| {
                    let a = 2.0 * PI * f_mhz * i as f64 / fs;
                    Complex::new(0.5 * a.cos(), 0.5 * a.sin())
                })
                .collect()
        };
        let peak = |s: &[f64]| s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let on_bin = peak(&compute_spectrum_padded(&tone(10.0 * bin), n, fs, FftWindow::Hanning, 16).0);
        let half_bin = peak(&compute_spectrum_padded(&tone(10.5 * bin), n, fs, FftWindow::Hanning, 16).0);
        let raw_half_bin = peak(&compute_spectrum_padded(&tone(10.5 * bin), n, fs, FftWindow::Hanning, 1).0);

        assert!(
            (on_bin - half_bin).abs() < 0.1,
            "padded peak should not depend on bin alignment: {:.2} vs {:.2} dBFS",
            on_bin,
            half_bin
        );
        // Hanning scallops by ~1.4 dB at the half-bin worst case without padding.
        assert!(
            on_bin - raw_half_bin > 1.0,
            "expected the unpadded half-bin tone to read low, got {:.2} dB of loss",
            on_bin - raw_half_bin
        );
    }

    /// Build a capture the way `app.rs` does, and run it through the pipeline.
    fn run_pipeline(
        decim: crate::rfdc::DecimationFactor,
        detail: SpectrumDetail,
        tones: &[(f64, f64)],
    ) -> ProcessedSignal {
        use crate::rfdc::{AdcTile, MixerMode, MixerType};
        use crate::signal::{SignalGenerator, Tone, ToneModulation};

        let sim_fs = 15000.0;
        let mut tile = AdcTile::new(0);
        tile.sample_rate_gsps = 4.0;
        {
            let b = &mut tile.blocks[0];
            b.decimation = decim;
            b.mixer_settings.mixer_type = MixerType::Fine;
            b.mixer_settings.mixer_mode = MixerMode::RealToIq;
            b.mixer_settings.freq = -tones[0].0;
        }
        let block = tile.blocks[0].clone();

        let oversampling = sim_fs / tile.sample_rate_mhz();
        let num = next_smooth_size(
            ((required_tile_samples(decim.factor(), detail) as f64 * oversampling).ceil() as usize)
                .clamp(4096, MAX_WIDEBAND_SAMPLES),
        );

        let sig_gen = SignalGenerator {
            tones: tones
                .iter()
                .map(|&(f, a)| Tone {
                    frequency_mhz: f,
                    amplitude_dbfs: a,
                    phase_deg: 0.0,
                    bandwidth_mhz: 0.0,
                    modulation: ToneModulation::Cw,
                })
                .collect(),
            noise_floor_dbfs: -110.0,
            noise_enabled: false,
        };
        let samples = sig_gen.generate(num, sim_fs);
        process_adc_block(
            &samples,
            sim_fs,
            &block,
            &tile,
            None,
            None,
            SpectrumAnalysis { window: DEFAULT_DISPLAY_WINDOW, detail },
        )
    }

    /// The bug this whole mechanism exists to fix: the old `ANALYSIS_FFT_SIZE / decimation`
    /// policy made the bin count shrink at exactly the rate the span did, pinning the output
    /// RBW at `Fs / 2048` for every factor. Decimating now buys resolution instead.
    #[test]
    fn output_resolution_improves_with_decimation() {
        use crate::rfdc::DecimationFactor;

        let detail = SpectrumDetail::Balanced;
        let mut previous_rbw = f64::INFINITY;

        for decim in DecimationFactor::ALL {
            let out = run_pipeline(decim, detail, &[(300.0, -6.0)]);

            // Balanced never outruns the sample budget, so the request is honoured exactly.
            assert_eq!(
                out.output_bins_analysed,
                detail.output_bins(),
                "×{} fell short of the requested bin count",
                decim.factor()
            );

            let expected = out.output_sample_rate_mhz / detail.output_bins() as f64;
            assert!(
                (out.output_rbw_mhz - expected).abs() < expected * 1e-9,
                "×{}: RBW {} MHz, expected {expected} MHz",
                decim.factor(),
                out.output_rbw_mhz
            );

            // Strictly finer at every step up, which the old policy never was.
            assert!(
                out.output_rbw_mhz < previous_rbw,
                "×{}: RBW {} MHz did not improve on {previous_rbw} MHz",
                decim.factor(),
                out.output_rbw_mhz
            );
            previous_rbw = out.output_rbw_mhz;
        }
    }

    /// The user-visible payoff: two tones far closer than the old 1.95 MHz resolution bandwidth
    /// separate into distinct peaks, and drop back into one when the detail is turned down.
    #[test]
    fn detail_resolves_two_close_tones() {
        use crate::rfdc::DecimationFactor;
        use crate::ui::spectrum_view::find_spectral_peaks;

        // 1.5 MHz apart at ×16, inside the 1.95 MHz the old policy resolved to. That is ~12
        // Balanced bins — clear of the Blackman-Harris main lobe — but only ~3 Fast bins,
        // which is inside it.
        let tones = [(300.0, -6.0), (301.5, -9.0)];

        let fine = run_pipeline(DecimationFactor::X16, SpectrumDetail::Balanced, &tones);
        let fine_peaks = find_spectral_peaks(
            &fine.output_spectrum_dbfs,
            &fine.output_freq_axis_mhz,
            -100.0,
            fine.output_rbw_mhz,
            fine.display_window,
        );
        assert!(
            fine_peaks.len() >= 2,
            "Balanced resolved {} peak(s) at RBW {} MHz; expected both tones",
            fine_peaks.len(),
            fine.output_rbw_mhz
        );
        let separation = (fine_peaks[0].freq_mhz - fine_peaks[1].freq_mhz).abs();
        assert!(
            (separation - 1.5).abs() < fine.output_rbw_mhz * 2.0,
            "peaks {separation} MHz apart, expected ~1.5 MHz"
        );

        // Fast puts both tones inside one main lobe, so only one peak survives.
        let coarse = run_pipeline(DecimationFactor::X16, SpectrumDetail::Fast, &tones);
        let coarse_peaks = find_spectral_peaks(
            &coarse.output_spectrum_dbfs,
            &coarse.output_freq_axis_mhz,
            -100.0,
            coarse.output_rbw_mhz,
            coarse.display_window,
        );
        assert_eq!(
            coarse_peaks.len(),
            1,
            "Fast should merge tones 0.5 MHz apart at RBW {} MHz",
            coarse.output_rbw_mhz
        );
    }

    /// When the wideband budget cannot supply the requested record, the pane has to report the
    /// resolution it actually achieved rather than the one it asked for.
    #[test]
    fn budget_clipping_is_reported_honestly() {
        use crate::rfdc::DecimationFactor;

        let detail = SpectrumDetail::Max;
        let out = run_pipeline(DecimationFactor::X40, detail, &[(300.0, -6.0)]);

        assert!(
            out.output_bins_analysed < out.output_bins_requested,
            "×40 at Max was expected to outrun the {MAX_WIDEBAND_SAMPLES}-sample budget"
        );
        assert_eq!(out.output_bins_requested, detail.output_bins());

        // The reported RBW tracks what was transformed, not what was requested.
        let expected = out.output_sample_rate_mhz / out.output_bins_analysed as f64;
        assert!(
            (out.output_rbw_mhz - expected).abs() < expected * 1e-9,
            "RBW {} MHz does not match {} analysed bins",
            out.output_rbw_mhz,
            out.output_bins_analysed
        );
    }

    /// The tabulated resampler kernel has to agree with evaluating the windowed sinc directly,
    /// which is what it replaced — the tabulation is a speed change, not a modelling one.
    #[test]
    fn tabulated_resampler_matches_direct_evaluation() {
        /// The original per-sample evaluation, kept here as the reference.
        fn direct(w: &[f64], sim: f64, tile: f64) -> Vec<f64> {
            let ratio = sim / tile;
            let num = (w.len() as f64 / ratio).floor() as usize;
            let radius = RESAMPLER_RADIUS;
            let len = w.len() as isize;
            (0..num)
                .map(|n| {
                    let pos = n as f64 * ratio;
                    let centre = pos.floor() as isize;
                    let (mut val, mut weight_sum) = (0.0, 0.0);
                    for k in (centre - radius)..=(centre + radius) {
                        if k < 0 || k >= len {
                            continue;
                        }
                        let dx = pos - k as f64;
                        let abs_dx = dx.abs();
                        let norm_x = abs_dx / (radius as f64 + 1.0);
                        if norm_x >= 1.0 {
                            continue;
                        }
                        let sinc = if abs_dx < 1e-9 {
                            1.0
                        } else {
                            (PI * dx).sin() / (PI * dx)
                        };
                        let window = 0.35875
                            + 0.48829 * (PI * norm_x).cos()
                            + 0.14128 * (2.0 * PI * norm_x).cos()
                            + 0.01168 * (3.0 * PI * norm_x).cos();
                        val += w[k as usize] * sinc * window;
                        weight_sum += sinc * window;
                    }
                    if weight_sum.abs() > 1e-9 { val / weight_sum } else { 0.0 }
                })
                .collect()
        }

        let sim_fs = 15000.0;
        // 4000/2457.6/5000 divide the simulation rate into phases the table holds exactly;
        // 3930 does not, and exercises the interpolation between neighbouring phases.
        for tile_fs in [4000.0, 3930.0, 2457.6, 5000.0] {
            let samples: Vec<f64> = (0..4000)
                .map(|i| {
                    let t = i as f64 / sim_fs;
                    (2.0 * PI * 1234.5 * t).sin()
                        + 0.3 * (2.0 * PI * 6789.0 * t).cos()
                        + 0.05 * (2.0 * PI * 137.0 * t).sin()
                })
                .collect();

            let reference = direct(&samples, sim_fs, tile_fs);
            let tabulated = sample_adc_at_tile_rate(&samples, sim_fs, tile_fs);
            assert_eq!(reference.len(), tabulated.len());

            let worst = reference
                .iter()
                .zip(&tabulated)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let rms = (reference.iter().map(|s| s * s).sum::<f64>()
                / reference.len() as f64)
                .sqrt();

            // Comfortably under the kernel's own ~100 dB spur floor, so the tabulation is
            // never what limits the resampler.
            let error_dbc = 20.0 * (worst / rms).log10();
            assert!(
                error_dbc < -140.0,
                "Fs={tile_fs}: tabulation differs by {error_dbc:.1} dBc"
            );
        }
    }

    #[test]
    fn smooth_size_is_smooth_and_never_shrinks() {
        for n in [1usize, 2, 3, 100, 4095, 4096, 4097, 100_000, 124_000, 125_001] {
            let size = next_smooth_size(n);
            assert!(size >= n, "next_smooth_size({n}) = {size} is smaller than its input");
            assert!(
                smooth_size_cost(size).is_some(),
                "next_smooth_size({n}) = {size} is not 5-smooth"
            );
        }

        // The wideband ceiling is itself a power of two, so sizing up from the clamped target
        // lands exactly on it rather than overshooting the budget.
        assert_eq!(
            next_smooth_size(MAX_WIDEBAND_SAMPLES),
            MAX_WIDEBAND_SAMPLES,
            "sizing overshot the wideband sample budget"
        );
    }

    /// The point of the cost model: the smallest smooth length above the target is often heavy
    /// in radix 5, and a slightly larger power of two beats it outright.
    #[test]
    fn smooth_size_avoids_radix_five_heavy_lengths() {
        // 125000 = 2^3·5^6 is the smallest 5-smooth length above 124 000, and is measurably
        // slower to transform than 124416 = 2^9·3^5 despite being larger.
        let size = next_smooth_size(124_000);
        assert_ne!(size, 125_000, "picked the radix-5-heavy length");
        assert!(smooth_size_cost(size).unwrap() < smooth_size_cost(125_000).unwrap());
    }

    /// The ADC-rate panes ride along on whatever record the DDC output needed.
    #[test]
    fn analysis_fft_grows_with_the_available_record() {
        assert_eq!(analysis_fft_size(0), ANALYSIS_FFT_SIZE);
        assert_eq!(analysis_fft_size(ANALYSIS_FFT_SIZE - 1), ANALYSIS_FFT_SIZE);
        assert_eq!(analysis_fft_size(ANALYSIS_FFT_SIZE), ANALYSIS_FFT_SIZE);
        // Rounds down to a power of two rather than up past what is available.
        assert_eq!(analysis_fft_size(ANALYSIS_FFT_SIZE * 2 - 1), ANALYSIS_FFT_SIZE);
        assert_eq!(analysis_fft_size(ANALYSIS_FFT_SIZE * 2), ANALYSIS_FFT_SIZE * 2);
        assert_eq!(analysis_fft_size(usize::MAX), ANALYSIS_FFT_MAX);
    }

    /// Every spectrum handed to the UI should arrive with enough points to draw, whatever the
    /// decimation does to the native transform length.
    #[test]
    fn every_display_spectrum_reaches_the_display_point_count() {
        use crate::rfdc::{AdcTile, DecimationFactor, MixerMode, MixerType};
        use crate::signal::{SignalGenerator, Tone, ToneModulation};

        for decim in DecimationFactor::ALL {
            let sim_fs = 15000.0;
            let mut tile = AdcTile::new(0);
            tile.sample_rate_gsps = 4.0;
            {
                let b = &mut tile.blocks[0];
                b.decimation = decim;
                b.mixer_settings.mixer_type = MixerType::Fine;
                b.mixer_settings.mixer_mode = MixerMode::RealToIq;
                b.mixer_settings.freq = -300.0;
            }
            let block = tile.blocks[0].clone();

            let oversampling = sim_fs / tile.sample_rate_mhz();
            let needed = required_tile_samples(decim.factor(), SpectrumDetail::default());
            let num = next_smooth_size(
                ((needed as f64 * oversampling).ceil() as usize)
                    .clamp(4096, MAX_WIDEBAND_SAMPLES),
            );

            let sig_gen = SignalGenerator {
                tones: vec![Tone {
                    frequency_mhz: 300.0,
                    amplitude_dbfs: -6.0,
                    phase_deg: 0.0,
                    bandwidth_mhz: 0.0,
                    modulation: ToneModulation::Cw,
                }],
                noise_floor_dbfs: -80.0,
                noise_enabled: true,
            };
            let samples = sig_gen.generate(num, sim_fs);
            let out = process_adc_block(&samples, sim_fs, &block, &tile, None, None, SpectrumAnalysis::default());

            // One-sided spectra land on DISPLAY_FFT_SIZE/2 + 1 points, two-sided on the full count.
            let floor = DISPLAY_FFT_SIZE / 2;
            for (name, len) in [
                ("input", out.input_spectrum_dbfs.len()),
                ("folded", out.folded_spectrum_dbfs.len()),
                ("post-mixer", out.post_mixer_spectrum_dbfs.len()),
                ("output", out.output_spectrum_dbfs.len()),
            ] {
                assert!(
                    len >= floor,
                    "{name} spectrum at ×{} has only {len} points",
                    decim.factor()
                );
            }
        }
    }




    /// The leakage envelope drives peak picking, so it has to bound the window's real
    /// sidelobes rather than just quote a datasheet. Measures each window's own transform.
    #[test]
    fn window_leakage_envelope_matches_the_real_transform() {
        let n = 512;
        for window in FftWindow::ALL {
            // A DC tone: the transform of the window itself.
            let ones: Vec<Complex<f64>> = (0..n).map(|_| Complex::new(1.0, 0.0)).collect();
            let (spec, freq) = compute_spectrum_padded(&ones, n, 1.0, window, 16);
            let peak = spec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let bin = 1.0 / n as f64;

            let mut worst_excess: f64 = f64::NEG_INFINITY;
            let mut worst_at = 0.0;
            for (i, &f) in freq.iter().enumerate() {
                let bins = f.abs() / bin;
                // Beyond the main lobe, and short of the floor where rounding dominates.
                if !(2.0..=64.0).contains(&bins) || spec[i] < peak - 160.0 {
                    continue;
                }
                let predicted = peak - window.leakage_envelope_db(bins);
                if spec[i] - predicted > worst_excess {
                    worst_excess = spec[i] - predicted;
                    worst_at = bins;
                }
            }
            assert!(
                worst_excess < 3.0,
                "{window} sidelobes rise {worst_excess:.1} dB above the modelled envelope at \
                 {worst_at:.1} bins; the model must bound them from below"
            );
        }
    }


    /// The window is a display choice the user makes, so the pipeline has to actually use the
    /// one it is handed — every spectrum, not just some.
    #[test]
    fn pipeline_honours_the_selected_display_window() {
        use crate::rfdc::{AdcTile, DecimationFactor, MixerMode, MixerType};
        use crate::signal::{SignalGenerator, Tone, ToneModulation};

        let sim_fs = 15000.0;
        let mut tile = AdcTile::new(0);
        tile.sample_rate_gsps = 4.0;
        {
            let b = &mut tile.blocks[0];
            b.decimation = DecimationFactor::X8;
            b.mixer_settings.mixer_type = MixerType::Fine;
            b.mixer_settings.mixer_mode = MixerMode::RealToIq;
            b.mixer_settings.freq = -250.0;
        }
        let block = tile.blocks[0].clone();
        let oversampling = sim_fs / tile.sample_rate_mhz();
        let num = next_smooth_size(
            ((required_tile_samples(8, SpectrumDetail::default()) as f64 * oversampling).ceil()
                as usize)
                .clamp(4096, MAX_WIDEBAND_SAMPLES),
        );
        let sig_gen = SignalGenerator {
            tones: vec![Tone {
                frequency_mhz: 300.0,
                amplitude_dbfs: -6.0,
                phase_deg: 0.0,
                bandwidth_mhz: 0.0,
                modulation: ToneModulation::Cw,
            }],
            noise_floor_dbfs: -80.0,
            noise_enabled: false,
        };
        let samples = sig_gen.generate(num, sim_fs);

        // Worst leakage well outside either window's main lobe.
        let skirt_db = |w: FftWindow| {
            let out = process_adc_block(
                &samples,
                sim_fs,
                &block,
                &tile,
                None,
                None,
                SpectrumAnalysis { window: w, ..Default::default() },
            );
            let spec = &out.output_spectrum_dbfs;
            let freq = &out.output_freq_axis_mhz;
            let peak = spec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let pk_f = freq[spec.iter().position(|&v| v == peak).unwrap()];
            let mut worst = f64::NEG_INFINITY;
            for (i, &f) in freq.iter().enumerate() {
                if (f - pk_f).abs() / out.output_rbw_mhz > 12.0 {
                    worst = worst.max(spec[i]);
                }
            }
            (peak, peak - worst)
        };

        let (han_peak, han_skirt) = skirt_db(FftWindow::Hanning);
        let (bh_peak, bh_skirt) = skirt_db(FftWindow::BlackmanHarris);

        // Both must read the tone at the same level — the window changes leakage, not amplitude.
        assert!(
            (han_peak - bh_peak).abs() < 0.5,
            "window changed the reported tone level: {han_peak:.2} vs {bh_peak:.2} dBFS"
        );
        assert!(
            bh_skirt > han_skirt + 20.0,
            "Blackman-Harris should push the leakage skirt far below Hanning's; got {bh_skirt:.1} vs {han_skirt:.1} dB"
        );
    }

    /// The threaded transform is a scheduling change, not a modelling one: the factorisation
    /// evaluates the same butterflies in a different order, so it has to agree with the
    /// single-threaded transform to the last few bits — checked both ways round, and at the
    /// lengths the wideband record actually uses.
    #[test]
    fn parallel_transform_matches_the_single_threaded_one() {
        // A real transform of `n` runs a complex one of `n/2`, so `n` has to be twice
        // the threshold to qualify at all.
        for n in [2 * PARALLEL_FFT_MIN, 1 << 18, 1 << 19] {
            // Something with structure and something without, so a bin-aligned tone cannot
            // hide an indexing mistake that broadband noise would expose.
            let mut seed = 0x1234_5678_9ABC_DEF0u64;
            let x: Vec<f64> = (0..n)
                .map(|i| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let noise = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                    (2.0 * PI * 97.0 * i as f64 / n as f64).cos() + 0.25 * noise
                })
                .collect();

            // Forward: against realfft, which is what the fallback path uses.
            let parallel = parallel_real_fft_forward(&x).expect("length should qualify");
            let fft = REAL_FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(n));
            let mut input = x.clone();
            let mut reference = fft.make_output_vec();
            fft.process(&mut input, &mut reference).unwrap();

            let scale = reference.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
            let worst = parallel
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).norm())
                .fold(0.0_f64, f64::max);
            assert!(
                worst < 1e-9 * scale,
                "n = {n}: forward differs by {worst:.3e} against a peak of {scale:.3e}"
            );

            // Inverse: round-tripping has to give the record back, scaled by n as the
            // single-threaded path scales it.
            let mut spectrum = parallel.clone();
            let back = parallel_real_fft_inverse(&mut spectrum, n).expect("length should qualify");
            let worst = back
                .iter()
                .zip(&x)
                .map(|(a, b)| (a / n as f64 - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(worst < 1e-12, "n = {n}: round trip differs by {worst:.3e}");

            // And against realfft's inverse on the same spectrum, bit for bit in practice.
            let ifft = REAL_FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(n));
            let mut spectrum = parallel.clone();
            let mut ref_back = vec![0.0; n];
            ifft.process(&mut spectrum, &mut ref_back).unwrap();
            let scale = ref_back.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let worst = back
                .iter()
                .zip(&ref_back)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                worst < 1e-9 * scale,
                "n = {n}: inverse differs by {worst:.3e} against a peak of {scale:.3e}"
            );
        }
    }

    /// The real-input transform is a speed change, not a modelling one: it has to agree with
    /// running the same samples through the complex transform with a zero imaginary part.
    #[test]
    fn real_spectrum_matches_the_complex_transform() {
        let n = 4096;
        let fs = 4000.0;
        let real: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 317.0 * t).cos() + 0.01 * (2.0 * PI * 1123.0 * t).sin()
            })
            .collect();
        let complex: Vec<Complex<f64>> = real.iter().map(|&v| Complex::new(v, 0.0)).collect();

        for window in FftWindow::ALL {
            for pad in [1usize, 4] {
                let (a, fa) = compute_spectrum_positive_padded_real(&real, n, fs, window, pad);
                let (b, fb) = compute_spectrum_positive_padded(&complex, n, fs, window, pad);
                assert_eq!(a.len(), b.len(), "{window} pad {pad}: bin count differs");
                assert_eq!(fa, fb, "{window} pad {pad}: frequency axis differs");
                let peak = b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let worst_of = |floor_db: f64| {
                    a.iter()
                        .zip(&b)
                        .filter(|&(_, &y)| y > peak - floor_db)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0_f64, f64::max)
                };
                // Anywhere a signal lives, the two transforms agree to well past display
                // precision. The last 100 dB is each algorithm's own rounding noise, where a
                // vanishing absolute difference still reads as a visible one in dB.
                assert!(
                    worst_of(150.0) < 1e-6,
                    "{window} pad {pad}: differs by {:.3e} dB above −150 dBc",
                    worst_of(150.0)
                );
                assert!(
                    worst_of(400.0) < 1e-3,
                    "{window} pad {pad}: differs by {:.3e} dB at the floor",
                    worst_of(400.0)
                );
            }
        }
    }

    /// Every per-sample stage is chunked across threads above [`PAR_MIN_LEN`]. A chunk derives
    /// its state from its own start index, so crossing that threshold must not change the
    /// result — which this checks by running the same stage either side of it and comparing
    /// the overlap.
    #[test]
    fn chunked_stages_agree_across_the_parallel_threshold() {
        let short = PAR_MIN_LEN / 2;
        let long = PAR_MIN_LEN * 2 + 77; // deliberately not a multiple of the chunk size
        let sim_fs = 15000.0;
        let tile_fs = 4000.0;
        let wave = |i: usize| {
            let t = i as f64 / sim_fs;
            (2.0 * PI * 1234.5 * t).sin() + 0.3 * (2.0 * PI * 6789.0 * t).cos()
        };
        let short_in: Vec<f64> = (0..short).map(wave).collect();
        let long_in: Vec<f64> = (0..long).map(wave).collect();

        let worst = |a: &[f64], b: &[f64]| {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
        };

        // Resampler: the sequential run is a prefix of the parallel one, bar the last few
        // outputs whose kernel runs off the end of the shorter buffer.
        let a = sample_adc_at_tile_rate(&short_in, sim_fs, tile_fs);
        let b = sample_adc_at_tile_rate(&long_in, sim_fs, tile_fs);
        let settled = a.len() - RESAMPLER_TAPS;
        assert!(
            worst(&a[..settled], &b[..settled]) < 1e-12,
            "resampler disagrees either side of the parallel threshold"
        );

        // Mixer: a phasor stepped within a chunk has to track the phase the direct evaluation
        // would have given, at both lengths.
        let ms = crate::rfdc::MixerSettings {
            mixer_type: MixerType::Fine,
            freq: 700.0,
            phase_offset: 33.0,
            ..Default::default()
        };
        let phase0 = 33.0 * PI / 180.0;
        let omega = -2.0 * PI * 700.0 / tile_fs;
        for input in [&short_in, &long_in] {
            let mixed = apply_mixer(input, &ms, 700.0, tile_fs, tile_fs, 1.0);
            let worst_err = mixed
                .iter()
                .enumerate()
                .map(|(i, m)| {
                    let angle = omega * i as f64 - phase0;
                    let want = Complex::new(angle.cos(), angle.sin()) * input[i];
                    (m - want).norm()
                })
                .fold(0.0_f64, f64::max);
            assert!(
                worst_err < 1e-9,
                "stepped NCO drifted from the direct evaluation by {worst_err:.3e}"
            );
        }

        // Digital non-idealities: the noise realisation is seeded from the absolute index, so
        // the shorter run has to reproduce the longer one's opening samples exactly.
        let mut non = crate::rfdc::AdcNonIdealities::default();
        non.enabled = true;
        non.enob = 10.0;
        let (a, _) = apply_digital_non_idealities(&short_in, &non);
        let (b, _) = apply_digital_non_idealities(&long_in, &non);
        assert_eq!(
            worst(&a, &b[..a.len()]),
            0.0,
            "the ADC noise floor is not reproducible across the parallel threshold"
        );
    }
}













