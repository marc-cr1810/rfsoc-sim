# 5. Signal Generation & Analysis

This document details how the simulator generates complex test vectors and how it visually analyzes the output, bridging the gap between theoretical math and professional RF instrumentation.

## Signal Generation (`src/signal.rs`)

To rigorously test the RFSoC datapath, the simulator includes a wideband generator producing any number of parallel tones. **Every waveform is real**, because the analog domain carries one real voltage and the converter samples only that — a "complex tone" option would be indistinguishable from a cosine once it reached the ADC pin, which is why there is one carrier variant rather than three.

| Waveform | What it produces |
| --- | --- |
| **CW carrier** | A single line. Give it a non-zero **channel bandwidth** and it becomes a modulated channel instead: the in-band FFT bins are filled with random phase and transformed back, which is what a carrier with data on it looks like — flat, exactly the requested width, and carrying the same power as the tone it replaces. |
| **Square / Sawtooth / Triangle** | Built by summing only the harmonics that fit below Nyquist (odd·1/n, all·1/n, odd·1/n² respectively). Evaluating the ideal shape at the sample rate instead folds every harmonic above Nyquist back into the band, and in a simulator built to teach Nyquist behaviour those aliases are indistinguishable from real signals. The harmonic ceiling doubles as the finite rise time every real generator has. |
| **AM** | Carrier-referenced, as instruments are: sidebands at $m/2$, and an envelope peaking at $(1+m)$ times the carrier — so deep modulation of a hot carrier really does overdrive what follows. |
| **FM** | Sidebands land on the Bessel functions $J_n(\beta)$, including the carrier null at $\beta = 2.405$. |
| **FMCW chirp** | Linear sweep across the bandwidth, with sawtooth or **triangular** retrace. Triangular carries the phase through the turn, which is what lets a real FMCW radar separate range from Doppler. |
| **Pulsed radar** | Coherent pulse train with a **raised-cosine edge** (a perfectly rectangular pulse has skirts reaching to infinity) and an optional **intra-pulse LFM chirp** for pulse compression, the waveform most real radars actually transmit. |
| **Frequency hopping** | Hops a channel grid on a hashed sequence. The previous stride-based sequence shared a factor with any channel count that was a multiple of the stride — seven channels sat on channel 3 and never moved. |
| **QPSK** | Pseudo-random symbols with **root-raised-cosine** shaping, occupying $(1+\alpha)R_s$. The previous symbol sequence repeated every four symbols, making a handful of discrete lines rather than a modulated channel. |
| **AWGN floor** | Injected into the real voltage only — the one quantity a wire carries — so the measured floor lands on its setting rather than 3 dB below it. This is a *total power* figure, so its on-screen level moves with FFT length; the chain's own **physical** noise is the $kTB$ model in `src/node_graph/components.rs`, which is properly a power spectral density. |

### Time-Domain Continuity

The generator is fully time-aware: every waveform is a closed form in the absolute simulation timestamp rather than a running accumulator, so a frame starting at $t$ contains exactly the samples a longer run through $t$ would have produced. There is a test asserting this for every modulation, because anything that resets a phase accumulator shows up as a seam in the waterfall.

Two exceptions are deliberate, and both are what the real thing does: a sawtooth chirp jumps in phase at retrace, and frequency hops are not phase-coherent with each other.

### Verifying the Waveforms

Each modulation is checked against the textbook result rather than merely for "some energy": harmonic amplitude ratios and the absence of aliases, AM sideband depth, FM Bessel sidebands and the carrier null, chirp occupancy and continuity through the turn, pulse duty cycle and PRF line spacing, the energy a raised-cosine edge removes, hop-sequence coverage of the channel grid, and QPSK occupied bandwidth against $(1+\alpha)R_s$.

## Spectral Analysis (FFT & Windowing)

The simulator includes a professional-grade spectral analyzer UI (`src/ui/spectrum_view.rs`) built on top of high-performance Rust FFTs (`rustfft`).

### Windowing & Leakage
Because the discrete Fourier transform assumes the input signal is infinitely periodic, any non-periodic signal will suffer from "spectral leakage" (where energy smears across adjacent bins). To combat this, the simulator applies a **Blackman-Harris window** before the FFT.

### dBFS Calibration (Coherent Gain)
The Y-axis of the spectrum analyzer is strictly calibrated to **dBFS (Decibels relative to Full Scale)**.
- A physical sine wave touching the absolute maximum limit of the ADC ($+1.0$ to $-1.0$) corresponds to $0 \text{ dBFS}$.
- Because the Blackman-Harris window suppresses the edges of the time-domain signal, it mathematically reduces the total energy of the signal. The simulator calculates the **Coherent Gain** of the window (the sum of all window coefficients divided by the window length) and scales the FFT output to perfectly restore the amplitude, ensuring that a $1.0$ amplitude input always reads exactly as $0 \text{ dBFS}$ on the plot.

## Real vs. Complex Spectrums

The simulator intelligently adapts its visualization based on the mathematical nature of the data it is displaying:

1. **One-Sided Spectrums (Pre-ADC):** Real signals have symmetrical positive and negative frequencies. The UI hides the negative frequencies and scales the positive frequencies by a factor of $2\times$ (or $+6 \text{ dB}$) to correctly represent the combined energy of the folded negative image.
2. **Two-Sided Spectrums (Post-DDC):** Complex signals (I/Q data) are asymmetrical. The UI centers the plot at $0 \text{ Hz}$ (DC) and spans from $-F_s/2$ to $+F_s/2$, revealing both the positive and negative spectral images generated by the DDC mixer.

## Dynamic Range and Waterfall Spectrograms

The simulator dynamically calculates the number of samples required to maintain high-resolution FFT bins. If a heavy decimation factor (e.g., $\times 40$) reduces the output sample rate drastically, the system dynamically generates more input samples to prevent the FFT from starving for data, maintaining a crisp visual resolution.

The UI also includes a **Waterfall Spectrogram**, which stacks FFT outputs vertically over time, allowing users to visualize frequency sweeps, frequency-hopping algorithms, and transient pulsed radar signals visually as they evolve.
