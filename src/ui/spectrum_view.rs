//! Spectrum visualization plots using egui_plot and waterfall spectrogram.

use crate::dsp::{FftWindow, ProcessedSignal, SpectrumDetail};
use crate::ui::theme::Theme;
use egui_plot::{Line, Plot, PlotBounds, PlotPoints};
use std::cell::RefCell;
use std::collections::VecDeque;

/// Left margin every stacked plot reserves for its y axis.
///
/// `link_axis` synchronises the data range, not the geometry: two plots showing the same range
/// on plot areas of different width put the same frequency at different screen x, so the
/// waterfall drifts out of register with the spectrum above it. Pinning the axis width on both
/// keeps their plot areas identical. Wide enough for "-150" plus padding.
const PLOT_Y_AXIS_WIDTH: f32 = 56.0;

/// Widest waterfall texture built. The texture covers only the visible frequency range, so
/// columns track screen pixels; this is the ceiling for a very wide window on a HiDPI display.
const WATERFALL_MAX_COLUMNS: usize = 4096;

/// Narrowest waterfall texture, so a hard zoom onto a handful of bins still uploads something
/// the interpolator can work with.
const WATERFALL_MIN_COLUMNS: usize = 64;

struct WaterfallState {
    paused: bool,
    min_db: f64,
    max_db: f64,
    /// Rows of history kept. User-adjustable: deep history shows slow drift, shallow history
    /// gives each row more vertical pixels.
    depth: usize,
    /// `ProcessedSignal::sequence` of the newest row, so a repainted frame does not push a
    /// duplicate. The pipeline runs slower than the frame rate at high detail.
    last_sequence: u64,
    /// Cached texture and the inputs it was built from. Rebuilding is proportional to
    /// columns × rows, which is why it only happens when one of those inputs moves.
    texture: Option<egui::TextureHandle>,
    texture_key: Option<WaterfallKey>,
}

/// Everything the waterfall texture's pixels depend on.
#[derive(Debug, Clone, Copy, PartialEq)]
struct WaterfallKey {
    sequence: u64,
    rows: usize,
    columns: usize,
    /// Visible frequency range, in bits so the key can derive `PartialEq` and `Copy`.
    x_min_bits: u64,
    x_max_bits: u64,
    min_db_bits: u64,
    max_db_bits: u64,
}

impl Default for WaterfallState {
    fn default() -> Self {
        Self {
            paused: false,
            min_db: -140.0,
            max_db: 0.0,
            depth: 256,
            last_sequence: 0,
            texture: None,
            texture_key: None,
        }
    }
}

thread_local! {
    static WATERFALL_BUFFER: RefCell<VecDeque<Vec<f64>>> = RefCell::new(VecDeque::new());
    static WATERFALL_STATE: RefCell<WaterfallState> = RefCell::new(WaterfallState::default());
}

/// Magma-style ramp for a 0..1 intensity: black → blue → purple → red → yellow.
fn magma(norm: f64) -> egui::Color32 {
    let r = (norm * 3.0 - 1.0).clamp(0.0, 1.0);
    let g = (norm * 3.0 - 2.0).clamp(0.0, 1.0);
    let b = (norm * 3.0).clamp(0.0, 1.0) - (norm * 3.0 - 1.0).clamp(0.0, 1.0);
    egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Full frequency range of an output spectrum: `±Fout/2` for complex I/Q, `0..Fout/2` for real.
///
/// A real-output block's spectrum comes back one-sided, so assuming a two-sided axis would
/// stretch it across twice its true span and mislabel every frequency on it.
fn output_span_mhz(output_rate_mhz: f64, complex_output: bool) -> (f64, f64) {
    if complex_output {
        (-output_rate_mhz / 2.0, output_rate_mhz / 2.0)
    } else {
        (0.0, output_rate_mhz / 2.0)
    }
}

/// Half-open bin range of `len` covering `[view_min, view_max]` within the axis `[full_min, full_max]`.
///
/// The frequency axis is uniform, so this is a straight linear map. Clamped to the axis and
/// widened to at least one bin, so a view entirely outside the span still yields something
/// drawable rather than an empty slice.
fn visible_bin_range(
    len: usize,
    full_min: f64,
    full_max: f64,
    view_min: f64,
    view_max: f64,
) -> (usize, usize) {
    let span = full_max - full_min;
    if len == 0 || span <= 0.0 {
        return (0, len);
    }
    let to_bin = |x: f64| (((x - full_min) / span) * len as f64).floor();
    let lo = (to_bin(view_min).max(0.0) as usize).min(len.saturating_sub(1));
    // Ceil the high edge so a partially covered bin is still included.
    let hi = ((to_bin(view_max).max(0.0) as usize) + 1).clamp(lo + 1, len);
    (lo, hi)
}

/// A detected spectral peak.
#[derive(Debug, Clone, Copy)]
pub struct SpectralPeak {
    pub freq_mhz: f64,
    pub mag_dbfs: f64,
}

/// Margin a candidate must clear above a stronger peak's leakage to count as its own signal.
/// The measured envelope tracks theory to a few hundredths of a dB, so this only has to cover
/// noise jitter on the trace.
const LEAKAGE_MARGIN_DB: f64 = 3.0;

/// Find local spectral peaks above threshold_dbfs, strongest first.
///
/// Display traces are zero-padded, which resolves the analysis window's sidelobes into genuine
/// local maxima — a single tone produces dozens. `rbw_mhz` (the pane's real resolution
/// bandwidth, not its point spacing) lets this drop any candidate buried under a stronger
/// peak's leakage, so what comes back is signals rather than one signal's skirt.
///
/// A real tone sitting below the leakage envelope of a stronger neighbour is dropped too. That
/// is honest: at that separation and level it is not distinguishable from leakage.
///
/// Pass 0 for `rbw_mhz` to get the raw local maxima.
pub fn find_spectral_peaks(
    spectrum: &[f64],
    freq_axis: &[f64],
    threshold_dbfs: f64,
    rbw_mhz: f64,
    window: FftWindow,
) -> Vec<SpectralPeak> {
    let mut peaks = Vec::new();
    if spectrum.len() < 3 || spectrum.len() != freq_axis.len() {
        return peaks;
    }
    for i in 1..spectrum.len() - 1 {
        if spectrum[i] > threshold_dbfs
            && spectrum[i] > spectrum[i - 1]
            && spectrum[i] > spectrum[i + 1]
        {
            peaks.push(SpectralPeak {
                freq_mhz: freq_axis[i],
                mag_dbfs: spectrum[i],
            });
        }
    }
    peaks.sort_by(|a, b| b.mag_dbfs.partial_cmp(&a.mag_dbfs).unwrap_or(std::cmp::Ordering::Equal));

    if rbw_mhz <= 0.0 {
        return peaks;
    }
    // Strongest first, so every candidate is tested against the peaks that could bury it.
    let mut kept: Vec<SpectralPeak> = Vec::new();
    for pk in peaks {
        // Leakage from several tones lands on the same bin and adds, so sum it in power
        // rather than testing each stronger peak on its own — one skirt may not reach the
        // candidate while two together do.
        let leakage_power: f64 = kept
            .iter()
            .map(|k| {
                let bins = (k.freq_mhz - pk.freq_mhz).abs() / rbw_mhz;
                let db = k.mag_dbfs - window.leakage_envelope_db(bins);
                10.0_f64.powf(db / 10.0)
            })
            .sum();
        if leakage_power <= 0.0 {
            kept.push(pk);
            continue;
        }
        let leakage_dbfs = 10.0 * leakage_power.log10();
        if pk.mag_dbfs >= leakage_dbfs + LEAKAGE_MARGIN_DB {
            kept.push(pk);
        }
    }
    kept
}

/// Render the multi-pane spectrum display with waterfall and peak markers.
/// Renders the spectrum panes. Returns true if the user changed the analysis settings, so the
/// caller can recompute against them immediately instead of waiting out its pacing interval.
pub fn show_spectrum_view(
    ui: &mut egui::Ui,
    processed: &Option<ProcessedSignal>,
    tile_fs_mhz: f64,
    display_window: &mut FftWindow,
    detail: &mut SpectrumDetail,
) -> bool {
    let mut window_changed = false;
    if let Some(signal) = processed {
        let available_height = ui.available_height();
        let plot_height = (available_height / 3.2).max(140.0);

        ui.horizontal(|ui| {
            ui.heading("📊 Spectrum Analysis");
            if signal.rf_chain_response_db.is_some() {
                ui.separator();
                ui.colored_label(Theme::ACCENT_WARN, "⚡ Overlay: RF Chain H(f)");
            }
            if signal.overrange {
                ui.separator();
                ui.colored_label(Theme::ACCENT_ERROR, "⚠ OVR");
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let before = *display_window;
                egui::ComboBox::from_id_salt("fft_window")
                    .selected_text(display_window.to_string())
                    .width(140.0)
                    .show_ui(ui, |ui| {
                        for w in FftWindow::ALL {
                            ui.selectable_value(display_window, w, w.to_string())
                                .on_hover_text(window_hint(w));
                        }
                    })
                    .response
                    .on_hover_text(
                        "Trades main-lobe width against how far a tone's leakage skirt spreads. \
                         Low sidelobes reveal weak signals beside a strong one; a narrow lobe \
                         separates two close ones.",
                    );
                ui.label(egui::RichText::new("FFT window:").color(Theme::TEXT_SECONDARY));
                window_changed = *display_window != before;
            });
        });

        // Resolution sits on its own row beside the bandwidth it produces: the two only mean
        // anything together, and the heading row is already crowded at narrow widths.
        ui.horizontal(|ui| {
            ui.colored_label(
                Theme::TEXT_SECONDARY,
                format!(
                    "Post-DDC RBW: {} over {} bins",
                    format_rbw(signal.output_rbw_mhz),
                    signal.output_bins_analysed
                ),
            );
            // The record is capped, so a high detail setting at a high decimation factor can ask
            // for more bins than the capture can fill. Say so rather than quietly under-deliver.
            if signal.output_bins_analysed < signal.output_bins_requested {
                ui.colored_label(
                    Theme::ACCENT_WARN,
                    format!(
                        "⚠ sample-budget limited (requested {})",
                        signal.output_bins_requested
                    ),
                )
                .on_hover_text(
                    "This decimation factor would need a longer capture than the wideband \
                     sample budget allows. Lower the resolution or the decimation factor to \
                     get the requested bin count.",
                );
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let before_detail = *detail;
                egui::ComboBox::from_id_salt("spectrum_detail")
                    .selected_text(detail.to_string())
                    .width(150.0)
                    .show_ui(ui, |ui| {
                        for d in SpectrumDetail::ALL {
                            ui.selectable_value(detail, d, d.to_string())
                                .on_hover_text(detail_hint(d, signal.output_sample_rate_mhz));
                        }
                    })
                    .response
                    .on_hover_text(
                        "Bins the post-DDC transform resolves, held constant across decimation \
                         factors — so decimating narrows the resolution bandwidth instead of \
                         leaving it fixed. Each step up needs proportionally more wideband \
                         signal generated per capture, and the spectrum update rate drops to \
                         pay for it.",
                    );
                ui.label(egui::RichText::new("Resolution:").color(Theme::TEXT_SECONDARY));
                window_changed |= *detail != before_detail;
            });
        });
        ui.separator();

        let rf_chain_overlay = match (&signal.rf_chain_response_db, &signal.rf_chain_freq_axis_mhz) {
            (Some(resp), Some(freq)) => Some(ResponseOverlay {
                values_db: resp,
                freq_axis_mhz: freq,
                name: "RF chain H(f)",
                color: Theme::ACCENT_WARN,
                dashed: false,
                floor_db: -150.0,
            }),
            _ => None,
        };

        let raw_source_overlay = signal
            .raw_source_spectrum_dbfs
            .as_ref()
            .map(|raw| (raw.as_slice(), signal.input_freq_axis_mhz.as_slice()));

        let plot_height = plot_height * 0.8; // four stacked panes now

        // 1. Input Spectrum with RF Chain overlay
        show_single_spectrum(
            ui,
            SpectrumPlot {
                id: "input_spectrum",
                title: "Input Spectrum at ADC Pin (Pre-Sampling, Real Voltage)".to_string(),
                spectrum: &signal.input_spectrum_dbfs,
                freq_axis: &signal.input_freq_axis_mhz,
                height: plot_height,
                color: Theme::ACCENT_PRIMARY,
                span_rate_mhz: tile_fs_mhz,
                legend: {
                    let mut items = vec![("at ADC pin", Theme::ACCENT_PRIMARY)];
                    if raw_source_overlay.is_some() {
                        items.push(("raw source", Theme::TEXT_SECONDARY));
                    }
                    if rf_chain_overlay.is_some() {
                        items.push(("RF chain H(f)", Theme::ACCENT_WARN));
                    }
                    items
                },
                response_overlay: rf_chain_overlay,
                raw_source_overlay,
                show_nyquist_zones: true,
                ..Default::default()
            },
        );

        ui.add_space(4.0);

        // 2. Folded Spectrum (what the converter actually digitises)
        show_single_spectrum(
            ui,
            SpectrumPlot {
                id: "folded_spectrum",
                title: "Folded Spectrum (ADC Digital Output, 0–Fs/2)".to_string(),
                spectrum: &signal.folded_spectrum_dbfs,
                freq_axis: &signal.folded_freq_axis_mhz,
                height: plot_height,
                color: Theme::ZONE_1,
                span_rate_mhz: tile_fs_mhz,
                show_nyquist_zones: true,
                ..Default::default()
            },
        );

        ui.add_space(4.0);

        // 3. Post-mixer spectrum at the full ADC rate, with the decimation filter drawn on
        //    top. This is the stage that explains the baseband plot below: the mixer's
        //    real-to-I/Q image and any out-of-band signal are visible here, and the shaded
        //    band is the only part the decimation chain passes to the PL.
        let keep_band = if signal.output_sample_rate_mhz > 0.0 {
            Some((
                crate::dsp::DDC_PASSBAND_FRAC * signal.output_sample_rate_mhz,
                signal.output_sample_rate_mhz / 2.0,
            ))
        } else {
            None
        };

        show_single_spectrum(
            ui,
            SpectrumPlot {
                id: "post_mixer_spectrum",
                title: format!(
                    "Post-Mixer DDC Spectrum (at Fs = {:.1} MHz, NCO = {:+.3} MHz)",
                    tile_fs_mhz, signal.resolved_nco_freq_mhz
                ),
                legend: vec![
                    ("mixer output", Theme::ZONE_3),
                    ("decimation filter H(f)", Theme::ACCENT_WARN),
                    ("kept for PL", Theme::ACCENT_SECONDARY),
                ],
                spectrum: &signal.post_mixer_spectrum_dbfs,
                freq_axis: &signal.post_mixer_freq_axis_mhz,
                height: plot_height,
                color: Theme::ZONE_3,
                span_rate_mhz: tile_fs_mhz,
                two_sided: signal.complex_output,
                response_overlay: Some(ResponseOverlay {
                    values_db: &signal.decimation_response_db,
                    freq_axis_mhz: &signal.post_mixer_freq_axis_mhz,
                    name: "Decimation filter H(f)",
                    color: Theme::ACCENT_WARN,
                    dashed: true,
                    // Well below any realistic noise floor, so nothing meaningful is hidden.
                    floor_db: -120.0,
                }),
                ddc_keep_band_mhz: keep_band,
                show_dc_marker: signal.complex_output,
                ..Default::default()
            },
        );

        ui.add_space(4.0);

        let link_group = ui.id().with("baseband_x_link");

        // 4. Post-DDC Spectrum — what the PL receives
        let output_bounds = show_single_spectrum(
            ui,
            SpectrumPlot {
                id: "output_spectrum",
                title: format!(
                    "Post-DDC Baseband Spectrum to PL (Output Rate: {:.1} MHz, Span: {}{:.1} MHz)",
                    signal.output_sample_rate_mhz,
                    if signal.complex_output { "±" } else { "0–" },
                    signal.output_sample_rate_mhz / 2.0
                ),
                spectrum: &signal.output_spectrum_dbfs,
                freq_axis: &signal.output_freq_axis_mhz,
                height: plot_height,
                color: Theme::ACCENT_SECONDARY,
                span_rate_mhz: signal.output_sample_rate_mhz,
                two_sided: signal.complex_output,
                show_dc_marker: signal.complex_output,
                show_peak_markers: true,
                window: signal.display_window,
                resolution_bw_mhz: signal.output_rbw_mhz,
                usable_band_mhz: Some(
                    crate::dsp::DDC_PASSBAND_FRAC * signal.output_sample_rate_mhz,
                ),
                link_group: Some(link_group),
                ..Default::default()
            },
        );

        ui.add_space(4.0);

        // 4. Real-Time Oscilloscope & Constellation Viewers
        ui.collapsing("📈 Real-Time Baseband Oscilloscope & IQ Constellation", |ui| {
            ui.columns(2, |cols| {
                // Column 1: Time-domain Oscilloscope
                cols[0].group(|ui| {
                    ui.label(
                        egui::RichText::new("📉 Time-Domain Oscilloscope")
                            .strong()
                            .color(Theme::ACCENT_PRIMARY),
                    );

                    let num_pts = signal.output_time_samples.len().min(300);
                    let dt = 1.0 / signal.output_sample_rate_mhz; // µs

                    let i_points: PlotPoints = signal.output_time_samples[..num_pts]
                        .iter()
                        .enumerate()
                        .map(|(idx, c)| [idx as f64 * dt, c.re])
                        .collect();

                    let q_points: PlotPoints = signal.output_time_samples[..num_pts]
                        .iter()
                        .enumerate()
                        .map(|(idx, c)| [idx as f64 * dt, c.im])
                        .collect();

                    let env_points: PlotPoints = signal.output_time_samples[..num_pts]
                        .iter()
                        .enumerate()
                        .map(|(idx, c)| [idx as f64 * dt, c.norm()])
                        .collect();

                    let line_i = Line::new("I(t)", i_points)
                        .color(egui::Color32::from_rgb(0, 200, 255))
                        .width(1.5);
                    let line_q = Line::new("Q(t)", q_points)
                        .color(egui::Color32::from_rgb(255, 180, 0))
                        .width(1.5);
                    let line_env = Line::new("|I+jQ|", env_points)
                        .color(egui::Color32::from_rgb(200, 100, 255))
                        .width(1.0)
                        .style(egui_plot::LineStyle::dashed_dense());

                    Plot::new("oscilloscope_plot")
                        .height(150.0)
                        .x_axis_label("Time (µs)")
                        .y_axis_label("Amplitude")
                        .include_y(-1.2)
                        .include_y(1.2)
                        .show(ui, |plot_ui| {
                            plot_ui.line(line_i);
                            plot_ui.line(line_q);
                            plot_ui.line(line_env);
                        });
                });

                // Column 2: IQ Constellation Scatter Plot
                cols[1].group(|ui| {
                    ui.label(
                        egui::RichText::new("🎯 IQ Constellation Diagram")
                            .strong()
                            .color(Theme::ACCENT_SECONDARY),
                    );

                    let num_pts = signal.output_time_samples.len().min(400);
                    let points: PlotPoints = signal.output_time_samples[..num_pts]
                        .iter()
                        .map(|c| [c.re, c.im])
                        .collect();

                    let traj_line = Line::new("IQ_trajectory", points)
                        .color(Theme::ACCENT_SECONDARY.linear_multiply(0.6))
                        .width(1.0);

                    let points_scatter = egui_plot::Points::new("IQ_symbols", signal.output_time_samples[..num_pts]
                        .iter()
                        .map(|c| [c.re, c.im])
                        .collect::<PlotPoints>())
                        .color(Theme::ACCENT_PRIMARY)
                        .radius(2.0);

                    Plot::new("iq_constellation_plot")
                        .height(150.0)
                        .x_axis_label("In-Phase I")
                        .y_axis_label("Quadrature Q")
                        .include_x(-1.2)
                        .include_x(1.2)
                        .include_y(-1.2)
                        .include_y(1.2)
                        .show(ui, |plot_ui| {
                            plot_ui.line(traj_line);
                            plot_ui.points(points_scatter);
                        });
                });
            });
        });

        ui.add_space(4.0);

        // 5. Spectrogram / Waterfall Display (Always Open & Prominent)
        WATERFALL_STATE.with(|state_rc| {
            let mut state = state_rc.borrow_mut();

            WATERFALL_BUFFER.with(|buf| {
                let mut history = buf.borrow_mut();
                let new_len = signal.output_spectrum_dbfs.len();

                // Clear history if spectrum length changed (e.g. decimation or FFT size updated)
                if let Some(front) = history.front() {
                    if front.len() != new_len {
                        history.clear();
                    }
                }

                // One row per capture, not per frame. The pipeline is paced independently of
                // the repaint rate, so keying off the sequence number is what stops a slow
                // capture from being smeared across a run of identical rows.
                let fresh = signal.sequence != state.last_sequence;
                if new_len > 0 && fresh && !state.paused {
                    history.push_front(signal.output_spectrum_dbfs.clone());
                }
                state.last_sequence = signal.sequence;
                while history.len() > state.depth {
                    history.pop_back();
                }

                // Deliberately not wrapped in ui.group: the group inset would shift this
                // pane a few pixels relative to the spectrum panes above, and a linked x
                // axis only matches ranges, not screen geometry.
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("🌊 Real-Time Spectrogram / Waterfall Plot")
                            .strong()
                            .size(13.0)
                            .color(Theme::ACCENT_PRIMARY),
                    );
                    ui.colored_label(
                        Theme::TEXT_SECONDARY,
                        "(Vertical: Time [latest on top], Horizontal: Frequency, Color: dBFS intensity)",
                    );

                    // Controls
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("🗑 Clear").clicked() {
                            history.clear();
                            // Let the current capture repopulate the first row immediately
                            // rather than leaving the pane blank until the next one lands.
                            state.last_sequence = 0;
                        }
                        if ui.button(if state.paused { "▶ Resume" } else { "⏸ Freeze" }).clicked() {
                            state.paused = !state.paused;
                        }
                        
                        ui.add(
                            egui::Slider::new(&mut state.depth, 32..=1024)
                                .text("History")
                                .logarithmic(true),
                        )
                        .on_hover_text(
                            "Captures kept. Deeper history reaches further back; shallower \
                             gives each capture more vertical pixels.",
                        );
                        ui.add(egui::Slider::new(&mut state.max_db, -150.0..=20.0).text("Max dB"));
                        ui.add(egui::Slider::new(&mut state.min_db, -150.0..=20.0).text("Min dB"));

                        // Colorbar Legend
                        let mut legend_pixels = Vec::with_capacity(60 * 10);
                        for _y in 0..10 {
                            for x in 0..60 {
                                legend_pixels.push(magma(x as f64 / 59.0));
                            }
                        }
                        let legend_img = egui::ColorImage::new([60, 10], legend_pixels);
                        let legend_tex = ui.ctx().load_texture("waterfall_legend", legend_img, egui::TextureOptions::LINEAR);
                        ui.add(egui::Image::new(&legend_tex).fit_to_exact_size(egui::vec2(60.0, 10.0)));
                        ui.label(egui::RichText::new("Legend:").color(Theme::TEXT_SECONDARY));
                    });
                });

                // A real-output block hands back a one-sided 0..Fout/2 spectrum; assuming
                // ±Fout/2 would stretch it across twice its span and mislabel every frequency.
                let (full_min, full_max) =
                    output_span_mhz(signal.output_sample_rate_mhz, signal.complex_output);

                // Render only what is on screen. The x axis is linked to the pane above, so
                // these are its bounds: zooming in then buys the waterfall real detail instead
                // of magnifying columns it folded away at full span.
                let view_min = output_bounds.min()[0].max(full_min);
                let view_max = output_bounds.max()[0].min(full_max);
                let (bin_lo, bin_hi) = if view_max > view_min {
                    visible_bin_range(new_len, full_min, full_max, view_min, view_max)
                } else {
                    (0, new_len)
                };
                // Frequencies the texture's edges actually land on, so the image is placed on
                // the bin boundaries it was built from rather than on the requested view.
                let bin_width = (full_max - full_min) / new_len.max(1) as f64;
                let tex_min = full_min + bin_lo as f64 * bin_width;
                let tex_max = full_min + bin_hi as f64 * bin_width;

                // One column per screen pixel: past that the extra columns land inside a pixel,
                // short of it the plot upscales a texture coarser than the display.
                let visible_bins = bin_hi - bin_lo;
                let pixel_width = (ui.available_width() - PLOT_Y_AXIS_WIDTH).max(0.0) as usize;
                let width = pixel_width
                    .min(visible_bins)
                    .clamp(WATERFALL_MIN_COLUMNS.min(visible_bins.max(1)), WATERFALL_MAX_COLUMNS);
                let height = history.len();

                if width > 0 && height > 0 {
                    let key = WaterfallKey {
                        sequence: state.last_sequence,
                        rows: height,
                        columns: width,
                        x_min_bits: tex_min.to_bits(),
                        x_max_bits: tex_max.to_bits(),
                        min_db_bits: state.min_db.to_bits(),
                        max_db_bits: state.max_db.to_bits(),
                    };

                    // Rebuilding costs columns × rows and uploads the lot, so it happens only
                    // when one of the inputs it depends on has actually moved.
                    if state.texture_key != Some(key) || state.texture.is_none() {
                        let mut pixels = Vec::with_capacity(width * height);
                        let range_db = (state.max_db - state.min_db).max(0.1);

                        for row in history.iter() {
                            if row.len() != new_len {
                                continue;
                            }
                            for col in 0..width {
                                // Columns fold together by max-hold rather than picking one
                                // bin, so a tone narrower than a column still paints it
                                // instead of flickering in and out between captures. Zoomed
                                // in far enough that bins outnumber columns no longer holds,
                                // and the texture filter interpolates instead.
                                let lo = bin_lo + col * visible_bins / width;
                                let hi = (bin_lo + (col + 1) * visible_bins / width)
                                    .max(lo + 1)
                                    .min(bin_hi);
                                let mag = row[lo..hi]
                                    .iter()
                                    .copied()
                                    .fold(f64::NEG_INFINITY, f64::max);

                                let norm = ((mag - state.min_db) / range_db).clamp(0.0, 1.0);
                                pixels.push(magma(norm));
                            }
                        }

                        if pixels.len() == width * height {
                            let img = egui::ColorImage::new([width, height], pixels);
                            state.texture = Some(ui.ctx().load_texture(
                                "waterfall_texture",
                                img,
                                egui::TextureOptions::LINEAR,
                            ));
                            state.texture_key = Some(key);
                        }
                    }

                    if let Some(texture) = state.texture.clone() {
                        // The y axis is shown, and both plots pin the same width, so the
                        // waterfall's columns stay under the matching spectrum bins.
                        // Newest row is at the top, so the label counts back from there.
                        let plot = egui_plot::Plot::new("waterfall_plot")
                            .height(250.0)
                            .allow_drag(true)
                            .allow_zoom(true)
                            .link_axis(link_group, [true, false])
                            .show_axes([true, true])
                            .x_axis_label("Baseband Offset Frequency (MHz)")
                            .y_axis_label("Captures Ago")
                            .y_axis_min_width(PLOT_Y_AXIS_WIDTH)
                            .y_axis_formatter(move |mark, _| {
                                let ago = height as f64 - mark.value;
                                if !(0.0..=height as f64).contains(&ago) {
                                    String::new()
                                } else {
                                    format!("{ago:.0}")
                                }
                            })
                            .show_grid(false);

                        let inner = plot.show(ui, |plot_ui| {
                            let image = egui_plot::PlotImage::new(
                                "waterfall_img",
                                texture.id(),
                                egui_plot::PlotPoint::new(
                                    (tex_max + tex_min) / 2.0,
                                    height as f64 / 2.0,
                                ),
                                [(tex_max - tex_min) as f32, height as f32],
                            );
                            plot_ui.image(image);

                            plot_ui.pointer_coordinate()
                        });

                        if let Some(coord) = inner.inner {
                            inner.response.on_hover_ui_at_pointer(|ui| {
                                // Newest row is drawn at the top (y = height), and
                                // history[0] is the newest, so the index counts down.
                                let frame_idx = (height as f64 - coord.y)
                                    .round()
                                    .clamp(0.0, height.saturating_sub(1) as f64)
                                    as usize;

                                // Index the spectrum, not the texture: the texture folded
                                // bins together, and the readout should quote the real bin
                                // under the cursor.
                                let span = full_max - full_min;
                                let x_norm = ((coord.x - full_min) / span).clamp(0.0, 1.0);
                                let bin_idx = ((x_norm * (new_len as f64 - 1.0)).round() as usize)
                                    .min(new_len.saturating_sub(1));

                                let mag = history
                                    .get(frame_idx)
                                    .and_then(|row| row.get(bin_idx))
                                    .copied()
                                    .unwrap_or(-150.0);

                                ui.label(
                                    egui::RichText::new(format!("Freq: {:.3} MHz", coord.x))
                                        .strong(),
                                );
                                ui.label(format!("Time: {frame_idx} captures ago"));
                                ui.label(format!("Magnitude: {mag:.1} dBFS"));
                            });
                        }
                    }
                }
            });
        });
    } else {
        ui.centered_and_justified(|ui| {
            ui.heading("No signal processed yet");
            ui.label("Configure a signal source and ADC tile to see the spectrum.");
        });
    }
    window_changed
}

/// One-line summary of what each window buys, for the picker's tooltips.
fn window_hint(window: FftWindow) -> &'static str {
    match window {
        FftWindow::Hanning => "General purpose. Sidelobes -31 dB, so a strong tone's skirt is visible on the trace.",
        FftWindow::Hamming => "Lowest first sidelobe of the narrow windows, but its skirt decays slowly.",
        FftWindow::BlackmanHarris => "Sidelobes -92 dB: a tone shows one clean lobe. ~36% wider than Hanning at -3 dB.",
        FftWindow::FlatTop => "Amplitude accuracy: reads a tone's true level however it falls between bins. Widest lobe.",
        FftWindow::Rectangular => "No window. Sharpest lobe, worst leakage — only for coherently sampled tones.",
    }
}

/// Hover text for a detail level, quoting the resolution bandwidth it buys at the current span.
fn detail_hint(detail: SpectrumDetail, output_rate_mhz: f64) -> String {
    let rbw = if detail.output_bins() > 0 {
        output_rate_mhz / detail.output_bins() as f64
    } else {
        0.0
    };
    let cost = match detail {
        SpectrumDetail::Fast => "Cheapest capture; use when sweeping settings.",
        SpectrumDetail::Balanced => "Default. Comfortable on most configurations.",
        SpectrumDetail::Fine => "4× the capture length; the update rate starts to drop.",
        SpectrumDetail::Max => {
            "16× the capture length. Expect a visibly slow update rate, and the sample budget \
             may cap it at high decimation."
        }
    };
    format!("{} bins → RBW {} here. {cost}", detail.output_bins(), format_rbw(rbw))
}

/// Resolution bandwidth in whichever unit keeps it readable.
fn format_rbw(rbw_mhz: f64) -> String {
    if rbw_mhz >= 1.0 {
        format!("{rbw_mhz:.2} MHz")
    } else if rbw_mhz >= 0.001 {
        format!("{:.1} kHz", rbw_mhz * 1000.0)
    } else {
        format!("{:.1} Hz", rbw_mhz * 1e6)
    }
}

/// A transfer-function curve drawn over a spectrum, in dB rather than dBFS.
struct ResponseOverlay<'a> {
    values_db: &'a [f64],
    freq_axis_mhz: &'a [f64],
    name: &'a str,
    color: egui::Color32,
    dashed: bool,
    /// Clamp the drawn curve here. Deep stopband ripple is real but tangles with the noise
    /// floor, and the passband and transition are what the pane is for.
    floor_db: f64,
}

/// Configuration for one stacked spectrum pane.
struct SpectrumPlot<'a> {
    id: &'a str,
    title: String,
    spectrum: &'a [f64],
    freq_axis: &'a [f64],
    height: f32,
    color: egui::Color32,
    /// Sample rate the pane's span is derived from: 0..rate/2, or ±rate/2 when `two_sided`.
    span_rate_mhz: f64,
    two_sided: bool,
    /// Secondary transfer-function curve, kept visually distinct from the spectrum trace.
    response_overlay: Option<ResponseOverlay<'a>>,
    raw_source_overlay: Option<(&'a [f64], &'a [f64])>,
    show_nyquist_zones: bool,
    /// Band the decimation chain keeps: (passband edge, output Nyquist) in MHz.
    ddc_keep_band_mhz: Option<(f64, f64)>,
    /// Window `spectrum` was computed with, so peak picking knows the shape of a tone's own
    /// leakage skirt.
    window: FftWindow,
    /// Resolution bandwidth of `spectrum`, in MHz. The trace is zero-padded for display, so
    /// its point spacing is finer than this; peak picking needs the real figure. 0 disables
    /// main-lobe suppression.
    resolution_bw_mhz: f64,
    /// Draw the ±0.4·Fout usable-bandwidth edges on an output plot.
    usable_band_mhz: Option<f64>,
    show_dc_marker: bool,
    show_peak_markers: bool,
    /// Colour key rendered next to the title, since the panes carry no plot legend.
    legend: Vec<(&'a str, egui::Color32)>,
    link_group: Option<egui::Id>,
}

impl Default for SpectrumPlot<'_> {
    fn default() -> Self {
        Self {
            id: "",
            title: String::new(),
            spectrum: &[],
            freq_axis: &[],
            height: 160.0,
            color: Theme::ACCENT_PRIMARY,
            span_rate_mhz: 0.0,
            two_sided: false,
            response_overlay: None,
            raw_source_overlay: None,
            show_nyquist_zones: false,
            ddc_keep_band_mhz: None,
            usable_band_mhz: None,
            show_dc_marker: false,
            show_peak_markers: false,
            window: crate::dsp::DEFAULT_DISPLAY_WINDOW,
            resolution_bw_mhz: 0.0,
            legend: Vec::new(),
            link_group: None,
        }
    }
}

/// Draws one spectrum pane and returns the plot bounds it settled on.
///
/// The waterfall is drawn straight after the post-DDC pane and shares its linked x axis, so
/// those bounds are exactly the range it needs to render — read this frame, not lagged a frame
/// behind as reading its own bounds would be.
fn show_single_spectrum(ui: &mut egui::Ui, p: SpectrumPlot<'_>) -> PlotBounds {
    let SpectrumPlot {
        id,
        ref title,
        spectrum,
        freq_axis,
        height,
        color,
        span_rate_mhz: fs_mhz,
        two_sided,
        response_overlay: rf_overlay,
        raw_source_overlay,
        show_nyquist_zones,
        ddc_keep_band_mhz,
        usable_band_mhz,
        show_dc_marker,
        show_peak_markers,
        window,
        resolution_bw_mhz,
        ref legend,
        link_group,
    } = p;

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(title).strong().size(13.0));

        // Painted swatches rather than box-drawing glyphs, which the bundled fonts lack.
        for (label, color) in legend {
            ui.add_space(6.0);
            let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 3.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 1.0, *color);
            ui.label(
                egui::RichText::new(*label)
                    .color(Theme::TEXT_SECONDARY)
                    .size(11.0),
            );
        }

        if show_peak_markers {
            let peaks = find_spectral_peaks(spectrum, freq_axis, -100.0, resolution_bw_mhz, window);
            if peaks.len() >= 2 {
                let p1 = peaks[0];
                let p2 = peaks[1];
                let delta_f = (p2.freq_mhz - p1.freq_mhz).abs();
                let delta_m = p2.mag_dbfs - p1.mag_dbfs;
                ui.separator();
                ui.colored_label(
                    Theme::ACCENT_PRIMARY,
                    format!("Peak 1: {:.1} MHz ({:.1} dBFS)", p1.freq_mhz, p1.mag_dbfs),
                );
                ui.colored_label(
                    Theme::ACCENT_SECONDARY,
                    format!("Peak 2: {:.1} MHz ({:.1} dBFS)", p2.freq_mhz, p2.mag_dbfs),
                );
                ui.colored_label(
                    Theme::ACCENT_WARN,
                    format!("Δf: {:.1} MHz, ΔM: {:.1} dB", delta_f, delta_m),
                );
            }
        }
    });

    let points: PlotPoints = freq_axis
        .iter()
        .zip(spectrum.iter())
        .map(|(&f, &mag)| [f, mag.max(-150.0)])
        .collect();

    let line = Line::new(id, points).color(color).width(1.5);

    let label_fmt = |pos: &egui_plot::HoverPosition<'_>| {
        match pos {
            egui_plot::HoverPosition::NearDataPoint { plot_name, position, .. } => {
                if plot_name.is_empty() {
                    Some(format!("Freq: {:.2} MHz\nMag: {:.1} dBFS", position.x, position.y))
                } else {
                    Some(format!("{plot_name}\nFreq: {:.2} MHz\nMag: {:.1} dBFS", position.x, position.y))
                }
            }
            egui_plot::HoverPosition::Elsewhere { position } => {
                Some(format!("Freq: {:.2} MHz\nMag: {:.1} dBFS", position.x, position.y))
            }
        }
    };

    let mut plot = Plot::new(id)
        .height(height)
        .label_formatter(label_fmt)
        .x_axis_label(if two_sided {
            "Baseband Offset Frequency (MHz)"
        } else {
            "Frequency (MHz)"
        })
        .y_axis_label("Magnitude (dBFS)")
        .y_axis_min_width(PLOT_Y_AXIS_WIDTH)
        .include_y(-150.0)
        .include_y(10.0);

    if let Some(lg) = link_group {
        plot = plot.link_axis(lg, [true, false]);
    }

    if two_sided {
        let span = fs_mhz / 2.0;
        plot = plot.include_x(-span).include_x(span);
    } else {
        let max_freq = fs_mhz / 2.0;
        plot = plot.include_x(0.0).include_x(max_freq);
    }

    plot.show(ui, |plot_ui| {
        let min_freq = freq_axis.first().copied().unwrap_or(0.0);
        let max_freq = freq_axis.last().copied().unwrap_or(fs_mhz / 2.0);

        if show_nyquist_zones {
            // Draw Nyquist zones in the background
            let nyquist_bw = fs_mhz / 2.0;
            let max_zone = (max_freq / nyquist_bw).ceil() as usize;
            for zone in 1..=max_zone {
                let start_f = (zone as f64 - 1.0) * nyquist_bw;
                let end_f = (zone as f64) * nyquist_bw;

                let poly_points: PlotPoints = vec![
                    [start_f, -150.0],
                    [end_f, -150.0],
                    [end_f, 10.0],
                    [start_f, 10.0],
                ].into();

                let poly = egui_plot::Polygon::new(
                    format!("Nyquist Zone {zone}"),
                    poly_points,
                )
                .fill_color(Theme::zone_color(zone).linear_multiply(0.05))
                .stroke(egui::Stroke::NONE);
                plot_ui.polygon(poly);

                let boundary = zone as f64 * nyquist_bw;
                if boundary <= max_freq {
                    let zone_line_points: PlotPoints = vec![[boundary, -150.0], [boundary, 10.0]].into();
                    let zone_line = Line::new(format!("{id}_zone_bound_{zone}"), zone_line_points)
                        .color(Theme::zone_color(zone).linear_multiply(0.8))
                        .width(1.5)
                        .style(egui_plot::LineStyle::dashed_dense());
                    plot_ui.line(zone_line);
                }
            }
        }

        // Shade the band the decimation chain hands to the PL, in the same green the output
        // pane uses so the two read as the same band. Two tints: the guaranteed-flat
        // 0.4·Fout passband, and the strip out to the output Nyquist where the filter is
        // still in transition. Strokes are suppressed so only the fills show.
        if let Some((pass_edge, nyq_edge)) = ddc_keep_band_mhz {
            for (edge, alpha, name) in [
                (nyq_edge, 0.07_f32, "DDC output span (±Fout/2)"),
                (pass_edge, 0.12, "DDC usable band (±0.4·Fout)"),
            ] {
                let band: PlotPoints = vec![
                    [-edge, -150.0],
                    [edge, -150.0],
                    [edge, 10.0],
                    [-edge, 10.0],
                ]
                .into();
                plot_ui.polygon(
                    egui_plot::Polygon::new(name, band)
                        .fill_color(Theme::ACCENT_SECONDARY.linear_multiply(alpha))
                        .stroke(egui::Stroke::NONE),
                );
            }
            // Crisp edges at the two boundaries.
            for (edge, color) in [
                (nyq_edge, Theme::TEXT_SECONDARY.linear_multiply(0.45)),
                (pass_edge, Theme::ACCENT_SECONDARY.linear_multiply(0.55)),
            ] {
                for side in [-1.0_f64, 1.0] {
                    let x = side * edge;
                    let pts: PlotPoints = vec![[x, -150.0], [x, 10.0]].into();
                    plot_ui.line(
                        Line::new("", pts)
                            .color(color)
                            .width(1.0)
                            .style(egui_plot::LineStyle::dashed_loose()),
                    );
                }
            }
        }

        // Mark the guaranteed-flat usable bandwidth on the output plot.
        if let Some(edge) = usable_band_mhz {
            for side in [-1.0_f64, 1.0] {
                let x = side * edge;
                let pts: PlotPoints = vec![[x, -150.0], [x, 10.0]].into();
                plot_ui.line(
                    Line::new("Usable BW edge (±0.4·Fout)", pts)
                        .color(Theme::ACCENT_SECONDARY.linear_multiply(0.5))
                        .width(1.0)
                        .style(egui_plot::LineStyle::dashed_loose()),
                );
            }
        }

        // Render Raw Source reference line first
        if let Some((raw_db, raw_freq)) = raw_source_overlay {
            let raw_points: PlotPoints = raw_freq
                .iter()
                .zip(raw_db.iter())
                .filter(|&(&f, _)| f >= min_freq && f <= max_freq)
                .map(|(&f, &mag)| [f, mag.max(-150.0)])
                .collect();

            let raw_line = Line::new(format!("{id}_raw_source"), raw_points)
                .color(Theme::TEXT_SECONDARY.linear_multiply(0.4))
                .width(1.0)
                .style(egui_plot::LineStyle::dashed_dense());
            plot_ui.line(raw_line);
        }

        // Render main filtered spectrum line
        plot_ui.line(line);

        // Render the response overlay (RF chain H(f), or the decimation filter)
        if let Some(ov) = rf_overlay {
            let overlay_points: PlotPoints = ov
                .freq_axis_mhz
                .iter()
                .zip(ov.values_db.iter())
                .filter(|&(&f, _)| f >= min_freq && f <= max_freq)
                .map(|(&f, &db)| [f, db.max(ov.floor_db)])
                .collect();

            let overlay_line = Line::new(ov.name, overlay_points)
                .color(ov.color)
                .width(2.0)
                .style(if ov.dashed {
                    egui_plot::LineStyle::dashed_loose()
                } else {
                    egui_plot::LineStyle::Solid
                });
            plot_ui.line(overlay_line);
        }

        if show_dc_marker {
            // Render vertical line for 0 Hz (DC / Tuned RF Center)
            let dc_line_points: PlotPoints = vec![[0.0, -150.0], [0.0, 10.0]].into();
            let dc_line = Line::new(format!("{id}_dc_center"), dc_line_points)
                .color(Theme::ACCENT_PRIMARY)
                .width(1.5)
                .style(egui_plot::LineStyle::dashed_dense());
            plot_ui.line(dc_line);
        }

        if show_peak_markers {
            // Render Peak markers as vertical lines
            let peaks = find_spectral_peaks(spectrum, freq_axis, -100.0, resolution_bw_mhz, window);
            for (idx, pk) in peaks.iter().take(2).enumerate() {
                let pk_line_points: PlotPoints = vec![[pk.freq_mhz, -150.0], [pk.freq_mhz, pk.mag_dbfs]].into();
                let pk_color = if idx == 0 { Theme::ACCENT_PRIMARY } else { Theme::ACCENT_SECONDARY };
                let pk_line = Line::new(format!("{id}_pk_{idx}"), pk_line_points)
                    .color(pk_color)
                    .width(1.5)
                    .style(egui_plot::LineStyle::Solid);
                plot_ui.line(pk_line);
            }
        }

        plot_ui.plot_bounds()
    })
    .inner
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waterfall_buffer_handles_length_changes() {
        WATERFALL_BUFFER.with(|buf| {
            let mut history = buf.borrow_mut();
            history.clear();

            // Push 5 frames of length 100
            for _ in 0..5 {
                history.push_front(vec![-50.0; 100]);
            }

            // Simulate changing FFT size to 200: length mismatch
            let new_len = 200;
            if let Some(front) = history.front() {
                if front.len() != new_len {
                    history.clear();
                }
            }
            history.push_front(vec![-50.0; new_len]);

            assert_eq!(history.len(), 1);
            assert_eq!(history.front().unwrap().len(), 200);
        });
    }

    /// Drive the whole pane — detail picker, both spectrum plots, and the waterfall texture
    /// build — through a headless egui context.
    ///
    /// The waterfall lives at the bottom of a scroll area, so in a running window it is easy
    /// for it to be culled and never execute. This runs it for real: a screen tall enough that
    /// nothing is culled, over several frames so the history accumulates and the texture cache
    /// is exercised on both the miss and the hit.
    fn drive_spectrum_view(
        detail: &mut SpectrumDetail,
        signals: &[ProcessedSignal],
    ) -> Vec<egui::FullOutput> {
        let ctx = egui::Context::default();
        let mut window = FftWindow::BlackmanHarris;
        signals
            .iter()
            .map(|signal| {
                let input = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1600.0, 4000.0),
                    )),
                    ..Default::default()
                };
                ctx.run_ui(input, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        show_spectrum_view(ui, &Some(signal.clone()), 4000.0, &mut window, detail);
                    });
                })
            })
            .collect()
    }

    /// Build a capture through the real pipeline so the pane sees the shapes it will in the app.
    fn capture(decim: crate::rfdc::DecimationFactor, complex_out: bool, seq: u64) -> ProcessedSignal {
        use crate::dsp::{process_adc_block, SpectrumAnalysis};
        use crate::rfdc::{AdcTile, MixerMode, MixerType};
        use crate::signal::{SignalGenerator, Tone, ToneModulation};

        let sim_fs = 15000.0;
        let mut tile = AdcTile::new(0);
        tile.sample_rate_gsps = 4.0;
        {
            let b = &mut tile.blocks[0];
            b.decimation = decim;
            b.mixer_settings.mixer_type = MixerType::Fine;
            b.mixer_settings.mixer_mode = if complex_out {
                MixerMode::RealToIq
            } else {
                MixerMode::RealToReal
            };
            b.mixer_settings.freq = -300.0;
        }
        let block = tile.blocks[0].clone();

        let sig_gen = SignalGenerator {
            tones: vec![Tone {
                frequency_mhz: 300.0,
                amplitude_dbfs: -6.0,
                phase_deg: 0.0,
                bandwidth_mhz: 0.0,
                modulation: ToneModulation::Cw,
            }],
            noise_floor_dbfs: -110.0,
            noise_enabled: false,
        };
        let samples = sig_gen.generate(120_000, sim_fs);
        let mut out = process_adc_block(
            &samples,
            sim_fs,
            &block,
            &tile,
            None,
            None,
            SpectrumAnalysis::default(),
        );
        out.sequence = seq;
        out
    }

    #[test]
    fn spectrum_view_renders_complex_and_real_outputs() {
        use crate::rfdc::DecimationFactor;

        for complex_out in [true, false] {
            WATERFALL_BUFFER.with(|b| b.borrow_mut().clear());
            WATERFALL_STATE.with(|s| *s.borrow_mut() = WaterfallState::default());

            let captures: Vec<ProcessedSignal> = (1..=4)
                .map(|seq| capture(DecimationFactor::X16, complex_out, seq))
                .collect();
            assert_eq!(captures[0].complex_output, complex_out);

            let mut detail = SpectrumDetail::Balanced;
            let outputs = drive_spectrum_view(&mut detail, &captures);
            assert_eq!(outputs.len(), 4);

            // One row per capture, and a texture actually built for them.
            WATERFALL_BUFFER.with(|b| assert_eq!(b.borrow().len(), 4));
            WATERFALL_STATE.with(|s| {
                let s = s.borrow();
                assert!(s.texture.is_some(), "waterfall texture was never built");
                let key = s.texture_key.expect("texture built without a key");
                assert_eq!(key.rows, 4);
                assert!(key.columns > 0);

                // The texture spans the spectrum's real axis, not an assumed two-sided one.
                let (lo, hi) = output_span_mhz(captures[0].output_sample_rate_mhz, complex_out);
                assert!(f64::from_bits(key.x_min_bits) >= lo - 1e-9);
                assert!(f64::from_bits(key.x_max_bits) <= hi + 1e-9);
            });
        }
    }

    /// A repaint with no new capture must not push a duplicate row — the pipeline runs slower
    /// than the frame rate, so this happens constantly at high detail.
    #[test]
    fn repaints_without_a_new_capture_add_no_rows() {
        use crate::rfdc::DecimationFactor;

        WATERFALL_BUFFER.with(|b| b.borrow_mut().clear());
        WATERFALL_STATE.with(|s| *s.borrow_mut() = WaterfallState::default());

        // Same sequence number four times over: one capture, repainted.
        let one = capture(DecimationFactor::X8, true, 7);
        let repeats = vec![one.clone(), one.clone(), one.clone(), one];

        let mut detail = SpectrumDetail::Balanced;
        drive_spectrum_view(&mut detail, &repeats);

        WATERFALL_BUFFER.with(|b| {
            assert_eq!(
                b.borrow().len(),
                1,
                "repainting the same capture pushed duplicate waterfall rows"
            )
        });
    }

    #[test]
    fn waterfall_history_trims_to_the_configured_depth() {
        let mut history: VecDeque<Vec<f64>> = VecDeque::new();
        for i in 0..40 {
            history.push_front(vec![i as f64; 8]);
        }

        // Lowering the depth drops the oldest rows and keeps the newest.
        let depth = 10;
        while history.len() > depth {
            history.pop_back();
        }

        assert_eq!(history.len(), depth);
        assert_eq!(history.front().unwrap()[0], 39.0);
        assert_eq!(history.back().unwrap()[0], 30.0);
    }

    /// A real-output block's spectrum is one-sided; assuming ±Fout/2 stretched it across twice
    /// its span and mislabelled every frequency on the waterfall's axis.
    #[test]
    fn output_span_matches_the_spectrum_sidedness() {
        assert_eq!(output_span_mhz(250.0, true), (-125.0, 125.0));
        assert_eq!(output_span_mhz(250.0, false), (0.0, 125.0));
    }

    #[test]
    fn visible_bin_range_covers_the_whole_axis_when_unzoomed() {
        // Two-sided: the full ±125 MHz view maps to every bin.
        assert_eq!(visible_bin_range(1024, -125.0, 125.0, -125.0, 125.0), (0, 1024));
        // One-sided likewise.
        assert_eq!(visible_bin_range(1024, 0.0, 125.0, 0.0, 125.0), (0, 1024));
    }

    #[test]
    fn visible_bin_range_narrows_with_the_view() {
        // Middle half of a two-sided axis.
        let (lo, hi) = visible_bin_range(1000, -100.0, 100.0, -50.0, 50.0);
        assert_eq!((lo, hi), (250, 751));

        // A hard zoom still yields a usable slice rather than an empty one.
        let (lo, hi) = visible_bin_range(1000, -100.0, 100.0, 0.0, 0.0001);
        assert!(hi > lo, "zoomed range collapsed to {lo}..{hi}");

        // Views past the axis clamp instead of indexing out of bounds.
        let (lo, hi) = visible_bin_range(1000, -100.0, 100.0, -500.0, 500.0);
        assert_eq!((lo, hi), (0, 1000));
    }

    #[test]
    fn visible_bin_range_survives_degenerate_input() {
        assert_eq!(visible_bin_range(0, -1.0, 1.0, -1.0, 1.0), (0, 0));
        assert_eq!(visible_bin_range(64, 0.0, 0.0, 0.0, 0.0), (0, 64));
    }

    fn hanning_tone(n: usize, fs: f64, entries: &[(f64, f64)]) -> Vec<num_complex::Complex<f64>> {
        (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                entries.iter().fold(
                    num_complex::Complex::new(0.0, 0.0),
                    |acc, &(f_mhz, amp)| {
                        let a = 2.0 * std::f64::consts::PI * f_mhz * t;
                        acc + num_complex::Complex::new(amp * a.cos(), amp * a.sin())
                    },
                )
            })
            .collect()
    }

    /// Display padding resolves the window's sidelobes into real local maxima — a single tone
    /// produces dozens. The marker readout must report one signal, not a skirt.
    #[test]
    fn peak_finder_reports_one_peak_for_one_tone() {
        use crate::dsp::{compute_spectrum_padded, FftWindow};

        let n = 512;
        let fs = 500.0;
        let rbw = fs / n as f64;
        let samples = hanning_tone(n, fs, &[(100.0, 0.5)]);
        let (spec, freq) = compute_spectrum_padded(&samples, n, fs, FftWindow::Hanning, 8);

        let raw = find_spectral_peaks(&spec, &freq, -100.0, 0.0, FftWindow::Hanning);
        assert!(
            raw.len() > 10,
            "padded trace should expose sidelobes as local maxima, else this proves nothing; got {}",
            raw.len()
        );

        let peaks = find_spectral_peaks(&spec, &freq, -100.0, rbw, FftWindow::Hanning);
        assert_eq!(
            peaks.len(),
            1,
            "one tone should report one peak, got {} (first extra at {:.2} MHz, {:.1} dBFS)",
            peaks.len(),
            peaks.get(1).map(|p| p.freq_mhz).unwrap_or(0.0),
            peaks.get(1).map(|p| p.mag_dbfs).unwrap_or(0.0),
        );
        assert!((peaks[0].freq_mhz - 100.0).abs() < 1.0);
    }

    /// Suppression must not swallow a genuine second signal standing above the leakage.
    #[test]
    fn peak_finder_keeps_a_real_second_tone() {
        use crate::dsp::{compute_spectrum_padded, FftWindow};

        let n = 512;
        let fs = 500.0;
        let rbw = fs / n as f64;
        // Second tone 12 dB down — far above Hanning's -31.5 dB first sidelobe.
        let samples = hanning_tone(n, fs, &[(100.0, 0.5), (150.0, 0.125)]);
        let (spec, freq) = compute_spectrum_padded(&samples, n, fs, FftWindow::Hanning, 8);

        let peaks = find_spectral_peaks(&spec, &freq, -100.0, rbw, FftWindow::Hanning);
        assert_eq!(peaks.len(), 2, "expected exactly the two tones");
        assert!((peaks[0].freq_mhz - 100.0).abs() < 1.0);
        assert!(
            (peaks[1].freq_mhz - 150.0).abs() < 1.0,
            "second peak should be the 150 MHz tone, got {:.2} MHz",
            peaks[1].freq_mhz
        );
    }

    /// A weak tone close in should still be found once it clears the neighbour's skirt.
    #[test]
    fn peak_finder_keeps_a_close_tone_above_the_skirt() {
        use crate::dsp::{compute_spectrum_padded, FftWindow};

        let n = 512;
        let fs = 500.0;
        let rbw = fs / n as f64;
        // 8 bins out, 25 dB down: Hanning leakage there is ~-53 dB, so this stands clear.
        let samples = hanning_tone(n, fs, &[(100.0, 0.5), (100.0 + 8.0 * rbw, 0.028)]);
        let (spec, freq) = compute_spectrum_padded(&samples, n, fs, FftWindow::Hanning, 8);

        let peaks = find_spectral_peaks(&spec, &freq, -100.0, rbw, FftWindow::Hanning);
        assert_eq!(peaks.len(), 2, "close-in tone above the skirt should survive");
    }
}
