//! Spectrum visualization plots using egui_plot and waterfall spectrogram.

use crate::dsp::ProcessedSignal;
use crate::ui::theme::Theme;
use egui_plot::{Line, Plot, PlotPoints};
use std::cell::RefCell;
use std::collections::VecDeque;

struct WaterfallState {
    paused: bool,
    min_db: f64,
    max_db: f64,
}

impl Default for WaterfallState {
    fn default() -> Self {
        Self {
            paused: false,
            min_db: -140.0,
            max_db: 0.0,
        }
    }
}

thread_local! {
    static WATERFALL_BUFFER: RefCell<VecDeque<Vec<f64>>> = RefCell::new(VecDeque::new());
    static WATERFALL_STATE: RefCell<WaterfallState> = RefCell::new(WaterfallState::default());
}

/// A detected spectral peak.
#[derive(Debug, Clone, Copy)]
pub struct SpectralPeak {
    pub freq_mhz: f64,
    pub mag_dbfs: f64,
}

/// Find local spectral peaks above threshold_dbfs.
pub fn find_spectral_peaks(
    spectrum: &[f64],
    freq_axis: &[f64],
    threshold_dbfs: f64,
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
    peaks
}

/// Render the multi-pane spectrum display with waterfall and peak markers.
pub fn show_spectrum_view(
    ui: &mut egui::Ui,
    processed: &Option<ProcessedSignal>,
    tile_fs_mhz: f64,
) {
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
        });
        ui.separator();

        let rf_chain_overlay = match (&signal.rf_chain_response_db, &signal.rf_chain_freq_axis_mhz) {
            (Some(resp), Some(freq)) => Some((resp.as_slice(), freq.as_slice())),
            _ => None,
        };

        let raw_source_overlay = signal
            .raw_source_spectrum_dbfs
            .as_ref()
            .map(|raw| (raw.as_slice(), signal.input_freq_axis_mhz.as_slice()));

        // 1. Input Spectrum with RF Chain overlay
        show_single_spectrum(
            ui,
            "input_spectrum",
            "Input Spectrum (Pre-ADC)",
            &signal.input_spectrum_dbfs,
            &signal.input_freq_axis_mhz,
            plot_height,
            Theme::ACCENT_PRIMARY,
            tile_fs_mhz,
            rf_chain_overlay,
            raw_source_overlay,
            false,
            None,
        );

        ui.add_space(4.0);

        // 2. Folded Spectrum (what ADC sees)
        show_single_spectrum(
            ui,
            "folded_spectrum",
            "Folded Spectrum (ADC Output, 0–Fs/2)",
            &signal.folded_spectrum_dbfs,
            &signal.folded_freq_axis_mhz,
            plot_height,
            Theme::ZONE_1,
            tile_fs_mhz,
            None,
            None,
            false,
            None,
        );

        ui.add_space(4.0);

        let link_group = ui.id().with("baseband_x_link");

        // 3. Post-DDC Spectrum with Peak & Delta Markers
        show_single_spectrum(
            ui,
            "output_spectrum",
            &format!(
                "Post-DDC Complex Baseband Spectrum (Output Rate: {:.1} MHz, Span: ±{:.1} MHz)",
                signal.output_sample_rate_mhz,
                signal.output_sample_rate_mhz / 2.0
            ),
            &signal.output_spectrum_dbfs,
            &signal.output_freq_axis_mhz,
            plot_height,
            Theme::ACCENT_SECONDARY,
            signal.output_sample_rate_mhz,
            None,
            None,
            true,
            Some(link_group),
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

                if new_len > 0 && !state.paused {
                    if history.len() >= 256 {
                        history.pop_back();
                    }
                    history.push_front(signal.output_spectrum_dbfs.clone());
                }

                ui.group(|ui| {
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
                            }
                            if ui.button(if state.paused { "▶ Resume" } else { "⏸ Freeze" }).clicked() {
                                state.paused = !state.paused;
                            }
                            
                            ui.add(egui::Slider::new(&mut state.max_db, -150.0..=20.0).text("Max dB"));
                            ui.add(egui::Slider::new(&mut state.min_db, -150.0..=20.0).text("Min dB"));
                            
                            // Colorbar Legend
                            let mut legend_pixels = Vec::with_capacity(60 * 10);
                            for _y in 0..10 {
                                for x in 0..60 {
                                    let norm = x as f64 / 59.0;
                                    let r = (norm * 3.0 - 1.0).clamp(0.0, 1.0);
                                    let g = (norm * 3.0 - 2.0).clamp(0.0, 1.0);
                                    let b = (norm * 3.0).clamp(0.0, 1.0) - (norm * 3.0 - 1.0).clamp(0.0, 1.0);
                                    legend_pixels.push(egui::Color32::from_rgb(
                                        (r * 255.0) as u8,
                                        (g * 255.0) as u8,
                                        (b * 255.0) as u8,
                                    ));
                                }
                            }
                            let legend_img = egui::ColorImage::new([60, 10], legend_pixels);
                            let legend_tex = ui.ctx().load_texture("waterfall_legend", legend_img, egui::TextureOptions::LINEAR);
                            ui.add(egui::Image::new(&legend_tex).fit_to_exact_size(egui::vec2(60.0, 10.0)));
                            ui.label(egui::RichText::new("Legend:").color(Theme::TEXT_SECONDARY));
                        });
                    });

                    let width = new_len;
                    let height = history.len();
                    if width > 0 && height > 0 {
                        let mut pixels = Vec::with_capacity(width * height);
                        let range_db = (state.max_db - state.min_db).max(0.1);

                        for row in history.iter() {
                            if row.len() == width {
                                for &mag in row.iter() {
                                    // Map dBFS using dynamic range
                                    let norm = ((mag - state.min_db) / range_db).clamp(0.0, 1.0);
                                    
                                    // Magma/Inferno style colormap: Black -> Blue -> Purple -> Red -> Yellow
                                    let r = (norm * 3.0 - 1.0).clamp(0.0, 1.0);
                                    let g = (norm * 3.0 - 2.0).clamp(0.0, 1.0);
                                    let b = (norm * 3.0).clamp(0.0, 1.0) - (norm * 3.0 - 1.0).clamp(0.0, 1.0);

                                    pixels.push(egui::Color32::from_rgb(
                                        (r * 255.0) as u8,
                                        (g * 255.0) as u8,
                                        (b * 255.0) as u8,
                                    ));
                                }
                            }
                        }

                        if pixels.len() == width * height {
                            let img = egui::ColorImage::new([width, height], pixels);
                            let texture = ui.ctx().load_texture(
                                "waterfall_texture",
                                img,
                                egui::TextureOptions::LINEAR,
                            );
                            
                            let plot = egui_plot::Plot::new("waterfall_plot")
                                .height(250.0)
                                .allow_drag(true)
                                .allow_zoom(true)
                                .link_axis(link_group, [true, false])
                                .show_axes([true, false])
                                .x_axis_label("Baseband Offset Frequency (MHz)")
                                .show_grid(false);
                            
                            let inner = plot.show(ui, |plot_ui| {
                                let x_min = -signal.output_sample_rate_mhz / 2.0;
                                let x_max = signal.output_sample_rate_mhz / 2.0;
                                
                                let image = egui_plot::PlotImage::new(
                                    "waterfall_img",
                                    texture.id(),
                                    egui_plot::PlotPoint::new((x_max + x_min) / 2.0, height as f64 / 2.0),
                                    [(x_max - x_min) as f32, height as f32],
                                );
                                plot_ui.image(image);
                                
                                plot_ui.pointer_coordinate()
                            });

                            if let Some(coord) = inner.inner {
                                inner.response.on_hover_ui_at_pointer(|ui| {
                                    let frame_idx = (coord.y.round() as usize).clamp(0, height.saturating_sub(1));
                                    
                                    let x_min = -signal.output_sample_rate_mhz / 2.0;
                                    let x_max = signal.output_sample_rate_mhz / 2.0;
                                    let freq_span = x_max - x_min;
                                    let x_norm = ((coord.x - x_min) / freq_span).clamp(0.0, 1.0);
                                    let bin_idx = (x_norm * (width as f64 - 1.0)).round() as usize;

                                    let mag = history.get(frame_idx)
                                        .and_then(|row| row.get(bin_idx))
                                        .copied()
                                        .unwrap_or(-150.0);

                                    ui.label(egui::RichText::new(format!("Freq: {:.3} MHz", coord.x)).strong());
                                    ui.label(format!("Time: {} frames ago", frame_idx));
                                    ui.label(format!("Magnitude: {:.1} dBFS", mag));
                                });
                            }
                        }
                    }
                });
            });
        });
    } else {
        ui.centered_and_justified(|ui| {
            ui.heading("No signal processed yet");
            ui.label("Configure a signal source and ADC tile to see the spectrum.");
        });
    }
}

fn show_single_spectrum(
    ui: &mut egui::Ui,
    id: &str,
    title: &str,
    spectrum: &[f64],
    freq_axis: &[f64],
    height: f32,
    color: egui::Color32,
    fs_mhz: f64,
    rf_overlay: Option<(&[f64], &[f64])>,
    raw_source_overlay: Option<(&[f64], &[f64])>,
    is_complex_baseband: bool,
    link_group: Option<egui::Id>,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(title).strong().size(13.0));

        if is_complex_baseband {
            let peaks = find_spectral_peaks(spectrum, freq_axis, -100.0);
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
        .x_axis_label(if is_complex_baseband {
            "Baseband Offset Frequency (MHz)"
        } else {
            "Frequency (MHz)"
        })
        .y_axis_label("Magnitude (dBFS)")
        .include_y(-150.0)
        .include_y(10.0);

    if let Some(lg) = link_group {
        plot = plot.link_axis(lg, [true, false]);
    }

    if is_complex_baseband {
        let span = fs_mhz / 2.0;
        plot = plot.include_x(-span).include_x(span);
    } else {
        let max_freq = fs_mhz / 2.0;
        plot = plot.include_x(0.0).include_x(max_freq);
    }

    plot.show(ui, |plot_ui| {
        let min_freq = freq_axis.first().copied().unwrap_or(0.0);
        let max_freq = freq_axis.last().copied().unwrap_or(fs_mhz / 2.0);

        if !is_complex_baseband {
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
                .fill_color(Theme::zone_color(zone).linear_multiply(0.05));
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

        // Render RF Chain H(f) overlay if provided
        if let Some((resp_db, resp_freq)) = rf_overlay {
            let overlay_points: PlotPoints = resp_freq
                .iter()
                .zip(resp_db.iter())
                .filter(|&(&f, _)| f >= min_freq && f <= max_freq)
                .map(|(&f, &db)| [f, db.max(-150.0)])
                .collect();

            let overlay_line = Line::new(format!("{id}_rf_overlay"), overlay_points)
                .color(Theme::ACCENT_WARN)
                .width(2.0)
                .style(egui_plot::LineStyle::Solid);
            plot_ui.line(overlay_line);
        }

        if is_complex_baseband {
            // Render vertical line for 0 Hz (DC / Tuned RF Center)
            let dc_line_points: PlotPoints = vec![[0.0, -150.0], [0.0, 10.0]].into();
            let dc_line = Line::new(format!("{id}_dc_center"), dc_line_points)
                .color(Theme::ACCENT_PRIMARY)
                .width(1.5)
                .style(egui_plot::LineStyle::dashed_dense());
            plot_ui.line(dc_line);

            // Render Peak markers as vertical lines
            let peaks = find_spectral_peaks(spectrum, freq_axis, -100.0);
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
    });
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
}
