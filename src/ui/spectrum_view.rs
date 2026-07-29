//! Spectrum visualization plots using egui_plot.

use crate::dsp::ProcessedSignal;
use crate::ui::theme::Theme;
use egui_plot::{Line, Plot, PlotPoints};

/// Render the multi-pane spectrum display.
pub fn show_spectrum_view(
    ui: &mut egui::Ui,
    processed: &Option<ProcessedSignal>,
    tile_fs_mhz: f64,
) {
    if let Some(signal) = processed {
        // Use a vertical layout with multiple plots
        let available_height = ui.available_height();
        let plot_height = (available_height / 3.0).max(150.0);

        ui.horizontal(|ui| {
            ui.heading("📊 Spectrum Analysis");
            if signal.rf_chain_response_db.is_some() {
                ui.separator();
                ui.colored_label(
                    Theme::ACCENT_WARN,
                    "⚡ Overlay: RF Chain H(f)",
                );
                ui.colored_label(
                    Theme::TEXT_SECONDARY,
                    " (Gold = Transfer Function H(f), Gray = Raw Source, Blue = Filtered)",
                );
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
        );

        ui.add_space(4.0);

        // 3. Post-DDC Spectrum
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
        );
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
) {
    ui.label(egui::RichText::new(title).strong().size(13.0));

    let points: PlotPoints = freq_axis
        .iter()
        .zip(spectrum.iter())
        .map(|(&f, &mag)| [f, mag.max(-150.0)]) // Clamp floor
        .collect();

    let line = Line::new(id, points)
        .color(color)
        .width(1.5);

    let mut plot = Plot::new(id)
        .height(height)
        .x_axis_label(if is_complex_baseband { "Baseband Offset Frequency (MHz)" } else { "Frequency (MHz)" })
        .y_axis_label("Magnitude (dBFS)")
        .include_y(-150.0)
        .include_y(10.0);

    if is_complex_baseband {
        let span = fs_mhz / 2.0;
        plot = plot.include_x(-span).include_x(span);
    } else {
        let max_freq = fs_mhz / 2.0;
        plot = plot.include_x(0.0).include_x(max_freq);
    }

    plot.show(ui, |plot_ui| {
            // Render Raw Source reference line first (behind)
            if let Some((raw_db, raw_freq)) = raw_source_overlay {
                let raw_points: PlotPoints = raw_freq
                    .iter()
                    .zip(raw_db.iter())
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
                    .map(|(&f, &db)| [f, db])
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
            } else {
                // Draw Nyquist zone boundaries as vertical lines for wideband plots
                let nyquist_bw = fs_mhz / 2.0;
                for zone in 1..=6 {
                    let boundary = zone as f64 * nyquist_bw;
                    let zone_line_points: PlotPoints = vec![[boundary, -150.0], [boundary, 10.0]].into();
                    let zone_line = Line::new(
                        format!("{id}_zone_{zone}"),
                        zone_line_points,
                    )
                        .color(Theme::zone_color(zone).linear_multiply(0.4))
                        .width(1.0)
                        .style(egui_plot::LineStyle::dashed_dense());
                    plot_ui.line(zone_line);
                }
            }
        });
}
