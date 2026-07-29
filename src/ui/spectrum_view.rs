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

        ui.heading("📊 Spectrum Analysis");
        ui.separator();

        // 1. Input Spectrum
        show_single_spectrum(
            ui,
            "input_spectrum",
            "Input Spectrum (Pre-ADC)",
            &signal.input_spectrum_dbfs,
            &signal.input_freq_axis_mhz,
            plot_height,
            Theme::ACCENT_PRIMARY,
            tile_fs_mhz,
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
        );

        ui.add_space(4.0);

        // 3. Post-DDC Spectrum
        show_single_spectrum(
            ui,
            "output_spectrum",
            &format!(
                "Post-DDC Spectrum (Output Rate: {:.1} MHz)",
                signal.output_sample_rate_mhz
            ),
            &signal.output_spectrum_dbfs,
            &signal.output_freq_axis_mhz,
            plot_height,
            Theme::ACCENT_SECONDARY,
            signal.output_sample_rate_mhz,
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

    Plot::new(id)
        .height(height)
        .x_axis_label("Frequency (MHz)")
        .y_axis_label("Magnitude (dBFS)")
        .include_y(-150.0)
        .include_y(10.0)
        .show(ui, |plot_ui| {
            plot_ui.line(line);

            // Draw Nyquist zone boundaries as vertical lines
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
        });
}
