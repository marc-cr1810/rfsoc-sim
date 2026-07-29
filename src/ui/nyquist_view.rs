//! Nyquist zone visualization.

use crate::ui::theme::Theme;

/// Render a Nyquist zone strip diagram showing how zones fold.
pub fn show_nyquist_view(
    ui: &mut egui::Ui,
    fs_mhz: f64,
    num_zones: usize,
    selected_zone: usize,
) {
    ui.heading("🔢 Nyquist Zone Map");
    ui.separator();

    let nyquist_bw = fs_mhz / 2.0;

    ui.label(format!(
        "Fs = {:.0} MHz  |  Nyquist BW = {:.0} MHz",
        fs_mhz, nyquist_bw
    ));
    ui.add_space(4.0);

    // Draw zone strip
    let available_width = ui.available_width();
    let strip_height = 40.0;
    let (response, painter) = ui.allocate_painter(
        egui::Vec2::new(available_width, strip_height),
        egui::Sense::hover(),
    );
    let rect = response.rect;

    let _total_freq = nyquist_bw * num_zones as f64;

    for zone in 0..num_zones {
        let zone_num = zone + 1;
        let x_start = rect.left() + (zone as f32 / num_zones as f32) * rect.width();
        let x_end = rect.left() + ((zone + 1) as f32 / num_zones as f32) * rect.width();

        let zone_rect = egui::Rect::from_min_max(
            egui::Pos2::new(x_start, rect.top()),
            egui::Pos2::new(x_end, rect.bottom()),
        );

        // Fill colour based on zone, highlighted if selected
        let mut color = Theme::zone_color(zone_num);
        if zone_num != selected_zone {
            color = color.linear_multiply(0.3);
        }

        painter.rect_filled(zone_rect, 0.0, color);

        // Zone label
        let label = format!("Z{}", zone_num);
        painter.text(
            zone_rect.center(),
            egui::Align2::CENTER_CENTER,
            &label,
            egui::FontId::proportional(14.0),
            Theme::TEXT_PRIMARY,
        );

        // Frequency labels at boundaries
        let freq_label = format!("{:.0}", zone_num as f64 * nyquist_bw);
        painter.text(
            egui::Pos2::new(x_end, rect.bottom() + 2.0),
            egui::Align2::CENTER_TOP,
            &freq_label,
            egui::FontId::proportional(10.0),
            Theme::TEXT_SECONDARY,
        );
    }

    // Starting 0 label
    painter.text(
        egui::Pos2::new(rect.left(), rect.bottom() + 2.0),
        egui::Align2::CENTER_TOP,
        "0",
        egui::FontId::proportional(10.0),
        Theme::TEXT_SECONDARY,
    );

    ui.add_space(18.0);

    // Folding explanation
    ui.label(egui::RichText::new("Zone Folding Rules:").strong());
    egui::Grid::new("nyquist_folding_rules_grid")
        .num_columns(2)
        .spacing([24.0, 4.0])
        .show(ui, |ui| {
            for zone in 1..=num_zones {
                let freq_start = (zone - 1) as f64 * nyquist_bw;
                let freq_end = zone as f64 * nyquist_bw;
                let fold_dir = if zone % 2 == 1 { "☐ direct" } else { "☐ mirrored" };
                let color = Theme::zone_color(zone);
                let is_selected = zone == selected_zone;

                ui.horizontal(|ui| {
                    if is_selected {
                        ui.colored_label(color, egui::RichText::new(format!("▶ Zone {zone}")).strong());
                        ui.label(egui::RichText::new(format!("({:.0}–{:.0} MHz) {fold_dir}", freq_start, freq_end)).strong());
                    } else {
                        ui.colored_label(color, format!("Zone {zone}"));
                        ui.label(format!("({:.0}–{:.0} MHz) {fold_dir}", freq_start, freq_end));
                    }
                });

                if zone % 2 == 0 {
                    ui.end_row();
                }
            }
        });
}
