//! Tile/block overview grid.

use crate::rfdc::RfdcConfig;
use crate::ui::theme::Theme;

/// Render the tile/block overview grid.
pub fn show_tile_overview(
    ui: &mut egui::Ui,
    config: &RfdcConfig,
    selected_tile: &mut usize,
    selected_block: &mut usize,
) {
    ui.heading("📋 ADC Tile Overview");
    ui.separator();

    egui::Grid::new("tile_overview_grid")
        .striped(true)
        .min_col_width(140.0)
        .spacing([8.0, 6.0])
        .show(ui, |ui| {
            // Header row
            ui.strong("Tile");
            ui.strong("Status");
            ui.strong("Fs (GSPS)");
            ui.strong("NZ");
            ui.strong("Block 0");
            ui.strong("Block 1");
            ui.end_row();

            for (ti, tile) in config.adc_tiles.iter().enumerate() {
                // Tile index
                let is_selected = ti == *selected_tile;
                let label = format!("ADC Tile {}", ti);
                if ui
                    .selectable_label(is_selected, &label)
                    .clicked()
                {
                    *selected_tile = ti;
                    *selected_block = 0;
                }

                // Status
                if tile.enabled {
                    ui.colored_label(Theme::ENABLED, "● Enabled");
                } else {
                    ui.colored_label(Theme::DISABLED, "● Disabled");
                }

                // Sample rate
                if tile.enabled {
                    ui.label(format!("{:.2}", tile.sample_rate_gsps));
                } else {
                    ui.colored_label(Theme::TEXT_SECONDARY, "—");
                }

                // Nyquist zone (show from block 0 as representative)
                if tile.enabled {
                    let b0 = &tile.blocks[0];
                    let zone_color = Theme::zone_color(b0.planner_zone as usize);
                    ui.colored_label(zone_color, b0.nyquist_zone.to_string());
                } else {
                    ui.colored_label(Theme::TEXT_SECONDARY, "—");
                }

                // Block 0
                if tile.enabled {
                    let b = &tile.blocks[0];
                    let block_text = if b.enabled {
                        format!("{} | {}", b.mixer_settings.mixer_type, b.decimation)
                    } else {
                        "Disabled".to_string()
                    };
                    let color = if b.enabled { Theme::ENABLED } else { Theme::DISABLED };
                    if ui
                        .colored_label(color, &block_text)
                        .on_hover_text(format!("Block 0: {}", block_text))
                        .clicked()
                    {
                        *selected_tile = ti;
                        *selected_block = 0;
                    }
                } else {
                    ui.colored_label(Theme::TEXT_SECONDARY, "—");
                }

                // Block 1
                if tile.enabled {
                    let b = &tile.blocks[1];
                    let block_text = if b.enabled {
                        format!("{} | {}", b.mixer_settings.mixer_type, b.decimation)
                    } else {
                        "Disabled".to_string()
                    };
                    let color = if b.enabled { Theme::ENABLED } else { Theme::DISABLED };
                    if ui
                        .colored_label(color, &block_text)
                        .on_hover_text(format!("Block 1: {}", block_text))
                        .clicked()
                    {
                        *selected_tile = ti;
                        *selected_block = 1;
                    }
                } else {
                    ui.colored_label(Theme::TEXT_SECONDARY, "—");
                }

                ui.end_row();
            }
        });
}
