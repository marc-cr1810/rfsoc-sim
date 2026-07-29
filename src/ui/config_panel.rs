//! Configuration sidebar for tile/block parameters.

use crate::rfdc::*;
use crate::ui::theme::Theme;

/// Render the RFDC configuration panel in a side panel.
pub fn show_config_panel(ui: &mut egui::Ui, config: &mut RfdcConfig, selected_tile: &mut usize, selected_block: &mut usize) {
    ui.heading("⚙ RFDC Configuration");
    ui.separator();

    // Tile selector
    ui.horizontal(|ui| {
        ui.label("ADC Tile:");
        egui::ComboBox::from_id_salt("tile_select")
            .selected_text(format!("Tile {}", selected_tile))
            .show_ui(ui, |ui| {
                for i in 0..4 {
                    let label = if config.adc_tiles[i].enabled {
                        format!("Tile {} ✓", i)
                    } else {
                        format!("Tile {} ✗", i)
                    };
                    ui.selectable_value(selected_tile, i, label);
                }
            });
    });

    let tile = &mut config.adc_tiles[*selected_tile];

    // Tile enable/disable
    ui.horizontal(|ui| {
        ui.checkbox(&mut tile.enabled, "Tile Enabled");
        if tile.enabled {
            ui.colored_label(Theme::ENABLED, "●");
        } else {
            ui.colored_label(Theme::DISABLED, "●");
        }
    });

    if !tile.enabled {
        ui.colored_label(Theme::TEXT_SECONDARY, "Tile is disabled. Enable to configure.");
        return;
    }

    ui.separator();
    ui.label("Tile Configuration");

    // Sample rate
    ui.horizontal(|ui| {
        ui.label("Sample Rate:");
        ui.add(
            egui::DragValue::new(&mut tile.sample_rate_gsps)
                .range(0.5..=5.0)
                .suffix(" GSPS")
                .speed(0.1),
        );
    });

    ui.label(format!("Nyquist BW: {:.0} MHz", tile.nyquist_bw_mhz()));

    // Nyquist zone
    ui.horizontal(|ui| {
        ui.label("Nyquist Zone:");
        for zone in NyquistZone::ALL {
            ui.selectable_value(&mut tile.nyquist_zone, zone, zone.to_string());
        }
    });

    // PLL
    ui.horizontal(|ui| {
        ui.checkbox(&mut tile.pll_enabled, "PLL Enabled");
        if tile.pll_enabled {
            ui.add(
                egui::DragValue::new(&mut tile.ref_clk_mhz)
                    .range(10.0..=1000.0)
                    .suffix(" MHz")
                    .speed(1.0),
            );
        }
    });

    ui.separator();

    let selected_block_idx = *selected_block;
    let tile_fs = tile.sample_rate_gsps;
    let mut apply_auto_tune: Option<AutoTuneResult> = None;

    // Block selector
    ui.horizontal(|ui| {
        ui.label("Block:");
        ui.selectable_value(selected_block, 0, "Block 0");
        ui.selectable_value(selected_block, 1, "Block 1");
    });

    let block = &mut tile.blocks[selected_block_idx];

    // Block enable/disable
    ui.horizontal(|ui| {
        ui.checkbox(&mut block.enabled, "Block Enabled");
        if block.enabled {
            ui.colored_label(Theme::ENABLED, "●");
        } else {
            ui.colored_label(Theme::DISABLED, "●");
        }
    });

    if !block.enabled {
        ui.colored_label(Theme::TEXT_SECONDARY, "Block is disabled.");
        return;
    }

    ui.separator();
    ui.label("DDC Configuration");

    // Mixer mode
    ui.horizontal(|ui| {
        ui.label("Mixer:");
        egui::ComboBox::from_id_salt("mixer_mode")
            .selected_text(block.mixer_mode.to_string())
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut block.mixer_mode, MixerMode::Bypass, "Bypass");
                for freq in CoarseMixFreq::ALL {
                    ui.selectable_value(
                        &mut block.mixer_mode,
                        MixerMode::CoarseMix(freq),
                        format!("Coarse ({})", freq),
                    );
                }
                ui.selectable_value(&mut block.mixer_mode, MixerMode::FineMix, "Fine (NCO)");
            });
    });

    // NCO frequency (only when fine mix is active)
    if matches!(block.mixer_mode, MixerMode::FineMix) {
        ui.horizontal(|ui| {
            ui.label("NCO Freq:");
            ui.add(
                egui::DragValue::new(&mut block.nco_freq_mhz)
                    .range(-tile_fs * 1000.0 / 2.0..=tile_fs * 1000.0 / 2.0)
                    .suffix(" MHz")
                    .speed(10.0),
            );
        });
    }

    // Decimation
    ui.horizontal(|ui| {
        ui.label("Decimation:");
        egui::ComboBox::from_id_salt("decimation")
            .selected_text(block.decimation.to_string())
            .show_ui(ui, |ui| {
                for dec in DecimationFactor::ALL {
                    ui.selectable_value(&mut block.decimation, dec, dec.to_string());
                }
            });
    });

    let output_rate = block.output_rate_mhz(tile_fs);
    ui.label(format!("Output Rate: {:.1} MHz", output_rate));

    // Calibration mode
    ui.horizontal(|ui| {
        ui.label("Cal Mode:");
        ui.selectable_value(&mut block.calibration_mode, CalibrationMode::Mode1, "Mode 1");
        ui.selectable_value(&mut block.calibration_mode, CalibrationMode::Mode2, "Mode 2");
    });

    ui.separator();
    ui.collapsing("⚡ Auto-Tune & SDR Nyquist Planner", |ui| {
        ui.label(
            egui::RichText::new("Center-tune target RF frequency to 0 Hz complex baseband:")
                .small()
                .color(Theme::TEXT_SECONDARY),
        );

        let mut target_freq = ui.data_mut(|d| {
            *d.get_temp_mut_or_insert_with(egui::Id::new("auto_tune_target_freq"), || 300.0)
        });

        ui.horizontal(|ui| {
            ui.label("RF Target:");
            ui.add(
                egui::DragValue::new(&mut target_freq)
                    .range(1.0..=20000.0)
                    .suffix(" MHz")
                    .speed(10.0),
            );
        });

        ui.data_mut(|d| {
            d.insert_temp(egui::Id::new("auto_tune_target_freq"), target_freq);
        });

        // Compute auto-tune using sample rate from tile_fs
        let temp_tile = AdcTile {
            sample_rate_gsps: tile_fs,
            ..AdcTile::new(0)
        };
        let res = temp_tile.auto_tune(target_freq);

        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label("Zone:");
                ui.colored_label(
                    Theme::ACCENT_PRIMARY,
                    format!("Zone {} ({})", res.zone_index, if res.is_even_zone { "Even" } else { "Odd" }),
                );
            });
            ui.horizontal(|ui| {
                ui.label("Mode:");
                ui.label(res.nyquist_zone.to_string());
            });
            ui.horizontal(|ui| {
                ui.label("Folded Alias:");
                ui.label(format!("{:.1} MHz", res.alias_freq_mhz));
            });
            ui.horizontal(|ui| {
                ui.label("NCO Downshift:");
                ui.colored_label(
                    Theme::ACCENT_SECONDARY,
                    format!("{:.1} MHz", res.nco_freq_mhz),
                );
            });
        });

        if ui.button("⚡ Apply SDR Nyquist Zone & NCO").clicked() {
            block.mixer_mode = MixerMode::FineMix;
            block.nco_freq_mhz = res.nco_freq_mhz;
            apply_auto_tune = Some(res);
        }
    });

    if let Some(res) = apply_auto_tune {
        tile.nyquist_zone = res.nyquist_zone;
    }
}
