//! Configuration sidebar for tile/block parameters.

use crate::rfdc::*;
use crate::ui::theme::Theme;

fn help_label(ui: &mut egui::Ui, text: &str, help_text: &str) {
    ui.label(text);
    ui.label(egui::RichText::new(egui_phosphor::regular::INFO).color(Theme::TEXT_SECONDARY))
        .on_hover_text(help_text);
}

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
                        format!("Tile {} {}", i, egui_phosphor::regular::CHECK)
                    } else {
                        format!("Tile {} {}", i, egui_phosphor::regular::X)
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
            ui.colored_label(Theme::ENABLED, egui_phosphor::regular::CIRCLE);
        } else {
            ui.colored_label(Theme::DISABLED, egui_phosphor::regular::CIRCLE);
        }
    });

    if !tile.enabled {
        ui.colored_label(Theme::TEXT_SECONDARY, "Tile is disabled. Enable to configure.");
        return;
    }

    ui.separator();
    ui.label("Tile Configuration");

    // Sample rate & PLL
    ui.horizontal(|ui| {
        help_label(ui, "Sample Rate:", "The physical sampling rate of the ADC. Must be within hardware limits (e.g., 0.5 to 10 GSPS).");
        ui.add(
            egui::DragValue::new(&mut tile.sample_rate_gsps)
                .range(0.5..=10.0)
                .suffix(" GSPS")
                .speed(0.1),
        );
    });

    ui.horizontal(|ui| {
        ui.checkbox(&mut tile.pll_enabled, "Internal PLL");
        ui.label(egui::RichText::new(egui_phosphor::regular::INFO).color(Theme::TEXT_SECONDARY)).on_hover_text("Enables the on-chip phase-locked loop (PLL) to generate the sampling clock from a lower frequency reference clock.");
        if tile.pll_enabled {
            ui.add(
                egui::DragValue::new(&mut tile.ref_clk_mhz)
                    .range(10.0..=1000.0)
                    .suffix(" MHz Ref")
                    .speed(1.0),
            );
        }
    });

    if let Some(err) = tile.validate_pll() {
        ui.colored_label(Theme::ACCENT_ERROR, err);
    } else if tile.pll_enabled {
        let mult = tile.sample_rate_mhz() / tile.ref_clk_mhz;
        ui.colored_label(Theme::TEXT_SECONDARY, format!("PLL Multiplier: {:.2}x", mult));
    }

    ui.label(format!("Nyquist BW: {:.0} MHz", tile.nyquist_bw_mhz()));



    ui.separator();

    let selected_block_idx = *selected_block;
    let tile_fs = tile.sample_rate_gsps;


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
            ui.colored_label(Theme::ENABLED, egui_phosphor::regular::CIRCLE);
        } else {
            ui.colored_label(Theme::DISABLED, egui_phosphor::regular::CIRCLE);
        }
    });

    if !block.enabled {
        ui.colored_label(Theme::TEXT_SECONDARY, "Block is disabled.");
        return;
    }

    ui.separator();
    ui.label("DDC Configuration");

    let errors = block.validate(tile_fs * 1000.0);
    if !errors.is_empty() {
        ui.group(|ui| {
            ui.colored_label(Theme::ACCENT_ERROR, "Hardware Configuration Errors:");
            for err in errors {
                ui.colored_label(Theme::ACCENT_ERROR, format!("• {}", err));
            }
        });
        ui.separator();
    }

    // Nyquist zone
    ui.horizontal(|ui| {
        help_label(ui, "Planner Zone:", "The target Nyquist zone you want to operate in. The hardware will automatically configure the physical Nyquist Zone (Even/Odd) based on this selection.");
        egui::ComboBox::from_id_salt(format!("nyquist_zone_b{}", block.index))
            .selected_text(format!("Zone {}", block.planner_zone))
            .show_ui(ui, |ui| {
                for zone_idx in 1..=16 {
                    if ui.selectable_value(&mut block.planner_zone, zone_idx, format!("Zone {}", zone_idx)).clicked() {
                        block.nyquist_zone = if zone_idx % 2 == 0 { NyquistZone::Even } else { NyquistZone::Odd };
                    }
                }
            });
    });
    ui.label(format!("Hardware Zone: {}", block.nyquist_zone));

    // DSA Attenuation
    ui.horizontal(|ui| {
        help_label(ui, "DSA Attn:", "Digital Step Attenuator. Applies analog attenuation at the RF front-end before the ADC to prevent clipping (0 to 27 dB).");
        ui.add(
            egui::DragValue::new(&mut block.dsa_db)
                .range(0.0..=27.0)
                .suffix(" dB")
                .speed(1.0),
        );
    });

    // Mixer Type & Mode
    ui.horizontal(|ui| {
        help_label(ui, "Mixer Type:", "Type of digital down-conversion mixing. Coarse mixing uses fixed frequency steps, Fine mixing allows arbitrary NCO frequencies.");
        egui::ComboBox::from_id_salt("mixer_type")
            .selected_text(block.mixer_settings.mixer_type.to_string())
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut block.mixer_settings.mixer_type, MixerType::Off, "Off");
                ui.selectable_value(&mut block.mixer_settings.mixer_type, MixerType::Coarse, "Coarse");
                ui.selectable_value(&mut block.mixer_settings.mixer_type, MixerType::Fine, "Fine");
            });
    });

    if block.mixer_settings.mixer_type != MixerType::Off {
        ui.horizontal(|ui| {
            help_label(ui, "Mixer Mode:", "Determines the signal path (Real to I/Q, I/Q to I/Q, etc.) for the digital mixer.");
            egui::ComboBox::from_id_salt("mixer_mode")
                .selected_text(block.mixer_settings.mixer_mode.to_string())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut block.mixer_settings.mixer_mode, MixerMode::RealToReal, "Real -> Real");
                    ui.selectable_value(&mut block.mixer_settings.mixer_mode, MixerMode::RealToIq, "Real -> I/Q");
                    ui.selectable_value(&mut block.mixer_settings.mixer_mode, MixerMode::IqToIq, "I/Q -> I/Q");
                    ui.selectable_value(&mut block.mixer_settings.mixer_mode, MixerMode::ComplexToReal, "I/Q -> Real");
                });
        });

        if block.mixer_settings.mixer_type == MixerType::Coarse {
            ui.horizontal(|ui| {
                help_label(ui, "Coarse Freq:", "Fixed frequency shift for coarse mixing (Fs/2, Fs/4, -Fs/4, etc.).");
                egui::ComboBox::from_id_salt("coarse_mix_freq")
                    .selected_text(block.mixer_settings.coarse_mix_freq.to_string())
                    .show_ui(ui, |ui| {
                        for freq in CoarseMixFreq::ALL {
                            ui.selectable_value(&mut block.mixer_settings.coarse_mix_freq, freq, freq.to_string());
                        }
                    });
            });
        }

        if block.mixer_settings.mixer_type == MixerType::Fine {
            ui.horizontal(|ui| {
                help_label(ui, "NCO Freq:", "Numerically Controlled Oscillator frequency for fine mixing. Shifts the spectrum by this exact amount.");
                ui.add(
                    egui::DragValue::new(&mut block.mixer_settings.freq)
                        .range(-tile_fs * 1000.0 / 2.0..=tile_fs * 1000.0 / 2.0)
                        .suffix(" MHz")
                        .speed(10.0),
                );
            });
            ui.horizontal(|ui| {
                help_label(ui, "NCO Phase:", "Initial phase offset for the NCO (in degrees). Useful for aligning multiple channels.");
                ui.add(
                    egui::DragValue::new(&mut block.mixer_settings.phase_offset)
                        .range(-180.0..=180.0)
                        .suffix(" °")
                        .speed(1.0),
                );
            });
        }
        
        ui.horizontal(|ui| {
            help_label(ui, "Mixer Scale:", "Scaling factor applied after the mixer to prevent overflow or maximize dynamic range. Auto is recommended.");
            egui::ComboBox::from_id_salt("mixer_scale")
                .selected_text(block.mixer_settings.fine_mixer_scale.to_string())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut block.mixer_settings.fine_mixer_scale, FineMixerScale::Auto, "Auto");
                    ui.selectable_value(&mut block.mixer_settings.fine_mixer_scale, FineMixerScale::OnePointZero, "1.0");
                    ui.selectable_value(&mut block.mixer_settings.fine_mixer_scale, FineMixerScale::ZeroPointSeven, "0.7");
                });
        });
    }

    // Decimation
    ui.horizontal(|ui| {
        help_label(ui, "Decimation:", "Reduces the sample rate by this factor after mixing. Essential for lowering the data rate sent to the FPGA fabric.");
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

    ui.horizontal(|ui| {
        help_label(ui, "AXI Words/Clk:", "Number of digital samples transferred per FPGA clock cycle on the AXI-Stream interface.");
        egui::ComboBox::from_id_salt("axi_words")
            .selected_text(block.axi_words_per_clock.to_string())
            .show_ui(ui, |ui| {
                for &words in &[1, 2, 4, 8, 16] {
                    ui.selectable_value(&mut block.axi_words_per_clock, words, words.to_string());
                }
            });
    });

    // Calibration mode
    ui.horizontal(|ui| {
        help_label(ui, "Cal Mode:", "Background calibration mode for the ADC. Mode 1 is standard, Mode 2 is optimized for specific frequency plans.");
        ui.selectable_value(&mut block.calibration_mode, CalibrationMode::Mode1, "Mode 1");
        ui.selectable_value(&mut block.calibration_mode, CalibrationMode::Mode2, "Mode 2");
    });

    ui.separator();
    ui.collapsing("⚖ QMC Settings", |ui| {
        ui.horizontal(|ui| {
            help_label(ui, "Gain:", "Quadrature Modulation Correction (QMC) gain adjustment to compensate for I/Q amplitude imbalance.");
            ui.add(egui::DragValue::new(&mut block.qmc_settings.gain).speed(0.01));
        });
        ui.horizontal(|ui| {
            help_label(ui, "Phase:", "QMC phase adjustment (in degrees) to correct I/Q phase imbalance.");
            ui.add(egui::DragValue::new(&mut block.qmc_settings.phase).suffix(" °").speed(0.1));
        });
        ui.horizontal(|ui| {
            help_label(ui, "Offset:", "DC offset correction for the signal.");
            ui.add(egui::DragValue::new(&mut block.qmc_settings.offset).speed(1.0));
        });
    });

    ui.separator();
    ui.collapsing("🔬 ADC Hardware Non-Idealities", |ui| {
        let non = &mut block.non_idealities;
        ui.checkbox(&mut non.enabled, "Enable Hardware Distortion");
        if non.enabled {
            ui.horizontal(|ui| {
                help_label(ui, "ENOB:", "Effective Number Of Bits. Models the true dynamic range of the ADC by adding broadband noise.");
                ui.add(
                    egui::DragValue::new(&mut non.enob)
                        .range(4.0..=16.0)
                        .speed(0.1)
                        .suffix(" bits"),
                );
            });
            ui.horizontal(|ui| {
                help_label(ui, "Quantization:", "Simulates the bit depth of the ADC (typically 12 or 14 bits for RFSoC). Truncates the analog signal precision.");
                ui.add(
                    egui::DragValue::new(&mut non.quantization_bits)
                        .range(4..=16)
                        .speed(1)
                        .suffix(" bits"),
                );
            });
            ui.horizontal(|ui| {
                help_label(ui, "HD2:", "Second Harmonic Distortion (in dBc). Adds a spurious signal at 2x the fundamental frequency.");
                ui.add(
                    egui::DragValue::new(&mut non.hd2_dbc)
                        .range(-150.0..=0.0)
                        .speed(1.0)
                        .suffix(" dBc"),
                );
            });
            ui.horizontal(|ui| {
                help_label(ui, "HD3:", "Third Harmonic Distortion (in dBc). Adds a spurious signal at 3x the fundamental frequency.");
                ui.add(
                    egui::DragValue::new(&mut non.hd3_dbc)
                        .range(-150.0..=0.0)
                        .speed(1.0)
                        .suffix(" dBc"),
                );
            });
            ui.horizontal(|ui| {
                help_label(ui, "IL Spur:", "Interleaving Spur (in dBc). Simulates artifacts caused by the time-interleaved sub-ADCs in the RFSoC.");
                ui.add(
                    egui::DragValue::new(&mut non.interleaving_spur_dbc)
                        .range(-150.0..=0.0)
                        .speed(1.0)
                        .suffix(" dBc"),
                );
            });
        }
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

        // Preview auto-tune result using tile sample rate
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
            block.auto_tune(tile_fs, target_freq);
        }
    });
}
