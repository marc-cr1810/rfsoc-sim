//! Main application state and eframe::App implementation.

use crate::dsp::{self, ProcessedSignal};
use crate::node_graph::nodes::RfNode;
use crate::node_graph::viewer;
use crate::rfdc::RfdcConfig;
use crate::signal::SignalGenerator;
use crate::ui::{config_panel, nyquist_view, spectrum_view, theme::Theme, tile_overview};
use egui_snarl::Snarl;

/// The active tab in the main content area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Overview,
    RfChain,
    Spectrum,
}

/// Main application state.
pub struct RfSocSimApp {
    /// RFDC configuration (all tiles and blocks).
    pub rfdc: RfdcConfig,
    /// Node graph for the RF front end.
    pub snarl: Snarl<RfNode>,
    /// Currently selected ADC tile index (0–3).
    pub selected_tile: usize,
    /// Currently selected block index (0–1).
    pub selected_block: usize,
    /// Active tab in the main area.
    pub active_tab: Tab,
    /// Cached processed signal for the selected block.
    pub processed_signal: Option<ProcessedSignal>,
    /// Whether to auto-recompute the spectrum.
    pub auto_compute: bool,
    /// Signal generator (used when no node graph source is connected).
    pub signal_gen: SignalGenerator,
    /// Whether the theme has been applied.
    theme_applied: bool,
}

impl Default for RfSocSimApp {
    fn default() -> Self {
        Self {
            rfdc: RfdcConfig::default(),
            snarl: Snarl::new(),
            selected_tile: 0,
            selected_block: 0,
            active_tab: Tab::Overview,
            processed_signal: None,
            auto_compute: true,
            signal_gen: SignalGenerator::default(),
            theme_applied: false,
        }
    }
}

impl RfSocSimApp {
    /// Recompute the processed signal for the selected tile/block.
    fn recompute_signal(&mut self) {
        let tile = &self.rfdc.adc_tiles[self.selected_tile];
        if !tile.enabled {
            self.processed_signal = None;
            return;
        }

        let block = &tile.blocks[self.selected_block];
        if !block.enabled {
            self.processed_signal = None;
            return;
        }

        // Generate or evaluate input signal from graph
        let num_samples = 4096;
        let input_sample_rate_mhz = 10000.0; // 10 GHz wideband

        let samples = crate::node_graph::nodes::evaluate_graph(
            &self.snarl,
            self.selected_tile,
            self.selected_block,
            num_samples,
            input_sample_rate_mhz,
        )
        .unwrap_or_else(|| self.signal_gen.generate(num_samples, input_sample_rate_mhz));

        // Process through ADC chain
        self.processed_signal = Some(dsp::process_adc_block(
            &samples,
            input_sample_rate_mhz,
            block,
            tile,
        ));
    }
}

impl eframe::App for RfSocSimApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Apply theme once
        if !self.theme_applied {
            Theme::apply(ui.ctx());
            self.theme_applied = true;
        }

        // Auto-recompute
        if self.auto_compute {
            self.recompute_signal();
        }

        // Top panel with tabs
        egui::Panel::top("top_panel").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading(
                    egui::RichText::new("RFSoC Simulator")
                        .strong()
                        .color(Theme::ACCENT_PRIMARY),
                );
                ui.label(
                    egui::RichText::new("ZU48DR / ZCU208")
                        .color(Theme::TEXT_SECONDARY)
                        .italics(),
                );
                ui.separator();

                ui.selectable_value(&mut self.active_tab, Tab::Overview, "📋 Overview");
                ui.selectable_value(&mut self.active_tab, Tab::RfChain, "🔗 RF Chain");
                ui.selectable_value(&mut self.active_tab, Tab::Spectrum, "📊 Spectrum");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut self.auto_compute, "Auto-compute");
                    if ui.button("🔄 Recompute").clicked() {
                        self.recompute_signal();
                    }
                });
            });
        });

        // Right side panel — RFDC config
        egui::Panel::right("config_panel")
            .min_size(280.0)
            .default_size(320.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    config_panel::show_config_panel(
                        ui,
                        &mut self.rfdc,
                        &mut self.selected_tile,
                        &mut self.selected_block,
                    );

                    ui.separator();

                    // Quick signal generator controls
                    ui.heading("🎵 Signal Generator");
                    ui.separator();

                    let sig = &mut self.signal_gen;

                    // Number of tones
                    ui.horizontal(|ui| {
                        ui.label("Tones:");
                        if ui.button("+").clicked() {
                            sig.tones.push(crate::signal::Tone::default());
                        }
                        if sig.tones.len() > 1 && ui.button("−").clicked() {
                            sig.tones.pop();
                        }
                    });

                    for (i, tone) in sig.tones.iter_mut().enumerate() {
                        ui.group(|ui| {
                            ui.label(format!("Tone {}", i));
                            ui.horizontal(|ui| {
                                ui.label("f:");
                                ui.add(
                                    egui::DragValue::new(&mut tone.frequency_mhz)
                                        .range(0.1..=10000.0)
                                        .suffix(" MHz")
                                        .speed(10.0),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.label("A:");
                                ui.add(
                                    egui::DragValue::new(&mut tone.amplitude_dbfs)
                                        .range(-120.0..=0.0)
                                        .suffix(" dBFS")
                                        .speed(0.5),
                                );
                            });
                        });
                    }

                    ui.horizontal(|ui| {
                        ui.checkbox(&mut sig.noise_enabled, "Noise");
                        if sig.noise_enabled {
                            ui.add(
                                egui::DragValue::new(&mut sig.noise_floor_dbfs)
                                    .range(-200.0..=0.0)
                                    .suffix(" dBFS")
                                    .speed(1.0),
                            );
                        }
                    });
                });
            });

        // Central region content — determined by active tab
        match self.active_tab {
            Tab::Overview => {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    tile_overview::show_tile_overview(
                        ui,
                        &self.rfdc,
                        &mut self.selected_tile,
                        &mut self.selected_block,
                    );

                    ui.add_space(16.0);

                    let tile = &self.rfdc.adc_tiles[self.selected_tile];
                    nyquist_view::show_nyquist_view(
                        ui,
                        tile.sample_rate_mhz(),
                        4,
                        match tile.nyquist_zone {
                            crate::rfdc::NyquistZone::First => 1,
                            crate::rfdc::NyquistZone::Second => 2,
                        },
                    );
                });
            }
            Tab::RfChain => {
                viewer::show_node_graph(ui, &mut self.snarl);
            }
            Tab::Spectrum => {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let tile = &self.rfdc.adc_tiles[self.selected_tile];
                    spectrum_view::show_spectrum_view(
                        ui,
                        &self.processed_signal,
                        tile.sample_rate_mhz(),
                    );
                });
            }
        }
    }
}
