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
    /// Real-time simulation state (Play/Pause).
    pub is_running: bool,
    /// Global simulation timestamp in microseconds.
    pub simulation_time_us: f64,
    /// Simulation playback speed multiplier.
    pub sim_speed: f64,
    /// Instant of the last frame update.
    last_update_instant: Option<std::time::Instant>,
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
            is_running: true,
            simulation_time_us: 0.0,
            sim_speed: 1.0,
            last_update_instant: None,
        }
    }
}

impl RfSocSimApp {
    /// Recompute the processed signal for the selected tile/block at current simulation time.
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

        // Evaluate input signal & cumulative transfer function from graph
        let num_samples = 4096;
        let input_sample_rate_mhz = 15000.0; // 15 GHz wideband to support signals up to 7.5 GHz

        let graph_res = crate::node_graph::nodes::evaluate_graph(
            &self.snarl,
            self.selected_tile,
            self.selected_block,
            num_samples,
            input_sample_rate_mhz,
            &self.signal_gen,
            self.simulation_time_us,
        );

        let (samples, rf_chain_response) = match graph_res {
            Some(res) => (
                res.samples,
                Some((res.rf_chain_response_db, res.rf_chain_freq_axis_mhz)),
            ),
            None => (
                self.signal_gen.generate_at_time(num_samples, input_sample_rate_mhz, self.simulation_time_us),
                None,
            ),
        };

        let raw_samples = self.signal_gen.generate_at_time(num_samples, input_sample_rate_mhz, self.simulation_time_us);

        // Process through ADC chain
        self.processed_signal = Some(dsp::process_adc_block(
            &samples,
            input_sample_rate_mhz,
            block,
            tile,
            Some(&raw_samples),
            rf_chain_response,
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

        // Advance simulation clock if running
        let now = std::time::Instant::now();
        if self.is_running {
            let dt_secs = match self.last_update_instant {
                Some(last) => now.duration_since(last).as_secs_f64().min(0.1),
                None => 0.016,
            };
            self.simulation_time_us += dt_secs * 1_000_000.0 * self.sim_speed;
            ui.ctx().request_repaint();
        }
        self.last_update_instant = Some(now);

        // Auto-recompute
        if self.auto_compute {
            self.recompute_signal();
        }

        // Top panel with tabs and realtime simulation controls
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

                    ui.separator();

                    // Speed selector
                    egui::ComboBox::from_id_salt("sim_speed_select")
                        .selected_text(format!("{:.2}x", self.sim_speed))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.sim_speed, 0.1, "0.10x");
                            ui.selectable_value(&mut self.sim_speed, 0.25, "0.25x");
                            ui.selectable_value(&mut self.sim_speed, 0.5, "0.50x");
                            ui.selectable_value(&mut self.sim_speed, 1.0, "1.00x");
                            ui.selectable_value(&mut self.sim_speed, 2.0, "2.00x");
                            ui.selectable_value(&mut self.sim_speed, 5.0, "5.00x");
                        });

                    if ui.button("⏮ Reset").clicked() {
                        self.simulation_time_us = 0.0;
                    }

                    if !self.is_running && ui.button("⏭ Step").clicked() {
                        self.simulation_time_us += 10_000.0 * self.sim_speed; // step 10 ms
                        self.recompute_signal();
                    }

                    let play_label = if self.is_running { "⏸ Pause" } else { "▶ Play" };
                    if ui.button(play_label).clicked() {
                        self.is_running = !self.is_running;
                    }

                    ui.label(
                        egui::RichText::new(format!("⏱ {:.2} ms", self.simulation_time_us / 1000.0))
                            .strong()
                            .color(Theme::ACCENT_SECONDARY),
                    );
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

                            // Modulation type selector
                            ui.horizontal(|ui| {
                                ui.label("Mod:");
                                egui::ComboBox::from_id_salt(format!("tone_mod_{i}"))
                                    .selected_text(tone.modulation.to_string())
                                    .show_ui(ui, |ui| {
                                        if ui.selectable_label(matches!(tone.modulation, crate::signal::ToneModulation::Cw), "CW (Tone)").clicked() {
                                            tone.modulation = crate::signal::ToneModulation::Cw;
                                        }
                                        if ui.selectable_label(matches!(tone.modulation, crate::signal::ToneModulation::SweptChirp { .. }), "FMCW Chirp Sweep").clicked() {
                                            tone.modulation = crate::signal::ToneModulation::SweptChirp { sweep_period_ms: 10.0 };
                                        }
                                        if ui.selectable_label(matches!(tone.modulation, crate::signal::ToneModulation::FmModulated { .. }), "FM Modulated").clicked() {
                                            tone.modulation = crate::signal::ToneModulation::FmModulated { dev_mhz: 50.0, mod_freq_khz: 10.0 };
                                        }
                                        if ui.selectable_label(matches!(tone.modulation, crate::signal::ToneModulation::PulsedRadar { .. }), "Pulsed Radar").clicked() {
                                            tone.modulation = crate::signal::ToneModulation::PulsedRadar { pulse_width_us: 20.0, pri_us: 100.0 };
                                        }
                                        if ui.selectable_label(matches!(tone.modulation, crate::signal::ToneModulation::FreqHopping { .. }), "Frequency Hopping").clicked() {
                                            tone.modulation = crate::signal::ToneModulation::FreqHopping { hop_step_mhz: 50.0, num_channels: 8, hop_rate_hz: 500.0 };
                                        }
                                        if ui.selectable_label(matches!(tone.modulation, crate::signal::ToneModulation::DigitalQpsk { .. }), "Digital QPSK").clicked() {
                                            tone.modulation = crate::signal::ToneModulation::DigitalQpsk { symbol_rate_ksps: 100.0 };
                                        }
                                    });
                            });

                            // Modulation-specific parameters
                            match &mut tone.modulation {
                                crate::signal::ToneModulation::SweptChirp { sweep_period_ms } => {
                                    ui.horizontal(|ui| {
                                        ui.label("BW:");
                                        ui.add(
                                            egui::DragValue::new(&mut tone.bandwidth_mhz)
                                                .range(1.0..=2000.0)
                                                .suffix(" MHz")
                                                .speed(5.0),
                                        );
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Period:");
                                        ui.add(
                                            egui::DragValue::new(sweep_period_ms)
                                                .range(0.1..=1000.0)
                                                .suffix(" ms")
                                                .speed(1.0),
                                        );
                                    });
                                }
                                crate::signal::ToneModulation::FmModulated { dev_mhz, mod_freq_khz } => {
                                    ui.horizontal(|ui| {
                                        ui.label("Dev:");
                                        ui.add(
                                            egui::DragValue::new(dev_mhz)
                                                .range(0.1..=500.0)
                                                .suffix(" MHz")
                                                .speed(1.0),
                                        );
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("f_m:");
                                        ui.add(
                                            egui::DragValue::new(mod_freq_khz)
                                                .range(0.1..=1000.0)
                                                .suffix(" kHz")
                                                .speed(1.0),
                                        );
                                    });
                                }
                                crate::signal::ToneModulation::PulsedRadar { pulse_width_us, pri_us } => {
                                    ui.horizontal(|ui| {
                                        ui.label("PW:");
                                        ui.add(
                                            egui::DragValue::new(pulse_width_us)
                                                .range(0.5..=1000.0)
                                                .suffix(" µs")
                                                .speed(1.0),
                                        );
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("PRI:");
                                        ui.add(
                                            egui::DragValue::new(pri_us)
                                                .range(1.0..=5000.0)
                                                .suffix(" µs")
                                                .speed(5.0),
                                        );
                                    });
                                }
                                crate::signal::ToneModulation::FreqHopping { hop_step_mhz, num_channels, hop_rate_hz } => {
                                    ui.horizontal(|ui| {
                                        ui.label("Step:");
                                        ui.add(
                                            egui::DragValue::new(hop_step_mhz)
                                                .range(1.0..=500.0)
                                                .suffix(" MHz")
                                                .speed(2.0),
                                        );
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Chans:");
                                        ui.add(
                                            egui::DragValue::new(num_channels)
                                                .range(2..=32)
                                                .speed(1),
                                        );
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Rate:");
                                        ui.add(
                                            egui::DragValue::new(hop_rate_hz)
                                                .range(10.0..=10000.0)
                                                .suffix(" Hz")
                                                .speed(50.0),
                                        );
                                    });
                                }
                                crate::signal::ToneModulation::DigitalQpsk { symbol_rate_ksps } => {
                                    ui.horizontal(|ui| {
                                        ui.label("Sym Rate:");
                                        ui.add(
                                            egui::DragValue::new(symbol_rate_ksps)
                                                .range(1.0..=10000.0)
                                                .suffix(" kSPS")
                                                .speed(10.0),
                                        );
                                    });
                                }
                                crate::signal::ToneModulation::Cw => {
                                    ui.horizontal(|ui| {
                                        ui.label("BW:");
                                        ui.add(
                                            egui::DragValue::new(&mut tone.bandwidth_mhz)
                                                .range(0.0..=1000.0)
                                                .suffix(" MHz")
                                                .speed(1.0),
                                        );
                                    });
                                }
                            }
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
                    let nyquist_bw = tile.sample_rate_mhz() / 2.0;
                    let num_zones = (15000.0 / nyquist_bw).ceil() as usize;
                    nyquist_view::show_nyquist_view(
                        ui,
                        tile.sample_rate_mhz(),
                        num_zones,
                        tile.nyquist_zone_index as usize,
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
