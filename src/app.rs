//! Main application state and eframe::App implementation.

use crate::dsp::{self, ProcessedSignal};
use crate::node_graph::nodes::{ChainEnvironment, RfNode};
use crate::node_graph::viewer::{self, GraphAnnotations};
use crate::rfdc::RfdcConfig;
use crate::signal::SignalGenerator;
use crate::ui::{config_panel, nyquist_view, spectrum_view, theme::Theme, tile_overview};
use egui_snarl::Snarl;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct SimulatorState {
    snarl: Snarl<RfNode>,
    rfdc: RfdcConfig,
    #[serde(default)]
    chain_env: ChainEnvironment,
}

/// Wideband simulation rate in MHz, high enough to carry signals up to 7.5 GHz.
pub const SIM_SAMPLE_RATE_MHZ: f64 = 15000.0;

/// Cascaded RF budget of the chain feeding the selected block.
#[derive(Debug, Clone, Copy)]
pub struct ChainBudget {
    pub gain_db: f64,
    pub noise_figure_db: f64,
    pub oip3_dbm: f64,
    pub analysis_freq_mhz: f64,
    pub compressing: bool,
    pub has_cycle: bool,
}

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
    /// FFT window used for every displayed spectrum. A display choice, not hardware — it
    /// trades main-lobe width against how far a tone's leakage skirt spreads across the trace.
    pub display_window: dsp::FftWindow,
    /// Signal generator (used when no node graph source is connected).
    pub signal_gen: SignalGenerator,
    /// Physical environment of the RF chain: temperature and thermal noise.
    pub chain_env: ChainEnvironment,
    /// Cascaded budget of the chain, from the last evaluation.
    pub chain_budget: Option<ChainBudget>,
    /// Per-node annotations painted onto the graph.
    pub graph_annotations: GraphAnnotations,
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
        let mut snarl = Snarl::new();
        
        let src_id = snarl.insert_node(
            egui::pos2(-400.0, 0.0),
            RfNode::SignalSource(crate::node_graph::nodes::SignalSourceNode::default()),
        );
        let adc_id = snarl.insert_node(
            egui::pos2(400.0, 0.0),
            RfNode::AdcInput(crate::node_graph::nodes::AdcInputNode::default()),
        );
        snarl.connect(
            egui_snarl::OutPinId { node: src_id, output: 0 },
            egui_snarl::InPinId { node: adc_id, input: 0 },
        );

        Self {
            rfdc: RfdcConfig::default(),
            snarl,
            selected_tile: 0,
            selected_block: 0,
            active_tab: Tab::Overview,
            processed_signal: None,
            auto_compute: true,
            display_window: dsp::DEFAULT_DISPLAY_WINDOW,
            signal_gen: SignalGenerator::default(),
            chain_env: ChainEnvironment::default(),
            chain_budget: None,
            graph_annotations: GraphAnnotations::default(),
            theme_applied: false,
            is_running: true,
            simulation_time_us: 0.0,
            sim_speed: 1.0,
            last_update_instant: None,
        }
    }
}

impl RfSocSimApp {
    /// Frequency of the strongest tone driving the graph, used as the RF budget's reference.
    ///
    /// A stage's gain, loss and noise figure all depend on frequency, so a cascaded figure
    /// quoted at a fixed frequency says nothing useful about a chain carrying a signal
    /// somewhere else — a 1 GHz low-pass looks lossless right up until the tone moves.
    fn dominant_tone_mhz(&self) -> f64 {
        let mut best: Option<(f64, f64)> = None; // (amplitude dBFS, frequency)
        let mut consider = |source: &SignalGenerator| {
            for tone in &source.tones {
                if best.is_none_or(|(a, _)| tone.amplitude_dbfs > a) {
                    best = Some((tone.amplitude_dbfs, tone.frequency_mhz));
                }
            }
        };

        // Local sources on the graph take priority over the global generator.
        let mut saw_local = false;
        for (_, node) in self.snarl.node_ids() {
            if let RfNode::SignalSource(src) = node {
                if matches!(
                    src.source_type,
                    crate::node_graph::nodes::SourceType::LocalGenerator
                ) {
                    consider(&src.generator);
                    saw_local = true;
                }
            }
        }
        if !saw_local {
            consider(&self.signal_gen);
        }

        best.map(|(_, f)| f).unwrap_or(1000.0).max(1.0)
    }

    /// Toolbar above the node graph: the chain's environment and its cascaded RF budget.
    fn show_chain_toolbar(&mut self, ui: &mut egui::Ui) {
        egui::Frame::new()
            .fill(Theme::BG_CARD)
            .inner_margin(egui::Margin::symmetric(10, 6))
            .corner_radius(6.0)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut self.chain_env.thermal_noise, "Thermal noise")
                        .on_hover_text(
                            "Each stage contributes (F-1)·G·kTB of its own noise, so where an \
                             LNA sits in the chain changes the SNR reaching the converter.",
                        );
                    ui.add_enabled_ui(self.chain_env.thermal_noise, |ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.chain_env.temperature_k)
                                .range(4.0..=500.0)
                                .suffix(" K")
                                .speed(1.0),
                        )
                        .on_hover_text("Physical temperature of the chain. 290 K is the IEEE reference.");
                    });

                    ui.separator();

                    match self.chain_budget {
                        Some(b) => {
                            let metric = |ui: &mut egui::Ui,
                                          label: &str,
                                          value: String,
                                          color: egui::Color32,
                                          hover: &str| {
                                ui.label(
                                    egui::RichText::new(label)
                                        .small()
                                        .color(Theme::TEXT_SECONDARY),
                                );
                                ui.label(egui::RichText::new(value).strong().color(color))
                                    .on_hover_text(hover);
                                ui.add_space(6.0);
                            };

                            metric(
                                ui,
                                "Gain",
                                format!("{:+.2} dB", b.gain_db),
                                Theme::TEXT_PRIMARY,
                                "Cascaded gain of every stage feeding the selected ADC block.",
                            );
                            metric(
                                ui,
                                "NF",
                                format!("{:.2} dB", b.noise_figure_db),
                                if b.noise_figure_db > 10.0 {
                                    Theme::ACCENT_WARN
                                } else {
                                    Theme::ACCENT_SECONDARY
                                },
                                "Friis cascaded noise figure. The first stage dominates, which \
                                 is why an LNA belongs as close to the antenna as possible.",
                            );
                            metric(
                                ui,
                                "OIP3",
                                if b.oip3_dbm.is_finite() {
                                    format!("{:+.1} dBm", b.oip3_dbm)
                                } else {
                                    "linear".to_string()
                                },
                                Theme::TEXT_PRIMARY,
                                "Cascaded output third-order intercept. Passive stages are taken \
                                 as ideally linear and do not degrade it.",
                            );
                            metric(
                                ui,
                                "@",
                                format!("{:.0} MHz", b.analysis_freq_mhz),
                                Theme::ACCENT_PRIMARY,
                                "The budget above is evaluated here, at the strongest tone \
                                 driving the chain.",
                            );

                            if b.compressing {
                                ui.label(
                                    egui::RichText::new("⚠ stage compressing")
                                        .strong()
                                        .color(Theme::ACCENT_WARN),
                                )
                                .on_hover_text(
                                    "A stage is more than 1 dB into compression. Reduce the \
                                     drive level or raise its P1dB.",
                                );
                            }
                            if b.has_cycle {
                                ui.label(
                                    egui::RichText::new("⚠ feedback loop")
                                        .strong()
                                        .color(Theme::ACCENT_ERROR),
                                )
                                .on_hover_text(
                                    "Part of the graph is wired in a loop and cannot be \
                                     evaluated by a forward-only chain model.",
                                );
                            }
                        }
                        None => {
                            ui.label(
                                egui::RichText::new(format!(
                                    "No chain reaches Tile {} Block {} — add an ADC Input node \
                                     and wire a source to it",
                                    self.selected_tile, self.selected_block
                                ))
                                .small()
                                .color(Theme::ACCENT_WARN),
                            );
                        }
                    }
                });
            });
        ui.add_space(4.0);
    }

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

        let input_sample_rate_mhz = SIM_SAMPLE_RATE_MHZ;

        // The ADC-rate and DDC FFTs consume samples at the *tile* rate, so the wideband
        // buffer has to be scaled by the oversampling ratio — sizing it in wideband samples
        // starves those FFTs of resolution whenever Fs is well below the simulation rate.
        let oversampling = (input_sample_rate_mhz / tile.sample_rate_mhz().max(1.0)).max(1.0);
        let needed_tile_samples = dsp::required_tile_samples(block.decimation.factor());
        let num_samples = dsp::next_smooth_size(
            ((needed_tile_samples as f64 * oversampling).ceil() as usize).clamp(4096, 131_072),
        );

        // The RF budget is only meaningful at a frequency, so report it where the signal
        // actually is: the strongest tone driving this chain.
        self.chain_env.analysis_freq_mhz = self.dominant_tone_mhz();

        let graph_res = crate::node_graph::nodes::evaluate_graph(
            &self.snarl,
            self.selected_tile,
            self.selected_block,
            num_samples,
            input_sample_rate_mhz,
            &self.signal_gen,
            self.simulation_time_us,
            &self.chain_env,
        );

        let (samples, rf_chain_response) = match graph_res {
            Some(res) => {
                self.chain_budget = Some(ChainBudget {
                    gain_db: res.cascaded_gain_db,
                    noise_figure_db: res.cascaded_nf_db,
                    oip3_dbm: res.cascaded_oip3_dbm,
                    analysis_freq_mhz: res.analysis_freq_mhz,
                    compressing: res.compressing,
                    has_cycle: !res.cycle_nodes.is_empty(),
                });
                self.graph_annotations = GraphAnnotations {
                    stats: res.node_stats,
                    cycle_nodes: res.cycle_nodes,
                    analysis_freq_mhz: res.analysis_freq_mhz,
                };
                (
                    res.samples,
                    Some((res.rf_chain_response_db, res.rf_chain_freq_axis_mhz)),
                )
            }
            None => {
                self.chain_budget = None;
                self.graph_annotations = GraphAnnotations {
                    analysis_freq_mhz: self.chain_env.analysis_freq_mhz,
                    ..Default::default()
                };
                let empty_gen = SignalGenerator {
                    tones: vec![],
                    noise_floor_dbfs: self.signal_gen.noise_floor_dbfs,
                    noise_enabled: self.signal_gen.noise_enabled,
                };
                (
                    empty_gen.generate_at_time(num_samples, input_sample_rate_mhz, self.simulation_time_us),
                    None,
                )
            }
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
            self.display_window,
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

                    if ui.button("📂 Load").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("JSON Graph", &["json"])
                            .pick_file()
                        {
                            if let Ok(content) = std::fs::read_to_string(&path) {
                                if let Ok(state) = serde_json::from_str::<SimulatorState>(&content) {
                                    self.snarl = state.snarl;
                                    self.rfdc = state.rfdc;
                                    self.chain_env = state.chain_env;
                                    self.recompute_signal();
                                }
                            }
                        }
                    }

                    if ui.button("💾 Save").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("JSON Graph", &["json"])
                            .save_file()
                        {
                            let state = SimulatorState {
                                snarl: self.snarl.clone(),
                                rfdc: self.rfdc.clone(),
                                chain_env: self.chain_env,
                            };
                            if let Ok(content) = serde_json::to_string_pretty(&state) {
                                let _ = std::fs::write(&path, content);
                            }
                        }
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

                    crate::ui::tone_editor::generator_editor(
                        ui,
                        "sidebar",
                        &mut self.signal_gen,
                        SIM_SAMPLE_RATE_MHZ,
                    );
                });
            });

        // Central region content — determined by active tab
        // The window picker lives in the spectrum pane, so its change has to be reported back
        // out here where recompute is reachable.
        let mut window_changed = false;

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
                    let block = &tile.blocks[self.selected_block];
                    let nyquist_bw = tile.sample_rate_mhz() / 2.0;
                    let num_zones = (15000.0 / nyquist_bw).ceil() as usize;
                    nyquist_view::show_nyquist_view(
                        ui,
                        tile.sample_rate_mhz(),
                        num_zones,
                        block.planner_zone as usize,
                    );
                });
            }
            Tab::RfChain => {
                self.show_chain_toolbar(ui);
                viewer::show_node_graph(ui, &mut self.snarl, &self.graph_annotations);
            }
            Tab::Spectrum => {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let tile = &self.rfdc.adc_tiles[self.selected_tile];
                    let fs_mhz = tile.sample_rate_mhz();
                    window_changed = spectrum_view::show_spectrum_view(
                        ui,
                        &self.processed_signal,
                        fs_mhz,
                        &mut self.display_window,
                    );
                });
            }
        }

        // Auto-compute already refreshes every frame; this covers the manual case.
        if window_changed && !self.auto_compute {
            self.recompute_signal();
        }
    }
}
