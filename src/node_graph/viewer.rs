//! SnarlViewer implementation for the RF front end node graph.

use super::components::*;
use super::nodes::*;
use crate::ui::theme::Theme;
use egui_snarl::ui::{PinInfo, SnarlStyle, SnarlViewer, SnarlWidget};
use egui_snarl::{InPin, NodeId, OutPin, Snarl};

/// Wire color for RF signal connections.
const RF_WIRE_COLOR: egui::Color32 = egui::Color32::from_rgb(0, 180, 255);
/// Wire color for a coupler's tap-off arm.
const COUPLED_WIRE_COLOR: egui::Color32 = egui::Color32::from_rgb(180, 140, 255);

/// Results of the last chain evaluation, so each node can annotate itself with what it is
/// actually doing to the signal rather than only what it is configured to do.
#[derive(Default, Clone)]
pub struct GraphAnnotations {
    pub stats: Vec<(NodeId, NodeStats)>,
    pub cycle_nodes: Vec<NodeId>,
    pub analysis_freq_mhz: f64,
}

impl GraphAnnotations {
    fn get(&self, id: NodeId) -> Option<NodeStats> {
        self.stats.iter().find(|(n, _)| *n == id).map(|(_, s)| *s)
    }

    fn in_cycle(&self, id: NodeId) -> bool {
        self.cycle_nodes.contains(&id)
    }
}

/// Our viewer that implements SnarlViewer for RfNode.
pub struct RfNodeViewer<'a> {
    annotations: &'a GraphAnnotations,
}

/// Small dimmed caption used for the live readouts under each node's controls.
fn readout(ui: &mut egui::Ui, text: impl Into<String>) {
    ui.add(
        egui::Label::new(
            egui::RichText::new(text.into())
                .small()
                .color(Theme::TEXT_SECONDARY),
        )
        .wrap_mode(egui::TextWrapMode::Extend),
    );
}

/// Coloured caption, for anything the user needs to notice.
fn warning(ui: &mut egui::Ui, text: impl Into<String>, color: egui::Color32) {
    ui.add(
        egui::Label::new(egui::RichText::new(text.into()).small().strong().color(color))
            .wrap_mode(egui::TextWrapMode::Extend),
    );
}

impl SnarlViewer<RfNode> for RfNodeViewer<'_> {
    fn title(&mut self, node: &RfNode) -> String {
        node.title().to_string()
    }

    fn inputs(&mut self, node: &RfNode) -> usize {
        node.num_inputs()
    }

    fn outputs(&mut self, node: &RfNode) -> usize {
        node.num_outputs()
    }

    fn header_frame(
        &mut self,
        default: egui::Frame,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        snarl: &Snarl<RfNode>,
    ) -> egui::Frame {
        let n = &snarl[node];
        let mut color = match n {
            RfNode::SignalSource(_) => Theme::NODE_SOURCE,
            RfNode::Balun(_)
            | RfNode::Filter(_)
            | RfNode::Attenuator(_)
            | RfNode::Splitter(_)
            | RfNode::Combiner(_)
            | RfNode::PhaseShifter(_)
            | RfNode::DirectionalCoupler(_)
            | RfNode::S2p(_) => Theme::NODE_PASSIVE,
            RfNode::Amplifier(_) | RfNode::Mixer(_) => Theme::NODE_ACTIVE,
            RfNode::AdcInput(_) => Theme::NODE_SINK,
        };
        // A stage in trouble recolours its own header, so it is visible without reading text.
        if self.annotations.in_cycle(node) {
            color = Theme::ACCENT_ERROR;
        } else if self
            .annotations
            .get(node)
            .is_some_and(|s| s.compression_db < -1.0)
        {
            color = Theme::ACCENT_WARN;
        }
        default
            .fill(color.linear_multiply(0.25))
            .inner_margin(egui::Margin::same(6))
    }

    fn show_header(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<RfNode>,
    ) {
        let n = &snarl[node];
        let icon = match n {
            RfNode::SignalSource(_) => egui_phosphor::regular::WAVE_SAWTOOTH,
            RfNode::Balun(_) => egui_phosphor::regular::ARROWS_LEFT_RIGHT,
            RfNode::Filter(_) => egui_phosphor::regular::FUNNEL,
            RfNode::Amplifier(_) => egui_phosphor::regular::SPEAKER_HIFI,
            RfNode::Attenuator(_) => egui_phosphor::regular::SLIDERS_HORIZONTAL,
            RfNode::Splitter(_) => egui_phosphor::regular::GIT_MERGE,
            RfNode::Combiner(_) => egui_phosphor::regular::ARROWS_IN_SIMPLE,
            RfNode::Mixer(_) => egui_phosphor::regular::WAVES,
            RfNode::PhaseShifter(_) => egui_phosphor::regular::CLOCK,
            RfNode::DirectionalCoupler(_) => egui_phosphor::regular::ROWS,
            RfNode::S2p(_) => egui_phosphor::regular::FILE_TEXT,
            RfNode::AdcInput(_) => egui_phosphor::regular::CPU,
        };
        let title = n.title().to_string();
        let stats = self.annotations.get(node);

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(icon).strong().color(Theme::TEXT_PRIMARY));
            ui.label(egui::RichText::new(title).strong().color(Theme::TEXT_PRIMARY));

            // Signal level leaving this stage, so a budget can be read straight off the graph.
            if let Some(s) = stats {
                let level = if s.output_level_dbfs > -299.0 {
                    format!("{:+.1} dBFS", s.output_level_dbfs)
                } else {
                    "—".to_string()
                };
                let color = if s.compression_db < -1.0 {
                    Theme::ACCENT_WARN
                } else {
                    Theme::TEXT_SECONDARY
                };
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(level).small().color(color))
                        .on_hover_text(format!(
                            "RMS level out of this stage at {:.0} MHz.\nCumulative chain gain: {:+.2} dB",
                            self.annotations.analysis_freq_mhz, s.cumulative_gain_db
                        ));
                });
            }
        });
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<RfNode>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        let node = &snarl[pin.id.node];
        match node {
            RfNode::Combiner(_) => {
                ui.label(format!("In {}", pin.id.input));
            }
            _ => {
                ui.label("RF In");
            }
        }
        PinInfo::circle()
            .with_fill(RF_WIRE_COLOR)
            .with_stroke(egui::Stroke::NONE)
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<RfNode>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        let node = &snarl[pin.id.node];
        let label = node.output_label(pin.id.output);
        let coupled = matches!(node, RfNode::DirectionalCoupler(_))
            && pin.id.output == COUPLER_COUPLED_PORT;
        let resp = ui.label(label);
        if coupled {
            resp.on_hover_text("Tap-off arm, at the coupling factor below the input.");
        }
        PinInfo::circle()
            .with_fill(if coupled {
                COUPLED_WIRE_COLOR
            } else {
                RF_WIRE_COLOR
            })
            .with_stroke(egui::Stroke::NONE)
    }

    fn has_body(&mut self, _node: &RfNode) -> bool {
        true
    }

    fn show_body(
        &mut self,
        node_id: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<RfNode>,
    ) {
        let f_a = self.annotations.analysis_freq_mhz;
        let stats = self.annotations.get(node_id);
        let in_cycle = self.annotations.in_cycle(node_id);
        let node = &mut snarl[node_id];

        ui.vertical(|ui| {
            if in_cycle {
                warning(
                    ui,
                    "⚠ In a feedback loop — not evaluated",
                    Theme::ACCENT_ERROR,
                );
            }
            match node {
                RfNode::SignalSource(src) => show_source_body(ui, node_id, src),
                RfNode::Balun(balun) => show_balun_body(ui, node_id, balun, f_a),
                RfNode::Filter(filter) => show_filter_body(ui, node_id, filter, f_a),
                RfNode::Amplifier(amp) => show_amplifier_body(ui, node_id, amp, stats),
                RfNode::Attenuator(att) => {
                    egui::Grid::new(format!("atten_grid_{node_id:?}"))
                        .num_columns(2)
                        .spacing([4.0, 3.0])
                        .show(ui, |ui| {
                            ui.label("Atten:");
                            ui.add(
                                egui::DragValue::new(&mut att.model.attenuation_db)
                                    .range(0.0..=60.0)
                                    .suffix(" dB")
                                    .speed(0.5),
                            )
                            .on_hover_text(
                                "Resistive pad. Being passive and at ambient, its noise figure \
                                 equals its loss — 6 dB of pad costs 6 dB of noise figure.",
                            );
                            ui.end_row();
                        });
                    readout(ui, format!("NF {:.1} dB · linear", att.model.attenuation_db));
                }
                RfNode::Splitter(spl) => {
                    egui::Grid::new(format!("split_grid_{node_id:?}"))
                        .num_columns(2)
                        .spacing([4.0, 3.0])
                        .show(ui, |ui| {
                            ui.label("Ports:");
                            ui.add(egui::DragValue::new(&mut spl.model.num_outputs).range(2..=8))
                                .on_hover_text("Number of output arms. Each takes 1/N of the power.");
                            ui.end_row();

                            ui.label("Excess:");
                            ui.add(
                                egui::DragValue::new(&mut spl.model.excess_loss_db)
                                    .range(0.0..=10.0)
                                    .suffix(" dB")
                                    .speed(0.1),
                            )
                            .on_hover_text("Dissipative loss on top of the ideal split ratio.");
                            ui.end_row();
                        });
                    readout(
                        ui,
                        format!(
                            "{:.2} dB per port ({:.2} ideal + {:.2})",
                            spl.model.total_loss_db(),
                            10.0 * (spl.model.num_outputs.max(1) as f64).log10(),
                            spl.model.excess_loss_db
                        ),
                    );
                }
                RfNode::Combiner(comb) => {
                    egui::Grid::new(format!("comb_grid_{node_id:?}"))
                        .num_columns(2)
                        .spacing([4.0, 3.0])
                        .show(ui, |ui| {
                            ui.label("Ports:");
                            ui.add(egui::DragValue::new(&mut comb.model.num_inputs).range(2..=8))
                                .on_hover_text(
                                    "Number of input arms. Voltages sum, then the split ratio \
                                     applies: two coherent inputs gain 3 dB, two uncorrelated \
                                     ones gain nothing.",
                                );
                            ui.end_row();

                            ui.label("Excess:");
                            ui.add(
                                egui::DragValue::new(&mut comb.model.excess_loss_db)
                                    .range(0.0..=10.0)
                                    .suffix(" dB")
                                    .speed(0.1),
                            )
                            .on_hover_text("Dissipative loss on top of the ideal combining ratio.");
                            ui.end_row();
                        });
                    readout(ui, format!("{:.2} dB per port", comb.model.total_loss_db()));
                }
                RfNode::Mixer(mix) => show_mixer_body(ui, node_id, mix, f_a),
                RfNode::PhaseShifter(ps) => show_phase_body(ui, node_id, ps, f_a),
                RfNode::DirectionalCoupler(dc) => show_coupler_body(ui, node_id, dc),
                RfNode::S2p(s2p) => show_s2p_body(ui, node_id, s2p, f_a),
                RfNode::AdcInput(adc) => {
                    egui::Grid::new(format!("adc_grid_{node_id:?}"))
                        .num_columns(2)
                        .spacing([4.0, 3.0])
                        .show(ui, |ui| {
                            ui.label("Tile:");
                            ui.add(egui::DragValue::new(&mut adc.tile_index).range(0..=3))
                                .on_hover_text("Which ADC tile this input feeds.");
                            ui.end_row();

                            ui.label("Block:");
                            ui.add(egui::DragValue::new(&mut adc.block_index).range(0..=1))
                                .on_hover_text("Which block within the tile.");
                            ui.end_row();
                        });
                }
            }

            // Per-stage budget line: what this node contributes at the analysis frequency.
            if let Some(s) = stats {
                if !matches!(node, RfNode::SignalSource(_) | RfNode::AdcInput(_)) {
                    ui.separator();
                    let mut line = format!("{:+.2} dB · NF {:.2} dB", s.gain_db, s.noise_figure_db);
                    if s.group_delay_ns.abs() > 0.001 {
                        line.push_str(&format!(" · {:.2} ns", s.group_delay_ns));
                    }
                    readout(ui, line);
                    if s.compression_db < -0.1 {
                        warning(
                            ui,
                            format!("⚠ compressing {:.1} dB", s.compression_db),
                            Theme::ACCENT_WARN,
                        );
                    }
                }
            }
        });
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<RfNode>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: egui::Pos2, ui: &mut egui::Ui, snarl: &mut Snarl<RfNode>) {
        ui.label("Add Node");
        ui.separator();
        let mut add = |ui: &mut egui::Ui, icon: &str, label: &str, node: RfNode| {
            if ui.button(format!("{icon} {label}")).clicked() {
                snarl.insert_node(pos, node);
                ui.close();
            }
        };
        add(
            ui,
            egui_phosphor::regular::WAVE_SAWTOOTH,
            "Signal Source",
            RfNode::SignalSource(SignalSourceNode::default()),
        );
        add(
            ui,
            egui_phosphor::regular::ARROWS_LEFT_RIGHT,
            "Balun",
            RfNode::Balun(BalunNode::default()),
        );
        ui.separator();
        for (label, ft) in [
            ("Low-Pass Filter", FilterType::LowPass),
            ("High-Pass Filter", FilterType::HighPass),
            ("Band-Pass Filter", FilterType::BandPass),
        ] {
            let mut f = FilterNode::default();
            f.model.filter_type = ft;
            if ft == FilterType::HighPass {
                f.model.cutoff_mhz = 100.0;
            }
            add(ui, egui_phosphor::regular::FUNNEL, label, RfNode::Filter(f));
        }
        ui.separator();
        add(
            ui,
            egui_phosphor::regular::SPEAKER_HIFI,
            "Amplifier",
            RfNode::Amplifier(AmplifierNode::default()),
        );
        add(
            ui,
            egui_phosphor::regular::SLIDERS_HORIZONTAL,
            "Attenuator",
            RfNode::Attenuator(AttenuatorNode::default()),
        );
        add(
            ui,
            egui_phosphor::regular::GIT_MERGE,
            "Splitter",
            RfNode::Splitter(SplitterNode::default()),
        );
        add(
            ui,
            egui_phosphor::regular::ARROWS_IN_SIMPLE,
            "Combiner",
            RfNode::Combiner(CombinerNode::default()),
        );
        add(
            ui,
            egui_phosphor::regular::WAVES,
            "Mixer",
            RfNode::Mixer(MixerNode::default()),
        );
        add(
            ui,
            egui_phosphor::regular::CLOCK,
            "Phase Shifter",
            RfNode::PhaseShifter(PhaseShifterNode::default()),
        );
        add(
            ui,
            egui_phosphor::regular::ROWS,
            "Directional Coupler",
            RfNode::DirectionalCoupler(DirectionalCouplerNode::default()),
        );
        add(
            ui,
            egui_phosphor::regular::FILE_TEXT,
            "Touchstone .s2p Block",
            RfNode::S2p(S2pNode::default()),
        );
        ui.separator();
        add(
            ui,
            egui_phosphor::regular::CPU,
            "ADC Input",
            RfNode::AdcInput(AdcInputNode::default()),
        );
    }

    fn has_node_menu(&mut self, _node: &RfNode) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<RfNode>,
    ) {
        if ui
            .button(format!("{} Duplicate", egui_phosphor::regular::COPY))
            .clicked()
        {
            let clone = snarl[node].clone();
            let pos = snarl
                .get_node_info(node)
                .map(|i| i.pos + egui::vec2(40.0, 60.0))
                .unwrap_or(egui::pos2(0.0, 0.0));
            snarl.insert_node(pos, clone);
            ui.close();
        }
        if ui
            .button(format!("{} Delete", egui_phosphor::regular::TRASH))
            .clicked()
        {
            snarl.remove_node(node);
            ui.close();
        }
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<RfNode>) {
        // A forward-only chain cannot evaluate feedback, so refuse the wire that would close
        // a loop rather than letting the evaluator recurse into it.
        if would_create_cycle(snarl, from.id.node, to.id.node) {
            return;
        }
        // Enforce 1:1 connections: an output can only go to one input, and an input can only
        // receive from one output.
        snarl.drop_outputs(from.id);
        snarl.drop_inputs(to.id);
        snarl.connect(from.id, to.id);
    }
}

// ---------------------------------------------------------------------------
// Per-node bodies
// ---------------------------------------------------------------------------

fn show_source_body(ui: &mut egui::Ui, node_id: NodeId, src: &mut SignalSourceNode) {
    ui.horizontal(|ui| {
        ui.label("Mode:");
        ui.selectable_value(&mut src.source_type, SourceType::GlobalGenerator, "Global");
        ui.selectable_value(&mut src.source_type, SourceType::LocalGenerator, "Local");
        ui.selectable_value(&mut src.source_type, SourceType::IqFile, "File");
    });
    match src.source_type {
        SourceType::GlobalGenerator => {
            ui.label(
                egui::RichText::new("Synced to Sidebar Signal Generator")
                    .small()
                    .color(Theme::ACCENT_PRIMARY),
            );
        }
        SourceType::LocalGenerator => {
            if let Some(tone) = src.generator.tones.first_mut() {
                egui::Grid::new(format!("src_gen_grid_{node_id:?}"))
                    .num_columns(2)
                    .spacing([4.0, 3.0])
                    .show(ui, |ui| {
                        ui.label("Type:");
                        let selected_text = tone.modulation.to_string();
                        egui::ComboBox::from_id_salt(format!("mod_combo_{node_id:?}"))
                            .selected_text(selected_text)
                            .show_ui(ui, |ui| {
                                use crate::signal::ToneModulation::*;
                                let opts = [
                                    ("CW (Complex Tone)", Cw),
                                    ("Cosine", RealCosine),
                                    ("Sine", RealSine),
                                    ("Square", Square),
                                    ("Sawtooth", Sawtooth),
                                    ("Triangle", Triangle),
                                    ("FMCW Chirp Sweep", SweptChirp { sweep_period_ms: 5.0 }),
                                    (
                                        "FM Modulated",
                                        FmModulated { dev_mhz: 10.0, mod_freq_khz: 10.0 },
                                    ),
                                    (
                                        "Pulsed Radar",
                                        PulsedRadar { pulse_width_us: 10.0, pri_us: 100.0 },
                                    ),
                                    (
                                        "Frequency Hopping",
                                        FreqHopping {
                                            hop_step_mhz: 10.0,
                                            num_channels: 10,
                                            hop_rate_hz: 100.0,
                                        },
                                    ),
                                    ("Digital QPSK", DigitalQpsk { symbol_rate_ksps: 100.0 }),
                                ];
                                for (label, variant) in opts {
                                    let selected =
                                        std::mem::discriminant(&tone.modulation)
                                            == std::mem::discriminant(&variant);
                                    if ui.selectable_label(selected, label).clicked() {
                                        tone.modulation = variant;
                                    }
                                }
                            });
                        ui.end_row();

                        ui.label("Freq:");
                        ui.add(
                            egui::DragValue::new(&mut tone.frequency_mhz)
                                .range(0.1..=10000.0)
                                .suffix(" MHz")
                                .speed(10.0),
                        );
                        ui.end_row();

                        ui.label("Amp:");
                        ui.add(
                            egui::DragValue::new(&mut tone.amplitude_dbfs)
                                .range(-120.0..=0.0)
                                .suffix(" dBFS")
                                .speed(0.5),
                        )
                        .on_hover_text(format!(
                            "Peak amplitude relative to converter full scale.\n\
                             0 dBFS = {:+.2} dBm at the ADC input.",
                            FULL_SCALE_DBM
                        ));
                        ui.end_row();

                        ui.label("Channel BW:");
                        ui.add(
                            egui::DragValue::new(&mut tone.bandwidth_mhz)
                                .range(0.0..=2000.0)
                                .suffix(" MHz")
                                .speed(1.0),
                        )
                        .on_hover_text(
                            "0 for a pure tone. Above zero the carrier is spread into a \
                             modulated channel of this width.",
                        );
                        ui.end_row();

                        ui.label("Noise:");
                        ui.add(
                            egui::DragValue::new(&mut src.generator.noise_floor_dbfs)
                                .range(-200.0..=0.0)
                                .suffix(" dBFS")
                                .speed(1.0),
                        )
                        .on_hover_text(
                            "Total AWGN power injected at the source, as a test vector. The \
                             chain's own physical noise is set by the thermal-noise control \
                             above the graph.",
                        );
                        ui.end_row();

                        use crate::signal::ToneModulation::*;
                        match &mut tone.modulation {
                            Cw | RealCosine | RealSine | Square | Sawtooth | Triangle => {}
                            SweptChirp { sweep_period_ms } => {
                                ui.label("Period:");
                                ui.add(
                                    egui::DragValue::new(sweep_period_ms)
                                        .suffix(" ms")
                                        .speed(0.1),
                                );
                                ui.end_row();
                            }
                            FmModulated { dev_mhz, mod_freq_khz } => {
                                ui.label("Dev:");
                                ui.add(egui::DragValue::new(dev_mhz).suffix(" MHz").speed(1.0));
                                ui.end_row();

                                ui.label("Mod Freq:");
                                ui.add(egui::DragValue::new(mod_freq_khz).suffix(" kHz").speed(1.0));
                                ui.end_row();
                            }
                            PulsedRadar { pulse_width_us, pri_us } => {
                                ui.label("Width:");
                                ui.add(egui::DragValue::new(pulse_width_us).suffix(" µs").speed(1.0));
                                ui.end_row();

                                ui.label("PRI:");
                                ui.add(egui::DragValue::new(pri_us).suffix(" µs").speed(1.0));
                                ui.end_row();
                            }
                            FreqHopping { hop_step_mhz, num_channels, hop_rate_hz } => {
                                ui.label("Step:");
                                ui.add(egui::DragValue::new(hop_step_mhz).suffix(" MHz").speed(1.0));
                                ui.end_row();

                                ui.label("Chans:");
                                ui.add(egui::DragValue::new(num_channels));
                                ui.end_row();

                                ui.label("Rate:");
                                ui.add(egui::DragValue::new(hop_rate_hz).suffix(" Hz").speed(1.0));
                                ui.end_row();
                            }
                            DigitalQpsk { symbol_rate_ksps } => {
                                ui.label("Sym Rate:");
                                ui.add(
                                    egui::DragValue::new(symbol_rate_ksps)
                                        .suffix(" ksps")
                                        .speed(10.0),
                                );
                                ui.end_row();
                            }
                        }
                    });
            }
        }
        SourceType::IqFile => {
            if let Some(path) = &src.file_loader.path {
                let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("File");
                ui.label(format!("File: {filename}"));
            } else {
                ui.label("No file loaded");
            }
            if ui.button("📁 Browse...").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("IQ Files", &["bin", "dat", "csv", "iq"])
                    .pick_file()
                {
                    src.file_loader.path = Some(path);
                    src.file_loader.clear_cache();
                }
            }

            egui::Grid::new(format!("iq_file_grid_{node_id:?}"))
                .num_columns(2)
                .spacing([4.0, 3.0])
                .show(ui, |ui| {
                    ui.label("Format:");
                    egui::ComboBox::from_id_salt(format!("iq_format_{node_id:?}"))
                        .selected_text(src.file_loader.format.to_string())
                        .show_ui(ui, |ui| {
                            use crate::signal::IqFormat;
                            for (fmt, label) in [
                                (IqFormat::BinaryF32, "fc32 (Binary f32)"),
                                (IqFormat::BinaryF64, "fc64 (Binary f64)"),
                                (IqFormat::Sc16, "sc16 (Binary i16)"),
                                (IqFormat::Csv, "CSV (I, Q)"),
                            ] {
                                if ui
                                    .selectable_value(&mut src.file_loader.format, fmt, label)
                                    .changed()
                                {
                                    src.file_loader.clear_cache();
                                }
                            }
                        });
                    ui.end_row();

                    ui.label("Capt. Rate:");
                    ui.add(
                        egui::DragValue::new(&mut src.file_loader.sample_rate_mhz)
                            .suffix(" MHz")
                            .speed(1.0)
                            .range(0.001..=10000.0),
                    );
                    ui.end_row();

                    ui.label("Repeat:");
                    ui.checkbox(&mut src.file_loader.repeat, "");
                    ui.end_row();

                    if src.file_loader.repeat {
                        ui.label("Idle Gap:");
                        ui.add(
                            egui::DragValue::new(&mut src.file_loader.repeat_period_us)
                                .suffix(" µs")
                                .speed(1.0)
                                .range(0.0..=1e6),
                        );
                        ui.end_row();
                    }
                });
            readout(
                ui,
                "Only the real part reaches the converter pin, as with any physical input.",
            );
        }
    }
}

fn show_balun_body(ui: &mut egui::Ui, node_id: NodeId, balun: &mut BalunNode, f_a: f64) {
    let preset = |name: &str, il: Vec<(f64, f64)>, lo: f64, hi: f64| BalunModel {
        name: name.to_string(),
        il_table: il,
        min_freq_mhz: lo,
        max_freq_mhz: hi,
        ..BalunModel::default()
    };

    let mut current_name = balun.model.name.clone();
    egui::ComboBox::from_id_salt(format!("balun_combo_{node_id:?}"))
        .selected_text(&current_name)
        .show_ui(ui, |ui| {
            if ui
                .selectable_value(&mut current_name, "TCM2-33WX+".to_string(), "TCM2-33WX+")
                .changed()
            {
                balun.model = BalunModel::default();
            }
            let bands: [(&str, &str, f64, f64, f64, f64); 4] = [
                ("XM655 Band 1", "XM655 Band 1 (10-1000 MHz)", 10.0, 1000.0, 1.0, 1.5),
                ("XM655 Band 2", "XM655 Band 2 (1-4 GHz)", 1000.0, 4000.0, 1.0, 2.0),
                ("XM655 Band 3", "XM655 Band 3 (4-5 GHz)", 4000.0, 5000.0, 1.5, 2.0),
                ("XM655 Band 4", "XM655 Band 4 (5-6 GHz)", 5000.0, 6000.0, 1.5, 2.5),
            ];
            for (name, label, lo, hi, il_lo, il_hi) in bands {
                if ui
                    .selectable_value(&mut current_name, name.to_string(), label)
                    .changed()
                {
                    balun.model = preset(name, vec![(lo, il_lo), (hi, il_hi)], lo, hi);
                }
            }
            if ui
                .selectable_value(&mut current_name, "Ideal".to_string(), "Ideal")
                .changed()
            {
                balun.model = BalunModel {
                    name: "Ideal".to_string(),
                    il_table: vec![(0.0, 0.0), (10000.0, 0.0)],
                    min_freq_mhz: 0.0,
                    max_freq_mhz: 0.0,
                    amplitude_imbalance_db: 0.0,
                    phase_imbalance_deg: 0.0,
                };
            }
            if ui
                .selectable_value(&mut current_name, "Custom".to_string(), "Custom")
                .changed()
            {
                balun.model = preset("Custom", vec![(0.0, 0.5), (10000.0, 0.5)], 10.0, 1000.0);
            }
        });

    egui::Grid::new(format!("balun_grid_{node_id:?}"))
        .num_columns(2)
        .spacing([4.0, 3.0])
        .show(ui, |ui| {
            if balun.model.name == "Custom" {
                ui.label("Min:");
                ui.add(
                    egui::DragValue::new(&mut balun.model.min_freq_mhz)
                        .range(0.0..=10000.0)
                        .suffix(" MHz")
                        .speed(10.0),
                )
                .on_hover_text("Low corner. A transformer is a high-pass below this point.");
                ui.end_row();
                ui.label("Max:");
                ui.add(
                    egui::DragValue::new(&mut balun.model.max_freq_mhz)
                        .range(0.0..=20000.0)
                        .suffix(" MHz")
                        .speed(10.0),
                )
                .on_hover_text("High corner, where parasitics take over.");
                ui.end_row();
            }

            ui.label("Amp Imb:");
            ui.add(
                egui::DragValue::new(&mut balun.model.amplitude_imbalance_db)
                    .range(0.0..=3.0)
                    .suffix(" dB")
                    .speed(0.05),
            )
            .on_hover_text(
                "Amplitude difference between the two differential arms. Barely affects the \
                 differential level; what it costs is common-mode rejection.",
            );
            ui.end_row();

            ui.label("Phase Imb:");
            ui.add(
                egui::DragValue::new(&mut balun.model.phase_imbalance_deg)
                    .range(0.0..=30.0)
                    .suffix("°")
                    .speed(0.5),
            )
            .on_hover_text("Departure from an ideal 180° between the arms.");
            ui.end_row();
        });

    if balun.model.name != "Custom" && balun.model.max_freq_mhz > 0.0 {
        readout(
            ui,
            format!(
                "{:.0}–{:.0} MHz",
                balun.model.min_freq_mhz, balun.model.max_freq_mhz
            ),
        );
    }
    readout(
        ui,
        format!(
            "IL {:.2} dB @ {:.0} MHz · CMRR {:.0} dB",
            balun.model.insertion_loss_at(f_a.max(1.0)),
            f_a,
            balun.model.cmrr_db()
        ),
    );
}

fn show_filter_body(ui: &mut egui::Ui, node_id: NodeId, filter: &mut FilterNode, f_a: f64) {
    ui.horizontal(|ui| {
        let mut ft = filter.model.filter_type;
        egui::ComboBox::from_id_salt(format!("filter_combo_{node_id:?}"))
            .selected_text(ft.to_string())
            .width(90.0)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut ft, FilterType::LowPass, "Low-Pass");
                ui.selectable_value(&mut ft, FilterType::HighPass, "High-Pass");
                ui.selectable_value(&mut ft, FilterType::BandPass, "Band-Pass");
            });
        filter.model.filter_type = ft;

        let is_cheby = matches!(filter.model.response, FilterResponse::Chebyshev { .. });
        egui::ComboBox::from_id_salt(format!("filter_resp_{node_id:?}"))
            .selected_text(if is_cheby { "Chebyshev" } else { "Butterworth" })
            .width(95.0)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(!is_cheby, "Butterworth")
                    .on_hover_text("Maximally flat passband, -3 dB exactly at the corner.")
                    .clicked()
                {
                    filter.model.response = FilterResponse::Butterworth;
                }
                if ui
                    .selectable_label(is_cheby, "Chebyshev")
                    .on_hover_text("Equiripple passband, steeper skirt for the same order.")
                    .clicked()
                {
                    filter.model.response = FilterResponse::Chebyshev { ripple_db: 0.5 };
                }
            });
    });

    egui::Grid::new(format!("filter_grid_{node_id:?}"))
        .num_columns(2)
        .spacing([4.0, 3.0])
        .show(ui, |ui| {
            ui.label(if filter.model.filter_type == FilterType::BandPass {
                "Centre:"
            } else {
                "Cutoff:"
            });
            ui.add(
                egui::DragValue::new(&mut filter.model.cutoff_mhz)
                    .range(1.0..=10000.0)
                    .suffix(" MHz")
                    .speed(10.0),
            )
            .on_hover_text("The -3 dB corner (or band centre for a band-pass).");
            ui.end_row();

            if filter.model.filter_type == FilterType::BandPass {
                ui.label("BW:");
                ui.add(
                    egui::DragValue::new(&mut filter.model.bandwidth_mhz)
                        .range(1.0..=5000.0)
                        .suffix(" MHz")
                        .speed(5.0),
                )
                .on_hover_text(
                    "-3 dB bandwidth. The skirts are geometrically symmetric about the centre, \
                     as a real band-pass is.",
                );
                ui.end_row();
            }

            ui.label("Order:");
            ui.add(egui::DragValue::new(&mut filter.model.order).range(1..=12))
                .on_hover_text("Number of poles. Butterworth rolls off 20·n dB/decade.");
            ui.end_row();

            if let FilterResponse::Chebyshev { ripple_db } = &mut filter.model.response {
                ui.label("Ripple:");
                ui.add(
                    egui::DragValue::new(ripple_db)
                        .range(0.01..=3.0)
                        .suffix(" dB")
                        .speed(0.05),
                )
                .on_hover_text("Peak-to-peak passband ripple.");
                ui.end_row();
            }

            ui.label("IL:");
            ui.add(
                egui::DragValue::new(&mut filter.model.insertion_loss_db)
                    .range(0.0..=10.0)
                    .suffix(" dB")
                    .speed(0.1),
            )
            .on_hover_text("Flat passband loss, as any real filter has.");
            ui.end_row();
        });

    let f_probe = f_a.max(1.0);
    readout(
        ui,
        format!(
            "{:.1} dB @ {:.0} MHz · {:.2} ns delay",
            filter.model.attenuation_at(f_probe),
            f_probe,
            filter.model.group_delay_ns(f_probe)
        ),
    );
}

fn show_amplifier_body(
    ui: &mut egui::Ui,
    node_id: NodeId,
    amp: &mut AmplifierNode,
    stats: Option<NodeStats>,
) {
    egui::Grid::new(format!("amp_grid_{node_id:?}"))
        .num_columns(2)
        .spacing([4.0, 3.0])
        .show(ui, |ui| {
            ui.label("Gain:");
            ui.add(
                egui::DragValue::new(&mut amp.model.gain_db)
                    .range(-20.0..=40.0)
                    .suffix(" dB")
                    .speed(0.5),
            )
            .on_hover_text("Small-signal gain, in band.");
            ui.end_row();

            ui.label("NF:");
            ui.add(
                egui::DragValue::new(&mut amp.model.noise_figure_db)
                    .range(0.0..=20.0)
                    .suffix(" dB")
                    .speed(0.1),
            )
            .on_hover_text(
                "Noise figure. This injects real noise into the waveform, so where you put \
                 this stage in the chain changes the SNR reaching the converter.",
            );
            ui.end_row();

            ui.label("P1dB:");
            ui.add(
                egui::DragValue::new(&mut amp.model.p1db_dbm)
                    .range(-40.0..=50.0)
                    .suffix(" dBm")
                    .speed(0.5),
            )
            .on_hover_text(
                "Output-referred 1 dB compression point. Drive the stage this hard and the \
                 gain really does drop a dB.",
            );
            ui.end_row();

            ui.label("OIP3:");
            ui.add(
                egui::DragValue::new(&mut amp.model.oip3_dbm)
                    .range(-30.0..=60.0)
                    .suffix(" dBm")
                    .speed(0.5),
            )
            .on_hover_text(
                "Output third-order intercept, which sets the IM3 products two tones make. \
                 Typically 9 to 15 dB above P1dB.",
            );
            ui.end_row();

            ui.label("BW:");
            ui.add(
                egui::DragValue::new(&mut amp.model.bandwidth_mhz)
                    .range(0.0..=20000.0)
                    .suffix(" MHz")
                    .speed(50.0),
            )
            .on_hover_text("-3 dB gain bandwidth; 0 for a flat, unbounded gain.");
            ui.end_row();
        });

    readout(
        ui,
        format!("input P1dB {:+.1} dBm", amp.model.input_p1db_dbm()),
    );
    if amp.model.oip3_dbm < amp.model.p1db_dbm {
        warning(
            ui,
            "⚠ OIP3 below P1dB — check the datasheet",
            Theme::ACCENT_WARN,
        );
    }
    if let Some(s) = stats {
        if s.compression_db < -0.1 {
            readout(
                ui,
                format!("driven to {:+.1} dBFS out", s.output_level_dbfs),
            );
        }
    }
}

fn show_mixer_body(ui: &mut egui::Ui, node_id: NodeId, mix: &mut MixerNode, f_a: f64) {
    egui::Grid::new(format!("mix_grid_{node_id:?}"))
        .num_columns(2)
        .spacing([4.0, 3.0])
        .show(ui, |ui| {
            ui.label("LO Freq:");
            ui.add(
                egui::DragValue::new(&mut mix.model.lo_freq_mhz)
                    .range(0.1..=10000.0)
                    .suffix(" MHz")
                    .speed(10.0),
            )
            .on_hover_text(
                "Local oscillator. A real LO is a real waveform, so every input tone comes \
                 out at both |f-LO| and f+LO — filter the one you do not want.",
            );
            ui.end_row();

            ui.label("Loss:");
            ui.add(
                egui::DragValue::new(&mut mix.model.conversion_loss_db)
                    .range(0.0..=30.0)
                    .suffix(" dB")
                    .speed(0.5),
            )
            .on_hover_text("Loss to the wanted product, as datasheets specify it.");
            ui.end_row();

            ui.label("NF:");
            ui.add(
                egui::DragValue::new(&mut mix.model.noise_figure_db)
                    .range(0.0..=30.0)
                    .suffix(" dB")
                    .speed(0.1),
            )
            .on_hover_text("Noise figure; for a passive diode mixer, close to the conversion loss.");
            ui.end_row();

            ui.label("OIP3:");
            ui.add(
                egui::DragValue::new(&mut mix.model.oip3_dbm)
                    .range(-30.0..=60.0)
                    .suffix(" dBm")
                    .speed(0.5),
            )
            .on_hover_text("Output third-order intercept of the conversion.");
            ui.end_row();

            ui.label("LO Leak:");
            ui.add(
                egui::DragValue::new(&mut mix.model.lo_leakage_dbfs)
                    .range(-140.0..=0.0)
                    .suffix(" dBFS")
                    .speed(1.0),
            )
            .on_hover_text("LO feedthrough at the output. Signal-independent, so it shows up even with no input.");
            ui.end_row();

            ui.label("LO H3:");
            ui.add(
                egui::DragValue::new(&mut mix.model.lo_harmonic3_dbc)
                    .range(-60.0..=0.0)
                    .suffix(" dBc")
                    .speed(1.0),
            )
            .on_hover_text(
                "Third-harmonic content of the LO drive, which is what puts 3×LO ± RF spurs \
                 on the output. Set to 0 dBc to disable.",
            );
            ui.end_row();
        });

    let (diff, sum) = mix.model.product_freqs(f_a.max(1.0));
    readout(
        ui,
        format!("{:.0} MHz -> {diff:.0} / {sum:.0} MHz", f_a.max(1.0)),
    );
}

fn show_phase_body(ui: &mut egui::Ui, node_id: NodeId, ps: &mut PhaseShifterNode, f_a: f64) {
    egui::ComboBox::from_id_salt(format!("ps_kind_{node_id:?}"))
        .selected_text(ps.model.kind.to_string())
        .show_ui(ui, |ui| {
            ui.selectable_value(
                &mut ps.model.kind,
                PhaseShiftKind::ConstantPhase,
                "Constant Phase",
            )
            .on_hover_text("Same phase shift at every frequency, as a vector modulator gives.");
            ui.selectable_value(&mut ps.model.kind, PhaseShiftKind::TrueDelay, "True Delay")
                .on_hover_text(
                    "A length of line: the shift is set at a reference frequency and scales \
                     linearly with frequency.",
                );
        });

    egui::Grid::new(format!("ps_grid_{node_id:?}"))
        .num_columns(2)
        .spacing([4.0, 3.0])
        .show(ui, |ui| {
            ui.label("Phase:");
            ui.add(
                egui::DragValue::new(&mut ps.model.phase_shift_deg)
                    .range(-360.0..=360.0)
                    .suffix("°")
                    .speed(1.0),
            )
            .on_hover_text("Phase shift. The amplitude is unaffected at any angle.");
            ui.end_row();

            if ps.model.kind == PhaseShiftKind::TrueDelay {
                ui.label("At:");
                ui.add(
                    egui::DragValue::new(&mut ps.model.ref_freq_mhz)
                        .range(1.0..=10000.0)
                        .suffix(" MHz")
                        .speed(10.0),
                )
                .on_hover_text("Frequency at which the phase above is specified.");
                ui.end_row();
            }

            ui.label("Loss:");
            ui.add(
                egui::DragValue::new(&mut ps.model.insertion_loss_db)
                    .range(0.0..=20.0)
                    .suffix(" dB")
                    .speed(0.1),
            );
            ui.end_row();
        });

    match ps.model.kind {
        PhaseShiftKind::TrueDelay => readout(
            ui,
            format!(
                "{:.3} ns · {:.0}° @ {:.0} MHz",
                ps.model.delay_ns(),
                ps.model.transfer_at(f_a.max(1.0), 0).arg().to_degrees(),
                f_a.max(1.0)
            ),
        ),
        PhaseShiftKind::ConstantPhase => {
            readout(ui, format!("{:.0}° at all frequencies", -ps.model.phase_shift_deg))
        }
    }
}

fn show_coupler_body(ui: &mut egui::Ui, node_id: NodeId, dc: &mut DirectionalCouplerNode) {
    egui::Grid::new(format!("dc_grid_{node_id:?}"))
        .num_columns(2)
        .spacing([4.0, 3.0])
        .show(ui, |ui| {
            ui.label("Coupling:");
            ui.add(
                egui::DragValue::new(&mut dc.model.coupling_db)
                    .range(3.0..=50.0)
                    .suffix(" dB")
                    .speed(0.5),
            )
            .on_hover_text("How far below the input the coupled arm sits.");
            ui.end_row();

            ui.label("IL:");
            ui.add(
                egui::DragValue::new(&mut dc.model.insertion_loss_db)
                    .range(0.0..=10.0)
                    .suffix(" dB")
                    .speed(0.1),
            )
            .on_hover_text("Dissipative loss, on top of the power tapped off to the coupled arm.");
            ui.end_row();

            ui.label("Dir:");
            ui.add(
                egui::DragValue::new(&mut dc.model.directivity_db)
                    .range(5.0..=50.0)
                    .suffix(" dB")
                    .speed(0.5),
            )
            .on_hover_text(
                "Reported for reference. Reverse-travelling waves are not simulated, so this \
                 does not affect the forward result.",
            );
            ui.end_row();
        });

    readout(
        ui,
        format!(
            "Main {:+.2} dB · Coupled {:+.2} dB",
            -dc.model.through_loss_db(),
            -(dc.model.coupling_db + dc.model.insertion_loss_db)
        ),
    );
}

fn show_s2p_body(ui: &mut egui::Ui, node_id: NodeId, s2p: &mut S2pNode, f_a: f64) {
    ui.add(
        egui::Label::new(egui::RichText::new(&s2p.model.name).small().strong())
            .wrap_mode(egui::TextWrapMode::Extend),
    );

    egui::Grid::new(format!("s2p_grid_{node_id:?}"))
        .num_columns(2)
        .spacing([4.0, 3.0])
        .show(ui, |ui| {
            ui.label("NF:");
            ui.add(
                egui::DragValue::new(&mut s2p.model.noise_figure_db)
                    .range(0.0..=20.0)
                    .suffix(" dB")
                    .speed(0.1),
            )
            .on_hover_text(
                "Noise figure. A passive block can never do better than its insertion loss, \
                 so the larger of the two is used.",
            );
            ui.end_row();

            ui.label("OIP3:");
            ui.add(
                egui::DragValue::new(&mut s2p.model.oip3_dbm)
                    .range(0.0..=60.0)
                    .suffix(" dBm")
                    .speed(0.5),
            );
            ui.end_row();

            ui.label("Phase:");
            ui.checkbox(&mut s2p.model.use_measured_phase, "")
                .on_hover_text(
                    "Apply the measured phase as well as the magnitude, giving the block its \
                     real group delay. Off makes it zero-phase.",
                );
            ui.end_row();
        });

    let f_probe = f_a.max(1.0);
    let mut line = format!(
        "{} pts · S21 {:+.2} dB @ {:.0} MHz",
        s2p.model.s21_table.len(),
        s2p.model.s21_gain_at(f_probe),
        f_probe
    );
    if let Some(vswr) = s2p.model.vswr(f_probe) {
        line.push_str(&format!(" · VSWR {vswr:.2}"));
    }
    readout(ui, line);

    if ui.button("📂 Load .s2p").clicked() {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Touchstone S2P", &["s2p"])
            .pick_file()
        {
            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("S2P Block");
                    match parse_touchstone_s2p(name, &content) {
                        Ok(parsed) => s2p.model = parsed,
                        Err(e) => tracing::warn!("failed to parse {}: {e}", path.display()),
                    }
                }
                Err(e) => tracing::warn!("failed to read {}: {e}", path.display()),
            }
        }
    }
}

/// Show the node graph in a UI area.
pub fn show_node_graph(
    ui: &mut egui::Ui,
    snarl: &mut Snarl<RfNode>,
    annotations: &GraphAnnotations,
) {
    let mut style = SnarlStyle::new();
    style.wire_width = Some(3.0);
    style.pin_size = Some(10.0);
    style.pin_placement = Some(egui_snarl::ui::PinPlacement::Edge);
    style.wire_style = Some(egui_snarl::ui::WireStyle::Bezier5);
    style.downscale_wire_frame = Some(true);
    style.crisp_magnified_text = Some(true);

    SnarlWidget::new()
        .style(style)
        .show(snarl, &mut RfNodeViewer { annotations }, ui);
}
