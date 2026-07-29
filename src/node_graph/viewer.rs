//! SnarlViewer implementation for the RF front end node graph.

use super::nodes::*;
use crate::ui::theme::Theme;
use egui_snarl::ui::{PinInfo, SnarlViewer, SnarlWidget, SnarlStyle};
use egui_snarl::{InPin, NodeId, OutPin, Snarl};

/// Wire color for RF signal connections.
const RF_WIRE_COLOR: egui::Color32 = egui::Color32::from_rgb(0, 180, 255);

/// Our viewer that implements SnarlViewer for RfNode.
pub struct RfNodeViewer;

impl SnarlViewer<RfNode> for RfNodeViewer {
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
        let color = match n {
            RfNode::SignalSource(_) => Theme::NODE_SOURCE,
            RfNode::Balun(_) | RfNode::Filter(_) | RfNode::Attenuator(_) | RfNode::Splitter(_) | RfNode::PhaseShifter(_) | RfNode::DirectionalCoupler(_) | RfNode::S2p(_) => Theme::NODE_PASSIVE,
            RfNode::Amplifier(_) | RfNode::Mixer(_) => Theme::NODE_ACTIVE,
            RfNode::AdcInput(_) => Theme::NODE_SINK,
        };
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
        let (icon, title) = match n {
            RfNode::SignalSource(_) => (egui_phosphor::regular::WAVE_SAWTOOTH, n.title()),
            RfNode::Balun(_) => (egui_phosphor::regular::ARROWS_LEFT_RIGHT, n.title()),
            RfNode::Filter(_) => (egui_phosphor::regular::FUNNEL, n.title()),
            RfNode::Amplifier(_) => (egui_phosphor::regular::SPEAKER_HIFI, n.title()),
            RfNode::Attenuator(_) => (egui_phosphor::regular::SLIDERS_HORIZONTAL, n.title()),
            RfNode::Splitter(_) => (egui_phosphor::regular::GIT_MERGE, n.title()),
            RfNode::Mixer(_) => (egui_phosphor::regular::WAVES, n.title()),
            RfNode::PhaseShifter(_) => (egui_phosphor::regular::CLOCK, n.title()),
            RfNode::DirectionalCoupler(_) => (egui_phosphor::regular::ROWS, n.title()),
            RfNode::S2p(_) => (egui_phosphor::regular::FILE_TEXT, n.title()),
            RfNode::AdcInput(_) => (egui_phosphor::regular::CPU, n.title()),
        };
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(icon).strong().color(Theme::TEXT_PRIMARY));
            ui.label(egui::RichText::new(title).strong().color(Theme::TEXT_PRIMARY));
        });
    }

    fn show_input(
        &mut self,
        _pin: &InPin,
        ui: &mut egui::Ui,
        _snarl: &mut Snarl<RfNode>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        ui.label("RF In");
        PinInfo::circle().with_fill(RF_WIRE_COLOR).with_stroke(egui::Stroke::NONE)
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<RfNode>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        let node = &snarl[pin.id.node];
        match node {
            RfNode::Splitter(_) => {
                ui.label(format!("Out {}", pin.id.output));
            }
            _ => {
                ui.label("RF Out");
            }
        }
        PinInfo::circle().with_fill(RF_WIRE_COLOR).with_stroke(egui::Stroke::NONE)
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
        let node = &mut snarl[node_id];
        ui.vertical(|ui| {
            match node {
                RfNode::SignalSource(src) => {
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
                                egui::Grid::new(format!("src_gen_grid_{:?}", node_id))
                                    .num_columns(2)
                                    .spacing([4.0, 3.0])
                                    .show(ui, |ui| {
                                        ui.label("Freq:");
                                        ui.add(egui::DragValue::new(&mut tone.frequency_mhz)
                                            .range(0.1..=10000.0)
                                            .suffix(" MHz")
                                            .speed(10.0));
                                        ui.end_row();

                                        ui.label("Amp:");
                                        ui.add(egui::DragValue::new(&mut tone.amplitude_dbfs)
                                            .range(-120.0..=0.0)
                                            .suffix(" dBFS")
                                            .speed(0.5));
                                        ui.end_row();

                                        ui.label("Noise:");
                                        ui.add(egui::DragValue::new(&mut src.generator.noise_floor_dbfs)
                                            .range(-200.0..=0.0)
                                            .suffix(" dBFS")
                                            .speed(1.0));
                                        ui.end_row();
                                    });
                            }
                        }
                        SourceType::IqFile => {
                            if let Some(path) = &src.file_loader.path {
                                let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("File");
                                ui.label(format!("File: {}", filename));
                            } else {
                                ui.label("No file loaded");
                            }
                            if ui.button("📁 Browse...").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("IQ Files", &["bin", "dat", "csv", "iq"])
                                    .pick_file()
                                {
                                    src.file_loader.path = Some(path);
                                }
                            }
                        }
                    }
                }
                RfNode::Balun(balun) => {
                    let mut current_name = balun.model.name.clone();
                    egui::ComboBox::from_id_salt(format!("balun_combo_{:?}", node_id))
                        .selected_text(&current_name)
                        .show_ui(ui, |ui| {
                            if ui.selectable_value(&mut current_name, "TCM2-33WX+".to_string(), "TCM2-33WX+").changed() {
                                balun.model = super::components::BalunModel::default();
                            }
                            if ui.selectable_value(&mut current_name, "Ideal".to_string(), "Ideal").changed() {
                                balun.model = super::components::BalunModel {
                                    name: "Ideal".to_string(),
                                    il_table: vec![(0.0, 0.0), (10000.0, 0.0)],
                                    min_freq_mhz: 0.0,
                                    max_freq_mhz: 10000.0,
                                };
                            }
                        });
                    ui.label(format!(
                        "{:.0}–{:.0} MHz",
                        balun.model.min_freq_mhz, balun.model.max_freq_mhz
                    ));
                }
                RfNode::Filter(filter) => {
                    let mut ft = filter.model.filter_type;
                    egui::ComboBox::from_id_salt(format!("filter_combo_{:?}", node_id))
                        .selected_text(ft.to_string())
                        .show_ui(ui, |ui| {
                            use super::components::FilterType;
                            ui.selectable_value(&mut ft, FilterType::LowPass, "Low-Pass");
                            ui.selectable_value(&mut ft, FilterType::HighPass, "High-Pass");
                            ui.selectable_value(&mut ft, FilterType::BandPass, "Band-Pass");
                        });
                    filter.model.filter_type = ft;

                    egui::Grid::new(format!("filter_grid_{:?}", node_id))
                        .num_columns(2)
                        .spacing([4.0, 3.0])
                        .show(ui, |ui| {
                            ui.label("Cutoff:");
                            ui.add(
                                egui::DragValue::new(&mut filter.model.cutoff_mhz)
                                    .range(1.0..=10000.0)
                                    .suffix(" MHz")
                                    .speed(10.0),
                            );
                            ui.end_row();

                            if filter.model.filter_type == super::components::FilterType::BandPass {
                                ui.label("BW:");
                                ui.add(
                                    egui::DragValue::new(&mut filter.model.bandwidth_mhz)
                                        .range(1.0..=5000.0)
                                        .suffix(" MHz")
                                        .speed(5.0),
                                );
                                ui.end_row();
                            }

                            ui.label("Order:");
                            ui.add(
                                egui::DragValue::new(&mut filter.model.order)
                                    .range(1..=12),
                            );
                            ui.end_row();
                        });
                }
                RfNode::Amplifier(amp) => {
                    egui::Grid::new(format!("amp_grid_{:?}", node_id))
                        .num_columns(2)
                        .spacing([4.0, 3.0])
                        .show(ui, |ui| {
                            ui.label("Gain:");
                            ui.add(
                                egui::DragValue::new(&mut amp.model.gain_db)
                                    .range(-20.0..=40.0)
                                    .suffix(" dB")
                                    .speed(0.5),
                            );
                            ui.end_row();

                            ui.label("NF:");
                            ui.add(
                                egui::DragValue::new(&mut amp.model.noise_figure_db)
                                    .range(0.0..=20.0)
                                    .suffix(" dB")
                                    .speed(0.1),
                            );
                            ui.end_row();

                            ui.label("P1dB:");
                            ui.add(
                                egui::DragValue::new(&mut amp.model.p1db_dbm)
                                    .range(-20.0..=50.0)
                                    .suffix(" dBm")
                                    .speed(0.5),
                            );
                            ui.end_row();
                        });
                }
                RfNode::Attenuator(att) => {
                    egui::Grid::new(format!("atten_grid_{:?}", node_id))
                        .num_columns(2)
                        .spacing([4.0, 3.0])
                        .show(ui, |ui| {
                            ui.label("Atten:");
                            ui.add(
                                egui::DragValue::new(&mut att.model.attenuation_db)
                                    .range(0.0..=60.0)
                                    .suffix(" dB")
                                    .speed(0.5),
                            );
                            ui.end_row();
                        });
                }
                RfNode::Splitter(spl) => {
                    egui::Grid::new(format!("split_grid_{:?}", node_id))
                        .num_columns(2)
                        .spacing([4.0, 3.0])
                        .show(ui, |ui| {
                            ui.label("Ports:");
                            ui.add(egui::DragValue::new(&mut spl.model.num_outputs).range(2..=8));
                            ui.end_row();

                            ui.label("Loss:");
                            ui.add(egui::DragValue::new(&mut spl.model.excess_loss_db)
                                .range(0.0..=10.0)
                                .suffix(" dB")
                                .speed(0.1));
                            ui.end_row();
                        });
                    ui.label(format!("Total Loss: {:.1} dB", spl.model.total_loss_db()));
                }
                RfNode::Mixer(mix) => {
                    egui::Grid::new(format!("mix_grid_{:?}", node_id))
                        .num_columns(2)
                        .spacing([4.0, 3.0])
                        .show(ui, |ui| {
                            ui.label("LO Freq:");
                            ui.add(egui::DragValue::new(&mut mix.model.lo_freq_mhz)
                                .range(0.1..=10000.0)
                                .suffix(" MHz")
                                .speed(10.0));
                            ui.end_row();

                            ui.label("Loss:");
                            ui.add(egui::DragValue::new(&mut mix.model.conversion_loss_db)
                                .range(0.0..=30.0)
                                .suffix(" dB")
                                .speed(0.5));
                            ui.end_row();
                        });
                }
                RfNode::PhaseShifter(ps) => {
                    egui::Grid::new(format!("ps_grid_{:?}", node_id))
                        .num_columns(2)
                        .spacing([4.0, 3.0])
                        .show(ui, |ui| {
                            ui.label("Phase:");
                            ui.add(egui::DragValue::new(&mut ps.model.phase_shift_deg)
                                .range(-360.0..=360.0)
                                .suffix("°")
                                .speed(1.0));
                            ui.end_row();

                            ui.label("Loss:");
                            ui.add(egui::DragValue::new(&mut ps.model.insertion_loss_db)
                                .range(0.0..=20.0)
                                .suffix(" dB")
                                .speed(0.1));
                            ui.end_row();
                        });
                }
                RfNode::DirectionalCoupler(dc) => {
                    egui::Grid::new(format!("dc_grid_{:?}", node_id))
                        .num_columns(2)
                        .spacing([4.0, 3.0])
                        .show(ui, |ui| {
                            ui.label("Cpl:");
                            ui.add(egui::DragValue::new(&mut dc.model.coupling_db)
                                .range(3.0..=50.0)
                                .suffix(" dB")
                                .speed(0.5));
                            ui.end_row();

                            ui.label("Loss:");
                            ui.add(egui::DragValue::new(&mut dc.model.insertion_loss_db)
                                .range(0.0..=10.0)
                                .suffix(" dB")
                                .speed(0.1));
                            ui.end_row();
                        });
                }
                RfNode::S2p(s2p) => {
                    ui.add(egui::Label::new(egui::RichText::new(&s2p.model.name).small().strong()).wrap_mode(egui::TextWrapMode::Extend));
                    ui.add(egui::Label::new(format!("Pts: {}", s2p.model.s21_table.len())).wrap_mode(egui::TextWrapMode::Extend));
                    
                    egui::Grid::new(format!("s2p_grid_{:?}", node_id))
                        .num_columns(2)
                        .spacing([4.0, 3.0])
                        .show(ui, |ui| {
                            ui.label("NF:");
                            ui.add(egui::DragValue::new(&mut s2p.model.noise_figure_db)
                                .range(0.0..=20.0)
                                .suffix(" dB")
                                .speed(0.1));
                            ui.end_row();

                            ui.label("OIP3:");
                            ui.add(egui::DragValue::new(&mut s2p.model.oip3_dbm)
                                .range(0.0..=60.0)
                                .suffix(" dBm")
                                .speed(0.5));
                            ui.end_row();
                        });

                    if ui.button("📂 Load .s2p").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Touchstone S2P", &["s2p"])
                            .pick_file()
                        {
                            if let Ok(content) = std::fs::read_to_string(&path) {
                                let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("S2P Block");
                                if let Ok(parsed) = super::components::parse_touchstone_s2p(name, &content) {
                                    s2p.model = parsed;
                                }
                            }
                        }
                    }
                }
                RfNode::AdcInput(adc) => {
                    egui::Grid::new(format!("adc_grid_{:?}", node_id))
                        .num_columns(2)
                        .spacing([4.0, 3.0])
                        .show(ui, |ui| {
                            ui.label("Tile:");
                            ui.add(egui::DragValue::new(&mut adc.tile_index).range(0..=3));
                            ui.end_row();

                            ui.label("Block:");
                            ui.add(egui::DragValue::new(&mut adc.block_index).range(0..=1));
                            ui.end_row();
                        });
                }
            }
        });
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<RfNode>) -> bool {
        true
    }

    fn show_graph_menu(
        &mut self,
        pos: egui::Pos2,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<RfNode>,
    ) {
        ui.label("Add Node");
        ui.separator();
        if ui.button(format!("{} Signal Source", egui_phosphor::regular::WAVE_SAWTOOTH)).clicked() {
            snarl.insert_node(pos, RfNode::SignalSource(SignalSourceNode::default()));
            ui.close();
        }
        if ui.button(format!("{} Balun", egui_phosphor::regular::ARROWS_LEFT_RIGHT)).clicked() {
            snarl.insert_node(pos, RfNode::Balun(BalunNode::default()));
            ui.close();
        }
        ui.separator();
        if ui.button(format!("{} Low-Pass Filter", egui_phosphor::regular::FUNNEL)).clicked() {
            let mut f = FilterNode::default();
            f.model.filter_type = super::components::FilterType::LowPass;
            snarl.insert_node(pos, RfNode::Filter(f));
            ui.close();
        }
        if ui.button(format!("{} High-Pass Filter", egui_phosphor::regular::FUNNEL)).clicked() {
            let mut f = FilterNode::default();
            f.model.filter_type = super::components::FilterType::HighPass;
            snarl.insert_node(pos, RfNode::Filter(f));
            ui.close();
        }
        if ui.button(format!("{} Band-Pass Filter", egui_phosphor::regular::FUNNEL)).clicked() {
            let mut f = FilterNode::default();
            f.model.filter_type = super::components::FilterType::BandPass;
            snarl.insert_node(pos, RfNode::Filter(f));
            ui.close();
        }
        ui.separator();
        if ui.button(format!("{} Amplifier", egui_phosphor::regular::SPEAKER_HIFI)).clicked() {
            snarl.insert_node(pos, RfNode::Amplifier(AmplifierNode::default()));
            ui.close();
        }
        if ui.button(format!("{} Attenuator", egui_phosphor::regular::SLIDERS_HORIZONTAL)).clicked() {
            snarl.insert_node(pos, RfNode::Attenuator(AttenuatorNode::default()));
            ui.close();
        }
        if ui.button(format!("{} Splitter", egui_phosphor::regular::GIT_MERGE)).clicked() {
            snarl.insert_node(pos, RfNode::Splitter(SplitterNode::default()));
            ui.close();
        }
        if ui.button(format!("{} Mixer", egui_phosphor::regular::WAVES)).clicked() {
            snarl.insert_node(pos, RfNode::Mixer(MixerNode::default()));
            ui.close();
        }
        if ui.button(format!("{} Phase Shifter", egui_phosphor::regular::CLOCK)).clicked() {
            snarl.insert_node(pos, RfNode::PhaseShifter(PhaseShifterNode::default()));
            ui.close();
        }
        if ui.button(format!("{} Directional Coupler", egui_phosphor::regular::ROWS)).clicked() {
            snarl.insert_node(pos, RfNode::DirectionalCoupler(DirectionalCouplerNode::default()));
            ui.close();
        }
        if ui.button(format!("{} Touchstone .s2p Block", egui_phosphor::regular::FILE_TEXT)).clicked() {
            snarl.insert_node(pos, RfNode::S2p(S2pNode::default()));
            ui.close();
        }
        ui.separator();
        if ui.button(format!("{} ADC Input", egui_phosphor::regular::CPU)).clicked() {
            snarl.insert_node(pos, RfNode::AdcInput(AdcInputNode::default()));
            ui.close();
        }
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
        if ui.button(format!("{} Delete", egui_phosphor::regular::TRASH)).clicked() {
            snarl.remove_node(node);
            ui.close();
        }
    }
}

/// Show the node graph in a UI area.
pub fn show_node_graph(ui: &mut egui::Ui, snarl: &mut Snarl<RfNode>) {
    let mut style = SnarlStyle::new();
    style.wire_width = Some(3.0);
    style.pin_size = Some(10.0);
    style.pin_placement = Some(egui_snarl::ui::PinPlacement::Edge);
    style.wire_style = Some(egui_snarl::ui::WireStyle::Bezier5);
    
    SnarlWidget::new()
        .style(style)
        .show(snarl, &mut RfNodeViewer, ui);
}


