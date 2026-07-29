//! Node enum definitions and graph pin mapping.

#![allow(dead_code)]

use super::components::*;
use crate::signal::{IqFileLoader, SignalGenerator};
use serde::{Deserialize, Serialize};

/// Pin types for the node graph — defines what data flows between nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PinType {
    /// RF signal (spectrum data).
    RfSignal,
}

/// A node in the RF front end signal chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RfNode {
    SignalSource(SignalSourceNode),
    Balun(BalunNode),
    Filter(FilterNode),
    Amplifier(AmplifierNode),
    Attenuator(AttenuatorNode),
    Splitter(SplitterNode),
    Mixer(MixerNode),
    PhaseShifter(PhaseShifterNode),
    DirectionalCoupler(DirectionalCouplerNode),
    S2p(S2pNode),
    AdcInput(AdcInputNode),
}

impl RfNode {
    /// Human-readable name for display in the node header.
    pub fn title(&self) -> &str {
        match self {
            RfNode::SignalSource(_) => "Signal Source",
            RfNode::Balun(_) => "Balun",
            RfNode::Filter(f) => match f.model.filter_type {
                FilterType::LowPass => "Low-Pass Filter",
                FilterType::HighPass => "High-Pass Filter",
                FilterType::BandPass => "Band-Pass Filter",
            },
            RfNode::Amplifier(_) => "Amplifier",
            RfNode::Attenuator(_) => "Attenuator",
            RfNode::Splitter(_) => "Splitter",
            RfNode::Mixer(_) => "Mixer",
            RfNode::PhaseShifter(_) => "Phase Shifter",
            RfNode::DirectionalCoupler(_) => "Directional Coupler",
            RfNode::S2p(s) => &s.model.name,
            RfNode::AdcInput(_) => "ADC Input",
        }
    }

    /// Number of input pins.
    pub fn num_inputs(&self) -> usize {
        match self {
            RfNode::SignalSource(_) => 0,
            RfNode::Splitter(_) => 1,
            RfNode::AdcInput(_) => 1,
            _ => 1,
        }
    }

    /// Number of output pins.
    pub fn num_outputs(&self) -> usize {
        match self {
            RfNode::AdcInput(_) => 0,
            RfNode::Splitter(s) => s.model.num_outputs as usize,
            _ => 1,
        }
    }

    /// Process an input spectrum through this node, returning the output spectrum.
    pub fn process(&self, input: &Spectrum) -> Spectrum {
        let mut output = input.clone();
        match self {
            RfNode::SignalSource(_) => {} // Source generates its own spectrum
            RfNode::Balun(b) => b.model.apply(&mut output),
            RfNode::Filter(f) => f.model.apply(&mut output),
            RfNode::Amplifier(a) => a.model.apply(&mut output),
            RfNode::Attenuator(a) => a.model.apply(&mut output),
            RfNode::Splitter(s) => s.model.apply(&mut output),
            RfNode::Mixer(m) => m.model.apply(&mut output),
            RfNode::PhaseShifter(p) => p.model.apply(&mut output),
            RfNode::DirectionalCoupler(d) => d.model.apply(&mut output),
            RfNode::S2p(s) => s.model.apply(&mut output),
            RfNode::AdcInput(_) => {} // Sink node, no processing
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Individual node structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalSourceNode {
    pub source_type: SourceType,
    pub generator: SignalGenerator,
    pub file_loader: IqFileLoader,
}

impl Default for SignalSourceNode {
    fn default() -> Self {
        Self {
            source_type: SourceType::GlobalGenerator,
            generator: SignalGenerator::default(),
            file_loader: IqFileLoader::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceType {
    GlobalGenerator,
    LocalGenerator,
    IqFile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalunNode {
    pub model: BalunModel,
}

impl Default for BalunNode {
    fn default() -> Self {
        Self {
            model: BalunModel::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterNode {
    pub model: FilterModel,
}

impl Default for FilterNode {
    fn default() -> Self {
        Self {
            model: FilterModel::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmplifierNode {
    pub model: AmplifierModel,
}

impl Default for AmplifierNode {
    fn default() -> Self {
        Self {
            model: AmplifierModel::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttenuatorNode {
    pub model: AttenuatorModel,
}

impl Default for AttenuatorNode {
    fn default() -> Self {
        Self {
            model: AttenuatorModel::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitterNode {
    pub model: SplitterModel,
}

impl Default for SplitterNode {
    fn default() -> Self {
        Self {
            model: SplitterModel::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixerNode {
    pub model: MixerModel,
}

impl Default for MixerNode {
    fn default() -> Self {
        Self {
            model: MixerModel::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseShifterNode {
    pub model: PhaseShifterModel,
}

impl Default for PhaseShifterNode {
    fn default() -> Self {
        Self {
            model: PhaseShifterModel::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalCouplerNode {
    pub model: DirectionalCouplerModel,
}

impl Default for DirectionalCouplerNode {
    fn default() -> Self {
        Self {
            model: DirectionalCouplerModel::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S2pNode {
    pub model: S2pModel,
}

impl Default for S2pNode {
    fn default() -> Self {
        Self {
            model: S2pModel::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdcInputNode {
    /// Which ADC tile this input feeds (0–3).
    pub tile_index: usize,
    /// Which block within the tile (0–1).
    pub block_index: usize,
}

impl Default for AdcInputNode {
    fn default() -> Self {
        Self {
            tile_index: 0,
            block_index: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Node Graph Traversal & Signal Evaluation
// ---------------------------------------------------------------------------

use num_complex::Complex;

/// Evaluation result containing generated samples, cumulative RF chain frequency response,
/// and dynamic cascaded RF metrics (Gain, Noise Figure, OIP3).
pub struct GraphEvaluationResult {
    pub samples: Vec<Complex<f64>>,
    pub rf_chain_response_db: Vec<f64>,
    pub rf_chain_freq_axis_mhz: Vec<f64>,
    pub cascaded_gain_db: f64,
    pub cascaded_nf_db: f64,
    pub cascaded_oip3_dbm: f64,
}

/// Traverses the Snarl node graph backwards from an `AdcInput(target_tile, target_block)` node,
/// generating complex IQ samples from an upstream `SignalSource` and applying each connected
/// component's DSP transfer function.
pub fn evaluate_graph(
    snarl: &egui_snarl::Snarl<RfNode>,
    target_tile: usize,
    target_block: usize,
    num_samples: usize,
    sample_rate_mhz: f64,
    global_signal_gen: &SignalGenerator,
    time_us: f64,
) -> Option<GraphEvaluationResult> {
    // Find AdcInput node matching target_tile and target_block
    let mut target_adc_id = None;
    for (id, node) in snarl.node_ids() {
        if let RfNode::AdcInput(adc) = node {
            if adc.tile_index == target_tile && adc.block_index == target_block {
                target_adc_id = Some(id);
                break;
            }
        }
    }
    let adc_node_id = target_adc_id?;

    // Trace backwards from the input pin of AdcInput node
    let in_pin = snarl.in_pin(egui_snarl::InPinId {
        node: adc_node_id,
        input: 0,
    });

    let mut current_pin = in_pin;
    let mut node_chain = Vec::new();

    while let Some(remote_out_pin_id) = current_pin.remotes.first() {
        let upstream_node_id = remote_out_pin_id.node;
        node_chain.push(upstream_node_id);

        let upstream_node = &snarl[upstream_node_id];
        if matches!(upstream_node, RfNode::SignalSource(_)) {
            break;
        }

        if upstream_node.num_inputs() == 0 {
            break;
        }

        current_pin = snarl.in_pin(egui_snarl::InPinId {
            node: upstream_node_id,
            input: 0,
        });
    }

    if node_chain.is_empty() {
        return None;
    }

    node_chain.reverse();

    let first_node = &snarl[node_chain[0]];
    let mut current_samples = match first_node {
        RfNode::SignalSource(src) => match src.source_type {
            SourceType::GlobalGenerator => global_signal_gen.generate_at_time(num_samples, sample_rate_mhz, time_us),
            SourceType::LocalGenerator => src.generator.generate_at_time(num_samples, sample_rate_mhz, time_us),
            SourceType::IqFile => src
                .file_loader
                .load()
                .unwrap_or_else(|_| src.generator.generate_at_time(num_samples, sample_rate_mhz, time_us)),
        },
        _ => return None,
    };

    for &node_id in &node_chain[1..] {
        let node = &snarl[node_id];
        current_samples = match node {
            RfNode::Balun(b) => b.model.process_samples(&current_samples, sample_rate_mhz),
            RfNode::Filter(f) => f.model.process_samples(&current_samples, sample_rate_mhz),
            RfNode::Amplifier(a) => a.model.process_samples(&current_samples),
            RfNode::Attenuator(a) => a.model.process_samples(&current_samples),
            RfNode::Splitter(s) => s.model.process_samples(&current_samples),
            RfNode::Mixer(m) => m.model.process_samples(&current_samples, sample_rate_mhz),
            RfNode::PhaseShifter(p) => p.model.process_samples(&current_samples),
            RfNode::DirectionalCoupler(d) => d.model.process_samples(&current_samples),
            RfNode::S2p(s) => s.model.process_samples(&current_samples, sample_rate_mhz),
            RfNode::SignalSource(_) | RfNode::AdcInput(_) => current_samples,
        };
    }

    // Compute cumulative frequency response curve across 0..sample_rate_mhz / 2
    let num_resp_bins = 256;
    let nyquist_max = sample_rate_mhz / 2.0;
    let rf_chain_freq_axis_mhz: Vec<f64> = (0..num_resp_bins)
        .map(|i| i as f64 * nyquist_max / (num_resp_bins - 1) as f64)
        .collect();

    let mut rf_chain_response_db = vec![0.0_f64; num_resp_bins];

    for &node_id in &node_chain {
        let node = &snarl[node_id];
        for (bin, &freq) in rf_chain_freq_axis_mhz.iter().enumerate() {
            match node {
                RfNode::Balun(b) => rf_chain_response_db[bin] -= b.model.insertion_loss_at(freq),
                RfNode::Filter(f) => rf_chain_response_db[bin] -= f.model.attenuation_at(freq),
                RfNode::Amplifier(a) => rf_chain_response_db[bin] += a.model.gain_db,
                RfNode::Attenuator(a) => rf_chain_response_db[bin] -= a.model.attenuation_db,
                RfNode::Splitter(s) => rf_chain_response_db[bin] -= s.model.total_loss_db(),
                RfNode::Mixer(m) => rf_chain_response_db[bin] -= m.model.conversion_loss_db,
                RfNode::PhaseShifter(p) => rf_chain_response_db[bin] -= p.model.insertion_loss_db,
                RfNode::DirectionalCoupler(d) => rf_chain_response_db[bin] -= d.model.insertion_loss_db,
                RfNode::S2p(s) => rf_chain_response_db[bin] += s.model.s21_gain_at(freq),
                RfNode::SignalSource(_) | RfNode::AdcInput(_) => {}
            }
        }
    }

    // Calculate dynamic Cascaded Gain, Friis Noise Figure, and OIP3
    let mut cum_gain_lin = 1.0_f64;
    let mut cum_noise_factor = 1.0_f64;
    let mut inv_cum_oip3_lin = 0.0_f64;

    for &node_id in &node_chain {
        let node = &snarl[node_id];
        let (stage_gain_db, stage_nf_db, stage_oip3_dbm) = match node {
            RfNode::Balun(b) => {
                let loss = b.model.insertion_loss_at(1000.0);
                (-loss, loss, 100.0) // Ideal passive
            }
            RfNode::Filter(f) => {
                let loss = f.model.attenuation_at(1000.0);
                (-loss, loss, 100.0)
            }
            RfNode::Amplifier(a) => (a.model.gain_db, a.model.noise_figure_db, a.model.p1db_dbm + 10.0),
            RfNode::Attenuator(a) => (-a.model.attenuation_db, a.model.attenuation_db, 10.0),
            RfNode::Splitter(s) => {
                let loss = s.model.total_loss_db();
                (-loss, loss, 100.0)
            }
            RfNode::Mixer(m) => (-m.model.conversion_loss_db, m.model.conversion_loss_db, 20.0),
            RfNode::PhaseShifter(p) => (-p.model.insertion_loss_db, p.model.insertion_loss_db, 100.0),
            RfNode::DirectionalCoupler(d) => (-d.model.insertion_loss_db, d.model.insertion_loss_db, 100.0),
            RfNode::S2p(s) => (s.model.s21_gain_at(1000.0), s.model.noise_figure_db, s.model.oip3_dbm),
            RfNode::SignalSource(_) | RfNode::AdcInput(_) => (0.0, 0.0, 100.0),
        };

        let g_stage = 10.0_f64.powf(stage_gain_db / 10.0);
        let f_stage = 10.0_f64.powf(stage_nf_db / 10.0);
        let oip3_stage = 10.0_f64.powf(stage_oip3_dbm / 10.0);

        if cum_gain_lin == 1.0 && cum_noise_factor == 1.0 {
            cum_noise_factor = f_stage;
        } else {
            cum_noise_factor += (f_stage - 1.0) / cum_gain_lin;
        }

        if oip3_stage < 1e9 {
            inv_cum_oip3_lin += 1.0 / (g_stage * oip3_stage);
        }

        cum_gain_lin *= g_stage;
    }

    let cascaded_gain_db = 10.0 * cum_gain_lin.max(1e-12).log10();
    let cascaded_nf_db = 10.0 * cum_noise_factor.max(1.0).log10();
    let cascaded_oip3_dbm = if inv_cum_oip3_lin > 1e-12 {
        10.0 * (1.0 / inv_cum_oip3_lin).log10()
    } else {
        100.0
    };

    Some(GraphEvaluationResult {
        samples: current_samples,
        rf_chain_response_db,
        rf_chain_freq_axis_mhz,
        cascaded_gain_db,
        cascaded_nf_db,
        cascaded_oip3_dbm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_snarl::Snarl;

    #[test]
    fn evaluate_graph_filters_signal() {
        let mut snarl = Snarl::<RfNode>::new();

        // Source node
        let src_id = snarl.insert_node(
            egui::pos2(0.0, 0.0),
            RfNode::SignalSource(SignalSourceNode::default()),
        );

        // Low-pass filter node (cutoff = 500 MHz)
        let mut fnode = FilterNode::default();
        fnode.model.cutoff_mhz = 500.0;
        let filter_id = snarl.insert_node(
            egui::pos2(200.0, 0.0),
            RfNode::Filter(fnode),
        );

        // ADC input node (Tile 0, Block 0)
        let adc_id = snarl.insert_node(
            egui::pos2(400.0, 0.0),
            RfNode::AdcInput(AdcInputNode {
                tile_index: 0,
                block_index: 0,
            }),
        );

        // Connect Source -> Filter -> ADC Input
        snarl.connect(
            egui_snarl::OutPinId { node: src_id, output: 0 },
            egui_snarl::InPinId { node: filter_id, input: 0 },
        );
        snarl.connect(
            egui_snarl::OutPinId { node: filter_id, output: 0 },
            egui_snarl::InPinId { node: adc_id, input: 0 },
        );

        let global_gen = crate::signal::SignalGenerator::default();
        let res = evaluate_graph(&snarl, 0, 0, 1024, 10000.0, &global_gen, 0.0);
        assert!(res.is_some());
        assert_eq!(res.unwrap().samples.len(), 1024);
    }

    #[test]
    fn friis_cascaded_noise_figure() {
        let mut snarl = Snarl::<RfNode>::new();
        let src_id = snarl.insert_node(
            egui::pos2(0.0, 0.0),
            RfNode::SignalSource(SignalSourceNode::default()),
        );

        // Stage 1 LNA: Gain = 20 dB, NF = 2 dB
        let mut amp1 = AmplifierNode::default();
        amp1.model.gain_db = 20.0;
        amp1.model.noise_figure_db = 2.0;
        let amp1_id = snarl.insert_node(egui::pos2(100.0, 0.0), RfNode::Amplifier(amp1));

        // Stage 2 Amp: Gain = 10 dB, NF = 8 dB
        let mut amp2 = AmplifierNode::default();
        amp2.model.gain_db = 10.0;
        amp2.model.noise_figure_db = 8.0;
        let amp2_id = snarl.insert_node(egui::pos2(200.0, 0.0), RfNode::Amplifier(amp2));

        let adc_id = snarl.insert_node(
            egui::pos2(300.0, 0.0),
            RfNode::AdcInput(AdcInputNode {
                tile_index: 0,
                block_index: 0,
            }),
        );

        snarl.connect(
            egui_snarl::OutPinId { node: src_id, output: 0 },
            egui_snarl::InPinId { node: amp1_id, input: 0 },
        );
        snarl.connect(
            egui_snarl::OutPinId { node: amp1_id, output: 0 },
            egui_snarl::InPinId { node: amp2_id, input: 0 },
        );
        snarl.connect(
            egui_snarl::OutPinId { node: amp2_id, output: 0 },
            egui_snarl::InPinId { node: adc_id, input: 0 },
        );

        let global_gen = crate::signal::SignalGenerator::default();
        let res = evaluate_graph(&snarl, 0, 0, 1024, 10000.0, &global_gen, 0.0).unwrap();

        // Total Gain = 30 dB
        assert!((res.cascaded_gain_db - 30.0).abs() < 1e-3);
        // Friis formula: F_total = F1 + (F2 - 1) / G1
        // F1 = 10^(2/10) = 1.5849, F2 = 10^(8/10) = 6.3095, G1 = 100
        // F_total = 1.5849 + 5.3095 / 100 = 1.6380 -> NF = 10 log10(1.6380) = 2.143 dB
        assert!((res.cascaded_nf_db - 2.143).abs() < 0.05);
    }
}
