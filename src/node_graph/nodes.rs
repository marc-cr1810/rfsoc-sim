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

/// Evaluation result containing generated samples and cumulative RF chain frequency response.
pub struct GraphEvaluationResult {
    pub samples: Vec<Complex<f64>>,
    pub rf_chain_response_db: Vec<f64>,
    pub rf_chain_freq_axis_mhz: Vec<f64>,
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
            SourceType::GlobalGenerator => global_signal_gen.generate(num_samples, sample_rate_mhz),
            SourceType::LocalGenerator => src.generator.generate(num_samples, sample_rate_mhz),
            SourceType::IqFile => src
                .file_loader
                .load()
                .unwrap_or_else(|_| src.generator.generate(num_samples, sample_rate_mhz)),
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
                RfNode::SignalSource(_) | RfNode::AdcInput(_) => {}
            }
        }
    }

    Some(GraphEvaluationResult {
        samples: current_samples,
        rf_chain_response_db,
        rf_chain_freq_axis_mhz,
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
        let res = evaluate_graph(&snarl, 0, 0, 1024, 10000.0, &global_gen);
        assert!(res.is_some());
        assert_eq!(res.unwrap().samples.len(), 1024);
    }
}
