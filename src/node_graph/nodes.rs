//! Node enum definitions, graph pin mapping and chain evaluation.

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
    Combiner(CombinerNode),
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
            RfNode::Combiner(_) => "Combiner",
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
            RfNode::Combiner(c) => c.model.num_inputs as usize,
            RfNode::AdcInput(_) => 1,
            _ => 1,
        }
    }

    /// Number of output pins.
    pub fn num_outputs(&self) -> usize {
        match self {
            RfNode::AdcInput(_) => 0,
            RfNode::Splitter(s) => s.model.num_outputs as usize,
            // Main line and coupled arm.
            RfNode::DirectionalCoupler(_) => 2,
            RfNode::Combiner(_) => 1,
            _ => 1,
        }
    }

    /// The component physics behind this node, if it has any.
    pub fn component(&self) -> Option<&dyn RfComponent> {
        match self {
            RfNode::Balun(n) => Some(&n.model),
            RfNode::Filter(n) => Some(&n.model),
            RfNode::Amplifier(n) => Some(&n.model),
            RfNode::Attenuator(n) => Some(&n.model),
            RfNode::Splitter(n) => Some(&n.model),
            RfNode::Combiner(n) => Some(&n.model),
            RfNode::Mixer(n) => Some(&n.model),
            RfNode::PhaseShifter(n) => Some(&n.model),
            RfNode::DirectionalCoupler(n) => Some(&n.model),
            RfNode::S2p(n) => Some(&n.model),
            RfNode::SignalSource(_) | RfNode::AdcInput(_) => None,
        }
    }

    /// Label for an output pin.
    pub fn output_label(&self, port: usize) -> String {
        match self {
            RfNode::Splitter(_) => format!("Out {port}"),
            RfNode::DirectionalCoupler(_) => {
                if port == COUPLER_COUPLED_PORT {
                    "Coupled".to_string()
                } else {
                    "Main".to_string()
                }
            }
            _ => "RF Out".to_string(),
        }
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BalunNode {
    pub model: BalunModel,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FilterNode {
    pub model: FilterModel,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AmplifierNode {
    pub model: AmplifierModel,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AttenuatorNode {
    pub model: AttenuatorModel,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SplitterNode {
    pub model: SplitterModel,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CombinerNode {
    pub model: CombinerModel,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MixerNode {
    pub model: MixerModel,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhaseShifterNode {
    pub model: PhaseShifterModel,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DirectionalCouplerNode {
    pub model: DirectionalCouplerModel,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct S2pNode {
    pub model: S2pModel,
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
        Self { tile_index: 0, block_index: 0 }
    }
}

// ---------------------------------------------------------------------------
// Node Graph Traversal & Signal Evaluation
// ---------------------------------------------------------------------------

use egui_snarl::{InPinId, NodeId, OutPinId, Snarl};
use num_complex::Complex;

/// Physical temperature and enable flag for the chain's thermal noise.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChainEnvironment {
    /// Whether components contribute their own thermal noise.
    pub thermal_noise: bool,
    /// Physical temperature of the chain in kelvin.
    pub temperature_k: f64,
    /// Frequency at which cascaded gain, noise figure and OIP3 are reported.
    pub analysis_freq_mhz: f64,
}

impl Default for ChainEnvironment {
    fn default() -> Self {
        Self { thermal_noise: true, temperature_k: 290.0, analysis_freq_mhz: 1000.0 }
    }
}

/// What one stage contributes, reported per node so the graph can annotate itself.
#[derive(Debug, Clone, Copy)]
pub struct NodeStats {
    /// Gain (positive) or loss (negative) at the analysis frequency, in dB.
    pub gain_db: f64,
    /// Noise figure at the analysis frequency, in dB.
    pub noise_figure_db: f64,
    /// Signal level leaving this node, in dBFS.
    pub output_level_dbfs: f64,
    /// Cumulative gain from the source up to and including this node, in dB.
    pub cumulative_gain_db: f64,
    /// Gain compression at this node's drive level, in dB. Zero for linear stages.
    pub compression_db: f64,
    /// Group delay contributed at the analysis frequency, in ns.
    pub group_delay_ns: f64,
}

/// Evaluation result: the waveform handed to the converter plus the chain's RF budget.
pub struct GraphEvaluationResult {
    pub samples: Vec<Complex<f64>>,
    pub rf_chain_response_db: Vec<f64>,
    pub rf_chain_freq_axis_mhz: Vec<f64>,
    pub cascaded_gain_db: f64,
    pub cascaded_nf_db: f64,
    pub cascaded_oip3_dbm: f64,
    /// Frequency the budget above was evaluated at, in MHz.
    pub analysis_freq_mhz: f64,
    /// Per-node annotations, in signal-flow order.
    pub node_stats: Vec<(NodeId, NodeStats)>,
    /// Nodes taking part in a feedback loop, which cannot be evaluated.
    pub cycle_nodes: Vec<NodeId>,
    /// True if any stage is compressing by more than 1 dB.
    pub compressing: bool,
}

/// Ceiling on the run-up generated ahead of a block, in wideband samples.
///
/// A 1 MHz filter at a 15 GHz simulation rate rings for tens of microseconds — hundreds of
/// thousands of samples — which is not worth computing every frame. Stages narrower than this
/// allows keep a little wrap-around; everything at a sane ratio of bandwidth to sample rate
/// settles well inside it.
const MAX_RUN_UP_SAMPLES: usize = 32768;

/// A deterministic per-node seed, so each stage's noise is independent but reproducible.
fn node_seed(id: NodeId) -> u64 {
    // NodeId's Debug form is stable within a session and unique per node.
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in format!("{id:?}").bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01B3);
    }
    h | 1
}

/// Follow input `port` of `node` back to whatever drives it.
fn upstream(snarl: &Snarl<RfNode>, node: NodeId, port: usize) -> Option<OutPinId> {
    snarl
        .in_pin(InPinId { node, input: port })
        .remotes
        .first()
        .copied()
}

/// Walk the graph backwards from `node`, gathering the waveform arriving at its input.
///
/// `path` carries the nodes currently being expanded so a feedback loop is detected rather
/// than recursed into forever; anything wired in a cycle is reported and evaluates to
/// nothing, which is the only sensible answer for a forward-only chain model.
fn evaluate_samples(
    snarl: &Snarl<RfNode>,
    node_id: NodeId,
    out_port: usize,
    num_samples: usize,
    ctx: &ChainCtx,
    global_signal_gen: &SignalGenerator,
    path: &mut Vec<NodeId>,
    cycles: &mut Vec<NodeId>,
    levels: &mut Vec<(NodeId, f64)>,
) -> Option<Vec<Complex<f64>>> {
    if path.contains(&node_id) {
        if !cycles.contains(&node_id) {
            cycles.push(node_id);
        }
        return None;
    }

    let node = &snarl[node_id];

    if let RfNode::SignalSource(src) = node {
        let raw = match src.source_type {
            SourceType::GlobalGenerator => {
                global_signal_gen.generate_at_time(num_samples, ctx.sample_rate_mhz, ctx.time_us)
            }
            SourceType::LocalGenerator => {
                src.generator
                    .generate_at_time(num_samples, ctx.sample_rate_mhz, ctx.time_us)
            }
            SourceType::IqFile => src
                .file_loader
                .generate_at_time(num_samples, ctx.sample_rate_mhz, ctx.time_us)
                .unwrap_or_else(|_| {
                    src.generator
                        .generate_at_time(num_samples, ctx.sample_rate_mhz, ctx.time_us)
                }),
        };
        // An antenna or a cable carries a real voltage, so this is where the analog domain
        // starts. Collapsing here rather than at the converter pin is what lets a mixer emit
        // both sidebands and a nonlinearity act on an unambiguous instantaneous voltage.
        let mut out: Vec<Complex<f64>> = raw.iter().map(|s| Complex::new(s.re, 0.0)).collect();
        // A matched source delivers kTB on top of whatever it is generating.
        add_source_noise(&mut out, &ctx.for_node(node_seed(node_id)));
        record_level(levels, node_id, &out);
        return Some(out);
    }

    path.push(node_id);

    let mut input_signals = Vec::new();
    for input_idx in 0..node.num_inputs() {
        if let Some(remote) = upstream(snarl, node_id, input_idx) {
            if let Some(samples) = evaluate_samples(
                snarl,
                remote.node,
                remote.output,
                num_samples,
                ctx,
                global_signal_gen,
                path,
                cycles,
                levels,
            ) {
                input_signals.push(samples);
            }
        }
    }

    path.pop();

    if input_signals.is_empty() {
        return None;
    }

    // A combiner sums voltages; everything else takes its first connected input.
    let mut combined = input_signals[0].clone();
    if matches!(node, RfNode::Combiner(_)) {
        for other in input_signals.iter().skip(1) {
            for (c, o) in combined.iter_mut().zip(other.iter()) {
                *c += *o;
            }
        }
    }

    let out = match node.component() {
        Some(c) => c.process(&combined, &ctx.for_node(node_seed(node_id)), out_port),
        None => combined,
    };
    record_level(levels, node_id, &out);
    Some(out)
}

/// Note the level leaving a node, replacing any earlier entry.
///
/// Levels are *measured* rather than accumulated from each stage's nominal gain, so they hold
/// up for a compressing amplifier or a source whose power is spread over a channel — neither of
/// which a running sum of decibels would get right.
fn record_level(levels: &mut Vec<(NodeId, f64)>, node_id: NodeId, samples: &[Complex<f64>]) {
    let level = rms_dbfs(samples);
    match levels.iter_mut().find(|(id, _)| *id == node_id) {
        Some(entry) => entry.1 = level,
        None => levels.push((node_id, level)),
    }
}

/// Root-mean-square level of a real waveform, in dBFS relative to a full-scale sine.
fn rms_dbfs(samples: &[Complex<f64>]) -> f64 {
    if samples.is_empty() {
        return -300.0;
    }
    let p: f64 = samples.iter().map(|s| s.re * s.re).sum::<f64>() / samples.len() as f64;
    // A full-scale sine has mean square 1/2, so that is the 0 dBFS reference.
    10.0 * (p / 0.5).max(1e-30).log10()
}

/// Trace the chain feeding an ADC input, from the source forward.
fn chain_order(
    snarl: &Snarl<RfNode>,
    adc_node_id: NodeId,
) -> (Vec<(NodeId, usize)>, Vec<NodeId>) {
    let mut chain: Vec<(NodeId, usize)> = Vec::new();
    let mut cycles = Vec::new();
    let mut seen: Vec<NodeId> = Vec::new();
    let mut cursor = upstream(snarl, adc_node_id, 0);

    while let Some(pin) = cursor {
        if seen.contains(&pin.node) {
            cycles.push(pin.node);
            break;
        }
        seen.push(pin.node);
        chain.push((pin.node, pin.output));

        let node = &snarl[pin.node];
        if matches!(node, RfNode::SignalSource(_)) || node.num_inputs() == 0 {
            break;
        }
        cursor = upstream(snarl, pin.node, 0);
    }

    chain.reverse();
    (chain, cycles)
}

/// Evaluate the RF chain feeding one ADC block.
pub fn evaluate_graph(
    snarl: &Snarl<RfNode>,
    target_tile: usize,
    target_block: usize,
    num_samples: usize,
    sample_rate_mhz: f64,
    global_signal_gen: &SignalGenerator,
    time_us: f64,
    env: &ChainEnvironment,
) -> Option<GraphEvaluationResult> {
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

    let (chain, mut cycle_nodes) = chain_order(snarl, adc_node_id);
    if chain.is_empty() {
        return None;
    }

    // Every frequency-domain stage multiplies in the frequency domain, which convolves
    // circularly: without a run-up, a filter's ringing wraps from the end of the block round
    // to the start and the result depends on the block length. Because the sources are
    // absolute-time aware, the fix is to start early and throw the run-up away. How early is
    // set by the slowest-ringing stage present.
    let settling_ns = chain
        .iter()
        .filter_map(|&(id, _)| snarl[id].component().map(|c| c.settling_ns()))
        .fold(0.0_f64, f64::max);
    // A stage far narrower than the simulation rate would ask for more run-up than is worth
    // computing; past this point the residual wrap-around is accepted.
    let want = (settling_ns * 1e-3 * sample_rate_mhz).ceil() as usize;
    let guard = crate::dsp::next_smooth_size(num_samples + want.clamp(512, MAX_RUN_UP_SAMPLES))
        - num_samples;
    let dt_us = 1.0 / sample_rate_mhz;
    let ctx = ChainCtx::new(
        sample_rate_mhz,
        time_us - guard as f64 * dt_us,
        env.thermal_noise.then_some(env.temperature_k),
    );

    let mut path = Vec::new();
    let mut levels: Vec<(NodeId, f64)> = Vec::new();
    let padded = evaluate_samples(
        snarl,
        adc_node_id,
        0,
        num_samples + guard,
        &ctx,
        global_signal_gen,
        &mut path,
        &mut cycle_nodes,
        &mut levels,
    )?;
    let samples: Vec<Complex<f64>> = padded.into_iter().skip(guard).collect();

    // Cumulative frequency response across the simulated band, for the spectrum overlay.
    let num_resp_bins = 256;
    let nyquist_max = sample_rate_mhz / 2.0;
    let rf_chain_freq_axis_mhz: Vec<f64> = (0..num_resp_bins)
        .map(|i| i as f64 * nyquist_max / (num_resp_bins - 1) as f64)
        .collect();

    let mut rf_chain_response_db = vec![0.0_f64; num_resp_bins];
    for &(node_id, port) in &chain {
        if let Some(c) = snarl[node_id].component() {
            for (bin, &freq) in rf_chain_freq_axis_mhz.iter().enumerate() {
                rf_chain_response_db[bin] += c.response_db(freq, port);
            }
        }
    }

    // Cascaded gain, Friis noise figure and IP3, all evaluated at the analysis frequency
    // rather than at a fixed 1 GHz, so a filter's stage loss tracks the signal it is passing.
    let f_a = env.analysis_freq_mhz.max(0.0);
    let mut cum_gain_lin = 1.0_f64;
    let mut cum_noise_factor = 1.0_f64;
    let mut inv_iip3 = 0.0_f64;
    let mut node_stats: Vec<(NodeId, NodeStats)> = Vec::new();
    let mut compressing = false;

    let measured = |id: NodeId| -> Option<f64> {
        levels.iter().find(|(n, _)| *n == id).map(|(_, l)| *l)
    };
    // Level arriving at the stage being examined, which is what sets its compression.
    let mut drive_dbfs = -300.0_f64;

    for &(node_id, port) in &chain {
        let node = &snarl[node_id];
        let out_level = measured(node_id).unwrap_or(-300.0);
        let Some(comp) = node.component() else {
            node_stats.push((
                node_id,
                NodeStats {
                    gain_db: 0.0,
                    noise_figure_db: 0.0,
                    output_level_dbfs: out_level,
                    cumulative_gain_db: 10.0 * cum_gain_lin.max(1e-12).log10(),
                    compression_db: 0.0,
                    group_delay_ns: 0.0,
                },
            ));
            drive_dbfs = out_level;
            continue;
        };

        let stage_gain_db = comp.response_db(f_a, port);
        let stage_nf_db = comp.noise_figure_db_at(f_a);
        let g_stage = 10.0_f64.powf(stage_gain_db / 10.0);
        let f_stage = 10.0_f64.powf(stage_nf_db / 10.0);

        // Friis: F_total = F₁ + (F₂−1)/G₁ + …, accumulated with the gain seen so far.
        cum_noise_factor += (f_stage - 1.0) / cum_gain_lin;

        // Input-referred IP3 cascade: 1/IIP3 = Σ G(before stage i)/IIP3ᵢ. Referring to the
        // input is what makes the recursion forward-only; the output figure follows from the
        // total gain at the end. Passive stages are ideally linear and contribute nothing.
        if let Some(oip3_dbm) = comp.oip3_dbm() {
            let iip3_lin = 10.0_f64.powf((oip3_dbm - stage_gain_db) / 10.0);
            if iip3_lin > 0.0 {
                inv_iip3 += cum_gain_lin / iip3_lin;
            }
        }

        // Compression at this stage's actual drive level.
        let mut compression_db = 0.0;
        if let RfNode::Amplifier(a) = node {
            if let Some(fit) = a.model.nonlinearity() {
                // RMS dBFS back to the peak amplitude of an equivalent sine.
                let drive_amp = 10.0_f64.powf(drive_dbfs / 20.0);
                compression_db = fit.compression_db(drive_amp);
            }
        }
        if compression_db < -1.0 {
            compressing = true;
        }

        let group_delay_ns = match node {
            RfNode::Filter(f) => f.model.group_delay_ns(f_a.max(1.0)),
            RfNode::PhaseShifter(p) if p.model.kind == PhaseShiftKind::TrueDelay => {
                p.model.delay_ns()
            }
            _ => 0.0,
        };

        cum_gain_lin *= g_stage;
        drive_dbfs = out_level;

        node_stats.push((
            node_id,
            NodeStats {
                gain_db: stage_gain_db,
                noise_figure_db: stage_nf_db,
                output_level_dbfs: out_level,
                cumulative_gain_db: 10.0 * cum_gain_lin.max(1e-12).log10(),
                compression_db,
                group_delay_ns,
            },
        ));
    }

    let cascaded_gain_db = 10.0 * cum_gain_lin.max(1e-12).log10();
    let cascaded_nf_db = 10.0 * cum_noise_factor.max(1.0).log10();
    let cascaded_oip3_dbm = if inv_iip3 > 0.0 {
        10.0 * (1.0 / inv_iip3).log10() + cascaded_gain_db
    } else {
        f64::INFINITY
    };

    Some(GraphEvaluationResult {
        samples,
        rf_chain_response_db,
        rf_chain_freq_axis_mhz,
        cascaded_gain_db,
        cascaded_nf_db,
        cascaded_oip3_dbm,
        analysis_freq_mhz: f_a,
        node_stats,
        cycle_nodes,
        compressing,
    })
}

/// Whether connecting `from` to `to` would close a loop in the graph.
///
/// A forward-only chain model cannot evaluate feedback, so the wiring is refused at the point
/// the user draws it rather than blowing the stack during evaluation.
pub fn would_create_cycle(snarl: &Snarl<RfNode>, from: NodeId, to: NodeId) -> bool {
    if from == to {
        return true;
    }
    // Walk backwards from `from`: if `to` already feeds it, the new wire closes a loop.
    let mut stack = vec![from];
    let mut seen: Vec<NodeId> = Vec::new();
    while let Some(node) = stack.pop() {
        if node == to {
            return true;
        }
        if seen.contains(&node) {
            continue;
        }
        seen.push(node);
        for input in 0..snarl[node].num_inputs() {
            if let Some(remote) = upstream(snarl, node, input) {
                stack.push(remote.node);
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_snarl::Snarl;

    fn adc(tile: usize, block: usize) -> RfNode {
        RfNode::AdcInput(AdcInputNode { tile_index: tile, block_index: block })
    }

    fn wire(snarl: &mut Snarl<RfNode>, from: NodeId, out: usize, to: NodeId, input: usize) {
        snarl.connect(
            OutPinId { node: from, output: out },
            InPinId { node: to, input },
        );
    }

    /// Source -> [stages] -> ADC, wired in order.
    ///
    /// The source runs off the generator handed to `evaluate_graph`, so a test controls the
    /// stimulus by passing one in rather than by reaching into the node.
    fn build_chain(stages: Vec<RfNode>) -> (Snarl<RfNode>, Vec<NodeId>) {
        let mut snarl = Snarl::<RfNode>::new();
        let src = SignalSourceNode {
            source_type: SourceType::GlobalGenerator,
            ..Default::default()
        };
        let mut ids = vec![snarl.insert_node(egui::pos2(0.0, 0.0), RfNode::SignalSource(src))];
        for (i, s) in stages.into_iter().enumerate() {
            ids.push(snarl.insert_node(egui::pos2(100.0 * (i + 1) as f32, 0.0), s));
        }
        let adc_id = snarl.insert_node(egui::pos2(900.0, 0.0), adc(0, 0));
        ids.push(adc_id);
        for w in ids.windows(2) {
            wire(&mut snarl, w[0], 0, w[1], 0);
        }
        (snarl, ids)
    }

    fn quiet_env() -> ChainEnvironment {
        ChainEnvironment { thermal_noise: false, ..Default::default() }
    }

    #[test]
    fn evaluate_graph_filters_signal() {
        let mut fnode = FilterNode::default();
        fnode.model.cutoff_mhz = 500.0;
        let (snarl, _) = build_chain(vec![RfNode::Filter(fnode)]);
        let global_gen = SignalGenerator::default();
        let res = evaluate_graph(&snarl, 0, 0, 1024, 10000.0, &global_gen, 0.0, &quiet_env());
        assert!(res.is_some());
        assert_eq!(res.unwrap().samples.len(), 1024);
    }

    #[test]
    fn friis_cascaded_noise_figure() {
        let amp = |gain, nf| {
            RfNode::Amplifier(AmplifierNode {
                model: AmplifierModel {
                    gain_db: gain,
                    noise_figure_db: nf,
                    p1db_dbm: 30.0,
                    oip3_dbm: 45.0,
                    bandwidth_mhz: 0.0,
                },
            })
        };
        let (snarl, _) = build_chain(vec![amp(20.0, 2.0), amp(10.0, 8.0)]);
        let global_gen = SignalGenerator::default();
        let res = evaluate_graph(&snarl, 0, 0, 1024, 10000.0, &global_gen, 0.0, &quiet_env()).unwrap();

        assert!((res.cascaded_gain_db - 30.0).abs() < 1e-3);
        // F_total = F1 + (F2-1)/G1 = 1.5849 + 5.3095/100 = 1.638 -> 2.143 dB
        assert!((res.cascaded_nf_db - 2.143).abs() < 0.05);
    }

    #[test]
    fn cascaded_oip3_cannot_beat_its_stages() {
        let stage = |g: f64, oip3: f64| {
            RfNode::S2p(S2pNode {
                model: S2pModel {
                    name: "stage".into(),
                    s21_table: vec![(1.0, g, 0.0), (10000.0, g, 0.0)],
                    s11_table: vec![],
                    s22_table: vec![],
                    noise_figure_db: 0.0,
                    oip3_dbm: oip3,
                    use_measured_phase: false,
                },
            })
        };
        let (snarl, _) = build_chain(vec![stage(10.0, 30.0), stage(10.0, 30.0)]);
        let g = SignalGenerator::default();
        let res = evaluate_graph(&snarl, 0, 0, 1024, 15000.0, &g, 0.0, &quiet_env()).unwrap();

        assert!((res.cascaded_gain_db - 20.0).abs() < 0.01);
        // Textbook: IIP3 per stage = 20 dBm, 1/IIP3 = 1/100 + 10/100 -> 9.59 dBm,
        // so OIP3 = 9.59 + 20 = 29.6 dBm. Never better than a single stage.
        assert!(
            (res.cascaded_oip3_dbm - 29.6).abs() < 0.3,
            "cascaded OIP3 {} dBm, expected 29.6",
            res.cascaded_oip3_dbm
        );
    }

    #[test]
    fn passive_pad_does_not_dominate_linearity() {
        // A resistive pad is essentially perfectly linear; putting one in front of an
        // amplifier must improve the cascade's linearity, never destroy it.
        let amp = RfNode::Amplifier(AmplifierNode {
            model: AmplifierModel {
                gain_db: 20.0,
                noise_figure_db: 1.5,
                p1db_dbm: 20.0,
                oip3_dbm: 30.0,
                bandwidth_mhz: 0.0,
            },
        });
        let pad = RfNode::Attenuator(AttenuatorNode {
            model: AttenuatorModel { attenuation_db: 6.0 },
        });
        let g = SignalGenerator::default();

        let (a, _) = build_chain(vec![amp.clone()]);
        let alone = evaluate_graph(&a, 0, 0, 1024, 15000.0, &g, 0.0, &quiet_env()).unwrap();
        let (b, _) = build_chain(vec![pad, amp]);
        let padded = evaluate_graph(&b, 0, 0, 1024, 15000.0, &g, 0.0, &quiet_env()).unwrap();

        assert!((alone.cascaded_oip3_dbm - 30.0).abs() < 0.01);
        // The pad shifts the whole chain's gain down 6 dB but the amp still saturates at the
        // same output power, so OIP3 is unchanged.
        assert!(
            (padded.cascaded_oip3_dbm - 30.0).abs() < 0.01,
            "a 6 dB pad should not change OIP3, got {}",
            padded.cascaded_oip3_dbm
        );
        // Noise figure does get worse by the full 6 dB, which is the real cost.
        assert!((padded.cascaded_nf_db - (alone.cascaded_nf_db + 6.0)).abs() < 0.1);
    }

    #[test]
    fn cascade_metrics_track_the_analysis_frequency() {
        // A 1 GHz low-pass is transparent at 100 MHz and deep in the stopband at 4 GHz.
        let mut f = FilterNode::default();
        f.model.cutoff_mhz = 1000.0;
        f.model.order = 4;
        f.model.insertion_loss_db = 0.0;
        let (snarl, _) = build_chain(vec![RfNode::Filter(f)]);
        let g = SignalGenerator::default();

        let at_100 = evaluate_graph(
            &snarl, 0, 0, 1024, 15000.0, &g, 0.0,
            &ChainEnvironment { analysis_freq_mhz: 100.0, ..quiet_env() },
        )
        .unwrap();
        let at_4000 = evaluate_graph(
            &snarl, 0, 0, 1024, 15000.0, &g, 0.0,
            &ChainEnvironment { analysis_freq_mhz: 4000.0, ..quiet_env() },
        )
        .unwrap();

        assert!(at_100.cascaded_gain_db.abs() < 0.1);
        assert!(
            at_4000.cascaded_gain_db < -40.0,
            "stopband stage loss should show up in the budget, got {}",
            at_4000.cascaded_gain_db
        );
        assert!(at_4000.cascaded_nf_db > 40.0);
    }

    #[test]
    fn lna_placement_changes_snr() {
        // The headline claim in the docs: where the LNA sits decides the system noise figure.
        // Friis puts a 20 dB / 2 dB NF LNA ahead of 20 dB of cable at
        //   F = 1.585 + (100−1)/100 = 2.575  ->  4.11 dB,
        // and behind it at
        //   F = 100 + (1.585−1)/0.01 = 158.5 ->  22.0 dB.
        // Both the reported budget and the waveform have to agree with that.
        let lna = RfNode::Amplifier(AmplifierNode {
            model: AmplifierModel {
                gain_db: 20.0,
                noise_figure_db: 2.0,
                p1db_dbm: 30.0,
                oip3_dbm: 45.0,
                bandwidth_mhz: 0.0,
            },
        });
        let cable = RfNode::Attenuator(AttenuatorNode {
            model: AttenuatorModel { attenuation_db: 20.0 },
        });

        let env = ChainEnvironment { thermal_noise: true, ..Default::default() };
        let fs = 15000.0;
        let n = 1 << 14;

        let measure = |lna_first: bool| -> (f64, f64) {
            let stages = if lna_first {
                vec![lna.clone(), cable.clone()]
            } else {
                vec![cable.clone(), lna.clone()]
            };
            let (snarl, _) = build_chain(stages);

            // Signal alone, then noise alone, so the SNR is exact.
            let mut tone = SignalGenerator::default();
            tone.tones[0].frequency_mhz = 1000.0;
            tone.tones[0].amplitude_dbfs = -40.0;
            tone.tones[0].modulation = crate::signal::ToneModulation::Cw;
            tone.noise_enabled = false;

            let sig = evaluate_graph(&snarl, 0, 0, n, fs, &tone, 0.0,
                &ChainEnvironment { thermal_noise: false, ..env }).unwrap();
            let p_sig: f64 = sig.samples.iter().map(|s| s.re * s.re).sum::<f64>() / n as f64;

            let quiet = SignalGenerator { tones: vec![], noise_enabled: false, ..Default::default() };
            let nz = evaluate_graph(&snarl, 0, 0, n, fs, &quiet, 0.0, &env).unwrap();
            let p_noise: f64 = nz.samples.iter().map(|s| s.re * s.re).sum::<f64>() / n as f64;

            (10.0 * (p_sig / p_noise).log10(), nz.cascaded_nf_db)
        };

        let (snr_first, nf_first) = measure(true);
        let (snr_last, nf_last) = measure(false);

        assert!((nf_first - 4.11).abs() < 0.3, "LNA-first NF {nf_first}");
        assert!((nf_last - 22.0).abs() < 0.3, "LNA-last NF {nf_last}");
        // The measured SNR penalty has to match the difference in noise figure, which is the
        // whole point: the budget and the waveform are the same physics, not two models.
        let penalty = snr_first - snr_last;
        let predicted = nf_last - nf_first;
        assert!(
            (penalty - predicted).abs() < 1.0,
            "measured penalty {penalty} dB against a predicted {predicted} dB \
             (first {snr_first}, last {snr_last})"
        );
        assert!(penalty > 15.0, "putting the LNA last should hurt badly, got {penalty} dB");
    }

    #[test]
    fn thermal_noise_floor_sits_at_ktb() {
        let quiet = SignalGenerator { tones: vec![], noise_enabled: false, ..Default::default() };
        let fs = 15000.0;
        let n = 1 << 14;

        let kt_b = thermal_noise_power(fs, 290.0);
        let env = ChainEnvironment { thermal_noise: true, ..Default::default() };
        let floor_db = |stages: Vec<RfNode>| -> f64 {
            let (snarl, _) = build_chain(stages);
            let res = evaluate_graph(&snarl, 0, 0, n, fs, &quiet, 0.0, &env).unwrap();
            let p: f64 = res.samples.iter().map(|s| s.re * s.re).sum::<f64>() / n as f64;
            10.0 * (p / kt_b).log10()
        };

        // A bare wire delivers the source's available noise and nothing more.
        assert!(floor_db(vec![]).abs() < 0.5, "bare wire at {} dB", floor_db(vec![]));

        // Any lossy passive at ambient still hands on exactly kTB, however lossy it is: the
        // loss it applies to the incoming noise is made up by its own contribution.
        for atten in [3.0, 10.0, 30.0] {
            let db = floor_db(vec![RfNode::Attenuator(AttenuatorNode {
                model: AttenuatorModel { attenuation_db: atten },
            })]);
            assert!(db.abs() < 0.6, "{atten} dB pad put the floor at {db} dB from kTB");
        }

        // An amplifier raises it by F·G, as its noise figure says it must.
        let db = floor_db(vec![RfNode::Amplifier(AmplifierNode {
            model: AmplifierModel {
                gain_db: 20.0,
                noise_figure_db: 3.0,
                p1db_dbm: 30.0,
                oip3_dbm: 45.0,
                bandwidth_mhz: 0.0,
            },
        })]);
        assert!((db - 23.0).abs() < 0.6, "amplified floor at {db} dB, expected 23");
    }

    #[test]
    fn feedback_loop_is_detected_not_recursed() {
        // Splitter -> amp -> combiner -> back into the splitter.
        let mut snarl = Snarl::<RfNode>::new();
        let src = snarl.insert_node(egui::pos2(0.0, 0.0), RfNode::SignalSource(SignalSourceNode::default()));
        let comb = snarl.insert_node(egui::pos2(100.0, 0.0), RfNode::Combiner(CombinerNode::default()));
        let split = snarl.insert_node(egui::pos2(200.0, 0.0), RfNode::Splitter(SplitterNode::default()));
        let amp = snarl.insert_node(egui::pos2(300.0, 0.0), RfNode::Amplifier(AmplifierNode::default()));
        let adc_id = snarl.insert_node(egui::pos2(400.0, 0.0), adc(0, 0));

        wire(&mut snarl, src, 0, comb, 0);
        wire(&mut snarl, comb, 0, split, 0);
        wire(&mut snarl, split, 0, amp, 0);
        wire(&mut snarl, split, 1, adc_id, 0);
        // The wire that closes the loop.
        wire(&mut snarl, amp, 0, comb, 1);

        assert!(would_create_cycle(&snarl, amp, comb));

        let g = SignalGenerator::default();
        // Must return rather than overflow the stack.
        let res = evaluate_graph(&snarl, 0, 0, 512, 15000.0, &g, 0.0, &quiet_env()).unwrap();
        assert!(!res.cycle_nodes.is_empty(), "the loop should be reported");
        assert_eq!(res.samples.len(), 512);
    }

    #[test]
    fn cycle_check_allows_legitimate_reconvergence() {
        // Splitter feeding two branches back into a combiner is a diamond, not a loop.
        let mut snarl = Snarl::<RfNode>::new();
        let src = snarl.insert_node(egui::pos2(0.0, 0.0), RfNode::SignalSource(SignalSourceNode::default()));
        let split = snarl.insert_node(egui::pos2(100.0, 0.0), RfNode::Splitter(SplitterNode::default()));
        let a = snarl.insert_node(egui::pos2(200.0, -50.0), RfNode::Attenuator(AttenuatorNode::default()));
        let b = snarl.insert_node(egui::pos2(200.0, 50.0), RfNode::Attenuator(AttenuatorNode::default()));
        let comb = snarl.insert_node(egui::pos2(300.0, 0.0), RfNode::Combiner(CombinerNode::default()));
        let adc_id = snarl.insert_node(egui::pos2(400.0, 0.0), adc(0, 0));

        wire(&mut snarl, src, 0, split, 0);
        wire(&mut snarl, split, 0, a, 0);
        wire(&mut snarl, split, 1, b, 0);
        wire(&mut snarl, a, 0, comb, 0);
        assert!(!would_create_cycle(&snarl, b, comb));
        wire(&mut snarl, b, 0, comb, 1);
        wire(&mut snarl, comb, 0, adc_id, 0);

        let g = SignalGenerator::default();
        let res = evaluate_graph(&snarl, 0, 0, 512, 15000.0, &g, 0.0, &quiet_env()).unwrap();
        assert!(res.cycle_nodes.is_empty());
    }

    #[test]
    fn coupler_ports_carry_different_levels() {
        let g = SignalGenerator::default();
        let coupler = DirectionalCouplerNode {
            model: DirectionalCouplerModel {
                coupling_db: 20.0,
                insertion_loss_db: 0.0,
                directivity_db: 25.0,
            },
        };

        // Main line into the ADC, then the coupled arm into the ADC, same graph shape.
        let level = |port: usize| -> f64 {
            let mut snarl = Snarl::<RfNode>::new();
            let mut src = SignalSourceNode::default();
            src.source_type = SourceType::LocalGenerator;
            src.generator.noise_enabled = false;
            src.generator.tones[0].modulation = crate::signal::ToneModulation::Cw;
            src.generator.tones[0].amplitude_dbfs = 0.0;
            let s = snarl.insert_node(egui::pos2(0.0, 0.0), RfNode::SignalSource(src));
            let c = snarl.insert_node(egui::pos2(100.0, 0.0), RfNode::DirectionalCoupler(coupler.clone()));
            let a = snarl.insert_node(egui::pos2(200.0, 0.0), adc(0, 0));
            wire(&mut snarl, s, 0, c, 0);
            wire(&mut snarl, c, port, a, 0);
            let res = evaluate_graph(&snarl, 0, 0, 4096, 15000.0, &g, 0.0, &quiet_env()).unwrap();
            rms_dbfs(&res.samples)
        };

        let main = level(0);
        let coupled = level(COUPLER_COUPLED_PORT);
        assert!((main - coupled - 20.0).abs() < 0.2, "main {main}, coupled {coupled}");
    }

    #[test]
    fn node_stats_track_levels_through_the_chain() {
        let amp = RfNode::Amplifier(AmplifierNode {
            model: AmplifierModel {
                gain_db: 20.0,
                noise_figure_db: 2.0,
                p1db_dbm: 30.0,
                oip3_dbm: 45.0,
                bandwidth_mhz: 0.0,
            },
        });
        let pad = RfNode::Attenuator(AttenuatorNode {
            model: AttenuatorModel { attenuation_db: 6.0 },
        });
        let (snarl, _) = build_chain(vec![amp, pad]);
        let mut g = SignalGenerator::default();
        g.tones[0].amplitude_dbfs = -40.0;
        g.tones[0].modulation = crate::signal::ToneModulation::Cw;
        g.noise_enabled = false;

        let res = evaluate_graph(&snarl, 0, 0, 4096, 15000.0, &g, 0.0, &quiet_env()).unwrap();
        // Source, amp, pad.
        assert_eq!(res.node_stats.len(), 3);
        let amp_stats = res.node_stats[1].1;
        let pad_stats = res.node_stats[2].1;
        assert!((amp_stats.gain_db - 20.0).abs() < 0.01);
        assert!((pad_stats.gain_db + 6.0).abs() < 0.01);
        assert!((pad_stats.cumulative_gain_db - 14.0).abs() < 0.01);
        // The reported output level is measured off the waveform rather than accumulated from
        // nominal gains. It covers the run-up as well as the kept block, so it agrees with the
        // trimmed view to a rounding error rather than exactly.
        let measured = rms_dbfs(&res.samples);
        assert!(
            (pad_stats.output_level_dbfs - measured).abs() < 0.05,
            "reported {} vs measured {measured}",
            pad_stats.output_level_dbfs
        );
    }

    #[test]
    fn levels_hold_up_for_a_wideband_channel_source() {
        // A source whose power is spread over a channel rather than concentrated in a line.
        // Accumulating decibels from each stage's nominal gain cannot see the difference;
        // measuring the waveform can.
        let pad = RfNode::Attenuator(AttenuatorNode {
            model: AttenuatorModel { attenuation_db: 6.0 },
        });
        let (snarl, _) = build_chain(vec![pad]);
        let channel = SignalGenerator {
            tones: vec![crate::signal::Tone {
                frequency_mhz: 2000.0,
                amplitude_dbfs: -10.0,
                phase_deg: 0.0,
                bandwidth_mhz: 200.0,
                modulation: crate::signal::ToneModulation::Cw,
            }],
            noise_floor_dbfs: -200.0,
            noise_enabled: false,
        };

        let res = evaluate_graph(&snarl, 0, 0, 8192, 15000.0, &channel, 0.0, &quiet_env()).unwrap();
        let src_level = res.node_stats[0].1.output_level_dbfs;
        let pad_level = res.node_stats[1].1.output_level_dbfs;
        // The channel carries the same power the carrier would have, so the source sits at its
        // configured level and the pad takes its 6 dB off.
        assert!((src_level + 10.0).abs() < 0.5, "source level {src_level}");
        assert!((pad_level - (src_level - 6.0)).abs() < 0.1, "pad level {pad_level}");
    }

    #[test]
    fn compression_is_flagged_when_an_amp_is_overdriven() {
        let amp = RfNode::Amplifier(AmplifierNode {
            model: AmplifierModel {
                gain_db: 20.0,
                noise_figure_db: 2.0,
                p1db_dbm: -20.0,
                oip3_dbm: -8.0,
                bandwidth_mhz: 0.0,
            },
        });
        let (snarl, _) = build_chain(vec![amp]);
        let mut hot = SignalGenerator::default();
        hot.tones[0].amplitude_dbfs = -3.0;
        hot.tones[0].modulation = crate::signal::ToneModulation::Cw;
        hot.noise_enabled = false;

        let res = evaluate_graph(&snarl, 0, 0, 4096, 15000.0, &hot, 0.0, &quiet_env()).unwrap();
        assert!(res.compressing, "an overdriven amp should raise the flag");
        assert!(res.node_stats[1].1.compression_db < -1.0);
    }

    #[test]
    fn filtered_output_is_independent_of_block_length() {
        // Every frequency-domain stage multiplies in the frequency domain, which convolves
        // circularly: without a run-up, the tail of the block folds back onto its head and the
        // answer depends on how many samples were asked for. A physical filter has no such
        // dependence, so evaluating a window twice at different lengths must agree sample for
        // sample. This is the property the pre-roll buys.
        let mut f = FilterNode::default();
        f.model.filter_type = FilterType::BandPass;
        f.model.cutoff_mhz = 1000.0;
        f.model.bandwidth_mhz = 40.0;
        f.model.order = 4;
        f.model.insertion_loss_db = 0.0;
        let (snarl, _) = build_chain(vec![RfNode::Filter(f)]);

        // A pulse train gives the filter a transient to ring on.
        let g = SignalGenerator {
            tones: vec![crate::signal::Tone {
                frequency_mhz: 1000.0,
                amplitude_dbfs: 0.0,
                phase_deg: 0.0,
                bandwidth_mhz: 0.0,
                modulation: crate::signal::ToneModulation::PulsedRadar {
                    pulse_width_us: 0.2,
                    pri_us: 0.5,
                    rise_ns: 0.0,
                    chirp_mhz: 0.0,
                },
            }],
            noise_floor_dbfs: -200.0,
            noise_enabled: false,
        };

        let short = evaluate_graph(&snarl, 0, 0, 4096, 15000.0, &g, 3.0, &quiet_env()).unwrap();
        let long = evaluate_graph(&snarl, 0, 0, 12288, 15000.0, &g, 3.0, &quiet_env()).unwrap();

        let peak = long.samples.iter().map(|s| s.re.abs()).fold(0.0, f64::max);
        assert!(peak > 0.1, "the test needs actual signal in the window, got {peak}");

        let worst = short
            .samples
            .iter()
            .zip(long.samples.iter())
            .map(|(a, b)| (a.re - b.re).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 1e-9 * peak,
            "block length changed the waveform by {worst} against a peak of {peak}"
        );
    }
}
