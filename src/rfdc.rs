//! RFDC hardware data model for the ZU48DR RFSoC.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Complete RFDC configuration for the ZU48DR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RfdcConfig {
    pub adc_tiles: [AdcTile; 4],
}

impl Default for RfdcConfig {
    fn default() -> Self {
        Self {
            adc_tiles: std::array::from_fn(|i| AdcTile::new(i)),
        }
    }
}

impl RfdcConfig {
    pub fn active_adc_blocks(&self) -> impl Iterator<Item = (usize, usize, &AdcBlock)> {
        self.adc_tiles.iter().enumerate().flat_map(|(ti, tile)| {
            if !tile.enabled {
                return Vec::new();
            }
            tile.blocks
                .iter()
                .enumerate()
                .filter(|(_, b)| b.enabled)
                .map(|(bi, block)| (ti, bi, block))
                .collect::<Vec<_>>()
        })
    }

    pub fn adc_block_mut(&mut self, tile: usize, block: usize) -> &mut AdcBlock {
        &mut self.adc_tiles[tile].blocks[block]
    }

    pub fn adc_block(&self, tile: usize, block: usize) -> &AdcBlock {
        &self.adc_tiles[tile].blocks[block]
    }
}

/// A single ADC tile containing 2 converter blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdcTile {
    pub index: usize,
    pub enabled: bool,
    pub sample_rate_gsps: f64,
    pub pll_enabled: bool,
    pub ref_clk_mhz: f64,
    pub blocks: [AdcBlock; 2],
    pub sync_group: Option<u8>,
    pub sysref_phase: f64,
}

impl AdcTile {
    pub fn new(index: usize) -> Self {
        Self {
            index,
            enabled: true,
            sample_rate_gsps: 4.0,
            pll_enabled: true,
            ref_clk_mhz: 250.0, // 250 MHz * 16 = 4000 MHz
            blocks: [AdcBlock::new(0), AdcBlock::new(1)],
            sync_group: None,
            sysref_phase: 0.0,
        }
    }

    pub fn validate_pll(&self) -> Option<String> {
        if !self.pll_enabled {
            return None;
        }
        let fs_mhz = self.sample_rate_mhz();
        let mult = fs_mhz / self.ref_clk_mhz;
        
        if mult < 2.0 || mult > 100.0 {
            return Some(format!("⚠ Target sample rate {:.1} MHz requires impossible PLL multiplier ({:.2}x).", fs_mhz, mult));
        }

        // Real RFSoC PLLs support integer and some fractional multipliers, but a totally 
        // irrational/unaligned ratio is unachievable. We'll warn if it's not a clean fraction.
        let is_clean = (mult * 1000.0).fract().abs() < 1e-6;
        if !is_clean {
            return Some(format!("⚠ Sample rate {:.1} MHz cannot be cleanly derived from {} MHz Ref Clock.", fs_mhz, self.ref_clk_mhz));
        }
        
        None
    }

    pub fn nyquist_bw_mhz(&self) -> f64 {
        self.sample_rate_gsps * 1000.0 / 2.0
    }

    pub fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_gsps * 1e9
    }

    pub fn sample_rate_mhz(&self) -> f64 {
        self.sample_rate_gsps * 1000.0
    }

    /// Calculate Nyquist zone and fine mixer NCO frequency for a target RF center frequency.
    pub fn auto_tune(&self, target_freq_mhz: f64) -> AutoTuneResult {
        let fs_mhz = self.sample_rate_mhz();
        let f_nyq = fs_mhz / 2.0;

        if f_nyq <= 0.0 || target_freq_mhz <= 0.0 {
            return AutoTuneResult {
                target_freq_mhz,
                zone_index: 1,
                is_even_zone: false,
                nyquist_zone: NyquistZone::Odd,
                alias_freq_mhz: target_freq_mhz,
                nco_freq_mhz: target_freq_mhz,
            };
        }

        let zone_index = (target_freq_mhz / f_nyq).floor() as u32 + 1;
        let is_even_zone = zone_index % 2 == 0;
        let nyquist_zone = if is_even_zone { NyquistZone::Even } else { NyquistZone::Odd };

        let alias_freq_mhz = if is_even_zone {
            (zone_index as f64 * f_nyq) - target_freq_mhz
        } else {
            target_freq_mhz - ((zone_index as f64 - 1.0) * f_nyq)
        };

        // NCO frequency to downconvert alias to 0 Hz baseband
        let nco_freq_mhz = if is_even_zone {
            -alias_freq_mhz
        } else {
            alias_freq_mhz
        };

        AutoTuneResult {
            target_freq_mhz,
            zone_index,
            is_even_zone,
            nyquist_zone,
            alias_freq_mhz,
            nco_freq_mhz,
        }
    }
}

/// Results of the Auto-Tune frequency calculation.
#[derive(Debug, Clone, Copy)]
pub struct AutoTuneResult {
    pub target_freq_mhz: f64,
    pub zone_index: u32,
    pub is_even_zone: bool,
    pub nyquist_zone: NyquistZone,
    pub alias_freq_mhz: f64,
    pub nco_freq_mhz: f64,
}

/// ADC Non-idealities configuration for hardware-level distortion simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdcNonIdealities {
    pub enabled: bool,
    pub enob: f64,
    pub quantization_bits: u8,
    pub hd2_dbc: f64,
    pub hd3_dbc: f64,
    pub interleaving_spur_dbc: f64,
}

impl Default for AdcNonIdealities {
    fn default() -> Self {
        Self {
            enabled: false,
            enob: 11.5,
            quantization_bits: 14, // Gen 3 ZU48DR is 14-bit
            hd2_dbc: -70.0,
            hd3_dbc: -75.0,
            interleaving_spur_dbc: -68.0,
        }
    }
}

/// A single ADC converter block with its DDC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdcBlock {
    pub index: usize,
    pub enabled: bool,
    pub dsa_db: f64,
    pub nyquist_zone: NyquistZone,
    pub planner_zone: u32,
    pub mixer_settings: MixerSettings,
    pub qmc_settings: QmcSettings,
    pub decimation: DecimationFactor,
    pub calibration_mode: CalibrationMode,
    pub non_idealities: AdcNonIdealities,
    pub axi_words_per_clock: u32,
}

impl AdcBlock {
    pub fn new(index: usize) -> Self {
        Self {
            index,
            enabled: true,
            dsa_db: 0.0,
            nyquist_zone: NyquistZone::Odd,
            planner_zone: 1,
            mixer_settings: MixerSettings::default(),
            qmc_settings: QmcSettings::default(),
            decimation: DecimationFactor::X1,
            calibration_mode: CalibrationMode::Mode1,
            non_idealities: AdcNonIdealities::default(),
            axi_words_per_clock: 4,
        }
    }

    pub fn output_rate_mhz(&self, tile_sample_rate_gsps: f64) -> f64 {
        tile_sample_rate_gsps * 1000.0 / self.decimation.factor() as f64
    }

    pub fn mixer_active(&self) -> bool {
        self.mixer_settings.mixer_type != MixerType::Off
    }

    pub fn validate(&self, tile_fs_mhz: f64) -> Vec<String> {
        let mut errors = Vec::new();
        let ms = &self.mixer_settings;

        if ms.mixer_type == MixerType::Fine && ms.mixer_mode == MixerMode::RealToReal {
            errors.push("Fine mixer cannot be used with Real-to-Real mode.".into());
        }
        if ms.mixer_mode == MixerMode::ComplexToReal {
            errors.push("ADC block cannot use Complex-to-Real mixer mode.".into());
        }
        if ms.mixer_type == MixerType::Coarse && ms.mixer_mode == MixerMode::RealToReal && ms.coarse_mix_freq != CoarseMixFreq::Bypass {
            errors.push("Coarse mixer with Real-to-Real mode requires Bypass frequency.".into());
        }
        if ms.mixer_type == MixerType::Coarse && ms.mixer_mode == MixerMode::IqToIq {
            errors.push("Coarse mixer with I/Q→I/Q mode is invalid on ADC (input is always real).".into());
        }
        if ms.mixer_type == MixerType::Fine && ms.freq.abs() < 1e-9 {
            errors.push("⚠ Fine mixer NCO frequency is 0 Hz (functionally a bypass).".into());
        }
        if self.decimation.factor() > 1 && ms.mixer_type == MixerType::Off {
            errors.push("⚠ Decimation > ×1 with mixer off: no anti-alias filtering applied.".into());
        }
        if self.dsa_db < 0.0 || self.dsa_db > 27.0 {
            errors.push("DSA attenuation must be between 0 and 27 dB.".into());
        }
        if self.planner_zone == 0 {
            errors.push("Planner zone must be ≥ 1.".into());
        }

        let output_rate = self.output_rate_mhz(tile_fs_mhz / 1000.0);
        let fabric_clk = output_rate / self.axi_words_per_clock as f64;
        if fabric_clk > 500.0 {
            errors.push(format!("⚠ Fabric clock {:.1} MHz exceeds 500 MHz limit (Output Rate / AXI Words = {:.1} / {}). Increase decimation or AXI words.", fabric_clk, output_rate, self.axi_words_per_clock));
        }
        
        errors
    }

    /// Auto-tune this block for a target RF frequency. Sets nyquist_zone, planner_zone,
    /// and mixer_settings (type=Fine, mode=R2IQ, freq=NCO) in one call.
    pub fn auto_tune(&mut self, tile_sample_rate_gsps: f64, target_freq_mhz: f64) -> AutoTuneResult {
        let fs_mhz = tile_sample_rate_gsps * 1000.0;
        let f_nyq = fs_mhz / 2.0;

        if f_nyq <= 0.0 || target_freq_mhz <= 0.0 {
            self.nyquist_zone = NyquistZone::Odd;
            self.planner_zone = 1;
            self.mixer_settings.mixer_type = MixerType::Fine;
            self.mixer_settings.mixer_mode = MixerMode::RealToIq;
            self.mixer_settings.freq = target_freq_mhz;
            return AutoTuneResult {
                target_freq_mhz,
                zone_index: 1,
                is_even_zone: false,
                nyquist_zone: NyquistZone::Odd,
                alias_freq_mhz: target_freq_mhz,
                nco_freq_mhz: target_freq_mhz,
            };
        }

        let zone_index = (target_freq_mhz / f_nyq).floor() as u32 + 1;
        let is_even_zone = zone_index % 2 == 0;
        let nyquist_zone = if is_even_zone { NyquistZone::Even } else { NyquistZone::Odd };

        let alias_freq_mhz = if is_even_zone {
            (zone_index as f64 * f_nyq) - target_freq_mhz
        } else {
            target_freq_mhz - ((zone_index as f64 - 1.0) * f_nyq)
        };

        let nco_freq_mhz = if is_even_zone {
            -alias_freq_mhz
        } else {
            alias_freq_mhz
        };

        self.nyquist_zone = nyquist_zone;
        self.planner_zone = zone_index;
        self.mixer_settings.mixer_type = MixerType::Fine;
        self.mixer_settings.mixer_mode = MixerMode::RealToIq;
        self.mixer_settings.freq = nco_freq_mhz;

        AutoTuneResult {
            target_freq_mhz,
            zone_index,
            is_even_zone,
            nyquist_zone,
            alias_freq_mhz,
            nco_freq_mhz,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QmcSettings {
    pub gain: f64,
    pub phase: f64,
    pub offset: f64,
}

impl Default for QmcSettings {
    fn default() -> Self {
        Self { gain: 1.0, phase: 0.0, offset: 0.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixerSettings {
    pub mixer_type: MixerType,
    pub mixer_mode: MixerMode,
    pub coarse_mix_freq: CoarseMixFreq,
    pub freq: f64,
    pub phase_offset: f64,
    pub fine_mixer_scale: FineMixerScale,
    pub event_source: EventSource,
}

impl Default for MixerSettings {
    fn default() -> Self {
        Self {
            mixer_type: MixerType::Off,
            mixer_mode: MixerMode::RealToIq,
            coarse_mix_freq: CoarseMixFreq::Off,
            freq: 0.0,
            phase_offset: 0.0,
            fine_mixer_scale: FineMixerScale::Auto,
            event_source: EventSource::Tile,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixerType { Off, Coarse, Fine }

impl std::fmt::Display for MixerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MixerType::Off => write!(f, "Off"),
            MixerType::Coarse => write!(f, "Coarse"),
            MixerType::Fine => write!(f, "Fine"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixerMode { RealToReal, RealToIq, IqToIq, ComplexToReal }

impl std::fmt::Display for MixerMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MixerMode::RealToReal => write!(f, "Real -> Real"),
            MixerMode::RealToIq => write!(f, "Real -> I/Q"),
            MixerMode::IqToIq => write!(f, "I/Q -> I/Q"),
            MixerMode::ComplexToReal => write!(f, "I/Q -> Real"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoarseMixFreq { Off, Bypass, FsOver4, MinusFsOver4, FsOver2 }
impl CoarseMixFreq {
    pub const ALL: [CoarseMixFreq; 5] = [
        CoarseMixFreq::Off, CoarseMixFreq::Bypass, CoarseMixFreq::FsOver4, CoarseMixFreq::MinusFsOver4, CoarseMixFreq::FsOver2,
    ];
}
impl std::fmt::Display for CoarseMixFreq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoarseMixFreq::Off => write!(f, "Off"),
            CoarseMixFreq::Bypass => write!(f, "Bypass"),
            CoarseMixFreq::FsOver4 => write!(f, "Fs/4"),
            CoarseMixFreq::MinusFsOver4 => write!(f, "−Fs/4"),
            CoarseMixFreq::FsOver2 => write!(f, "Fs/2"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FineMixerScale { Auto, OnePointZero, ZeroPointSeven }
impl std::fmt::Display for FineMixerScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FineMixerScale::Auto => write!(f, "Auto"),
            FineMixerScale::OnePointZero => write!(f, "1.0"),
            FineMixerScale::ZeroPointSeven => write!(f, "0.7"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventSource { Immediate, Slice, Tile, SysRef, Pl }
impl std::fmt::Display for EventSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventSource::Immediate => write!(f, "Immediate"),
            EventSource::Slice => write!(f, "Slice"),
            EventSource::Tile => write!(f, "Tile"),
            EventSource::SysRef => write!(f, "SYSREF"),
            EventSource::Pl => write!(f, "PL"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NyquistZone { Odd = 1, Even = 2 }

impl NyquistZone {
    pub fn is_even(&self) -> bool {
        matches!(self, NyquistZone::Even)
    }
}

impl std::fmt::Display for NyquistZone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NyquistZone::Odd => write!(f, "Zone 1 (Odd, Direct)"),
            NyquistZone::Even => write!(f, "Zone 2 (Even, Mirrored)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecimationFactor {
    X1, X2, X3, X4, X5, X6, X8, X10, X12, X16, X20, X24, X40,
}

impl DecimationFactor {
    pub const ALL: [DecimationFactor; 13] = [
        DecimationFactor::X1, DecimationFactor::X2, DecimationFactor::X3,
        DecimationFactor::X4, DecimationFactor::X5, DecimationFactor::X6,
        DecimationFactor::X8, DecimationFactor::X10, DecimationFactor::X12,
        DecimationFactor::X16, DecimationFactor::X20, DecimationFactor::X24,
        DecimationFactor::X40,
    ];

    pub fn factor(self) -> u32 {
        match self {
            Self::X1 => 1, Self::X2 => 2, Self::X3 => 3, Self::X4 => 4,
            Self::X5 => 5, Self::X6 => 6, Self::X8 => 8, Self::X10 => 10,
            Self::X12 => 12, Self::X16 => 16, Self::X20 => 20, Self::X24 => 24,
            Self::X40 => 40,
        }
    }
}

impl std::fmt::Display for DecimationFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "×{}", self.factor())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationMode {
    Mode1,
    Mode2,
}

impl std::fmt::Display for CalibrationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrationMode::Mode1 => write!(f, "Mode 1"),
            CalibrationMode::Mode2 => write!(f, "Mode 2"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_all_tiles_enabled() {
        let config = RfdcConfig::default();
        assert_eq!(config.adc_tiles.len(), 4);
        for tile in &config.adc_tiles {
            assert!(tile.enabled);
            for block in &tile.blocks {
                assert!(block.enabled);
            }
        }
    }

    #[test]
    fn active_blocks_respects_enable_flags() {
        let mut config = RfdcConfig::default();
        config.adc_tiles[1].enabled = false;
        config.adc_tiles[2].blocks[1].enabled = false;
        let active: Vec<_> = config.active_adc_blocks().collect();
        assert_eq!(active.len(), 5);
    }

    #[test]
    fn nyquist_bandwidth_calculation() {
        let tile = AdcTile::new(0);
        assert!((tile.nyquist_bw_mhz() - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn auto_tune_nyquist_zones() {
        let tile = AdcTile::new(0); // 4.0 GSPS -> F_nyq = 2000 MHz
        // 5800 MHz target -> 5800 / 2000 = 2.9 -> Zone 3 (Odd)
        let res = tile.auto_tune(5800.0);
        assert_eq!(res.zone_index, 3);
        assert!(!res.is_even_zone);
        assert_eq!(res.nyquist_zone, NyquistZone::Odd);
        assert!((res.alias_freq_mhz - 1800.0).abs() < 1e-6);

        // 3000 MHz target -> 3000 / 2000 = 1.5 -> Zone 2 (Even)
        let res2 = tile.auto_tune(3000.0);
        assert_eq!(res2.zone_index, 2);
        assert!(res2.is_even_zone);
        assert_eq!(res2.nyquist_zone, NyquistZone::Even);
        assert!((res2.alias_freq_mhz - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn validate_fine_r2r_error() {
        let mut block = AdcBlock::new(0);
        block.mixer_settings.mixer_type = MixerType::Fine;
        block.mixer_settings.mixer_mode = MixerMode::RealToReal;
        block.mixer_settings.freq = 100.0;

        let errors = block.validate(4000.0);
        assert!(errors.iter().any(|e| e.contains("Fine mixer cannot be used with Real-to-Real")));
    }

    #[test]
    fn validate_coarse_iq2iq_adc_error() {
        let mut block = AdcBlock::new(0);
        block.mixer_settings.mixer_type = MixerType::Coarse;
        block.mixer_settings.mixer_mode = MixerMode::IqToIq;
        block.mixer_settings.coarse_mix_freq = CoarseMixFreq::FsOver4;

        let errors = block.validate(4000.0);
        assert!(errors.iter().any(|e| e.contains("I/Q→I/Q mode is invalid on ADC")));
    }

    #[test]
    fn auto_tune_block_level() {
        let tile = AdcTile::new(0); // 4.0 GSPS
        let mut block = AdcBlock::new(0);

        // Auto-tune to 5800 MHz → Zone 3 (Odd), alias 1800 MHz
        let res = block.auto_tune(tile.sample_rate_gsps, 5800.0);

        assert_eq!(res.zone_index, 3);
        assert_eq!(block.planner_zone, 3);
        assert_eq!(block.nyquist_zone, NyquistZone::Odd);
        assert_eq!(block.mixer_settings.mixer_type, MixerType::Fine);
        assert_eq!(block.mixer_settings.mixer_mode, MixerMode::RealToIq);
        assert!((block.mixer_settings.freq - 1800.0).abs() < 1e-6);

        // Auto-tune to 3000 MHz → Zone 2 (Even), alias 1000 MHz, NCO -1000
        let res2 = block.auto_tune(tile.sample_rate_gsps, 3000.0);

        assert_eq!(res2.zone_index, 2);
        assert_eq!(block.planner_zone, 2);
        assert_eq!(block.nyquist_zone, NyquistZone::Even);
        assert!((block.mixer_settings.freq - (-1000.0)).abs() < 1e-6);
    }
}
