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
    pub nyquist_zone: NyquistZone,
    pub nyquist_zone_index: u32,
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
            nyquist_zone: NyquistZone::Zone1,
            nyquist_zone_index: 1,
            pll_enabled: true,
            ref_clk_mhz: 245.76,
            blocks: [AdcBlock::new(0), AdcBlock::new(1)],
            sync_group: None,
            sysref_phase: 0.0,
        }
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
                nyquist_zone: NyquistZone::Zone1,
                alias_freq_mhz: target_freq_mhz,
                nco_freq_mhz: target_freq_mhz,
            };
        }

        let zone_index = (target_freq_mhz / f_nyq).floor() as u32 + 1;
        let is_even_zone = zone_index % 2 == 0;
        let nyquist_zone = NyquistZone::from_index(zone_index);

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
            quantization_bits: 12,
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
    pub mixer_mode: MixerMode,
    pub nco_freq_mhz: f64,
    pub decimation: DecimationFactor,
    pub calibration_mode: CalibrationMode,
    pub non_idealities: AdcNonIdealities,
}

impl AdcBlock {
    pub fn new(index: usize) -> Self {
        Self {
            index,
            enabled: true,
            mixer_mode: MixerMode::Bypass,
            nco_freq_mhz: 0.0,
            decimation: DecimationFactor::X1,
            calibration_mode: CalibrationMode::Mode1,
            non_idealities: AdcNonIdealities::default(),
        }
    }

    pub fn output_rate_mhz(&self, tile_sample_rate_gsps: f64) -> f64 {
        tile_sample_rate_gsps * 1000.0 / self.decimation.factor() as f64
    }

    pub fn mixer_active(&self) -> bool {
        !matches!(self.mixer_mode, MixerMode::Bypass)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NyquistZone {
    Zone1,
    Zone2,
    Zone3,
    Zone4,
    Zone5,
    Zone6,
    Zone7,
    Zone8,
}

impl NyquistZone {
    pub const ALL: [NyquistZone; 8] = [
        NyquistZone::Zone1,
        NyquistZone::Zone2,
        NyquistZone::Zone3,
        NyquistZone::Zone4,
        NyquistZone::Zone5,
        NyquistZone::Zone6,
        NyquistZone::Zone7,
        NyquistZone::Zone8,
    ];

    pub fn index(&self) -> u32 {
        match self {
            Self::Zone1 => 1,
            Self::Zone2 => 2,
            Self::Zone3 => 3,
            Self::Zone4 => 4,
            Self::Zone5 => 5,
            Self::Zone6 => 6,
            Self::Zone7 => 7,
            Self::Zone8 => 8,
        }
    }

    pub fn from_index(index: u32) -> Self {
        match index {
            1 => Self::Zone1,
            2 => Self::Zone2,
            3 => Self::Zone3,
            4 => Self::Zone4,
            5 => Self::Zone5,
            6 => Self::Zone6,
            7 => Self::Zone7,
            _ => Self::Zone8,
        }
    }

    pub const FIRST: NyquistZone = NyquistZone::Zone1;
    pub const SECOND: NyquistZone = NyquistZone::Zone2;
}

impl std::fmt::Display for NyquistZone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let idx = self.index();
        let is_even = idx % 2 == 0;
        let mode_str = if is_even { "Even, Mirrored" } else { "Odd, Direct" };
        write!(f, "Zone {idx} ({mode_str})")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixerMode {
    Bypass,
    CoarseMix(CoarseMixFreq),
    FineMix,
}

impl MixerMode {
    pub const ALL_BASIC: [MixerMode; 3] = [
        MixerMode::Bypass,
        MixerMode::CoarseMix(CoarseMixFreq::FsOver4),
        MixerMode::FineMix,
    ];
}

impl std::fmt::Display for MixerMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MixerMode::Bypass => write!(f, "Bypass"),
            MixerMode::CoarseMix(freq) => write!(f, "Coarse ({freq})"),
            MixerMode::FineMix => write!(f, "Fine (NCO)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoarseMixFreq {
    FsOver4,
    MinusFsOver4,
    FsOver2,
}

impl CoarseMixFreq {
    pub const ALL: [CoarseMixFreq; 3] = [
        CoarseMixFreq::FsOver4,
        CoarseMixFreq::MinusFsOver4,
        CoarseMixFreq::FsOver2,
    ];
}

impl std::fmt::Display for CoarseMixFreq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoarseMixFreq::FsOver4 => write!(f, "Fs/4"),
            CoarseMixFreq::MinusFsOver4 => write!(f, "−Fs/4"),
            CoarseMixFreq::FsOver2 => write!(f, "Fs/2"),
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
        assert_eq!(res.nyquist_zone, NyquistZone::Zone3);
        assert!((res.alias_freq_mhz - 1800.0).abs() < 1e-6);

        // 3000 MHz target -> 3000 / 2000 = 1.5 -> Zone 2 (Even)
        let res2 = tile.auto_tune(3000.0);
        assert_eq!(res2.zone_index, 2);
        assert!(res2.is_even_zone);
        assert_eq!(res2.nyquist_zone, NyquistZone::Zone2);
        assert!((res2.alias_freq_mhz - 1000.0).abs() < 1e-6);
    }
}
