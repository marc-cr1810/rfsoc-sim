//! Built-in configuration presets for common use cases.

#![allow(dead_code)]

use crate::rfdc::*;

/// Create a default ZCU208 configuration.
/// All tiles at 4.0 GSPS, 1st Nyquist zone, mixer off, no decimation.
pub fn default_zcu208() -> RfdcConfig {
    RfdcConfig::default()
}

/// Wideband capture preset.
/// 5.0 GSPS, 2nd Nyquist zone, fine mixer enabled.
pub fn wideband_capture() -> RfdcConfig {
    let mut config = RfdcConfig::default();
    for tile in &mut config.adc_tiles {
        tile.sample_rate_gsps = 5.0;
        for block in &mut tile.blocks {
            block.nyquist_zone = NyquistZone::Even;
            block.planner_zone = 2;
            block.mixer_settings.mixer_type = MixerType::Fine;
            block.mixer_settings.mixer_mode = MixerMode::RealToIq;
            block.mixer_settings.freq = 1250.0;
        }
    }
    config
}

/// Narrowband DDC preset.
/// 2.0 GSPS, decimation ×16, NCO at 500 MHz.
pub fn narrowband_ddc() -> RfdcConfig {
    let mut config = RfdcConfig::default();
    for tile in &mut config.adc_tiles {
        tile.sample_rate_gsps = 2.0;
        for block in &mut tile.blocks {
            block.nyquist_zone = NyquistZone::Odd;
            block.planner_zone = 1;
            block.mixer_settings.mixer_type = MixerType::Fine;
            block.mixer_settings.mixer_mode = MixerMode::RealToIq;
            block.mixer_settings.freq = 500.0;
            block.decimation = DecimationFactor::X16;
        }
    }
    config
}

/// Single-tile configuration for minimal resource usage.
/// Only tile 0 enabled, blocks configured for simple capture.
pub fn single_tile() -> RfdcConfig {
    let mut config = RfdcConfig::default();
    for i in 1..4 {
        config.adc_tiles[i].enabled = false;
    }
    config
}
