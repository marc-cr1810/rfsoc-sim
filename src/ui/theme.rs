//! Application color theme and styling constants.

#![allow(dead_code)]

use egui::Color32;

/// Xilinx/AMD-inspired dark theme colours.
pub struct Theme;

impl Theme {
    // -- Background & surface --
    pub const BG_DARK: Color32 = Color32::from_rgb(18, 18, 24);
    pub const BG_PANEL: Color32 = Color32::from_rgb(28, 28, 38);
    pub const BG_CARD: Color32 = Color32::from_rgb(38, 38, 52);

    // -- Accent colours --
    pub const ACCENT_PRIMARY: Color32 = Color32::from_rgb(0, 164, 239);   // AMD blue
    pub const ACCENT_SECONDARY: Color32 = Color32::from_rgb(118, 185, 0); // AMD green
    pub const ACCENT_WARN: Color32 = Color32::from_rgb(255, 170, 0);
    pub const ACCENT_ERROR: Color32 = Color32::from_rgb(220, 50, 50);

    // -- Nyquist zone colours --
    pub const ZONE_1: Color32 = Color32::from_rgb(65, 135, 255);  // Blue
    pub const ZONE_2: Color32 = Color32::from_rgb(255, 150, 50);  // Orange
    pub const ZONE_3: Color32 = Color32::from_rgb(160, 90, 255);  // Purple
    pub const ZONE_4: Color32 = Color32::from_rgb(50, 200, 120);  // Green
    pub const ZONE_5: Color32 = Color32::from_rgb(255, 80, 120);  // Pink

    // -- Node colours --
    pub const NODE_SOURCE: Color32 = Color32::from_rgb(50, 180, 80);
    pub const NODE_PASSIVE: Color32 = Color32::from_rgb(70, 130, 200);
    pub const NODE_ACTIVE: Color32 = Color32::from_rgb(220, 140, 40);
    pub const NODE_SINK: Color32 = Color32::from_rgb(200, 60, 60);

    // -- Status --
    pub const ENABLED: Color32 = Color32::from_rgb(80, 200, 100);
    pub const DISABLED: Color32 = Color32::from_rgb(100, 100, 110);

    // -- Text --
    pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(220, 220, 230);
    pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(140, 140, 160);
    pub const TEXT_LABEL: Color32 = Color32::from_rgb(180, 180, 200);

    /// Get the colour for a Nyquist zone number (1-indexed).
    pub fn zone_color(zone: usize) -> Color32 {
        match zone {
            1 => Self::ZONE_1,
            2 => Self::ZONE_2,
            3 => Self::ZONE_3,
            4 => Self::ZONE_4,
            _ => Self::ZONE_5,
        }
    }

    /// Apply the dark theme to an egui context.
    pub fn apply(ctx: &egui::Context) {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = Self::BG_PANEL;
        visuals.window_fill = Self::BG_CARD;
        visuals.extreme_bg_color = Self::BG_DARK;
        visuals.widgets.noninteractive.bg_fill = Self::BG_CARD;
        visuals.selection.bg_fill = Self::ACCENT_PRIMARY.linear_multiply(0.3);
        visuals.selection.stroke = egui::Stroke::new(1.5, Self::ACCENT_PRIMARY);
        ctx.set_visuals(visuals);
    }
}
