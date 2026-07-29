import re

with open('src/ui/config_panel.rs', 'r') as f:
    content = f.read()

# Add helper function at the top
helper_fn = """
fn help_label(ui: &mut egui::Ui, text: &str, help_text: &str) {
    ui.label(text);
    ui.label(egui::RichText::new(egui_phosphor::regular::INFO).color(Theme::TEXT_SECONDARY))
        .on_hover_text(help_text);
}
"""
content = content.replace("use crate::ui::theme::Theme;\n", "use crate::ui::theme::Theme;\n" + helper_fn)

replacements = [
    (r'ui\.label\("Sample Rate:"\);', r'help_label(ui, "Sample Rate:", "The physical sampling rate of the ADC. Must be within hardware limits (e.g., 0.5 to 10 GSPS).");'),
    (r'ui\.checkbox\(&mut tile\.pll_enabled, "Internal PLL"\);', r'ui.checkbox(&mut tile.pll_enabled, "Internal PLL");\n        ui.label(egui::RichText::new(egui_phosphor::regular::INFO).color(Theme::TEXT_SECONDARY)).on_hover_text("Enables the on-chip phase-locked loop (PLL) to generate the sampling clock from a lower frequency reference clock.");'),
    (r'ui\.label\("Planner Zone:"\);', r'help_label(ui, "Planner Zone:", "The target Nyquist zone you want to operate in. The hardware will automatically configure the physical Nyquist Zone (Even/Odd) based on this selection.");'),
    (r'ui\.label\("DSA Attn:"\);', r'help_label(ui, "DSA Attn:", "Digital Step Attenuator. Applies analog attenuation at the RF front-end before the ADC to prevent clipping (0 to 27 dB).");'),
    (r'ui\.label\("Mixer Type:"\);', r'help_label(ui, "Mixer Type:", "Type of digital down-conversion mixing. Coarse mixing uses fixed frequency steps, Fine mixing allows arbitrary NCO frequencies.");'),
    (r'ui\.label\("Mixer Mode:"\);', r'help_label(ui, "Mixer Mode:", "Determines the signal path (Real to I/Q, I/Q to I/Q, etc.) for the digital mixer.");'),
    (r'ui\.label\("Coarse Freq:"\);', r'help_label(ui, "Coarse Freq:", "Fixed frequency shift for coarse mixing (Fs/2, Fs/4, -Fs/4, etc.).");'),
    (r'ui\.label\("NCO Freq:"\);', r'help_label(ui, "NCO Freq:", "Numerically Controlled Oscillator frequency for fine mixing. Shifts the spectrum by this exact amount.");'),
    (r'ui\.label\("NCO Phase:"\);', r'help_label(ui, "NCO Phase:", "Initial phase offset for the NCO (in degrees). Useful for aligning multiple channels.");'),
    (r'ui\.label\("Mixer Scale:"\);', r'help_label(ui, "Mixer Scale:", "Scaling factor applied after the mixer to prevent overflow or maximize dynamic range. Auto is recommended.");'),
    (r'ui\.label\("Decimation:"\);', r'help_label(ui, "Decimation:", "Reduces the sample rate by this factor after mixing. Essential for lowering the data rate sent to the FPGA fabric.");'),
    (r'ui\.label\("AXI Words/Clk:"\);', r'help_label(ui, "AXI Words/Clk:", "Number of digital samples transferred per FPGA clock cycle on the AXI-Stream interface.");'),
    (r'ui\.label\("Cal Mode:"\);', r'help_label(ui, "Cal Mode:", "Background calibration mode for the ADC. Mode 1 is standard, Mode 2 is optimized for specific frequency plans.");'),
    (r'ui\.label\("Gain:"\);', r'help_label(ui, "Gain:", "Quadrature Modulation Correction (QMC) gain adjustment to compensate for I/Q amplitude imbalance.");'),
    (r'ui\.label\("Phase:"\);', r'help_label(ui, "Phase:", "QMC phase adjustment (in degrees) to correct I/Q phase imbalance.");'),
    (r'ui\.label\("Offset:"\);', r'help_label(ui, "Offset:", "DC offset correction for the signal.");'),
    (r'ui\.label\("ENOB:"\);', r'help_label(ui, "ENOB:", "Effective Number Of Bits. Models the true dynamic range of the ADC by adding broadband noise.");'),
    (r'ui\.label\("Quantization:"\);', r'help_label(ui, "Quantization:", "Simulates the bit depth of the ADC (typically 12 or 14 bits for RFSoC). Truncates the analog signal precision.");'),
    (r'ui\.label\("HD2:"\);', r'help_label(ui, "HD2:", "Second Harmonic Distortion (in dBc). Adds a spurious signal at 2x the fundamental frequency.");'),
    (r'ui\.label\("HD3:"\);', r'help_label(ui, "HD3:", "Third Harmonic Distortion (in dBc). Adds a spurious signal at 3x the fundamental frequency.");'),
    (r'ui\.label\("IL Spur:"\);', r'help_label(ui, "IL Spur:", "Interleaving Spur (in dBc). Simulates artifacts caused by the time-interleaved sub-ADCs in the RFSoC.");'),
]

for old, new in replacements:
    content = re.sub(old, new, content)

with open('src/ui/config_panel.rs', 'w') as f:
    f.write(content)

