//! Shared editor for a signal generator's tones.
//!
//! The sidebar generator and the node graph's local sources both edit the same `Tone`, and
//! keeping two copies of the widget meant the two drifted — one grew a channel-bandwidth field
//! the other never got. There is one editor here and both call it.

use crate::signal::{top_harmonic_mhz, SignalGenerator, Tone, ToneModulation};
use crate::ui::theme::Theme;

/// Dimmed caption describing what the current settings actually produce.
fn readout(ui: &mut egui::Ui, text: String) {
    ui.add(
        egui::Label::new(
            egui::RichText::new(text)
                .small()
                .color(Theme::TEXT_SECONDARY),
        )
        .wrap_mode(egui::TextWrapMode::Extend),
    );
}

/// One line summarising the spectrum a tone will occupy.
///
/// Every modulation has a textbook expression for its occupied bandwidth, and showing it beats
/// making the user derive it from the sliders.
pub fn tone_summary(tone: &Tone, sample_rate_mhz: f64) -> String {
    let f = tone.frequency_mhz;
    match &tone.modulation {
        ToneModulation::Cw => {
            if tone.bandwidth_mhz > 0.0 {
                format!(
                    "{:.0} MHz channel, {:.0} to {:.0} MHz",
                    tone.bandwidth_mhz,
                    f - tone.bandwidth_mhz / 2.0,
                    f + tone.bandwidth_mhz / 2.0
                )
            } else {
                "single spectral line".to_string()
            }
        }
        ToneModulation::Square | ToneModulation::Sawtooth | ToneModulation::Triangle => {
            let top = top_harmonic_mhz(&tone.modulation, f, sample_rate_mhz);
            let which = match tone.modulation {
                ToneModulation::Sawtooth => "all harmonics",
                _ => "odd harmonics",
            };
            format!("{which} to {top:.0} MHz, band-limited")
        }
        ToneModulation::AmModulated { depth_percent, mod_freq_khz } => {
            let m = (depth_percent / 100.0).clamp(0.0, 1.0);
            let sideband_dbc = 20.0 * (m / 2.0).max(1e-6).log10();
            format!(
                "sidebands at {:+.0} kHz, {sideband_dbc:.1} dBc",
                mod_freq_khz
            )
        }
        ToneModulation::FmModulated { dev_mhz, mod_freq_khz } => {
            let f_m = mod_freq_khz / 1000.0;
            let beta = if f_m > 0.0 { dev_mhz / f_m } else { 0.0 };
            // Carson's rule: essentially all the power sits inside 2(dev + f_m).
            format!(
                "beta {beta:.2}, Carson BW {:.1} MHz",
                2.0 * (dev_mhz + f_m)
            )
        }
        ToneModulation::SweptChirp { sweep_period_us, triangular } => {
            let bw = if tone.bandwidth_mhz > 0.0 { tone.bandwidth_mhz } else { 100.0 };
            let period = sweep_period_us.max(1e-3);
            let rate = if *triangular { bw / (period / 2.0) } else { bw / period };
            format!(
                "{:.0} to {:.0} MHz at {rate:.1} MHz/us{}",
                f - bw / 2.0,
                f + bw / 2.0,
                if *triangular { ", both ways" } else { "" }
            )
        }
        ToneModulation::PulsedRadar { pulse_width_us, pri_us, chirp_mhz, .. } => {
            let pri = pri_us.max(1e-3);
            let duty = (pulse_width_us / pri * 100.0).min(100.0);
            let mut s = format!("duty {duty:.1}%, PRF {:.1} kHz", 1000.0 / pri);
            if *chirp_mhz > 0.0 {
                // Time-bandwidth product is the pulse compression ratio.
                s.push_str(&format!(
                    ", compression {:.0}x",
                    chirp_mhz * pulse_width_us
                ));
            }
            s
        }
        ToneModulation::FreqHopping { hop_step_mhz, num_channels, hop_rate_hz } => {
            let span = hop_step_mhz * (*num_channels as f64 - 1.0).max(0.0);
            format!(
                "{num_channels} channels over {span:.0} MHz, dwell {:.2} us",
                1e6 / hop_rate_hz.max(1e-3)
            )
        }
        ToneModulation::DigitalQpsk { symbol_rate_msps, rrc_alpha } => {
            format!(
                "occupies {:.1} MHz ((1+a) x Rs)",
                (1.0 + rrc_alpha.clamp(0.01, 1.0)) * symbol_rate_msps.max(1e-3)
            )
        }
    }
}

/// Edit one tone. `sample_rate_mhz` is the simulation rate, used for the summary line.
pub fn tone_editor(
    ui: &mut egui::Ui,
    id_salt: &str,
    tone: &mut Tone,
    sample_rate_mhz: f64,
) {
    egui::ComboBox::from_id_salt(format!("mod_{id_salt}"))
        .selected_text(tone.modulation.to_string())
        .width(150.0)
        .show_ui(ui, |ui| {
            for variant in ToneModulation::all_variants() {
                let selected = std::mem::discriminant(&tone.modulation)
                    == std::mem::discriminant(&variant);
                let label = variant.to_string();
                if ui.selectable_label(selected, label).clicked() {
                    tone.modulation = variant;
                }
            }
        });

    egui::Grid::new(format!("tone_{id_salt}"))
        .num_columns(2)
        .spacing([4.0, 3.0])
        .show(ui, |ui| {
            ui.label("Freq:");
            ui.add(
                egui::DragValue::new(&mut tone.frequency_mhz)
                    .range(0.1..=10000.0)
                    .suffix(" MHz")
                    .speed(10.0),
            )
            .on_hover_text("Carrier frequency. Above Fs/2 it will alias into a lower Nyquist zone.");
            ui.end_row();

            ui.label("Amp:");
            ui.add(
                egui::DragValue::new(&mut tone.amplitude_dbfs)
                    .range(-140.0..=0.0)
                    .suffix(" dBFS")
                    .speed(0.5),
            )
            .on_hover_text(
                "Peak amplitude relative to converter full scale. 0 dBFS is a sine that just \
                 touches the clipping limit.",
            );
            ui.end_row();

            ui.label("Phase:");
            ui.add(
                egui::DragValue::new(&mut tone.phase_deg)
                    .range(-360.0..=360.0)
                    .suffix(" deg")
                    .speed(1.0),
            )
            .on_hover_text("Starting phase. -90 deg turns a cosine into a sine.");
            ui.end_row();

            if let Some(label) = tone.modulation.bandwidth_label() {
                ui.label(format!("{label}:"));
                let hover = match tone.modulation {
                    ToneModulation::Cw => {
                        "0 for a pure tone. Above zero the carrier becomes a modulated channel \
                         of this width, filled like a real signal carrying data."
                    }
                    _ => "Total width the chirp sweeps across, centred on the carrier.",
                };
                ui.add(
                    egui::DragValue::new(&mut tone.bandwidth_mhz)
                        .range(0.0..=4000.0)
                        .suffix(" MHz")
                        .speed(1.0),
                )
                .on_hover_text(hover);
                ui.end_row();
            }

            match &mut tone.modulation {
                ToneModulation::Cw
                | ToneModulation::Square
                | ToneModulation::Sawtooth
                | ToneModulation::Triangle => {}

                ToneModulation::AmModulated { depth_percent, mod_freq_khz } => {
                    ui.label("Depth:");
                    ui.add(
                        egui::DragValue::new(depth_percent)
                            .range(0.0..=100.0)
                            .suffix(" %")
                            .speed(1.0),
                    )
                    .on_hover_text(
                        "Modulation depth. The envelope peaks at (1 + m) times the carrier, so \
                         deep modulation of a hot carrier will overdrive what follows.",
                    );
                    ui.end_row();

                    ui.label("Mod f:");
                    ui.add(
                        egui::DragValue::new(mod_freq_khz)
                            .range(0.001..=100_000.0)
                            .suffix(" kHz")
                            .speed(10.0),
                    )
                    .on_hover_text("Modulating frequency; the sidebands land this far out.");
                    ui.end_row();
                }

                ToneModulation::FmModulated { dev_mhz, mod_freq_khz } => {
                    ui.label("Dev:");
                    ui.add(
                        egui::DragValue::new(dev_mhz)
                            .range(0.0..=2000.0)
                            .suffix(" MHz")
                            .speed(1.0),
                    )
                    .on_hover_text("Peak frequency deviation.");
                    ui.end_row();

                    ui.label("Mod f:");
                    ui.add(
                        egui::DragValue::new(mod_freq_khz)
                            .range(0.001..=100_000.0)
                            .suffix(" kHz")
                            .speed(10.0),
                    )
                    .on_hover_text(
                        "Modulating frequency. Deviation over this is the modulation index, \
                         which sets the Bessel sideband pattern.",
                    );
                    ui.end_row();
                }

                ToneModulation::SweptChirp { sweep_period_us, triangular } => {
                    ui.label("Period:");
                    ui.add(
                        egui::DragValue::new(sweep_period_us)
                            .range(0.01..=100_000.0)
                            .suffix(" us")
                            .speed(1.0),
                    )
                    .on_hover_text(
                        "One full sweep. Shorter than the analysis window and the whole sweep \
                         shows up in one frame; longer and it crawls across the waterfall.",
                    );
                    ui.end_row();

                    ui.label("Retrace:");
                    ui.checkbox(triangular, "triangular")
                        .on_hover_text(
                            "Sweep back down instead of jumping. Triangular FMCW separates \
                             range from Doppler; sawtooth does not.",
                        );
                    ui.end_row();
                }

                ToneModulation::PulsedRadar { pulse_width_us, pri_us, rise_ns, chirp_mhz } => {
                    ui.label("Width:");
                    ui.add(
                        egui::DragValue::new(pulse_width_us)
                            .range(0.001..=10_000.0)
                            .suffix(" us")
                            .speed(0.1),
                    )
                    .on_hover_text("Pulse duration, which sets the width of the sinc skirts.");
                    ui.end_row();

                    ui.label("PRI:");
                    ui.add(
                        egui::DragValue::new(pri_us)
                            .range(0.01..=100_000.0)
                            .suffix(" us")
                            .speed(1.0),
                    )
                    .on_hover_text(
                        "Pulse repetition interval. Spectral lines appear at 1/PRI either side \
                         of the carrier.",
                    );
                    ui.end_row();

                    ui.label("Rise:");
                    ui.add(
                        egui::DragValue::new(rise_ns)
                            .range(0.0..=1000.0)
                            .suffix(" ns")
                            .speed(1.0),
                    )
                    .on_hover_text(
                        "Edge transition time. Zero gives an ideal rectangle, whose skirts \
                         reach out forever; a real transmitter has finite edges.",
                    );
                    ui.end_row();

                    ui.label("Chirp:");
                    ui.add(
                        egui::DragValue::new(chirp_mhz)
                            .range(0.0..=2000.0)
                            .suffix(" MHz")
                            .speed(1.0),
                    )
                    .on_hover_text(
                        "Sweep width inside each pulse, for pulse compression. Zero gives an \
                         unmodulated pulse.",
                    );
                    ui.end_row();
                }

                ToneModulation::FreqHopping { hop_step_mhz, num_channels, hop_rate_hz } => {
                    ui.label("Step:");
                    ui.add(
                        egui::DragValue::new(hop_step_mhz)
                            .range(0.001..=1000.0)
                            .suffix(" MHz")
                            .speed(1.0),
                    )
                    .on_hover_text("Channel spacing on the hop grid.");
                    ui.end_row();

                    ui.label("Channels:");
                    ui.add(egui::DragValue::new(num_channels).range(2..=128))
                        .on_hover_text("Channels in the grid, centred on the carrier.");
                    ui.end_row();

                    ui.label("Hop rate:");
                    ui.add(
                        egui::DragValue::new(hop_rate_hz)
                            .range(1.0..=100_000_000.0)
                            .suffix(" Hz")
                            .speed(1000.0),
                    )
                    .on_hover_text("Hops per second. The dwell time is the reciprocal.");
                    ui.end_row();
                }

                ToneModulation::DigitalQpsk { symbol_rate_msps, rrc_alpha } => {
                    ui.label("Rs:");
                    ui.add(
                        egui::DragValue::new(symbol_rate_msps)
                            .range(0.001..=2000.0)
                            .suffix(" Msps")
                            .speed(1.0),
                    )
                    .on_hover_text("Symbol rate. The channel occupies about (1 + alpha) times this.");
                    ui.end_row();

                    ui.label("RRC a:");
                    ui.add(
                        egui::DragValue::new(rrc_alpha)
                            .range(0.01..=1.0)
                            .speed(0.01),
                    )
                    .on_hover_text(
                        "Root-raised-cosine roll-off. Smaller is more spectrally efficient and \
                         harder on the transmitter's linearity.",
                    );
                    ui.end_row();
                }
            }
        });

    readout(ui, tone_summary(tone, sample_rate_mhz));
}

/// Edit a whole generator: its tones and its noise floor.
pub fn generator_editor(
    ui: &mut egui::Ui,
    id_salt: &str,
    generator: &mut SignalGenerator,
    sample_rate_mhz: f64,
) {
    ui.horizontal(|ui| {
        ui.label("Tones:");
        if ui
            .button("+")
            .on_hover_text("Add a tone. Two close together is the standard two-tone IMD test.")
            .clicked()
        {
            let mut next = generator.tones.last().cloned().unwrap_or_default();
            // Offset the copy so a second tone is immediately useful rather than coincident.
            next.frequency_mhz += 10.0;
            generator.tones.push(next);
        }
        if generator.tones.len() > 1 && ui.button("-").clicked() {
            generator.tones.pop();
        }
    });

    let multiple = generator.tones.len() > 1;
    for (i, tone) in generator.tones.iter_mut().enumerate() {
        ui.group(|ui| {
            if multiple {
                ui.label(
                    egui::RichText::new(format!("Tone {i}"))
                        .small()
                        .color(Theme::TEXT_LABEL),
                );
            }
            tone_editor(ui, &format!("{id_salt}_{i}"), tone, sample_rate_mhz);
        });
    }

    ui.horizontal(|ui| {
        ui.checkbox(&mut generator.noise_enabled, "AWGN")
            .on_hover_text(
                "Additive white Gaussian noise injected at the source, as a test vector. The \
                 chain's own physical noise is the thermal model on the RF Chain tab.",
            );
        ui.add_enabled_ui(generator.noise_enabled, |ui| {
            ui.add(
                egui::DragValue::new(&mut generator.noise_floor_dbfs)
                    .range(-200.0..=0.0)
                    .suffix(" dBFS")
                    .speed(1.0),
            )
            .on_hover_text("Total noise power across the whole simulated band.");
        });
    });
}
