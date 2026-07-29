//! RFSoC Simulator — main entry point.

mod app;
mod dsp;
mod node_graph;
mod rfdc;
mod signal;
mod ui;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 1000.0])
            .with_min_inner_size([1024.0, 600.0])
            .with_title("RFSoC Simulator — ZU48DR / ZCU208"),
        ..Default::default()
    };

    eframe::run_native(
        "RFSoC Simulator",
        options,
        Box::new(|cc| {
            let mut fonts = egui::FontDefinitions::default();
            egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
            cc.egui_ctx.set_fonts(fonts);
            Ok(Box::new(app::RfSocSimApp::default()))
        }),
    )
}
