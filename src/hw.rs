/// Hardware-aware runtime utilities.

const STOP_FILE: &str = "flux-lm.stop";

pub fn check_stop() -> bool {
    std::path::Path::new(STOP_FILE).exists()
}

pub fn core_budget(fraction: f64) -> usize {
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let f = fraction.clamp(0.0, 1.0);
    ((avail as f64 * f).ceil() as usize).max(1)
}
