/// CLI argument parsing helpers shared across commands.

pub fn arg_str(args: &[String], i: &mut usize) -> String {
    *i += 1;
    args.get(*i).map(|s| s.to_string()).unwrap_or_default()
}

pub fn arg_usize(args: &[String], i: &mut usize) -> usize {
    *i += 1;
    args.get(*i).and_then(|s| s.parse().ok()).unwrap_or(0)
}

pub fn arg_f64(args: &[String], i: &mut usize) -> f64 {
    *i += 1;
    args.get(*i).and_then(|s| s.parse().ok()).unwrap_or(0.0)
}

pub fn arg_u64(args: &[String], i: &mut usize) -> u64 {
    *i += 1;
    args.get(*i).and_then(|s| s.parse().ok()).unwrap_or(42)
}
