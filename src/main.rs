mod cli;
mod config;
mod float;
mod rng;
mod logger;
mod math;
mod checkpoint;
mod generate;
mod optim;
mod flux;
mod train;
mod hw;

use config::Config;
use logger::Logger;
use rng::Rng;
use float::Float;

#[cfg(feature = "f32")]
type F = f32;
#[cfg(not(feature = "f32"))]
type F = f64;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "train" => cmd_train(&args[2..]),
        "generate" => cmd_generate(&args[2..]),
        "info" => cmd_info(&args[2..]),
        _ => print_usage(),
    }
}

fn print_usage() {
    let prec = if F::BYTE_SIZE == 4 { "f32" } else { "f64" };
    eprintln!("flux-lm — Flux v3 byte-level language model [{}]", prec);
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  train    --corpus PATH");
    eprintln!("           [--d N] [--layers N] [--epochs N]");
    eprintln!("           [--lr F] [--seq-len N] [--seed N]");
    eprintln!("           [--ckpt PATH] [--ckpt-every N] [--log PATH]");
    eprintln!("           [--core-fraction F] [--print-every N]");
    eprintln!("  generate --ckpt PATH");
    eprintln!("           [--seed-text STR] [--length N]");
    eprintln!("           [--temperature F]");
    eprintln!("  info     --ckpt PATH");
}

fn cmd_train(args: &[String]) {
    let mut cfg = Config::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => { cfg.corpus_path = cli::arg_str(args, &mut i); }
            "--d" => { cfg.d = cli::arg_usize(args, &mut i); }
            "--layers" => { cfg.n_layers = cli::arg_usize(args, &mut i); }
            "--epochs" => { cfg.epochs = cli::arg_usize(args, &mut i); }
            "--lr" => { cfg.lr = cli::arg_f64(args, &mut i); }
            "--seq-len" => { cfg.seq_len = cli::arg_usize(args, &mut i); }
            "--seed" => { cfg.seed = cli::arg_u64(args, &mut i); }
            "--ckpt" => { cfg.ckpt_path = Some(cli::arg_str(args, &mut i)); }
            "--ckpt-every" => {
                cfg.ckpt_every = cli::arg_usize(args, &mut i);
            }
            "--log" => { cfg.log_path = Some(cli::arg_str(args, &mut i)); }
            "--print-every" => {
                cfg.print_every = cli::arg_usize(args, &mut i);
            }
            "--core-fraction" => {
                cfg.core_fraction = cli::arg_f64(args, &mut i);
            }
            _ => { eprintln!("Unknown flag: {}", args[i]); return; }
        }
        i += 1;
    }

    if cfg.corpus_path.is_empty() {
        eprintln!("ERROR: --corpus required");
        return;
    }

    let corpus = match std::fs::read(&cfg.corpus_path) {
        Ok(c) => c,
        Err(e) => { eprintln!("ERROR: {}", e); return; }
    };

    let logger = Logger::new(cfg.log_path.clone());
    let mut rng = Rng::new(cfg.seed);

    let (mut model, mut optim, start_epoch) =
        load_or_create_flux::<F>(&cfg, &logger, &mut rng);
    train::train_flux(
        &mut model, &mut optim, &corpus, &cfg, &logger,
        start_epoch,
    );
}

fn load_or_create_flux<T: Float>(
    cfg: &Config, logger: &Logger, rng: &mut Rng,
) -> (flux::FluxModel<T>, optim::EntropicAdam<T>, u32) {
    if let Some(ref path) = cfg.ckpt_path {
        if std::path::Path::new(path).exists() {
            match checkpoint::load_checkpoint::<T>(path) {
                Ok(ckpt) => {
                    let d = ckpt.d;
                    let nl = ckpt.n_layers;
                    let expected =
                        flux::flux_total_param_count(d, nl);
                    let mut params = ckpt.params;
                    let mut om = ckpt.optim_m;
                    let mut ov = ckpt.optim_v;

                    if params.len() < expected {
                        let old_len = params.len();
                        params.resize(expected, T::ZERO);
                        om.resize(expected, T::ZERO);
                        ov.resize(expected, T::ZERO);
                        let spm_base =
                            flux::flux_pre_spm_param_count(d, nl);
                        flux::spm::spm_init(
                            &mut params, spm_base, d, nl, rng,
                        );
                        logger.log(&format!(
                            "  SPM migration: {} -> {} params (+{})",
                            old_len, expected, expected - old_len,
                        ));
                    }

                    let np = params.len();
                    let mut optim =
                        optim::EntropicAdam::new(
                            np, cfg.lr, cfg.epochs,
                        );
                    optim.m = om;
                    optim.v = ov;
                    optim.t = ckpt.optim_t;
                    let model = flux::FluxModel {
                        d, n_layers: nl, params,
                    };
                    logger.log(&format!(
                        "Resumed from {} (epoch {}, loss={:.4})",
                        path, ckpt.epoch, ckpt.loss,
                    ));
                    return (model, optim, ckpt.epoch);
                }
                Err(e) => {
                    logger.log(&format!(
                        "Ckpt load failed ({}), fresh start", e,
                    ));
                }
            }
        }
    }
    let np = flux::flux_total_param_count(cfg.d, cfg.n_layers);
    let model = flux::FluxModel::new(cfg.d, cfg.n_layers, rng);
    let optim = optim::EntropicAdam::new(np, cfg.lr, cfg.epochs);
    (model, optim, 0)
}

fn cmd_generate(args: &[String]) {
    let mut ckpt_path = String::new();
    let mut seed_text = String::from("ROMEO:");
    let mut length = 500usize;
    let mut temperature = 0.8;
    let mut seed = 42u64;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ckpt" => { ckpt_path = cli::arg_str(args, &mut i); }
            "--seed-text" => { seed_text = cli::arg_str(args, &mut i); }
            "--length" => { length = cli::arg_usize(args, &mut i); }
            "--temperature" => { temperature = cli::arg_f64(args, &mut i); }
            "--seed" => { seed = cli::arg_u64(args, &mut i); }
            _ => { eprintln!("Unknown flag: {}", args[i]); return; }
        }
        i += 1;
    }

    if ckpt_path.is_empty() {
        eprintln!("ERROR: --ckpt required");
        return;
    }

    let ckpt = match checkpoint::load_checkpoint::<F>(&ckpt_path) {
        Ok(c) => c,
        Err(e) => { eprintln!("ERROR: {}", e); return; }
    };

    let mut rng = Rng::new(seed);
    let sb = seed_text.as_bytes();

    let model = flux::FluxModel {
        d: ckpt.d, n_layers: ckpt.n_layers,
        params: ckpt.params,
    };
    let generated = generate::generate_flux(
        &model, sb, length, temperature, &mut rng,
    );

    print!("{}", seed_text);
    print!("{}", String::from_utf8_lossy(&generated));
    println!();
}

fn cmd_info(args: &[String]) {
    let mut ckpt_path = String::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ckpt" => { ckpt_path = cli::arg_str(args, &mut i); }
            _ => { eprintln!("Unknown flag: {}", args[i]); return; }
        }
        i += 1;
    }

    if ckpt_path.is_empty() {
        eprintln!("ERROR: --ckpt required");
        return;
    }

    // Load as f64 for info display
    let ckpt = match checkpoint::load_checkpoint::<f64>(&ckpt_path) {
        Ok(c) => c,
        Err(e) => { eprintln!("ERROR: {}", e); return; }
    };

    let bpb = ckpt.loss / 0.6931471805599453;
    println!("Checkpoint: {}", ckpt_path);
    println!("  model:    Flux v3");
    println!("  d:        {}", ckpt.d);
    println!("  layers:   {}", ckpt.n_layers);
    println!("  params:   {}", ckpt.params.len());
    println!("  epoch:    {}", ckpt.epoch);
    println!("  loss:     {:.4}", ckpt.loss);
    println!("  bpb:      {:.4}", bpb);
}
