pub struct Config {
    pub d: usize,
    pub n_layers: usize,
    pub seq_len: usize,
    pub epochs: usize,
    pub lr: f64,
    pub seed: u64,
    pub corpus_path: String,
    pub ckpt_path: Option<String>,
    pub ckpt_every: usize,
    pub log_path: Option<String>,
    pub print_every: usize,
    pub core_fraction: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            d: 256,
            n_layers: 3,
            seq_len: 256,
            epochs: 50,
            lr: 0.001,
            seed: 42,
            corpus_path: String::new(),
            ckpt_path: None,
            ckpt_every: 10,
            log_path: None,
            print_every: 5,
            core_fraction: 0.75,
        }
    }
}
