use std::fs::OpenOptions;
use std::io::Write;
use std::time::SystemTime;

pub struct Logger {
    log_path: Option<String>,
}

impl Logger {
    pub fn new(log_path: Option<String>) -> Self {
        Self { log_path }
    }

    pub fn log(&self, msg: &str) {
        let ts = format_timestamp();
        let line = format!("[{}] {}", ts, msg);
        println!("{}", line);
        if let Some(ref path) = self.log_path {
            let _ = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .and_then(|mut f| writeln!(f, "{}", line));
        }
    }
}

fn format_timestamp() -> String {
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let (s, m, h) = (secs % 60, (secs / 60) % 60, (secs / 3600) % 24);
    let days = secs / 86400;
    let (y, mo, d) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", y, mo, d, h, m, s)
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut y = 1970;
    loop {
        let ylen = if is_leap(y) { 366 } else { 365 };
        if days < ylen { break; }
        days -= ylen;
        y += 1;
    }
    let leap = is_leap(y);
    let mdays = [
        31, if leap { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];
    let mut mo = 0;
    for (i, &ml) in mdays.iter().enumerate() {
        if days < ml { mo = i; break; }
        days -= ml;
    }
    (y, (mo + 1) as u64, days + 1)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}
