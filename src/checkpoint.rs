/// Checkpoint V3: precision-aware format + V2 backward compatibility.

use std::fs::File;
use std::io::{Read, Write, BufWriter, BufReader};

use crate::float::Float;
use crate::optim::EntropicAdam;

const MAGIC: &[u8; 4] = b"ELMC";
const VERSION_2: u32 = 2;
const VERSION_3: u32 = 3;

const PRECISION_F32: u8 = 0;
const PRECISION_F64: u8 = 1;

pub struct CkptData<F: Float> {
    pub d: usize,
    pub n_layers: usize,
    pub params: Vec<F>,
    pub optim_m: Vec<F>,
    pub optim_v: Vec<F>,
    pub optim_t: usize,
    pub epoch: u32,
    pub loss: f64,
}

fn precision_byte<F: Float>() -> u8 {
    if F::BYTE_SIZE == 4 { PRECISION_F32 } else { PRECISION_F64 }
}

pub fn save_v3<F: Float>(
    d: usize, n_layers: usize,
    params: &[F], opt_m: &[F], opt_v: &[F], opt_t: usize,
    epoch: u32, loss: f64, path: &str,
) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION_3.to_le_bytes())?;
    w.write_all(&[precision_byte::<F>()])?;
    w.write_all(&4u32.to_le_bytes())?; // model_type=Flux
    w.write_all(&(d as u32).to_le_bytes())?;
    w.write_all(&(n_layers as u32).to_le_bytes())?;
    w.write_all(&epoch.to_le_bytes())?;
    w.write_all(&loss.to_le_bytes())?; // always f64
    w.write_all(&(params.len() as u32).to_le_bytes())?;
    write_float_slice(&mut w, params)?;
    write_float_slice(&mut w, opt_m)?;
    write_float_slice(&mut w, opt_v)?;
    w.write_all(&(opt_t as u32).to_le_bytes())?;
    Ok(())
}

pub fn load_checkpoint<F: Float>(path: &str) -> std::io::Result<CkptData<F>> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, "bad magic",
        ));
    }
    let version = read_u32(&mut r)?;
    match version {
        VERSION_2 => load_v2_as::<F>(&mut r),
        VERSION_3 => load_v3_as::<F>(&mut r),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported checkpoint version {}", version),
        )),
    }
}

fn load_v3_as<F: Float>(r: &mut BufReader<File>) -> std::io::Result<CkptData<F>> {
    let mut prec_buf = [0u8; 1];
    r.read_exact(&mut prec_buf)?;
    let file_precision = prec_buf[0];

    let _mt = read_u32(r)?; // model_type (always 4=Flux)
    let d = read_u32(r)? as usize;
    let n_layers = read_u32(r)? as usize;
    let epoch = read_u32(r)?;
    let loss = read_f64(r)?;
    let n_params = read_u32(r)? as usize;

    let target_prec = precision_byte::<F>();

    let (params, optim_m, optim_v) = if file_precision == target_prec {
        let p = read_float_slice::<F>(r, n_params)?;
        let m = read_float_slice::<F>(r, n_params)?;
        let v = read_float_slice::<F>(r, n_params)?;
        (p, m, v)
    } else if file_precision == PRECISION_F64 {
        // File is f64, target is f32: read as f64, convert
        let p = read_f64_slice_convert::<F>(r, n_params)?;
        let m = read_f64_slice_convert::<F>(r, n_params)?;
        let v = read_f64_slice_convert::<F>(r, n_params)?;
        (p, m, v)
    } else {
        // File is f32, target is f64: read as f32, convert
        let p = read_f32_slice_convert::<F>(r, n_params)?;
        let m = read_f32_slice_convert::<F>(r, n_params)?;
        let v = read_f32_slice_convert::<F>(r, n_params)?;
        (p, m, v)
    };

    let optim_t = read_u32(r)? as usize;
    Ok(CkptData { d, n_layers, params, optim_m, optim_v, optim_t, epoch, loss })
}

/// Load V2 checkpoint (always f64 on disk) and convert to target F.
fn load_v2_as<F: Float>(r: &mut BufReader<File>) -> std::io::Result<CkptData<F>> {
    let mt_u32 = read_u32(r)?;
    if mt_u32 != 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("V2 checkpoint is model type {}, expected 4 (Flux)", mt_u32),
        ));
    }
    let d = read_u32(r)? as usize;
    let n_layers = read_u32(r)? as usize;
    let epoch = read_u32(r)?;
    let loss = read_f64(r)?;
    let n_params = read_u32(r)? as usize;

    let params = read_f64_slice_convert::<F>(r, n_params)?;
    let optim_m = read_f64_slice_convert::<F>(r, n_params)?;
    let optim_v = read_f64_slice_convert::<F>(r, n_params)?;
    let optim_t = read_u32(r)? as usize;

    Ok(CkptData { d, n_layers, params, optim_m, optim_v, optim_t, epoch, loss })
}

pub fn save_flux<F: Float>(
    model_d: usize, model_nl: usize,
    params: &[F], optim: &EntropicAdam<F>,
    epoch: u32, loss: f64, path: &str,
) -> std::io::Result<()> {
    save_v3(
        model_d, model_nl,
        params, &optim.m, &optim.v, optim.t,
        epoch, loss, path,
    )
}

// ---- I/O helpers ----

fn write_float_slice<F: Float, W: std::io::Write>(
    w: &mut W, s: &[F],
) -> std::io::Result<()> {
    for &val in s {
        val.write_le(w)?;
    }
    Ok(())
}

fn read_float_slice<F: Float>(
    r: &mut BufReader<File>, n: usize,
) -> std::io::Result<Vec<F>> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(F::read_le(r)?);
    }
    Ok(v)
}

fn read_f64_slice_convert<F: Float>(
    r: &mut BufReader<File>, n: usize,
) -> std::io::Result<Vec<F>> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf)?;
        v.push(F::from_f64(f64::from_le_bytes(buf)));
    }
    Ok(v)
}

fn read_f32_slice_convert<F: Float>(
    r: &mut BufReader<File>, n: usize,
) -> std::io::Result<Vec<F>> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        v.push(F::from_f64(f32::from_le_bytes(buf) as f64));
    }
    Ok(v)
}

fn read_u32(r: &mut BufReader<File>) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f64(r: &mut BufReader<File>) -> std::io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}
