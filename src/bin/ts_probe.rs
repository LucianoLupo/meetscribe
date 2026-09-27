//! Audit probe: plain vs DTW Whisper token timestamps on chunks (scratch only).
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use whisper_rs::{
    DtwMode, DtwModelPreset, DtwParameters, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters,
};

#[derive(Deserialize)]
struct Chunk { key: i64, t_start: f64, t_end: f64 }

#[derive(Serialize)]
struct Tok { text: String, t0: i64, t1: i64, t_dtw: i64, p: f32 }

#[derive(Serialize)]
struct Out { key: i64, mode: String, text: String, toks: Vec<Tok> }

fn read_wav(path: &str) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path)?;
    let s = r.spec();
    if s.sample_rate != 16000 || s.channels != 1 { bail!("need 16k mono"); }
    Ok(match s.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => r.samples::<i16>().map(|x| x.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>()?,
    })
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (model, wav, chunks, out) = (&a[1], &a[2], &a[3], &a[4]);
    let audio = read_wav(wav)?;
    let chunks: Vec<Chunk> = serde_json::from_str(&std::fs::read_to_string(chunks)?)?;
    let mut res = Vec::new();
    for dtw in [false, true] {
        let mut cp = WhisperContextParameters::default();
        if dtw {
            cp.dtw_parameters(DtwParameters {
                mode: DtwMode::ModelPreset { model_preset: DtwModelPreset::LargeV3 },
                ..Default::default()
            });
        }
        let ctx = WhisperContext::new_with_params(model, cp).context("load")?;
        let mut st = ctx.create_state()?;
        for c in &chunks {
            let a0 = (c.t_start * 16000.0) as usize;
            let b0 = ((c.t_end * 16000.0) as usize).min(audio.len());
            let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            p.set_language(Some("es"));
            p.set_n_threads(8);
            p.set_print_progress(false);
            p.set_print_realtime(false);
            p.set_print_timestamps(false);
            p.set_token_timestamps(true);
            st.full(p, &audio[a0..b0])?;
            let n = st.full_n_segments()?;
            let mut text = String::new();
            let mut toks = Vec::new();
            for i in 0..n {
                text.push_str(&st.full_get_segment_text_lossy(i).unwrap_or_default());
                for t in 0..st.full_n_tokens(i)? {
                    let d = st.full_get_token_data(i, t)?;
                    let tx = st.full_get_token_text_lossy(i, t).unwrap_or_default();
                    toks.push(Tok { text: tx, t0: d.t0, t1: d.t1, t_dtw: d.t_dtw, p: d.p });
                }
            }
            eprintln!("{} {} {}", if dtw { "dtw" } else { "plain" }, c.key, text.trim());
            res.push(Out { key: c.key, mode: if dtw { "dtw".into() } else { "plain".into() }, text, toks });
        }
    }
    std::fs::write(out, serde_json::to_string(&res)?)?;
    Ok(())
}
