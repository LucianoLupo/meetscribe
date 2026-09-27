//! Diarization-split spike: run meetscribe's far-end voice identity (embed → cluster → inherit →
//! auto-match) over an arbitrary list of windows, so Whisper-chunk windows and
//! Nemotron-split windows can be compared on the same meeting with the same enrolled voiceprints.
//!
//! Mirrors `voices::assemble` + `voices::apply_matches` using the real `spk` module; the only
//! difference is slicing: this reads one 16 kHz mono `system.wav` directly (single-roll sessions).
//!
//! ```text
//! cargo build --release --bin split_probe && ./target/release/split_probe \
//!     --wav <system.wav> --windows <windows.json> --enrolled <enrolled.json> \
//!     --model <campplus.onnx> --out <result.json>
//! ```
//! windows.json  = [{"key": 0, "t_start": 1.2, "t_end": 4.5}, …]
//! enrolled.json = [{"speaker_id": 3, "voiceprints": [[f32; dim], …]}, …]

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[path = "../spk.rs"]
mod spk;

#[derive(Deserialize)]
struct InWindow {
    key: i64,
    t_start: f64,
    t_end: f64,
}

#[derive(Deserialize)]
struct InEnrolled {
    speaker_id: i64,
    voiceprints: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct OutWindow {
    key: i64,
    t_start: f64,
    t_end: f64,
    embedded: bool,
    cluster: Option<String>,
    inherited: bool,
    speaker_id: Option<i64>,
}

#[derive(Serialize)]
struct OutCluster {
    cluster: String,
    n_windows: usize,
    speech_secs: f64,
    speaker_id: Option<i64>,
    match_score: Option<f32>,
}

#[derive(Serialize)]
struct Out {
    windows: Vec<OutWindow>,
    clusters: Vec<OutCluster>,
}

fn arg(args: &[String], name: &str) -> Result<PathBuf> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .with_context(|| format!("missing {name}"))
}

fn read_wav_16k(path: &PathBuf) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = r.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        bail!("{}: need 16 kHz mono, got {} Hz × {}", path.display(), spec.sample_rate, spec.channels);
    }
    Ok(match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => r.samples::<i16>().map(|s| s.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>()?,
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let audio = read_wav_16k(&arg(&args, "--wav")?)?;
    let windows: Vec<InWindow> = serde_json::from_slice(&std::fs::read(arg(&args, "--windows")?)?)?;
    let enrolled: Vec<spk::Enrolled> = serde_json::from_slice::<Vec<InEnrolled>>(&std::fs::read(arg(&args, "--enrolled")?)?)?
        .into_iter()
        .map(|e| spk::Enrolled { speaker_id: e.speaker_id, voiceprints: e.voiceprints })
        .collect();
    let mut emb = spk::Embedder::load(&arg(&args, "--model")?)?;

    // Embed every window at least MIN_WINDOW_MS long (the live path's rule).
    let min_len = spk::MIN_WINDOW_MS * 16;
    let mut embeddings: Vec<Option<Vec<f32>>> = Vec::with_capacity(windows.len());
    for w in &windows {
        let a = ((w.t_start * 16_000.0) as usize).min(audio.len());
        let b = ((w.t_end * 16_000.0) as usize).min(audio.len());
        embeddings.push(if b > a && b - a >= min_len { Some(emb.embed(&audio[a..b])?) } else { None });
    }

    // voices::assemble, step for step.
    let embedded: Vec<usize> = (0..windows.len()).filter(|&i| embeddings[i].is_some()).collect();
    let embs: Vec<&[f32]> = embedded.iter().map(|&i| embeddings[i].as_deref().expect("filtered")).collect();
    let assignments = spk::cluster(&embs, spk::CLUSTER_CUT);
    let secs: Vec<f64> = embedded.iter().map(|&i| windows[i].t_end - windows[i].t_start).collect();
    let labels = spk::label_clusters(&assignments, &secs);

    let mut clusters = Vec::with_capacity(labels.len());
    for (c, label) in labels.iter().enumerate() {
        let members: Vec<&[f32]> =
            assignments.iter().zip(&embs).filter(|(a, _)| **a == c).map(|(_, e)| *e).collect();
        let speech_secs: f64 = assignments.iter().zip(&secs).filter(|(a, _)| **a == c).map(|(_, s)| *s).sum();
        let centroid = spk::centroid(&members);
        // voices::apply_matches
        let m = spk::auto_match(&centroid, members.len(), &enrolled);
        clusters.push(OutCluster {
            cluster: label.clone(),
            n_windows: members.len(),
            speech_secs,
            speaker_id: m.map(|(id, _)| id),
            match_score: m.map(|(_, s)| s),
        });
    }

    let mut cluster_of: Vec<Option<(usize, bool)>> = vec![None; windows.len()];
    for (&i, &c) in embedded.iter().zip(&assignments) {
        cluster_of[i] = Some((c, false));
    }
    let anchors: Vec<(f64, usize)> = embedded.iter().zip(&assignments).map(|(&i, &c)| (windows[i].t_start, c)).collect();
    let short: Vec<usize> = (0..windows.len()).filter(|&i| embeddings[i].is_none()).collect();
    let short_starts: Vec<f64> = short.iter().map(|&i| windows[i].t_start).collect();
    for (&i, inherited) in short.iter().zip(spk::inherit_short(&anchors, &short_starts)) {
        cluster_of[i] = inherited.map(|c| (c, true));
    }

    let out = Out {
        windows: windows
            .iter()
            .enumerate()
            .map(|(i, w)| OutWindow {
                key: w.key,
                t_start: w.t_start,
                t_end: w.t_end,
                embedded: embeddings[i].is_some(),
                cluster: cluster_of[i].map(|(c, _)| labels[c].clone()),
                inherited: cluster_of[i].is_some_and(|(_, inh)| inh),
                speaker_id: cluster_of[i].and_then(|(c, _)| clusters[c].speaker_id),
            })
            .collect(),
        clusters,
    };
    std::fs::write(arg(&args, "--out")?, serde_json::to_vec_pretty(&out)?)?;
    Ok(())
}
