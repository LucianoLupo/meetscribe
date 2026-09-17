//! Batch D — speaker-embedding calibration spike (produces a number and a go/no-go, not a feature).
//!
//! Walks `~/.meetscribe/sessions/*/` directly (NOT the `meetings` table — several recordings have
//! audio but no DB row), embeds VAD speech windows from both channels with the 3D-Speaker CAM++
//! ONNX model, and measures on the FREE calibration set:
//!
//! - same-speaker  = `you` vs `you` across meetings (mic.wav is Luciano by construction), split by
//!   capture rate pair (16 kHz = Bluetooth HFP, 48 kHz = built-in mic) → codec-drift check;
//! - different     = `you` vs `others`;
//! - ROC / EER + the threshold at EER;
//! - coalesced-window PURITY: intra-window 3 s sub-chunk cosine on long `others` windows versus the
//!   single-speaker baseline from the `you` channel (a blended window scores below the baseline);
//! - wall-clock per stage (read / resample / VAD / embed) so embedding cost is measured, not guessed.
//!
//! ```text
//! cargo build --release --bin spk_probe && ./target/release/spk_probe [--sessions <dir>]
//!     [--model <onnx>] [--max-secs 900] [--per-channel 12] [--out report.json]
//!     [--dump-clip <16k.wav>] [--embed-wav <16k.wav>] [--dump-embeddings <json>]
//! ```
//!
//! `--embed-wav` prints one embedding as JSON — cross-checked against sherpa-onnx (an independent
//! implementation of the same model + fbank recipe) to prove the Rust front-end is right.
//! Capture-free → no TCC, no code-signing required.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use ndarray::Array3;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;

#[allow(dead_code)]
#[path = "../resample.rs"]
mod resample;
#[allow(dead_code)]
#[path = "../vad.rs"]
mod vad;

const RATE: usize = 16_000;
/// Windows shorter than this are not embedded (E's floor).
const MIN_WINDOW_MS: usize = 1500;
/// Sub-chunk length for the purity check.
const CHUNK_SECS: usize = 3;
/// Only `others` windows at least this long get the purity check (≥ 3 sub-chunks).
const PURITY_MIN_SECS: usize = 9;

// ---------------------------------------------------------------- embedder

struct Embedder {
    session: Session,
    input_name: String,
    dim: usize,
}

impl Embedder {
    fn load(path: &Path) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(path)
            .with_context(|| format!("load speaker model {}", path.display()))?;
        let input_name = session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .context("model has no inputs")?;
        Ok(Self { session, input_name, dim: 0 })
    }

    /// L2-normalised embedding of a 16 kHz mono clip.
    fn embed(&mut self, audio_16k: &[f32]) -> Result<Vec<f32>> {
        // knf: 25 ms / 10 ms Kaldi fbank, 80 bins, dither 0, then per-utterance mean subtraction
        // (== the model's `feature_normalize_type: global-mean`, `normalize_samples: 1`).
        let feats = knf_rs::compute_fbank(audio_16k).map_err(|e| anyhow::anyhow!("fbank: {e}"))?;
        let (t, bins) = feats.dim();
        let x: Array3<f32> = feats
            .into_shape_with_order((1, t, bins))
            .context("reshape fbank to (1,T,80)")?;
        let outputs = self.session.run(ort::inputs![
            self.input_name.as_str() => TensorRef::from_array_view(x.view())?
        ])?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
        let mut v = data.to_vec();
        let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-9);
        v.iter_mut().for_each(|a| *a /= norm);
        self.dim = v.len();
        Ok(v)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------- audio helpers

/// Mono 32-bit float WAV at any rate (what capture writes). Truncated to `max_secs` if given.
fn read_wav(path: &Path, max_secs: Option<f64>) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_format != hound::SampleFormat::Float || spec.bits_per_sample != 32 {
        bail!("{}: expected mono 32-bit float WAV", path.display());
    }
    let cap = max_secs.map(|s| (s * spec.sample_rate as f64) as usize);
    let mut samples = Vec::new();
    for s in reader.into_samples::<f32>() {
        samples.push(s.context("read sample")?);
        if let Some(c) = cap
            && samples.len() >= c
        {
            break;
        }
    }
    Ok((samples, spec.sample_rate))
}

fn write_wav_16k(path: &Path, audio: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: RATE as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for s in audio {
        w.write_sample(*s)?;
    }
    w.finalize()?;
    Ok(())
}

/// Tiny deterministic PRNG (no `rand` dep for a spike).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

// ---------------------------------------------------------------- per-meeting extraction

#[derive(Default)]
struct Timing {
    read: f64,
    resample: f64,
    vad: f64,
    embed: f64,
    audio_secs: f64,
    embedded_secs: f64,
    windows: usize,
}

struct Channel {
    rate: u32,
    /// Whole-window embeddings (subsampled, ≥ MIN_WINDOW_MS).
    windows: Vec<Vec<f32>>,
    /// (start_ms, end_ms) of each embedded window, same order as `windows`.
    window_times: Vec<(usize, usize)>,
    /// Per long window: its 3 s sub-chunk embeddings (purity check).
    chunked: Vec<Vec<Vec<f32>>>,
}

struct Meeting {
    stamp: String,
    you: Option<Channel>,
    others: Option<Channel>,
}

#[allow(clippy::too_many_arguments)]
fn extract_channel(
    path: &Path,
    emb: &mut Embedder,
    max_secs: Option<f64>,
    per_channel: usize,
    purity_windows: usize,
    rng: &mut Rng,
    timing: &mut Timing,
    dump_clip: &mut Option<PathBuf>,
) -> Result<Channel> {
    let t = Instant::now();
    let (samples, rate) = read_wav(path, max_secs)?;
    timing.read += t.elapsed().as_secs_f64();

    let t = Instant::now();
    let audio = resample::to_16k_mono(&samples, rate)?;
    timing.resample += t.elapsed().as_secs_f64();
    timing.audio_secs += audio.len() as f64 / RATE as f64;

    let t = Instant::now();
    let all = vad::speech_windows(&audio)?;
    timing.vad += t.elapsed().as_secs_f64();

    let usable: Vec<_> = all
        .into_iter()
        .filter(|w| w.end_ms.saturating_sub(w.start_ms) >= MIN_WINDOW_MS)
        .collect();
    // Evenly spaced subsample so the whole meeting is represented, not just its opening.
    let pick = |n: usize, k: usize| -> Vec<usize> {
        if n <= k {
            (0..n).collect()
        } else {
            (0..k).map(|i| i * n / k).collect()
        }
    };
    let mut windows = Vec::new();
    let mut window_times = Vec::new();
    let t = Instant::now();
    for i in pick(usable.len(), per_channel) {
        let (a, b) = usable[i].sample_range(audio.len());
        window_times.push((usable[i].start_ms, usable[i].end_ms));
        if let Some(p) = dump_clip.take() {
            write_wav_16k(&p, &audio[a..b])?;
            eprintln!("dumped first window ({:.2}s) → {}", (b - a) as f64 / RATE as f64, p.display());
        }
        windows.push(emb.embed(&audio[a..b])?);
        timing.embedded_secs += (b - a) as f64 / RATE as f64;
    }
    // Purity: random long windows → consecutive 3 s chunks.
    let long: Vec<_> = usable
        .iter()
        .filter(|w| w.end_ms.saturating_sub(w.start_ms) >= PURITY_MIN_SECS * 1000)
        .collect();
    let mut chunked = Vec::new();
    for _ in 0..purity_windows.min(long.len()) {
        let w = long[rng.below(long.len())];
        let (a, b) = w.sample_range(audio.len());
        let mut chunks = Vec::new();
        let mut s = a;
        while s + CHUNK_SECS * RATE <= b {
            chunks.push(emb.embed(&audio[s..s + CHUNK_SECS * RATE])?);
            timing.embedded_secs += CHUNK_SECS as f64;
            s += CHUNK_SECS * RATE;
        }
        if chunks.len() >= 3 {
            chunked.push(chunks);
        }
    }
    timing.embed += t.elapsed().as_secs_f64();
    timing.windows += windows.len();
    Ok(Channel { rate, windows, window_times, chunked })
}

// ---------------------------------------------------------------- statistics

fn stats(xs: &[f32]) -> BTreeMap<&'static str, f64> {
    let mut m = BTreeMap::new();
    if xs.is_empty() {
        return m;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize] as f64;
    m.insert("n", v.len() as f64);
    m.insert("mean", v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64);
    m.insert("p05", q(0.05));
    m.insert("p50", q(0.5));
    m.insert("p95", q(0.95));
    m.insert("min", v[0] as f64);
    m.insert("max", v[v.len() - 1] as f64);
    m
}

/// Equal error rate over cosine scores: returns (eer, threshold_at_eer).
fn eer(same: &[f32], diff: &[f32]) -> (f64, f64) {
    let mut best = (1.0f64, 0.0f64);
    let mut best_gap = f64::MAX;
    let mut t = -1.0f64;
    while t <= 1.0 {
        let frr = same.iter().filter(|s| (**s as f64) < t).count() as f64 / same.len().max(1) as f64;
        let far = diff.iter().filter(|s| (**s as f64) >= t).count() as f64 / diff.len().max(1) as f64;
        let gap = (frr - far).abs();
        if gap < best_gap {
            best_gap = gap;
            best = ((frr + far) / 2.0, t);
        }
        t += 0.005;
    }
    best
}

/// False-accept rate at the threshold that yields `frr_target` false rejects.
fn far_at_frr(same: &[f32], diff: &[f32], frr_target: f64) -> (f64, f64) {
    if same.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut s = same.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((s.len() as f64) * frr_target).floor() as usize;
    let thr = s[idx.min(s.len().saturating_sub(1))] as f64;
    let far = diff.iter().filter(|d| (**d as f64) >= thr).count() as f64 / diff.len().max(1) as f64;
    (far, thr)
}

// ---------------------------------------------------------------- main

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let home = std::env::var("HOME").unwrap_or_default();
    let mut sessions = PathBuf::from(format!("{home}/.meetscribe/sessions"));
    let mut model = PathBuf::from("models/speaker/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx");
    let mut max_secs: Option<f64> = Some(900.0);
    let mut per_channel = 12usize;
    let mut purity_windows = 4usize;
    let mut out: Option<PathBuf> = None;
    let mut dump_clip: Option<PathBuf> = None;
    let mut embed_wav: Option<PathBuf> = None;
    let mut dump_embeddings: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut i = 1;
    while i < argv.len() {
        let a = argv[i].as_str();
        let mut val = || {
            i += 1;
            argv.get(i).cloned().with_context(|| format!("{a} needs a value"))
        };
        match a {
            "--sessions" => sessions = PathBuf::from(val()?),
            "--model" => model = PathBuf::from(val()?),
            "--max-secs" => {
                let v: f64 = val()?.parse()?;
                max_secs = if v <= 0.0 { None } else { Some(v) };
            }
            "--per-channel" => per_channel = val()?.parse()?,
            "--purity-windows" => purity_windows = val()?.parse()?,
            "--limit" => limit = Some(val()?.parse()?),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--dump-clip" => dump_clip = Some(PathBuf::from(val()?)),
            "--embed-wav" => embed_wav = Some(PathBuf::from(val()?)),
            "--dump-embeddings" => dump_embeddings = Some(PathBuf::from(val()?)),
            other => bail!("unknown arg {other}"),
        }
        i += 1;
    }

    let mut emb = Embedder::load(&model)?;

    if let Some(p) = embed_wav {
        let (samples, rate) = read_wav(&p, None)?;
        if rate as usize != RATE {
            bail!("--embed-wav wants a 16 kHz clip (got {rate} Hz)");
        }
        let v = emb.embed(&samples)?;
        println!("{}", serde_json::to_string(&v)?);
        return Ok(());
    }

    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&sessions)
        .with_context(|| format!("read {}", sessions.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    if let Some(l) = limit {
        dirs.truncate(l);
    }

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut timing = Timing::default();
    let mut meetings: Vec<Meeting> = Vec::new();
    let mut skipped = 0usize;
    let t_all = Instant::now();
    for d in &dirs {
        let stamp = d.file_name().unwrap().to_string_lossy().to_string();
        let mic = d.join("mic.wav");
        let sys = d.join("system.wav");
        if !mic.exists() && !sys.exists() {
            eprintln!("skip {stamp}: no mic.wav/system.wav");
            skipped += 1;
            continue;
        }
        let mut m = Meeting { stamp: stamp.clone(), you: None, others: None };
        for (p, is_you) in [(&mic, true), (&sys, false)] {
            if !p.exists() {
                continue;
            }
            match extract_channel(p, &mut emb, max_secs, per_channel, purity_windows, &mut rng, &mut timing, &mut dump_clip) {
                Ok(c) => {
                    if is_you {
                        m.you = Some(c)
                    } else {
                        m.others = Some(c)
                    }
                }
                Err(e) => eprintln!("  {stamp} {}: {e:#}", p.file_name().unwrap().to_string_lossy()),
            }
        }
        eprintln!(
            "{stamp}: you {} win / others {} win @ {} Hz  [{:.0}s elapsed]",
            m.you.as_ref().map_or(0, |c| c.windows.len()),
            m.others.as_ref().map_or(0, |c| c.windows.len()),
            m.you.as_ref().or(m.others.as_ref()).map_or(0, |c| c.rate),
            t_all.elapsed().as_secs_f64()
        );
        meetings.push(m);
    }

    if let Some(p) = &dump_embeddings {
        let rows: Vec<serde_json::Value> = meetings
            .iter()
            .map(|m| {
                let ch = |c: &Option<Channel>| {
                    c.as_ref().map(|c| serde_json::json!({"rate": c.rate, "windows": c.windows, "window_times": c.window_times, "chunked": c.chunked}))
                };
                serde_json::json!({"stamp": m.stamp, "you": ch(&m.you), "others": ch(&m.others)})
            })
            .collect();
        std::fs::write(p, serde_json::to_string(&rows)?)?;
        eprintln!("embeddings → {}", p.display());
    }

    // ---- scores
    let mut same_cross: BTreeMap<String, Vec<f32>> = BTreeMap::new(); // by rate pair
    let mut same_within: Vec<f32> = Vec::new();
    let mut diff_same_meeting: Vec<f32> = Vec::new();
    let mut diff_cross: Vec<f32> = Vec::new();
    let yous: Vec<(usize, &Channel)> = meetings.iter().enumerate().filter_map(|(i, m)| m.you.as_ref().map(|c| (i, c))).collect();
    let others: Vec<(usize, &Channel)> = meetings.iter().enumerate().filter_map(|(i, m)| m.others.as_ref().map(|c| (i, c))).collect();

    for (a, (ia, ca)) in yous.iter().enumerate() {
        for w in 0..ca.windows.len() {
            for w2 in (w + 1)..ca.windows.len() {
                same_within.push(cosine(&ca.windows[w], &ca.windows[w2]));
            }
        }
        for (ib, cb) in yous.iter().skip(a + 1) {
            let _ = ib;
            let key = {
                let (r1, r2) = (ca.rate.min(cb.rate), ca.rate.max(cb.rate));
                format!("{}-{}", r1 / 1000, r2 / 1000)
            };
            let bucket = same_cross.entry(key).or_default();
            // up to 6 random pairs per meeting pair
            for _ in 0..6 {
                if ca.windows.is_empty() || cb.windows.is_empty() {
                    break;
                }
                let x = &ca.windows[rng.below(ca.windows.len())];
                let y = &cb.windows[rng.below(cb.windows.len())];
                bucket.push(cosine(x, y));
            }
        }
        for (ib, cb) in &others {
            if cb.windows.is_empty() || ca.windows.is_empty() {
                continue;
            }
            let dst = if ib == ia { &mut diff_same_meeting } else { &mut diff_cross };
            for _ in 0..(if ib == ia { 20 } else { 2 }) {
                let x = &ca.windows[rng.below(ca.windows.len())];
                let y = &cb.windows[rng.below(cb.windows.len())];
                dst.push(cosine(x, y));
            }
        }
    }

    // ---- purity
    let intra = |chs: &[Vec<Vec<f32>>]| -> (Vec<f32>, Vec<f32>) {
        let mut mins = Vec::new();
        let mut means = Vec::new();
        for chunks in chs {
            let mut cs = Vec::new();
            for i in 0..chunks.len() {
                for j in (i + 1)..chunks.len() {
                    cs.push(cosine(&chunks[i], &chunks[j]));
                }
            }
            if !cs.is_empty() {
                mins.push(cs.iter().cloned().fold(f32::MAX, f32::min));
                means.push(cs.iter().sum::<f32>() / cs.len() as f32);
            }
        }
        (mins, means)
    };
    let you_chunks: Vec<Vec<Vec<f32>>> = yous.iter().flat_map(|(_, c)| c.chunked.iter().cloned()).collect();
    let oth_chunks: Vec<Vec<Vec<f32>>> = others.iter().flat_map(|(_, c)| c.chunked.iter().cloned()).collect();
    let (you_min, you_mean) = intra(&you_chunks);
    let (oth_min, oth_mean) = intra(&oth_chunks);
    let you_p05 = stats(&you_min).get("p05").copied().unwrap_or(0.0);
    let blended_frac = if oth_min.is_empty() {
        0.0
    } else {
        oth_min.iter().filter(|m| (**m as f64) < you_p05).count() as f64 / oth_min.len() as f64
    };

    let all_same: Vec<f32> = same_cross.values().flatten().cloned().collect();
    let all_diff: Vec<f32> = diff_same_meeting.iter().chain(diff_cross.iter()).cloned().collect();
    let (eer_v, thr) = eer(&all_same, &all_diff);
    let (far1, thr1) = far_at_frr(&all_same, &all_diff, 0.01);
    let (eer_cross_only, thr_cross_only) = eer(&all_same, &diff_cross);

    let mut report = serde_json::Map::new();
    report.insert("model".into(), model.display().to_string().into());
    report.insert("embedding_dim".into(), emb.dim.into());
    report.insert("meetings_used".into(), meetings.len().into());
    report.insert("dirs_skipped".into(), skipped.into());
    report.insert("max_secs_per_channel".into(), serde_json::to_value(max_secs)?);
    report.insert("same_speaker_cross_meeting_by_rate".into(),
        serde_json::to_value(same_cross.iter().map(|(k, v)| (k.clone(), stats(v))).collect::<BTreeMap<_, _>>())?);
    report.insert("same_speaker_within_meeting".into(), serde_json::to_value(stats(&same_within))?);
    report.insert("different_same_meeting".into(), serde_json::to_value(stats(&diff_same_meeting))?);
    report.insert("different_cross_meeting".into(), serde_json::to_value(stats(&diff_cross))?);
    report.insert("eer".into(), eer_v.into());
    report.insert("threshold_at_eer".into(), thr.into());
    report.insert("far_at_1pct_frr".into(), far1.into());
    report.insert("threshold_at_1pct_frr".into(), thr1.into());
    report.insert("eer_vs_cross_meeting_others_only".into(), eer_cross_only.into());
    report.insert("threshold_eer_cross_only".into(), thr_cross_only.into());
    report.insert("purity_you_intra_min".into(), serde_json::to_value(stats(&you_min))?);
    report.insert("purity_you_intra_mean".into(), serde_json::to_value(stats(&you_mean))?);
    report.insert("purity_others_intra_min".into(), serde_json::to_value(stats(&oth_min))?);
    report.insert("purity_others_intra_mean".into(), serde_json::to_value(stats(&oth_mean))?);
    report.insert("purity_others_blended_fraction".into(), blended_frac.into());
    report.insert("timing".into(), serde_json::json!({
        "read_s": timing.read, "resample_s": timing.resample, "vad_s": timing.vad, "embed_s": timing.embed,
        "audio_secs_processed": timing.audio_secs, "embedded_secs": timing.embedded_secs,
        "windows_embedded": timing.windows,
        "embed_rtf": timing.embed / timing.embedded_secs.max(1e-9),
        "pipeline_rtf_excl_whisper": (timing.read + timing.resample + timing.vad + timing.embed) / timing.audio_secs.max(1e-9),
        "total_wall_s": t_all.elapsed().as_secs_f64(),
    }));

    let json = serde_json::to_string_pretty(&report)?;
    println!("{json}");
    if let Some(p) = out {
        std::fs::write(&p, &json)?;
        eprintln!("report → {}", p.display());
    }
    Ok(())
}
