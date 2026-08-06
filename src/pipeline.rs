//! The transcribe pipeline core (Phase 2/3), extracted from `main()`'s `transcribe` handler so
//! BOTH the CLI `transcribe` subcommand and the Phase-4 daemon run the SAME path: per channel
//! resample → VAD → whisper each speech window → tag You/Others → merge → **store AND export**
//! (Markdown+JSON). Storing without exporting would leave an auto-captured meeting with no
//! readable file — so the export is part of the core, not the CLI wrapper.
//!
//! A DB-store failure must NOT discard the transcript we spent real compute on: on error the core
//! logs, falls back to a synthetic row, and STILL writes the exports (`stored_id = None`).

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::{asr, db, export, render, resample, transcript, vad};

/// Inputs for one transcribe+store+export run. Designed against BOTH callers (CLI + daemon):
/// the daemon supplies an ABSOLUTE `model` path (under launchd `cwd=/`, a repo-relative path
/// resolves to `/models/...`); the CLI supplies its repo-relative default (run from the repo).
pub(crate) struct PipelineOpts {
    /// ggml model path. MUST be absolute for the daemon; repo-relative is fine for CLI-from-repo.
    pub model: PathBuf,
    pub lang: String,
    /// Meeting title; `None` derives it from the capture dir name.
    pub title: Option<String>,
    pub db_path: PathBuf,
    /// Where the Markdown+JSON land (defaults, at both call sites, to the capture/session dir).
    pub export_dir: PathBuf,
    pub no_store: bool,
}

/// Result of the pipeline. `segments`/`meeting_secs`/`rtf`/`wall_secs` let the CLI wrapper print
/// the same transcript+RTF summary it always did; `stored_id` is `Some` only when persisted.
pub(crate) struct PipelineOutput {
    pub segments: Vec<transcript::TranscriptSegment>,
    pub stored_id: Option<i64>,
    pub md_path: PathBuf,
    pub json_path: PathBuf,
    pub meeting_secs: f64,
    pub wall_secs: f64,
    pub rtf: f64,
}

/// Transcribe every channel segment under `dir`, merge into one time-ordered speaker-labeled
/// transcript, persist to the DB (unless `no_store`), and write Markdown+JSON exports.
pub(crate) fn transcribe_and_store(dir: &Path, opts: &PipelineOpts) -> Result<PipelineOutput> {
    let title = opts.title.clone().unwrap_or_else(|| {
        dir.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("meeting")
            .to_string()
    });

    let mic_files = discover_channel(dir, "mic");
    let sys_files = discover_channel(dir, "system");
    if mic_files.is_empty() && sys_files.is_empty() {
        anyhow::bail!("no mic.wav/system.wav found in {}", dir.display());
    }
    log::info!(
        "transcribe {} — mic segments: {}, system segments: {}",
        dir.display(),
        mic_files.len(),
        sys_files.len()
    );

    let model_str = opts
        .model
        .to_str()
        .with_context(|| format!("model path not valid UTF-8: {}", opts.model.display()))?;
    let mut asr = asr::Asr::load(model_str).with_context(|| format!("load model {model_str}"))?;
    // Inter-segment gaps the capture layer recorded on rate-roll boundaries (empty for the common
    // single-segment case). Both channels share the same manifest.
    let gaps = read_segment_gaps(dir)?;

    let t0 = Instant::now();
    let mut segs: Vec<transcript::TranscriptSegment> = Vec::new();
    let mut meeting_secs = 0.0f64;

    for (files, speaker) in [
        (&mic_files, transcript::Speaker::You),
        (&sys_files, transcript::Speaker::Others),
    ] {
        // Multi-segment (rate-roll) absolute-time offset = cumulative prior-segment duration + the
        // recorded inter-segment gap before each rolled segment (segments.txt).
        let mut offset = 0.0f64;
        for (i, path) in files.iter().enumerate() {
            if i >= 1 {
                offset += gaps.get(&(i as u32)).copied().unwrap_or(0.0);
            }
            let (samples, rate) = read_wav_any_rate(path)?;
            let audio16 = resample::to_16k_mono(&samples, rate)
                .with_context(|| format!("resample {}", path.display()))?;
            let dur = audio16.len() as f64 / resample::TARGET_RATE as f64;
            let windows =
                vad::speech_windows(&audio16).with_context(|| format!("vad {}", path.display()))?;
            log::info!(
                "  {} @ {} Hz → {:.1}s, {} speech windows",
                path.display(),
                rate,
                dur,
                windows.len()
            );
            for w in windows {
                let (a, b) = w.sample_range(audio16.len());
                if b <= a {
                    continue;
                }
                let (text, confidence) = asr.transcribe(&audio16[a..b], &opts.lang)?;
                if text.is_empty() {
                    continue;
                }
                segs.push(transcript::TranscriptSegment {
                    speaker,
                    text,
                    t_start: offset + w.start_ms as f64 / 1000.0,
                    t_end: offset + w.end_ms as f64 / 1000.0,
                    confidence,
                });
            }
            offset += dur;
        }
        meeting_secs = meeting_secs.max(offset);
    }

    let merged = transcript::merge(segs);
    let wall = t0.elapsed().as_secs_f64();
    let rtf = if meeting_secs > 0.0 { wall / meeting_secs } else { 0.0 };

    // started_at = the real meeting time (earliest capture-file birth time).
    let meta = db::MeetingMeta {
        title,
        source_dir: dir.display().to_string(),
        model: model_name(&opts.model),
        lang: opts.lang.clone(),
        started_at: earliest_capture_start(&mic_files, &sys_files),
        duration_secs: meeting_secs,
        created_at: crate::now_epoch(),
    };

    // Persist to the DB, then export. A storage failure must NOT discard the transcript: on error
    // we log and fall back to a synthetic row so the Markdown/JSON still get written.
    //
    // Vocabulary is loaded INSIDE this block_on, before `close()` — these exports are the files
    // the daemon writes for every captured meeting, so rendering them without corrections would
    // mean the feature never reaches the surface people actually read.
    let (row, stored_id, vocab_rows) = if opts.no_store {
        // No DB handle on this path: render raw, and say so rather than silently differing from
        // what `export <id>` would produce.
        log::info!("--no-store: exporting without vocabulary corrections (no database opened)");
        (synth_row(&meta, merged.len()), None, Vec::new())
    } else {
        let rt = crate::new_runtime()?;
        let stored = rt.block_on(async {
            let mut database = db::Db::open(&opts.db_path).await?;
            let id = database.insert_meeting(&meta, &merged).await?;
            let row = database
                .get_meeting(id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("meeting {id} vanished after insert"))?;
            let vocab = database.load_enabled_vocab().await?;
            database.close().await?;
            anyhow::Ok((id, row, vocab))
        });
        match stored {
            Ok((id, row, vocab)) => {
                log::info!("stored meeting id {id} → {}", opts.db_path.display());
                (row, Some(id), vocab)
            }
            Err(e) => {
                // Same rendering inputs as the --no-store branch, deliberately: if these two
                // diverged, a store failure would produce an on-disk transcript that no later
                // `export <id>` could reproduce.
                log::warn!(
                    "could not persist meeting ({e:#}); exporting without storage \
                     and without vocabulary corrections"
                );
                (synth_row(&meta, merged.len()), None, Vec::new())
            }
        }
    };

    let vocab = if vocab_rows.is_empty() {
        render::Vocab::empty()
    } else {
        let (v, warnings) = render::Vocab::compile(&vocab_rows);
        for w in &warnings {
            log::warn!("{w}");
        }
        log::info!("applying {} vocabulary correction(s)", vocab_rows.len() - warnings.len());
        v
    };
    // Identity is empty until the speaker-ID work lands; render already handles it.
    let rendered = render::render_fresh(&merged, &render::IdentityMap::empty(), &vocab);
    let (md_path, json_path) =
        export::write_exports(&opts.export_dir, "transcript", &row, &rendered)?;

    Ok(PipelineOutput {
        segments: merged,
        stored_id,
        md_path,
        json_path,
        meeting_secs,
        wall_secs: wall,
        rtf,
    })
}

/// Read a capture WAV of any sample rate → (mono f32 samples, rate).
fn read_wav_any_rate(path: &Path) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.sample_format != hound::SampleFormat::Float
        || spec.bits_per_sample != 32
    {
        anyhow::bail!(
            "{}: expected mono 32-bit float WAV (got {} ch, {:?}/{}-bit)",
            path.display(),
            spec.channels,
            spec.sample_format,
            spec.bits_per_sample
        );
    }
    let samples = reader
        .into_samples::<f32>()
        .collect::<std::result::Result<Vec<f32>, _>>()
        .context("read samples")?;
    Ok((samples, spec.sample_rate))
}

/// Parse `segments.txt` lines into `segment index → inter-segment gap (seconds)`. Each roll line
/// is `seg <N> mic=… system=… rate=<r> gap_frames=<g> …`; the gap is the silence the capture
/// layer recorded at the boundary *before* segment N.
fn parse_segment_gaps(content: &str) -> HashMap<u32, f64> {
    let mut gaps = HashMap::new();
    for line in content.lines() {
        let (mut seg, mut rate, mut gap) = (None, None, None);
        let mut toks = line.split_whitespace();
        while let Some(t) = toks.next() {
            if t == "seg" {
                seg = toks.next().and_then(|s| s.parse::<u32>().ok());
            } else if let Some(r) = t.strip_prefix("rate=") {
                rate = r.parse::<u32>().ok();
            } else if let Some(g) = t.strip_prefix("gap_frames=") {
                gap = g.parse::<u64>().ok();
            }
        }
        if let (Some(n), Some(r), Some(g)) = (seg, rate, gap)
            && r > 0
        {
            gaps.insert(n, g as f64 / r as f64);
        }
    }
    gaps
}

fn read_segment_gaps(dir: &Path) -> Result<HashMap<u32, f64>> {
    let path = dir.join("segments.txt");
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let content =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    Ok(parse_segment_gaps(&content))
}

/// Ordered segment files for a channel base ("mic"/"system"): base.wav, base.001.wav, …
fn discover_channel(dir: &Path, base: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let seg0 = dir.join(format!("{base}.wav"));
    if seg0.exists() {
        files.push(seg0);
    }
    let mut n = 1u32;
    loop {
        let p = dir.join(format!("{base}.{n:03}.wav"));
        if p.exists() {
            files.push(p);
            n += 1;
        } else {
            break;
        }
    }
    files
}

/// When a capture file started recording: its birth time (`created`), set when the WAV is first
/// opened at capture start. mtime would be ~recording END, so we prefer birth time and fall back
/// to mtime only if unavailable (macOS/APFS reports birth time).
fn file_start_epoch(p: &Path) -> Option<i64> {
    let meta = std::fs::metadata(p).ok()?;
    let t = meta.created().or_else(|_| meta.modified()).ok()?;
    t.duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs() as i64)
}

/// The real meeting start time = the earliest capture-file birth time (falls back to now).
fn earliest_capture_start(mic: &[PathBuf], sys: &[PathBuf]) -> i64 {
    mic.iter()
        .chain(sys.iter())
        .filter_map(|p| file_start_epoch(p))
        .min()
        .unwrap_or_else(crate::now_epoch)
}

/// Clean model name for storage/display (`models/ggml-large-v3.bin` → `ggml-large-v3`).
fn model_name(model: &Path) -> String {
    model
        .file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| model.to_string_lossy().into_owned())
}

/// A `MeetingRow` for the `--no-store`/store-failed export path (no DB id assigned).
fn synth_row(meta: &db::MeetingMeta, segment_count: usize) -> db::MeetingRow {
    db::MeetingRow {
        id: 0,
        title: meta.title.clone(),
        source_dir: meta.source_dir.clone(),
        model: meta.model.clone(),
        lang: meta.lang.clone(),
        started_at: meta.started_at,
        duration_secs: meta.duration_secs,
        segment_count: segment_count as i64,
        created_at: meta.created_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_gaps_parse_from_manifest_lines() {
        // Real manifest shape: `seg N mic=… system=… rate=<r> gap_frames=<g> prev_rate=… reason=…`
        let content = "\
seg 1 mic=capture/mic.001.wav system=capture/system.001.wav rate=48000 gap_frames=4800 prev_rate=16000 reason=route_change
seg 2 mic=capture/mic.002.wav system=capture/system.002.wav rate=16000 gap_frames=1600 prev_rate=48000 reason=route_change";
        let gaps = parse_segment_gaps(content);
        assert_eq!(gaps.len(), 2);
        assert!((gaps[&1] - 0.1).abs() < 1e-9, "4800/48000 = 0.1 s"); // gap before seg 1
        assert!((gaps[&2] - 0.1).abs() < 1e-9, "1600/16000 = 0.1 s"); // gap before seg 2
    }

    #[test]
    fn segment_gaps_empty_and_malformed_are_safe() {
        assert!(parse_segment_gaps("").is_empty());
        assert!(parse_segment_gaps("garbage line without fields").is_empty());
        // rate=0 must not divide-by-zero into the map
        assert!(parse_segment_gaps("seg 1 rate=0 gap_frames=100").is_empty());
    }

    #[test]
    fn model_name_strips_dir_and_extension() {
        assert_eq!(model_name(Path::new("models/ggml-large-v3.bin")), "ggml-large-v3");
        assert_eq!(
            model_name(Path::new("/Users/x/.meetscribe/models/ggml-large-v3.bin")),
            "ggml-large-v3"
        );
    }
}
