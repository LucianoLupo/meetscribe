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

use crate::{asr, db, diar, export, render, resample, split, spk, transcript, vad, voices};

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
    /// Speaker-embedding ONNX model. `None` or a missing file ⇒ transcribe without speaker
    /// identity (logged once), nothing else changes. MUST be absolute for the daemon.
    pub speaker_model: Option<PathBuf>,
    /// Split far-end chunks at diarizer voice changes before naming (rollback: false).
    pub split: bool,
    /// Diarizer runtime + model. `None`/missing ⇒ no split (logged once), nothing else changes.
    /// MUST be absolute for the daemon.
    pub diarizer_bin: PathBuf,
    pub diarizer_model: PathBuf,
    /// Evaluation only: replay this RTTM's turns instead of running the diarizer (single roll).
    pub diarizer_rttm: Option<PathBuf>,
}

/// Result of the pipeline. `segments`/`meeting_secs`/`rtf`/`wall_secs` let the CLI wrapper print
/// the same transcript+RTF summary it always did; `stored_id` is `Some` only when persisted.
///
/// `segments` are RENDERED, not raw: the CLI prints exactly what was written to disk. Handing
/// back raw segments here would make the terminal disagree with the file for anyone using
/// vocabulary corrections — two presentations of one transcript, which is the thing `render`
/// exists to prevent.
pub(crate) struct PipelineOutput {
    pub segments: Vec<render::RenderedSegment>,
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
    // The diarizer loads FIRST: DTW word timestamps are a load-time Whisper setting, enabled only
    // when there is something to split with — otherwise the context is exactly the pre-split one.
    let mut diarizer = load_diarizer(opts);
    // Rolls the diarizer covered / far-end rolls seen, and why any roll fell back (per-meeting
    // provenance log line).
    let (mut rolls_split, mut rolls_far, mut fallbacks) = (0usize, 0usize, Vec::<String>::new());
    // DTW uses large-v3's alignment heads; another Whisper model (turbo, small, …) can fail to
    // build its state with them. A split problem must never cost the transcript, so on any DTW
    // load failure: log, turn splitting off for this run, load exactly the pre-split context.
    let mut asr = match diarizer.as_ref().map(|_| asr::Asr::load(model_str, true)) {
        Some(Ok(a)) => a,
        Some(Err(e)) => {
            log::warn!(
                "far-end split: off — Whisper model {model_str} does not load with large-v3 DTW \
                 word timings ({e:#}); transcribing without splitting"
            );
            fallbacks.push(format!("DTW load failed: {e:#}"));
            diarizer = None;
            asr::Asr::load(model_str, false).with_context(|| format!("load model {model_str}"))?
        }
        None => asr::Asr::load(model_str, false).with_context(|| format!("load model {model_str}"))?,
    };
    let split_on = diarizer.is_some();
    let mut embedder = load_embedder(opts.speaker_model.as_deref());
    // Inter-segment gaps the capture layer recorded on rate-roll boundaries (empty for the common
    // single-segment case). Both channels share the same manifest.
    let gaps = read_segment_gaps(dir)?;

    let t0 = Instant::now();
    // Each segment travels with its far-end speaker embedding (`None` for the mic channel, for
    // windows under the embedding floor, and when there is no embedder) and, when splitting, its
    // pieces with their own embeddings (path B).
    type Pieces = Vec<(split::Piece, Option<Vec<f32>>)>;
    type Keyed = (Option<Vec<f32>>, Pieces);
    let mut segs: Vec<(transcript::TranscriptSegment, Keyed)> = Vec::new();
    let mut meeting_secs = 0.0f64;
    let embed_floor = spk::MIN_WINDOW_MS * resample::TARGET_RATE as usize / 1000;

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
            let far = speaker == transcript::Speaker::Others;
            // One diarizer run per far-end roll, started in the BACKGROUND so Whisper transcribes
            // the same roll meanwhile; its turns are collected after the windows. A failed roll
            // keeps its chunks whole (fail-open).
            let pending = match diarizer.as_mut() {
                Some(d) if far => {
                    rolls_far += 1;
                    match d.start(&audio16, dir) {
                        Ok(p) => Some(p),
                        Err(e) => {
                            log::warn!("far-end split: roll {i} falls back to whole chunks ({e:#})");
                            fallbacks.push(format!("roll {i}: {e:#}"));
                            None
                        }
                    }
                }
                _ => None,
            };
            // Far-end chunks of this roll waiting for the turns: (index in `segs`, words).
            let mut awaiting: Vec<(usize, Vec<asr::Word>)> = Vec::new();
            for w in windows {
                let (a, b) = w.sample_range(audio16.len());
                if b <= a {
                    continue;
                }
                // Far-end windows are decoded ONCE, with word timings when splitting.
                let (text, confidence, words) = if far && split_on {
                    asr.transcribe_words(&audio16[a..b], &opts.lang)?
                } else {
                    let (t, c) = asr.transcribe(&audio16[a..b], &opts.lang)?;
                    (t, c, Vec::new())
                };
                if text.is_empty() {
                    continue;
                }
                let embedding = match embedder.as_mut() {
                    Some(e) if speaker == transcript::Speaker::Others && b - a >= embed_floor => {
                        Some(e.embed(&audio16[a..b]).context("speaker embedding")?)
                    }
                    _ => None,
                };
                let (t_start, t_end) = (offset + w.start_ms as f64 / 1000.0, offset + w.end_ms as f64 / 1000.0);
                if far && split_on {
                    awaiting.push((segs.len(), words));
                }
                segs.push((
                    transcript::TranscriptSegment { speaker, text, t_start, t_end, confidence },
                    (embedding, Vec::new()),
                ));
            }
            if far && split_on {
                let turns = pending.and_then(|p| match p.wait() {
                    Ok(t) => {
                        rolls_split += 1;
                        Some(t)
                    }
                    Err(e) => {
                        log::warn!("far-end split: roll {i} falls back to whole chunks ({e:#})");
                        fallbacks.push(format!("roll {i}: {e:#}"));
                        None
                    }
                });
                for (k, words) in awaiting {
                    let (t_start, t_end) = (segs[k].0.t_start, segs[k].0.t_end);
                    // `chunk` is fixed up to the merged index after `merge_keyed`.
                    let mut ps = split::pieces_for_chunk(0, t_start, t_end, offset, turns.as_deref().unwrap_or(&[]));
                    split::assign_words(&mut ps, t_start, &words);
                    for p in ps {
                        let pa = (((p.t_start - offset) * resample::TARGET_RATE as f64).round() as usize).min(audio16.len());
                        let pb = (((p.t_end - offset) * resample::TARGET_RATE as f64).round() as usize).min(audio16.len());
                        let emb = match embedder.as_mut() {
                            Some(e) if pb > pa && pb - pa >= embed_floor => {
                                Some(e.embed(&audio16[pa..pb]).context("piece embedding")?)
                            }
                            _ => None,
                        };
                        segs[k].1.1.push((p, emb));
                    }
                }
            }
            offset += dur;
        }
        meeting_secs = meeting_secs.max(offset);
    }

    let (merged, keyed): (Vec<transcript::TranscriptSegment>, Vec<Keyed>) =
        transcript::merge_keyed(segs).into_iter().unzip();
    let mut embeddings: Vec<Option<Vec<f32>>> = Vec::with_capacity(keyed.len());
    let (mut pieces, mut piece_embs): (Vec<split::Piece>, Vec<Option<Vec<f32>>>) = (Vec::new(), Vec::new());
    for (ci, (emb, ps)) in keyed.into_iter().enumerate() {
        embeddings.push(emb);
        for (mut p, e) in ps {
            p.chunk = ci;
            pieces.push(p);
            piece_embs.push(e);
        }
    }
    let wall = t0.elapsed().as_secs_f64();

    // Far-end voice clusters for this meeting (empty without an embedder). Keys are indices into
    // `merged`, which is exactly what `insert_meeting_with_voices` expects.
    let windows: Vec<voices::Window> = merged
        .iter()
        .zip(&embeddings)
        .enumerate()
        .filter(|(_, (s, _))| s.speaker == transcript::Speaker::Others)
        .map(|(i, (s, e))| voices::Window { key: i as i64, t_start: s.t_start, t_end: s.t_end, embedding: e.clone() })
        .collect();
    let mut assembly = voices::assemble(&windows, spk::CLUSTER_CUT);
    // Path B: the same identity pipeline over the pieces (keys = piece index).
    let mut assembly_b = split_on.then(|| {
        let w: Vec<voices::Window> = pieces
            .iter()
            .zip(&piece_embs)
            .enumerate()
            .map(|(i, (p, e))| voices::Window { key: i as i64, t_start: p.t_start, t_end: p.t_end, embedding: e.clone() })
            .collect();
        voices::assemble(&w, spk::CLUSTER_CUT)
    });
    let split_reason = if !opts.split {
        "off".to_string()
    } else if !split_on {
        if fallbacks.is_empty() { "off (diarizer not loaded)".to_string() } else { format!("off ({})", fallbacks.join("; ")) }
    } else if fallbacks.is_empty() {
        "on".to_string()
    } else {
        format!("on, {} roll(s) whole: {}", fallbacks.len(), fallbacks.join("; "))
    };
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
    // Stored: (row, id, vocab rows, the segments re-read from the DB, speakers) — the exports are
    // then rendered from the SAME rows `export <id>` reads, so the daemon's transcript.md carries
    // the same names and corrections. Unstored: no identity, no vocab, rendered from `merged`.
    let stored = if opts.no_store {
        // No DB handle on this path: render raw, and say so rather than silently differing from
        // what `export <id>` would produce.
        log::info!(
            "--no-store: exporting without vocabulary corrections and without speaker names \
             (no database opened)"
        );
        None
    } else {
        let rt = crate::new_runtime()?;
        let res = rt.block_on(async {
            let mut database = db::Db::open(&opts.db_path).await?;
            let enrolled = voices::decode_enrolled(&database.load_enrolled().await?)?;
            let mut matched = voices::apply_matches(&mut assembly, &enrolled);
            // Split-then-name: match path B, apply rule C against path A, then turn the
            // named pieces back into stored segments. Without a diarizer: today's insert.
            let finalized = match assembly_b.as_mut() {
                Some(b) => {
                    matched = voices::apply_matches(b, &enrolled);
                    let parent: Vec<i64> = pieces.iter().map(|p| p.chunk as i64).collect();
                    let embedded: Vec<bool> = piece_embs.iter().map(Option::is_some).collect();
                    voices::rule_c(&assembly, b, &parent, &embedded);
                    Some(split::finalize(&merged, &embeddings, &pieces, b)?)
                }
                None => None,
            };
            let (final_segs, final_asm): (&[transcript::TranscriptSegment], &voices::Assembly) = match &finalized {
                Some((segs, asm)) => (segs, asm),
                None => (&merged, &assembly),
            };
            let id = database
                .insert_meeting_with_voices(&meta, final_segs, &final_asm.clusters, &final_asm.voices)
                .await?;
            let n_clusters = final_asm.clusters.len();
            let row = database
                .get_meeting(id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("meeting {id} vanished after insert"))?;
            let vocab = database.load_enabled_vocab().await?;
            let segments = database.load_segments(id).await?;
            let speakers = database.list_speakers().await?;
            database.close().await?;
            anyhow::Ok((id, row, vocab, segments, speakers, matched, n_clusters))
        });
        match res {
            Ok((id, row, vocab, segments, speakers, matched, n_clusters)) => {
                log::info!("stored meeting id {id} → {}", opts.db_path.display());
                if n_clusters > 0 {
                    log::info!(
                        "far-end voices: {n_clusters} cluster(s), {matched} recognised — `meetscribe speakers list {id}`"
                    );
                }
                log::info!(
                    "far-end split: {split_reason} — {} meeting {id}, rolls split {rolls_split}/{rolls_far}{}",
                    dir.display(),
                    diarizer.as_ref().map(|d| format!(", {}", d.provenance())).unwrap_or_default()
                );
                Some((row, id, vocab, segments, speakers))
            }
            Err(e) => {
                // Same rendering inputs as the --no-store branch, deliberately: if these two
                // diverged, a store failure would produce an on-disk transcript that no later
                // `export <id>` could reproduce.
                log::warn!(
                    "could not persist meeting ({e:#}); exporting without storage, without \
                     vocabulary corrections and without speaker names"
                );
                None
            }
        }
    };

    let (row, stored_id, rendered) = match stored {
        Some((row, id, vocab_rows, segments, speakers)) => {
            let vocab = compile_vocab(&vocab_rows);
            let ids = render::IdentityMap::from_db(&speakers);
            (row, Some(id), render::render(&segments, &ids, &vocab))
        }
        None => (
            synth_row(&meta, merged.len()),
            None,
            render::render_fresh(&merged, &render::IdentityMap::empty(), &render::Vocab::empty()),
        ),
    };
    let (md_path, json_path) =
        export::write_exports(&opts.export_dir, "transcript", &row, &rendered)?;

    Ok(PipelineOutput {
        segments: rendered,
        stored_id,
        md_path,
        json_path,
        meeting_secs,
        wall_secs: wall,
        rtf,
    })
}

fn compile_vocab(rows: &[db::VocabRow]) -> render::Vocab {
    if rows.is_empty() {
        return render::Vocab::empty();
    }
    let (v, warnings) = render::Vocab::compile(rows);
    for w in &warnings {
        log::warn!("{w}");
    }
    log::info!("applying {} vocabulary correction(s)", rows.len() - warnings.len());
    v
}

/// The far-end diarizer, or `None` with ONE log line explaining why splitting is off.
fn load_diarizer(opts: &PipelineOpts) -> Option<diar::Diarizer> {
    if !opts.split {
        log::info!("far-end split: off (split = false)");
        return None;
    }
    if let Some(rttm) = opts.diarizer_rttm.as_deref() {
        return match diar::Diarizer::from_rttm(rttm) {
            Ok(d) => {
                log::info!("far-end split: on ({})", d.provenance());
                Some(d)
            }
            Err(e) => {
                log::warn!("far-end split: off (cannot replay {}: {e:#})", rttm.display());
                None
            }
        };
    }
    match diar::Diarizer::load(&opts.diarizer_bin, &opts.diarizer_model) {
        Ok(d) => {
            log::info!("far-end split: on ({})", d.provenance());
            Some(d)
        }
        Err(e) => {
            log::warn!(
                "far-end split: off ({e:#}) — run `bash models/provision.sh` + \
                 `bash models/build-diarizer.sh`, then `meetscribe install`"
            );
            None
        }
    }
}

/// The speaker-embedding model, or `None` with ONE log line explaining why identity is off.
fn load_embedder(path: Option<&Path>) -> Option<spk::Embedder> {
    let path = path?;
    if !path.exists() {
        log::warn!(
            "speaker model not found at {} — transcribing without speaker identity \
             (run `bash models/provision.sh`, then `meetscribe install`)",
            path.display()
        );
        return None;
    }
    match spk::Embedder::load(path) {
        Ok(e) => Some(e),
        Err(e) => {
            log::warn!("speaker model failed to load ({e:#}) — transcribing without speaker identity");
            None
        }
    }
}

/// Read a capture WAV of any sample rate → (mono f32 samples, rate).
pub(crate) fn read_wav_any_rate(path: &Path) -> Result<(Vec<f32>, u32)> {
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

pub(crate) fn read_segment_gaps(dir: &Path) -> Result<HashMap<u32, f64>> {
    let path = dir.join("segments.txt");
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let content =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    Ok(parse_segment_gaps(&content))
}

/// Ordered segment files for a channel base ("mic"/"system"): base.wav, base.001.wav, …
pub(crate) fn discover_channel(dir: &Path, base: &str) -> Vec<PathBuf> {
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

    fn opts(split: bool, bin: &Path, model: &Path) -> PipelineOpts {
        PipelineOpts {
            model: PathBuf::from("unused.bin"),
            lang: "es".into(),
            title: None,
            db_path: PathBuf::from("unused.db"),
            export_dir: PathBuf::from("."),
            no_store: true,
            speaker_model: None,
            split,
            diarizer_bin: bin.to_path_buf(),
            diarizer_model: model.to_path_buf(),
            diarizer_rttm: None,
        }
    }

    /// No diarizer ⇒ `Asr::load(dtw = false)` ⇒ the Whisper context is exactly the pre-split one,
    /// and nothing can ever call the diarizer. Covers split off, unconfigured, and missing files.
    #[test]
    fn diarizer_stays_unloaded_when_split_is_off_or_it_is_missing() {
        let tmp = std::env::temp_dir().join(format!("meetscribe-loaddiar-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let bin = tmp.join("nemo-speech-diar");
        let model = tmp.join("m.gguf");
        std::fs::write(&bin, b"#!/bin/sh\n").unwrap();
        std::fs::write(&model, b"gguf").unwrap();
        assert!(load_diarizer(&opts(false, &bin, &model)).is_none(), "split = false");
        assert!(load_diarizer(&opts(true, &tmp.join("nope"), &model)).is_none(), "no bin");
        assert!(load_diarizer(&opts(true, &bin, &tmp.join("nope"))).is_none(), "no model");
        assert!(load_diarizer(&opts(true, &bin, &model)).is_some(), "both present");
        let _ = std::fs::remove_dir_all(&tmp);
    }

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
