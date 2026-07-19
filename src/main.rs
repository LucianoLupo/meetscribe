//! meetscribe — Phase 0 capture spike.
//!
//! Proves the SINGLE-aggregate design: one Core Audio aggregate device on one clock
//! capturing the mic ("You") and a global system tap ("Others") as two SEPARATE,
//! time-aligned channels. Writes `mic.wav` + `system.wav` and reports levels + the
//! observed buffer layout.
//!
//! Phase 0a: run during a REAL call → both WAVs non-zero + intelligible (your voice in
//!           mic.wav, the remote voices in system.wav).
//! Phase 0b: your voice then the remote voice must land in the right time order across
//!           the two files (single clock ⇒ they should be aligned frame-for-frame).
//!
//! Usage:
//!   meetscribe [--out <dir>] [--seconds <n>]
//!   (no --seconds ⇒ records until you press Enter)

mod capture;
mod resample;
mod vad;
mod asr;
mod transcript;
mod db;
mod export;

use anyhow::{Context, Result};
use capture::DualCapture;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

struct Args {
    out_dir: PathBuf,
    seconds: Option<u64>,
    /// Phase 1 Batch 1: force a rebuild this many seconds in, to exercise the
    /// request→poll→rebuild path without a real route change.
    rebuild_after: Option<u64>,
}

fn parse_args() -> Args {
    let mut out_dir = PathBuf::from("capture");
    let mut seconds = None;
    let mut rebuild_after = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" | "-o" => {
                if let Some(v) = it.next() {
                    out_dir = PathBuf::from(v);
                }
            }
            "--seconds" | "-s" => {
                seconds = it.next().and_then(|v| v.parse().ok());
            }
            "--rebuild-after" => {
                rebuild_after = it.next().and_then(|v| v.parse().ok());
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: meetscribe [--out <dir>] [--seconds <n>] [--rebuild-after <n>]"
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }
    Args {
        out_dir,
        seconds,
        rebuild_after,
    }
}

/// Read a capture WAV of any sample rate → (mono f32 samples, rate).
fn read_wav_any_rate(path: &Path) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_format != hound::SampleFormat::Float || spec.bits_per_sample != 32 {
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

/// Parse `segments.txt` lines into `segment index → inter-segment gap (seconds)`.
/// Each roll line is `seg <N> mic=… system=… rate=<r> gap_frames=<g> …`; the gap is
/// the silence the capture layer recorded at the boundary *before* segment N.
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

/// `meetscribe transcribe <dir> [--model <ggml.bin>] [--lang <code>]` — batch, post-capture.
/// Per channel: resample → VAD → whisper each speech window → tag You/Others → merge by time.
fn run_transcribe(argv: &[String]) -> Result<()> {
    let mut dir: Option<PathBuf> = None;
    let mut model = String::from("models/ggml-large-v3.bin");
    let mut lang = String::from("es");
    let mut title: Option<String> = None;
    let mut db_path: Option<PathBuf> = None;
    let mut export_dir: Option<PathBuf> = None;
    let mut no_store = false;
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" | "-m" => {
                if let Some(v) = it.next() {
                    model = v.clone();
                }
            }
            "--lang" | "-l" => {
                if let Some(v) = it.next() {
                    lang = v.clone();
                }
            }
            "--title" | "-t" => {
                if let Some(v) = it.next() {
                    title = Some(v.clone());
                }
            }
            "--db" => {
                if let Some(v) = it.next() {
                    db_path = Some(PathBuf::from(v));
                }
            }
            "--export-dir" => {
                if let Some(v) = it.next() {
                    export_dir = Some(PathBuf::from(v));
                }
            }
            "--no-store" => no_store = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: meetscribe transcribe <dir> [--title <t>] [--model <ggml.bin>] \
                     [--lang <code>] [--db <path>] [--export-dir <dir>] [--no-store]"
                );
                return Ok(());
            }
            s if !s.starts_with('-') && dir.is_none() => dir = Some(PathBuf::from(s)),
            _ => {}
        }
    }
    let dir = dir.context("transcribe: missing <dir> (e.g. `meetscribe transcribe capture`)")?;
    let title = title.unwrap_or_else(|| {
        dir.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("meeting")
            .to_string()
    });
    let db_path = db_path.unwrap_or_else(default_db_path);
    let export_dir = export_dir.unwrap_or_else(|| dir.clone());

    let mic_files = discover_channel(&dir, "mic");
    let sys_files = discover_channel(&dir, "system");
    if mic_files.is_empty() && sys_files.is_empty() {
        anyhow::bail!("no mic.wav/system.wav found in {}", dir.display());
    }
    log::info!(
        "transcribe {} — mic segments: {}, system segments: {}",
        dir.display(),
        mic_files.len(),
        sys_files.len()
    );

    let asr = asr::Asr::load(&model).with_context(|| format!("load model {model}"))?;
    // Inter-segment gaps the capture layer recorded on rate-roll boundaries (empty for the
    // common single-segment case). Both channels share the same manifest.
    let gaps = read_segment_gaps(&dir)?;

    let t0 = Instant::now();
    let mut segs: Vec<transcript::TranscriptSegment> = Vec::new();
    let mut meeting_secs = 0.0f64;

    for (files, speaker) in [
        (&mic_files, transcript::Speaker::You),
        (&sys_files, transcript::Speaker::Others),
    ] {
        // Multi-segment (rate-roll) absolute-time offset = cumulative prior-segment duration
        // + the recorded inter-segment gap before each rolled segment (segments.txt).
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
                let (text, confidence) = asr.transcribe(&audio16[a..b], &lang)?;
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

    println!("\n===== TRANSCRIPT ({}) =====", dir.display());
    for s in &merged {
        println!(
            "[{:7.2}-{:7.2}] {:<7} {}",
            s.t_start,
            s.t_end,
            s.speaker.label(),
            s.text
        );
    }

    let rtf = if meeting_secs > 0.0 { wall / meeting_secs } else { 0.0 };
    println!(
        "\nsegments: {} | meeting: {:.1}s | wall: {:.1}s | RTF: {:.3}x (both channels through whisper)",
        merged.len(),
        meeting_secs,
        wall,
        rtf
    );

    // Persist + export (Phase 3). started_at = the real meeting time (earliest capture mtime).
    let meta = db::MeetingMeta {
        title,
        source_dir: dir.display().to_string(),
        model: model_name(&model),
        lang: lang.clone(),
        started_at: earliest_capture_start(&mic_files, &sys_files),
        duration_secs: meeting_secs,
        created_at: now_epoch(),
    };

    // Persist to the DB, then export. A storage failure must NOT discard the transcript we
    // just spent real compute on: on error we log and fall back to a synthetic row so the
    // Markdown/JSON still get written. (Correctness of the DB round-trip is covered by the
    // db.rs unit test and the `export <id>` read path, not re-checked here every run.)
    let row = if no_store {
        synth_row(&meta, merged.len())
    } else {
        let rt = new_runtime()?;
        let stored = rt.block_on(async {
            let mut database = db::Db::open(&db_path).await?;
            let id = database.insert_meeting(&meta, &merged).await?;
            let row = database
                .get_meeting(id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("meeting {id} vanished after insert"))?;
            database.close().await?;
            anyhow::Ok((id, row))
        });
        match stored {
            Ok((id, row)) => {
                log::info!("stored meeting id {id} → {}", db_path.display());
                row
            }
            Err(e) => {
                log::warn!("could not persist meeting ({e:#}); exporting without storage");
                synth_row(&meta, merged.len())
            }
        }
    };

    let (md_path, json_path) = export::write_exports(&export_dir, "transcript", &row, &merged)?;
    println!("wrote {}", md_path.display());
    println!("wrote {}", json_path.display());
    Ok(())
}

/// `~/.meetscribe/meetscribe.db` (falls back to a repo-local path if `$HOME` is unset).
fn default_db_path() -> PathBuf {
    home_dir()
        .map(|h| h.join(".meetscribe/meetscribe.db"))
        .unwrap_or_else(|| PathBuf::from("meetscribe.db"))
}

/// `~/.meetscribe/exports/` (the default target for `export <id>`).
fn default_export_dir() -> PathBuf {
    home_dir()
        .map(|h| h.join(".meetscribe/exports"))
        .unwrap_or_else(|| PathBuf::from("exports"))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// When a capture file started recording: its birth time (`created`), set when the WAV is
/// first opened at capture start. mtime would be ~recording END (a WAV is written throughout
/// capture), so we prefer birth time and only fall back to mtime if unavailable (macOS/APFS
/// reports birth time).
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
        .unwrap_or_else(now_epoch)
}

/// Clean model name for storage/display (`models/ggml-large-v3.bin` → `ggml-large-v3`).
fn model_name(model: &str) -> String {
    Path::new(model)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| model.to_string())
}

/// A `MeetingRow` for the `--no-store` export path (no DB id assigned).
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

fn new_runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")
}

/// `meetscribe list [--db <path>]` — the stored meetings, newest first.
fn run_list(argv: &[String]) -> Result<()> {
    let mut db_path = default_db_path();
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--db" => {
                if let Some(v) = it.next() {
                    db_path = PathBuf::from(v);
                }
            }
            "-h" | "--help" => {
                eprintln!("usage: meetscribe list [--db <path>]");
                return Ok(());
            }
            _ => {}
        }
    }
    if !db_path.exists() {
        println!("no meetings yet — db {} does not exist", db_path.display());
        return Ok(());
    }

    let rt = new_runtime()?;
    let meetings = rt.block_on(async {
        let mut database = db::Db::open(&db_path).await?;
        let m = database.list_meetings().await?;
        database.close().await?;
        anyhow::Ok(m)
    })?;

    if meetings.is_empty() {
        println!("no meetings stored in {}", db_path.display());
        return Ok(());
    }
    println!(
        "{:>3}  {:<20}  {:>9}  {:>5}  TITLE",
        "ID", "DATE", "DURATION", "SEGS"
    );
    for m in &meetings {
        println!(
            "{:>3}  {:<20}  {:>9}  {:>5}  {}",
            m.id,
            export::fmt_utc(m.started_at),
            export::fmt_duration(m.duration_secs),
            m.segment_count,
            m.title
        );
    }
    Ok(())
}

/// `meetscribe export <id> [--db <path>] [--export-dir <dir>]` — re-export from the DB.
/// This exercises the real DB read path (round-trip proof).
fn run_export(argv: &[String]) -> Result<()> {
    let mut id: Option<i64> = None;
    let mut db_path = default_db_path();
    let mut export_dir: Option<PathBuf> = None;
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--db" => {
                if let Some(v) = it.next() {
                    db_path = PathBuf::from(v);
                }
            }
            "--export-dir" => {
                if let Some(v) = it.next() {
                    export_dir = Some(PathBuf::from(v));
                }
            }
            "-h" | "--help" => {
                eprintln!("usage: meetscribe export <id> [--db <path>] [--export-dir <dir>]");
                return Ok(());
            }
            s if !s.starts_with('-') && id.is_none() => {
                id = Some(
                    s.parse::<i64>()
                        .with_context(|| format!("export: invalid id '{s}'"))?,
                );
            }
            _ => {}
        }
    }
    let id = id.context("export: missing <id> (e.g. `meetscribe export 1`)")?;
    let export_dir = export_dir.unwrap_or_else(default_export_dir);
    if !db_path.exists() {
        anyhow::bail!("db {} does not exist — nothing to export", db_path.display());
    }

    let rt = new_runtime()?;
    let (row, segs) = rt.block_on(async {
        let mut database = db::Db::open(&db_path).await?;
        let row = database
            .get_meeting(id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("no meeting with id {id} in {}", db_path.display()))?;
        let segs = database.load_segments(id).await?;
        database.close().await?;
        anyhow::Ok((row, segs))
    })?;

    let (md_path, json_path) =
        export::write_exports(&export_dir, &format!("meeting-{id}"), &row, &segs)?;
    println!("wrote {}", md_path.display());
    println!("wrote {}", json_path.display());
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Subcommand dispatch. Anything unrecognized falls through to the capture flow
    // (back-compatible: `meetscribe --seconds 300` still records).
    let argv: Vec<String> = std::env::args().collect();
    match argv.get(1).map(String::as_str) {
        Some("transcribe") => return run_transcribe(&argv[2..]),
        Some("list") => return run_list(&argv[2..]),
        Some("export") => return run_export(&argv[2..]),
        _ => {}
    }

    let args = parse_args();
    std::fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("create output dir {}", args.out_dir.display()))?;

    log::info!("meetscribe capture spike — starting single-aggregate mic+tap capture");
    log::info!("(first run triggers TWO prompts: Microphone and \"record system audio\" — approve both)");

    // Startup self-check: on failure, print the classified, actionable remedy (TCC grant
    // missing vs mic/route dropped) rather than a generic error.
    let mut cap = match DualCapture::start() {
        Ok(c) => c,
        Err(e) => {
            log::error!("cannot start capture — {e}");
            return Err(anyhow::anyhow!("capture startup failed"));
        }
    };

    log::info!(
        "mic (\"You\"):   uid={} native={} Hz, {} ch",
        cap.mic_uid,
        cap.mic_native_rate,
        cap.mic_native_channels
    );
    log::info!(
        "tap (\"Others\"): native={} Hz, {} ch  (Teams typically 24000)",
        cap.tap_native_rate,
        cap.tap_native_channels
    );

    // The aggregate runs at the mic clock's rate; the tap's native rate is drift-resampled
    // into it, so one rate governs both WAVs. It's fixed for the device's lifetime, so we
    // read the seed once here rather than polling inside the RT proc.
    // `rate` and the output files can change if a rebuild can't hold the original rate —
    // then we roll to a new segment. In the common case (rate held) both stay put.
    let mut rate = cap.aggregate_rate().max(1);
    log::info!("aggregate rate = {rate} Hz (both channels)");

    let mut segment: u32 = 0;
    let mut mic_path = segment_path(&args.out_dir, "mic", segment);
    let mut sys_path = segment_path(&args.out_dir, "system", segment);
    let mut mic_w = WavStream::create(&mic_path, rate).context("open mic.wav")?;
    let mut sys_w = WavStream::create(&sys_path, rate).context("open system.wav")?;

    // Stop control: a --seconds deadline, else press Enter. Only wire the stdin thread
    // when there's no deadline — a non-interactive stdin hits EOF instantly and would
    // otherwise stop the capture on the first loop iteration.
    let stop = Arc::new(AtomicBool::new(false));
    if args.seconds.is_none() {
        let stop = stop.clone();
        std::thread::spawn(move || {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
            stop.store(true, Ordering::Release);
        });
        log::info!("recording… press Enter to stop.");
    } else {
        log::info!("recording for {}s…", args.seconds.unwrap());
    }

    let deadline = args.seconds.map(|s| Instant::now() + Duration::from_secs(s));
    let started_at = Instant::now();
    // One-shot forced-rebuild trigger (Batch 1 exercise). None once it has fired.
    let mut rebuild_deadline = args
        .rebuild_after
        .map(|s| started_at + Duration::from_secs(s));
    let mut rebuilds = 0u32;
    // Consecutive failed rebuild attempts (transient route-transition failures). A rebuild
    // failure is NEVER fatal — we re-arm and retry rather than tear down a live recording.
    let mut rebuild_failures = 0u32;
    // A single physical route change emits a BURST of HAL notifications; wait for the route
    // to settle before rebuilding once, so we don't spawn empty micro-segments.
    const REBUILD_SETTLE: Duration = Duration::from_millis(500);
    let mut rebuild_pending_since: Option<Instant> = None;
    // Mic-dry watchdog: the mic is the always-on clock master, so once the device has fired
    // its first callback, a mic that stops delivering FRAMES (not just goes quiet) for this
    // long means the input route dropped — a case a default-device listener can miss.
    const MIC_DRY_TIMEOUT: Duration = Duration::from_secs(3);
    let mut mic_last_seen = Instant::now();
    // Small per-drain scratch buffers (reused) — only a ring's-worth is ever resident.
    let mut mic_raw: Vec<f32> = Vec::new();
    let mut tap_raw: Vec<f32> = Vec::new();

    loop {
        mic_raw.clear();
        tap_raw.clear();
        cap.drain_into(&mut mic_raw, &mut tap_raw);
        mic_w.write(&mic_raw, cap.observed_mic_channels());
        sys_w.write(&tap_raw, cap.observed_tap_channels());

        // Mic-dry watchdog. The mic delivers frames continuously once the device has started,
        // even during system silence, so a dry mic = a dropped input route. Gated on the
        // device having started, and paused while a rebuild is already pending/settling.
        if !mic_raw.is_empty() {
            mic_last_seen = Instant::now();
        }
        if cap.first_num_buffers() > 0
            && rebuild_pending_since.is_none()
            && mic_last_seen.elapsed() > MIC_DRY_TIMEOUT
        {
            log::warn!(
                "mic delivered no frames for >{}s while the device is running → requesting \
                 rebuild (input route dropped?)",
                MIC_DRY_TIMEOUT.as_secs()
            );
            cap.request_rebuild();
            mic_last_seen = Instant::now();
        }

        // Fire the one-shot forced rebuild once its deadline passes.
        if let Some(d) = rebuild_deadline
            && Instant::now() >= d
        {
            log::info!("--rebuild-after fired → requesting rebuild");
            cap.request_rebuild();
            rebuild_deadline = None;
        }

        // A requested rebuild (forced trigger, the route-change listener, or the mic-dry
        // watchdog) starts a settle timer; we only rebuild once the burst has quieted.
        if cap.rebuild_requested() {
            rebuild_pending_since.get_or_insert_with(Instant::now);
        }
        let do_rebuild = rebuild_pending_since
            .map(|since| since.elapsed() >= REBUILD_SETTLE)
            .unwrap_or(false);
        if do_rebuild {
            rebuild_pending_since = None;
            // Synchronized final flush of BOTH channels so they end at equal frame counts
            // before the old device is torn down.
            mic_raw.clear();
            tap_raw.clear();
            cap.drain_into(&mut mic_raw, &mut tap_raw);
            mic_w.write(&mic_raw, cap.observed_mic_channels());
            sys_w.write(&tap_raw, cap.observed_tap_channels());

            let mic_before = mic_w.written();
            let sys_before = sys_w.written();

            // A rebuild failure is NEVER fatal. `rebuild()` fails on exactly the transient
            // operations (default input / process tap / aggregate assembly) that are most
            // likely to hiccup DURING a route change — the moment we rebuild. Killing the
            // session here would make the recovery machinery destroy what it exists to save.
            // So: log the classified remedy, re-arm the settle timer, and retry on a later
            // tick. `cap.device` is left None by a failed rebuild and recovers on a retry.
            match cap.rebuild() {
                Err(e) => {
                    rebuild_failures += 1;
                    if rebuild_failures == 1 || rebuild_failures.is_multiple_of(20) {
                        log::warn!(
                            "rebuild failed — {e} (capture paused; retry #{rebuild_failures})"
                        );
                    }
                    rebuild_pending_since = Some(Instant::now());
                    mic_last_seen = Instant::now();
                }
                Ok(out) => {
                    rebuilds += 1;
                    rebuild_failures = 0;
                    mic_last_seen = Instant::now(); // fresh watchdog window for the new device

                    if out.rate_changed {
                        // The new device couldn't hold the original rate → the fixed-header WAV
                        // can't continue. Finalize the current segment, roll to a fresh pair at
                        // the new rate, and record the boundary + gap in the manifest.
                        let (mn, mr, mp) = std::mem::replace(
                            &mut mic_w,
                            WavStream::create(
                                &segment_path(&args.out_dir, "mic", segment + 1),
                                out.actual_rate,
                            )
                            .context("open rolled mic segment")?,
                        )
                        .finalize()
                        .context("finalize rolled mic segment")?;
                        let (sn, sr, sp) = std::mem::replace(
                            &mut sys_w,
                            WavStream::create(
                                &segment_path(&args.out_dir, "system", segment + 1),
                                out.actual_rate,
                            )
                            .context("open rolled system segment")?,
                        )
                        .finalize()
                        .context("finalize rolled system segment")?;
                        report("mic seg (rolled)   ", &mic_path, mn, mr, mp, rate);
                        report("system seg (rolled)", &sys_path, sn, sr, sp, rate);

                        segment += 1;
                        mic_path = segment_path(&args.out_dir, "mic", segment);
                        sys_path = segment_path(&args.out_dir, "system", segment);
                        let old_rate = rate;
                        rate = out.actual_rate;
                        log::warn!(
                            "GAP MARKER rebuild #{rebuilds}: rate {old_rate}→{rate} Hz, \
                             ~{} frame gap → ROLLED to segment {segment}",
                            out.gap_frames
                        );
                        append_manifest(
                            &args.out_dir,
                            &format!(
                                "seg {segment} mic={} system={} rate={rate} gap_frames={} \
                                 prev_rate={old_rate} reason=route_change",
                                mic_path.display(),
                                sys_path.display(),
                                out.gap_frames,
                            ),
                        );
                    } else {
                        // Rate held → the same files continue. A rate-held rebuild can still
                        // have SWAPPED the input device, so re-latch each WavStream's channel
                        // count before padding — otherwise a device with a different channel
                        // count would be downmixed with the stale divisor and misalign the
                        // channels. Then pad BOTH channels with equal silence for the gap.
                        mic_w.reset_channels();
                        sys_w.reset_channels();
                        mic_w.write_silence(out.gap_frames);
                        sys_w.write_silence(out.gap_frames);
                        log::info!(
                            "GAP MARKER rebuild #{rebuilds}: padded {} silence frames into both \
                             channels at ~{rate} Hz (mic {mic_before}→{}, system {sys_before}→{})",
                            out.gap_frames,
                            mic_w.written(),
                            sys_w.written(),
                        );
                    }
                }
            }
        }

        if stop.load(Ordering::Acquire) {
            break;
        }
        if let Some(d) = deadline {
            if Instant::now() >= d {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    // Final flush of whatever is still buffered in the rings.
    mic_raw.clear();
    tap_raw.clear();
    cap.drain_into(&mut mic_raw, &mut tap_raw);
    mic_w.write(&mic_raw, cap.observed_mic_channels());
    sys_w.write(&tap_raw, cap.observed_tap_channels());

    let elapsed = started_at.elapsed().as_secs_f32();
    let mic_ch = cap.observed_mic_channels();
    let tap_ch = cap.observed_tap_channels();
    let num_buffers = cap.first_num_buffers();
    let dropped = cap.dropped();

    log::info!("stopped after {elapsed:.1}s ({rebuilds} rebuild(s))");
    log::info!(
        "OBSERVED IO-proc layout: number_buffers={num_buffers} (expect 2), \
         mic_buffer_channels={mic_ch}, tap_buffer_channels={tap_ch}"
    );
    if num_buffers != 2 {
        log::warn!(
            "number_buffers={num_buffers} != 2 — mic and tap did NOT arrive as two separate \
             buffers (or the last device hadn't fired yet post-rebuild). Report this number."
        );
    }
    if dropped > 0 {
        log::warn!("{dropped} samples dropped (ring full — drain fell behind); increase RING_CAPACITY or drain interval");
    }

    let (mic_n, mic_rms, mic_peak) = mic_w.finalize().context("finalize mic.wav")?;
    let (sys_n, sys_rms, sys_peak) = sys_w.finalize().context("finalize system.wav")?;
    report("mic.wav    (You)   ", &mic_path, mic_n, mic_rms, mic_peak, rate);
    report("system.wav (Others)", &sys_path, sys_n, sys_rms, sys_peak, rate);

    log::info!(
        "Phase 0a check: both files should be non-zero AND intelligible — YOUR voice in \
         mic.wav, the REMOTE voice(s) in system.wav. If they're swapped, buffer order is \
         tap-then-mic (a one-line fix)."
    );
    Ok(())
}

/// Incremental mono WAV writer: downmixes interleaved frames to mono and streams them to
/// disk as they're drained, so only a ring's-worth of audio is ever in memory. Carries a
/// per-source frame remainder so a drain that splits mid-frame never misaligns channels.
struct WavStream {
    writer: hound::WavWriter<BufWriter<File>>,
    channels: usize,
    pending: Vec<f32>,
    written: usize,
    sum_sq: f64,
    peak: f32,
}

impl WavStream {
    fn create(path: &Path, sample_rate: u32) -> Result<Self> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        Ok(Self {
            writer: hound::WavWriter::create(path, spec)?,
            channels: 0,
            pending: Vec::new(),
            written: 0,
            sum_sq: 0.0,
            peak: 0.0,
        })
    }

    /// Append a freshly-drained interleaved chunk; write every complete mono frame, keep the
    /// remainder. `observed_channels` latches on the first non-zero value (fixed per device).
    fn write(&mut self, interleaved: &[f32], observed_channels: u32) {
        if interleaved.is_empty() {
            return;
        }
        if self.channels == 0 && observed_channels > 0 {
            self.channels = observed_channels as usize;
        }
        let ch = self.channels.max(1);
        self.pending.extend_from_slice(interleaved);
        let full = self.pending.len() - self.pending.len() % ch;
        for frame in self.pending[..full].chunks_exact(ch) {
            let mono = frame.iter().sum::<f32>() / ch as f32;
            let _ = self.writer.write_sample(mono);
            self.written += 1;
            self.sum_sq += mono as f64 * mono as f64;
            self.peak = self.peak.max(mono.abs());
        }
        self.pending.drain(..full);
    }

    /// Forget the latched channel count so the next `write` re-latches from the (possibly
    /// new) device. Called on a rate-held rebuild in case the input device swapped to a
    /// different channel count. Drops any sub-frame remainder (< old channels) to avoid
    /// mixing the old and new channel layouts across the boundary.
    fn reset_channels(&mut self) {
        self.channels = 0;
        self.pending.clear();
    }

    /// Write `frames` mono zero-samples — used to pad a rebuild gap equally into both
    /// channels so they stay aligned to each other.
    fn write_silence(&mut self, frames: usize) {
        for _ in 0..frames {
            let _ = self.writer.write_sample(0.0f32);
            self.written += 1;
        }
    }

    /// Mono samples written to this segment so far.
    fn written(&self) -> usize {
        self.written
    }

    /// Returns (mono samples written, RMS, peak). Any trailing partial frame (< channels)
    /// is discarded — at most `channels - 1` samples.
    fn finalize(self) -> Result<(usize, f32, f32)> {
        let rms = if self.written > 0 {
            (self.sum_sq / self.written as f64).sqrt() as f32
        } else {
            0.0
        };
        self.writer.finalize()?;
        Ok((self.written, rms, self.peak))
    }
}

/// Segment 0 keeps the plain `mic.wav`/`system.wav` names; later segments (only produced
/// when a rebuild can't hold the rate) get a zero-padded suffix, e.g. `mic.001.wav`.
fn segment_path(dir: &Path, base: &str, segment: u32) -> PathBuf {
    if segment == 0 {
        dir.join(format!("{base}.wav"))
    } else {
        dir.join(format!("{base}.{segment:03}.wav"))
    }
}

/// Append one line to the session's segment manifest, recording each rebuild-induced
/// segment boundary + the gap between segments. Best-effort (a manifest write failure must
/// not abort capture).
fn append_manifest(dir: &Path, line: &str) {
    let path = dir.join("segments.txt");
    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{line}") {
                log::warn!("could not append to manifest {}: {e}", path.display());
            }
        }
        Err(e) => log::warn!("could not open manifest {}: {e}", path.display()),
    }
}

fn report(label: &str, path: &Path, n: usize, rms: f32, peak: f32, rate: u32) {
    let secs = if rate > 0 { n as f32 / rate as f32 } else { 0.0 };
    let verdict = if peak < 1e-5 {
        "SILENT — check permissions / source"
    } else {
        "signal present"
    };
    log::info!(
        "{label}: {n} samples, {secs:.1}s, RMS={rms:.5}, peak={peak:.5} → {verdict}  [{}]",
        path.display()
    );
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
}
