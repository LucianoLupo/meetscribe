//! meetscribe — local-first, background macOS meeting transcriber.
//!
//! CLI entry + subcommand dispatch over the proven layers:
//!   - default (no subcommand) = a capture session (mic "You" + system-tap "Others", one clock);
//!   - `transcribe <dir>`      = batch resample → VAD → whisper → merge → store + export;
//!   - `list` / `export <id>`  = read back stored meetings;
//!   - `detect [--watch]`      = which allowlisted app (if any) holds the mic (Phase-4 detector).
//!
//! Capture usage:
//!   meetscribe [--out <dir>] [--seconds <n>] [--rebuild-after <n>]
//!   (no --seconds ⇒ records until you press Enter)

mod capture;
mod resample;
mod vad;
mod asr;
mod transcript;
mod db;
mod export;
mod detect;
mod session;
mod pipeline;
mod daemon;
mod launchd;
mod config;
mod maintenance;
mod status;
mod tray;

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

struct Args {
    out_dir: PathBuf,
    seconds: Option<u64>,
    /// Debug hook: force a rebuild this many seconds in, to exercise the request→poll→rebuild
    /// path without a real route change.
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
                eprintln!("usage: meetscribe [--out <dir>] [--seconds <n>] [--rebuild-after <n>]");
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

/// `meetscribe transcribe <dir> [...]` — thin CLI wrapper over `pipeline::transcribe_and_store`:
/// parse args, run the pipeline (which stores AND exports), then print the transcript + RTF.
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
    let opts = pipeline::PipelineOpts {
        model: PathBuf::from(model),
        lang,
        title,
        db_path: db_path.unwrap_or_else(default_db_path),
        export_dir: export_dir.unwrap_or_else(|| dir.clone()),
        no_store,
    };
    let out = pipeline::transcribe_and_store(&dir, &opts)?;

    println!("\n===== TRANSCRIPT ({}) =====", dir.display());
    for s in &out.segments {
        println!(
            "[{:7.2}-{:7.2}] {:<7} {}",
            s.t_start,
            s.t_end,
            s.speaker.label(),
            s.text
        );
    }
    println!(
        "\nsegments: {} | meeting: {:.1}s | wall: {:.1}s | RTF: {:.3}x (both channels through whisper)",
        out.segments.len(),
        out.meeting_secs,
        out.wall_secs,
        out.rtf
    );
    println!("wrote {}", out.md_path.display());
    println!("wrote {}", out.json_path.display());
    Ok(())
}

/// `~/.meetscribe` — the app's data/config directory (`None` if `$HOME` is unset). The single
/// owner of the base-dir join; the daemon/launchd build their own validated-absolute variant.
pub(crate) fn base_dir() -> Option<PathBuf> {
    home_dir().map(|h| h.join(".meetscribe"))
}

/// `~/.meetscribe/meetscribe.db` (falls back to a repo-local path if `$HOME` is unset).
pub(crate) fn default_db_path() -> PathBuf {
    base_dir()
        .map(|b| b.join("meetscribe.db"))
        .unwrap_or_else(|| PathBuf::from("meetscribe.db"))
}

/// `~/.meetscribe/exports/` (the default target for `export <id>`).
pub(crate) fn default_export_dir() -> PathBuf {
    base_dir()
        .map(|b| b.join("exports"))
        .unwrap_or_else(|| PathBuf::from("exports"))
}

pub(crate) fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

pub(crate) fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub(crate) fn new_runtime() -> Result<tokio::runtime::Runtime> {
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

/// `meetscribe detect [--watch]` — print which allowlisted app (if any) currently holds the mic,
/// plus the live audio-active process table. The mechanical verify tool for the detector: run it
/// while a real meeting app holds the mic (Active) vs while only music plays (Idle).
fn run_detect(argv: &[String]) -> Result<()> {
    let watch = argv.iter().any(|a| a == "--watch" || a == "-w");
    // Mirror the daemon: honor ~/.meetscribe/config.toml so `detect` shows what the daemon sees.
    // Uses the PURE `Config::load` (no disk writes) — a diagnostic must not materialize config.
    let base = base_dir();
    let allowlist: Vec<String> = base
        .as_deref()
        .map(|b| config::Config::load(b).effective_allowlist())
        .unwrap_or_else(|| {
            detect::DEFAULT_ALLOWLIST
                .iter()
                .map(|s| (*s).to_string())
                .collect()
        });
    let al: Vec<&str> = allowlist.iter().map(String::as_str).collect();
    loop {
        let snap = detect::snapshot()?;
        let active = detect::active_app_in(&snap, &al);
        match &active {
            Some(app) => println!("MEETING ACTIVE — {app} holds the mic  ({} processes)", snap.len()),
            None => println!("idle — no allowlisted app holds the mic  ({} processes)", snap.len()),
        }
        for p in &snap {
            let Some(bid) = p.bundle_id.as_deref() else {
                continue;
            };
            if !p.input && !p.output {
                continue;
            }
            let allowed = detect::is_allowlisted(bid, &al);
            println!(
                "  {:<42} input={:<5} output={:<5} allowlisted={}",
                bid, p.input, p.output, allowed
            );
        }
        if !watch {
            break;
        }
        println!("---");
        std::thread::sleep(Duration::from_millis(1500));
    }
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
        Some("detect") => return run_detect(&argv[2..]),
        Some("daemon") => return daemon::run_daemon(&argv[2..]),
        Some("tray") => return tray::run_tray(&argv[2..]),
        Some("install") => return launchd::run_install(&argv[2..]),
        Some("uninstall") => return launchd::run_uninstall(&argv[2..]),
        _ => {}
    }

    let args = parse_args();

    log::info!("meetscribe capture — starting single-aggregate mic+tap capture");
    log::info!(
        "(first run triggers TWO prompts: Microphone and \"record system audio\" — approve both)"
    );

    // Stop control: a --seconds deadline, else press Enter. Only wire the stdin thread when there's
    // no deadline — a non-interactive stdin hits EOF instantly and would otherwise stop capture on
    // the first loop iteration. Both are folded into the `stop` predicate the session polls.
    let stop_flag = Arc::new(AtomicBool::new(false));
    if args.seconds.is_none() {
        let sf = stop_flag.clone();
        std::thread::spawn(move || {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
            sf.store(true, Ordering::Release);
        });
        log::info!("recording… press Enter to stop.");
    } else if let Some(s) = args.seconds {
        log::info!("recording for {s}s…");
    }
    let deadline = args.seconds.map(|s| Instant::now() + Duration::from_secs(s));
    let sf = stop_flag.clone();
    let stop = move || sf.load(Ordering::Acquire) || deadline.is_some_and(|d| Instant::now() >= d);

    let summary = session::run_capture(&args.out_dir, stop, args.rebuild_after)?;
    log::info!(
        "capture finished: {:.1}s, {} segment(s), {} rebuild(s), num_buffers={}, {} sample(s) dropped",
        summary.duration_secs,
        summary.segments,
        summary.rebuilds,
        summary.num_buffers,
        summary.dropped
    );
    Ok(())
}
