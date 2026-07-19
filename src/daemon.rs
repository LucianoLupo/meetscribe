//! The background daemon: a single-threaded state machine that auto-records meetings.
//!
//! IDLE — poll the detector every `POLL_INTERVAL`. When an allowlisted app takes the mic, start
//! a capture session (the proven `session::run_capture` loop) whose stop predicate is "the app
//! released the mic for `MEETING_END_DEBOUNCE`, OR we were signalled". On stop, finalize; if the
//! session ran at least `min_secs`, transcribe+store+export via the Phase-3 pipeline; then IDLE.
//!
//! Absolute paths only: under a launchd LaunchAgent `cwd=/` and `~` is not expanded, so the model,
//! DB, and session dirs are all resolved from an absolute `$HOME` (Batch 0 confirmed launchd sets
//! HOME for gui agents; `install` also pins it). A missing home is a loud startup error, never a
//! silent wrong-location write.
//!
//! Failure model (KeepAlive{SuccessfulExit:false} restarts on ANY non-zero exit): operational
//! errors inside the loop (capture start hiccup, transcription failure, a detector poll error) are
//! logged and the daemon keeps running — they never bubble to a non-zero exit / crash-loop. Only
//! startup misconfiguration (home unresolvable, another instance already holding the lock) exits
//! non-zero, and a correct `install` makes those impossible.

use anyhow::{Context, Result};
use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::detect::{self, MeetingDetector};
use crate::{export, pipeline, session};

/// How often to poll for a meeting while idle.
const POLL_INTERVAL: Duration = Duration::from_millis(1500);
/// The allowlisted app must have released the mic for this long before we finalize (survives a
/// transient route drop mid-call). Tunable, not load-bearing.
const MEETING_END_DEBOUNCE: Duration = Duration::from_secs(10);
/// While recording, re-poll the detector at most this often (the capture loop ticks every 50 ms;
/// enumerating every process 20×/s would be wasteful).
const DETECT_POLL_DURING_CAPTURE: Duration = Duration::from_millis(1000);
/// Default minimum session length to bother transcribing — drops sub-blip mic uses (voice search).
const DEFAULT_MIN_MEETING_SECS: f64 = 20.0;

/// Set by the SIGTERM/SIGINT handler; polled by the idle loop and folded into the capture stop
/// predicate so shutdown reuses the one clean finalize path.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_sig: libc::c_int) {
    // Async-signal-safe: a single atomic store.
    SHUTDOWN.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    // SAFETY: registering a handler that only does an atomic store is async-signal-safe.
    unsafe {
        libc::signal(libc::SIGTERM, on_signal as *const () as usize);
        libc::signal(libc::SIGINT, on_signal as *const () as usize);
    }
}

/// Hold the single-instance lock for the process lifetime (returned File keeps the fd open).
/// `LOCK_EX | LOCK_NB` — fails immediately if another daemon already holds it.
fn acquire_single_instance_lock(base: &Path) -> Result<std::fs::File> {
    let lock_path = base.join("daemon.lock");
    let f = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("open daemon lock {}", lock_path.display()))?;
    // SAFETY: f owns a valid fd for the duration of the call.
    let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        anyhow::bail!(
            "another meetscribe daemon is already running (lock held: {})",
            lock_path.display()
        );
    }
    Ok(f)
}

struct DaemonConfig {
    base: PathBuf,
    sessions_dir: PathBuf,
    db_path: PathBuf,
    model: PathBuf,
    lang: String,
    min_secs: f64,
    once: bool,
}

/// `meetscribe daemon [--min-secs <n>] [--lang <code>] [--model <path>] [--once]`.
pub(crate) fn run_daemon(argv: &[String]) -> Result<()> {
    let mut lang = String::from("es");
    let mut min_secs = DEFAULT_MIN_MEETING_SECS;
    let mut once = false;
    let mut model_override: Option<PathBuf> = None;
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--lang" | "-l" => {
                if let Some(v) = it.next() {
                    lang = v.clone();
                }
            }
            "--min-secs" => {
                if let Some(v) = it.next() {
                    min_secs = v.parse().unwrap_or(min_secs);
                }
            }
            "--model" | "-m" => {
                if let Some(v) = it.next() {
                    model_override = Some(PathBuf::from(v));
                }
            }
            "--once" => once = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: meetscribe daemon [--min-secs <n>] [--lang <code>] [--model <path>] [--once]"
                );
                return Ok(());
            }
            _ => {}
        }
    }

    // Absolute paths from an absolute home (loud error if unresolvable — never a relative write).
    let home = crate::home_dir()
        .context("daemon: cannot resolve $HOME — needed for absolute session/db/model paths under launchd")?;
    anyhow::ensure!(
        home.is_absolute(),
        "daemon: resolved home {} is not absolute",
        home.display()
    );
    let base = home.join(".meetscribe");
    let cfg = DaemonConfig {
        sessions_dir: base.join("sessions"),
        db_path: base.join("meetscribe.db"),
        model: model_override.unwrap_or_else(|| base.join("models/ggml-large-v3.bin")),
        base: base.clone(),
        lang,
        min_secs,
        once,
    };
    std::fs::create_dir_all(&cfg.sessions_dir)
        .with_context(|| format!("create sessions dir {}", cfg.sessions_dir.display()))?;

    // Single-instance (prevents a manual `daemon` colliding with the launchd one on the tap).
    let _lock = acquire_single_instance_lock(&cfg.base)?;
    install_signal_handlers();

    if !cfg.model.exists() {
        log::warn!(
            "model not found at {} — meetings will still be CAPTURED, but transcription will fail \
             until the model is provisioned (see `meetscribe install`).",
            cfg.model.display()
        );
    }
    log::info!(
        "meetscribe daemon up — watching for {} meeting apps (poll {}s, end-debounce {}s, \
         min {}s). db={} sessions={}",
        detect::DEFAULT_ALLOWLIST.len(),
        POLL_INTERVAL.as_secs_f32(),
        MEETING_END_DEBOUNCE.as_secs(),
        cfg.min_secs,
        cfg.db_path.display(),
        cfg.sessions_dir.display()
    );

    let detector = MeetingDetector::new();
    while !SHUTDOWN.load(Ordering::SeqCst) {
        match detector.active_app() {
            Ok(Some(app)) => {
                record_and_process(&cfg, &app);
                if cfg.once {
                    log::info!("--once: processed one meeting, exiting");
                    break;
                }
            }
            Ok(None) => {}
            Err(e) => log::warn!("detector poll failed (transient): {e}"),
        }
        // Interruptible idle wait.
        sleep_interruptible(POLL_INTERVAL);
    }

    log::info!("meetscribe daemon shutting down cleanly");
    Ok(())
}

/// Record one meeting (`app` holds the mic), then gate + transcribe+store+export. Never returns an
/// error — a capture/transcription failure is logged and the daemon returns to IDLE (the recovery
/// machinery must not take down the daemon).
fn record_and_process(cfg: &DaemonConfig, app: &str) {
    let started = crate::now_epoch();
    let dir = cfg.sessions_dir.join(export::stamp_compact(started));
    let label = detect::app_label(app);
    log::info!("meeting detected — {label} ({app}) took the mic → recording to {}", dir.display());

    // Stop predicate: the app released the mic for MEETING_END_DEBOUNCE, or we were signalled.
    // The detector is polled at most every DETECT_POLL_DURING_CAPTURE (the capture loop ticks far
    // faster). A transient detector error keeps the last state (don't drop a live recording).
    let detector = MeetingDetector::new();
    let mut last_poll = Instant::now();
    let mut active = true;
    let mut released_since: Option<Instant> = None;
    let stop = move || -> bool {
        if SHUTDOWN.load(Ordering::SeqCst) {
            return true;
        }
        if last_poll.elapsed() >= DETECT_POLL_DURING_CAPTURE {
            last_poll = Instant::now();
            match detector.active_app() {
                Ok(Some(_)) => {
                    active = true;
                    released_since = None;
                }
                Ok(None) => {
                    active = false;
                    released_since.get_or_insert_with(Instant::now);
                }
                Err(_) => {}
            }
        }
        !active && released_since.is_some_and(|s| s.elapsed() >= MEETING_END_DEBOUNCE)
    };

    let summary = match session::run_capture(&dir, stop, None) {
        Ok(s) => s,
        Err(e) => {
            log::error!("capture failed for {label} ({}): {e} — returning to idle", dir.display());
            return;
        }
    };
    log::info!(
        "meeting ended — {label}: captured {:.1}s ({} segment file(s), {} dropped)",
        summary.duration_secs,
        summary.segments,
        summary.dropped
    );

    if summary.duration_secs < cfg.min_secs {
        log::info!(
            "session {:.1}s < {:.0}s minimum — skipping transcription (WAVs kept at {})",
            summary.duration_secs,
            cfg.min_secs,
            dir.display()
        );
        return;
    }

    if SHUTDOWN.load(Ordering::SeqCst) {
        log::info!("shutdown requested — WAVs kept at {}, skipping transcription", dir.display());
        return;
    }

    let opts = pipeline::PipelineOpts {
        model: cfg.model.clone(),
        lang: cfg.lang.clone(),
        title: Some(format!("{label} — {}", export::fmt_utc(started))),
        db_path: cfg.db_path.clone(),
        export_dir: dir.clone(),
        no_store: false,
    };
    log::info!("transcribing {} …", dir.display());
    match pipeline::transcribe_and_store(&dir, &opts) {
        Ok(out) => log::info!(
            "stored{} → {} ({} segments)",
            out.stored_id.map(|id| format!(" meeting id {id}")).unwrap_or_default(),
            out.md_path.display(),
            out.segments.len()
        ),
        Err(e) => log::error!("transcription failed for {} — WAVs kept: {e}", dir.display()),
    }
}

/// Sleep in small slices so a signal breaks the idle wait promptly (SIGTERM → clean shutdown).
fn sleep_interruptible(total: Duration) {
    let step = Duration::from_millis(200);
    let mut left = total;
    while left > Duration::ZERO && !SHUTDOWN.load(Ordering::SeqCst) {
        let s = step.min(left);
        std::thread::sleep(s);
        left = left.saturating_sub(s);
    }
}
