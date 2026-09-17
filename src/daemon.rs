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

use crate::config::{self, Config};
use crate::detect::{self, MeetingDetector};
use crate::{export, pipeline, session, status};

/// How often to poll for a meeting while idle.
const POLL_INTERVAL: Duration = Duration::from_millis(1500);
/// The allowlisted app must have released the mic for this long before we finalize (survives a
/// transient route drop mid-call). Tunable, not load-bearing.
const MEETING_END_DEBOUNCE: Duration = Duration::from_secs(10);
/// While recording, re-poll the detector at most this often (the capture loop ticks every 50 ms;
/// enumerating every process 20×/s would be wasteful).
const DETECT_POLL_DURING_CAPTURE: Duration = Duration::from_millis(1000);
/// How often the idle loop runs disk/log hygiene (session prune + log rotation). Also runs once at
/// startup. A long-lived daemon rarely restarts, so the loop must handle rotation itself.
const MAINTENANCE_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);

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
    logs_dir: PathBuf,
    db_path: PathBuf,
    model: PathBuf,
    /// Speaker-embedding model (absolute). Missing ⇒ transcripts without speaker identity.
    speaker_model: PathBuf,
    lang: String,
    min_secs: f64,
    once: bool,
    /// Effective meeting-app allowlist (built-ins ∪ config extras) — the detectors match on it.
    allowlist: Vec<String>,
    /// Delete session dirs older than this many days (0 = keep forever).
    sessions_days: u64,
    /// Rotate the daemon logs over this many MB (0 = never).
    log_max_mb: u64,
}

/// `meetscribe daemon [--min-secs <n>] [--lang <code>] [--model <path>] [--once]`.
/// Values come from `~/.meetscribe/config.toml`; any CLI flag here overrides the config file.
pub(crate) fn run_daemon(argv: &[String]) -> Result<()> {
    // CLI overrides are Options so we can tell "flag given" from "use config/default".
    let mut lang_override: Option<String> = None;
    let mut min_secs_override: Option<f64> = None;
    let mut once = false;
    let mut model_override: Option<PathBuf> = None;
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--lang" | "-l" => {
                if let Some(v) = it.next() {
                    lang_override = Some(v.clone());
                }
            }
            "--min-secs" => {
                if let Some(v) = it.next() {
                    match v.parse::<f64>() {
                        Ok(n) => min_secs_override = Some(n),
                        Err(_) => log::warn!("daemon: ignoring invalid --min-secs '{v}'"),
                    }
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
                    "usage: meetscribe daemon [--min-secs <n>] [--lang <code>] [--model <path>] [--once]\n\
                     (config: ~/.meetscribe/config.toml — flags here override it)"
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

    // Load config (writes a default template on first run); CLI flags win over it.
    let file_cfg = Config::load_or_init(&base);
    let allowlist = file_cfg.effective_allowlist();
    let cfg = DaemonConfig {
        sessions_dir: base.join("sessions"),
        logs_dir: base.join("logs"),
        db_path: base.join("meetscribe.db"),
        model: model_override.unwrap_or_else(|| base.join("models/ggml-large-v3.bin")),
        speaker_model: base.join(crate::SPEAKER_MODEL_REL),
        lang: lang_override.unwrap_or(file_cfg.daemon.lang),
        min_secs: min_secs_override.unwrap_or(file_cfg.daemon.min_secs),
        allowlist,
        sessions_days: file_cfg.retention.sessions_days,
        log_max_mb: file_cfg.retention.log_max_mb,
        base: base.clone(),
        once,
    };
    std::fs::create_dir_all(&cfg.sessions_dir)
        .with_context(|| format!("create sessions dir {}", cfg.sessions_dir.display()))?;

    // Single-instance (prevents a manual `daemon` colliding with the launchd one on the tap).
    let _lock = acquire_single_instance_lock(&cfg.base)?;
    install_signal_handlers();

    if cfg.allowlist.is_empty() {
        log::warn!(
            "detector allowlist is EMPTY (use_builtin_allowlist=false with no allowlist_extra) — \
             no app will ever trigger a recording. Edit {}.",
            config::config_path(&base).display()
        );
    }
    if !cfg.model.exists() {
        log::warn!(
            "model not found at {} — meetings will still be CAPTURED, but transcription will fail \
             until the model is provisioned (see `meetscribe install`).",
            cfg.model.display()
        );
    }
    if !cfg.speaker_model.exists() {
        log::warn!(
            "speaker model not found at {} — meetings will be transcribed but far-end voices will \
             not be identified until it is provisioned (`bash models/provision.sh`, then \
             `meetscribe install`).",
            cfg.speaker_model.display()
        );
    }
    log::info!(
        "meetscribe daemon up — watching for {} meeting apps (poll {}s, end-debounce {}s, \
         min {}s). db={} sessions={}",
        cfg.allowlist.len(),
        POLL_INTERVAL.as_secs_f32(),
        MEETING_END_DEBOUNCE.as_secs(),
        cfg.min_secs,
        cfg.db_path.display(),
        cfg.sessions_dir.display()
    );

    let detector = MeetingDetector::with_allowlist(cfg.allowlist.clone());
    // Disk/log hygiene: once at startup, then daily (a long-lived daemon rarely restarts).
    run_maintenance(&cfg);
    let mut last_maintenance = Instant::now();

    // Tray IPC: publish state changes to status.json; honor the pause flag file. Start in the
    // state the flag file dictates (so a daemon (re)started while paused stays paused).
    let mut was_paused = status::is_paused(&cfg.base);
    publish_status(
        &cfg,
        if was_paused {
            status::DaemonState::Paused
        } else {
            status::DaemonState::Idle
        },
        None,
    );

    // Consecutive capture-start failures for the currently-detected app — throttles the error log
    // and backs off, so a persistently-denied grant doesn't spin+spam at the poll rate.
    let mut capture_fail_streak = 0u32;
    while !SHUTDOWN.load(Ordering::SeqCst) {
        if last_maintenance.elapsed() >= MAINTENANCE_INTERVAL {
            run_maintenance(&cfg);
            last_maintenance = Instant::now();
        }

        // Pause flag: publish the transition once, then skip STARTING new recordings while set.
        let paused_now = status::is_paused(&cfg.base);
        if paused_now != was_paused {
            was_paused = paused_now;
            if paused_now {
                log::info!("paused (flag {} present) — not starting new recordings", status::pause_path(&cfg.base).display());
                publish_status(&cfg, status::DaemonState::Paused, None);
            } else {
                log::info!("resumed (pause flag cleared)");
                publish_status(&cfg, status::DaemonState::Idle, None);
            }
        }
        if paused_now {
            sleep_interruptible(POLL_INTERVAL);
            continue;
        }

        match detector.active_app() {
            Ok(Some(app)) => {
                publish_status(
                    &cfg,
                    status::DaemonState::Recording,
                    Some(detect::app_label(&app).to_string()),
                );
                match record_and_process(&cfg, &app) {
                    Ok(()) => {
                        capture_fail_streak = 0;
                        publish_status(&cfg, status::DaemonState::Idle, None);
                        if cfg.once {
                            log::info!("--once: processed one meeting, exiting");
                            break;
                        }
                    }
                    Err(e) => {
                        capture_fail_streak += 1;
                        if capture_fail_streak == 1 || capture_fail_streak.is_multiple_of(20) {
                            log::error!("{e:#} (attempt #{capture_fail_streak}; backing off)");
                        }
                        publish_status(&cfg, status::DaemonState::Idle, None);
                        let backoff = (POLL_INTERVAL * capture_fail_streak.min(20))
                            .min(Duration::from_secs(30));
                        sleep_interruptible(backoff);
                        continue;
                    }
                }
            }
            Ok(None) => capture_fail_streak = 0,
            Err(e) => log::warn!("detector poll failed (transient): {e}"),
        }
        // Interruptible idle wait.
        sleep_interruptible(POLL_INTERVAL);
    }

    log::info!("meetscribe daemon shutting down cleanly");
    Ok(())
}

/// Publish a state change to `status.json` for the tray (best-effort — a write failure is logged,
/// never fatal). Called only on transitions, so it does not churn the file each poll.
fn publish_status(cfg: &DaemonConfig, state: status::DaemonState, app: Option<String>) {
    let now = crate::now_epoch();
    let s = status::Status {
        state,
        app,
        since_epoch: now,
        updated_epoch: now,
        pid: std::process::id(),
    };
    if let Err(e) = s.write(&cfg.base) {
        log::warn!("could not write status file: {e}");
    }
}

/// Record one meeting (`app` holds the mic), then gate + transcribe+store+export. Returns `Err`
/// ONLY if capture failed to start/run — the caller throttles + backs off, so a persistently-denied
/// grant (common on first run) doesn't spin+spam at the poll rate. A transcription failure is logged
/// internally and returns `Ok` (the WAVs are kept); the daemon then returns to IDLE.
fn record_and_process(cfg: &DaemonConfig, app: &str) -> Result<()> {
    let started = crate::now_epoch();
    let dir = cfg.sessions_dir.join(export::stamp_compact(started));
    let label = detect::app_label(app);
    log::info!("meeting detected — {label} ({app}) took the mic → recording to {}", dir.display());

    // Stop predicate: the app released the mic for MEETING_END_DEBOUNCE, or we were signalled.
    // The detector is polled at most every DETECT_POLL_DURING_CAPTURE (the capture loop ticks far
    // faster). A transient detector error keeps the last state (don't drop a live recording).
    // `released_since` is Some only while the app has been off the mic; it fully encodes the
    // "released?" state (no separate bool needed).
    let detector = MeetingDetector::with_allowlist(cfg.allowlist.clone());
    let mut last_poll = Instant::now();
    let mut released_since: Option<Instant> = None;
    let stop = move || -> bool {
        if SHUTDOWN.load(Ordering::SeqCst) {
            return true;
        }
        if last_poll.elapsed() >= DETECT_POLL_DURING_CAPTURE {
            last_poll = Instant::now();
            match detector.active_app() {
                Ok(Some(_)) => released_since = None,
                Ok(None) => {
                    released_since.get_or_insert_with(Instant::now);
                }
                Err(_) => {}
            }
        }
        released_since.is_some_and(|s| s.elapsed() >= MEETING_END_DEBOUNCE)
    };

    let summary = session::run_capture(&dir, stop, None)
        .with_context(|| format!("capture failed for {label} ({})", dir.display()))?;
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
        return Ok(());
    }

    if SHUTDOWN.load(Ordering::SeqCst) {
        log::info!("shutdown requested — WAVs kept at {}, skipping transcription", dir.display());
        return Ok(());
    }

    let opts = pipeline::PipelineOpts {
        model: cfg.model.clone(),
        lang: cfg.lang.clone(),
        title: Some(format!("{label} — {}", export::fmt_utc(started))),
        db_path: cfg.db_path.clone(),
        export_dir: dir.clone(),
        no_store: false,
        speaker_model: Some(cfg.speaker_model.clone()),
    };
    log::info!("transcribing {} …", dir.display());
    // Capture is done but this call blocks for minutes on a long meeting — publish the transition so
    // the tray stops showing "recording" the moment the call actually ended.
    publish_status(cfg, status::DaemonState::Transcribing, Some(label.to_string()));
    match pipeline::transcribe_and_store(&dir, &opts) {
        Ok(out) => log::info!(
            "stored{} → {} ({} segments)",
            out.stored_id.map(|id| format!(" meeting id {id}")).unwrap_or_default(),
            out.md_path.display(),
            out.segments.len()
        ),
        Err(e) => log::error!("transcription failed for {} — WAVs kept: {e}", dir.display()),
    }
    Ok(())
}

/// Disk/log hygiene: prune old session dirs + rotate the daemon logs (both config-driven, both
/// best-effort — a failure is logged, never fatal). Runs at startup and once per day.
fn run_maintenance(cfg: &DaemonConfig) {
    match crate::maintenance::prune_sessions(&cfg.sessions_dir, cfg.sessions_days, crate::now_epoch())
    {
        Ok(r) if r.removed > 0 => {
            log::info!("retention: pruned {} old session(s) (kept {})", r.removed, r.kept);
        }
        Ok(_) => {}
        Err(e) => log::warn!("retention: session prune failed: {e}"),
    }
    let max_bytes = cfg.log_max_mb.saturating_mul(1024 * 1024);
    for name in ["meetscribe.err.log", "meetscribe.out.log"] {
        let p = cfg.logs_dir.join(name);
        if let Err(e) = crate::maintenance::rotate_log(&p, max_bytes) {
            log::warn!("retention: log rotate failed for {}: {e}", p.display());
        }
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
