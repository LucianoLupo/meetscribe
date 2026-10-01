//! The background daemon: a capture state machine plus one transcription worker thread.
//!
//! IDLE — poll the detector every `POLL_INTERVAL`. When an allowlisted app takes the mic, start
//! a capture session (the proven `session::run_capture` loop) whose stop predicate is "the app
//! released the mic for `MEETING_END_DEBOUNCE`, OR we were signalled". On stop, finalize; if the
//! session ran at least `min_secs`, QUEUE it for transcribe+store+export; then IDLE at once.
//!
//! Transcription runs on a separate worker, one job at a time, in arrival order. It used to run
//! inline, and a long meeting's transcription (often 10–50 min) blinded the detector: a call that
//! started meanwhile went unrecorded until whisper finished. Capture stays on the main thread (Core
//! Audio handles are `!Send`); the worker owns nothing but paths and the pipeline it runs.
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
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::thread::JoinHandle;
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
    /// Far-end split ([speakers] split) + the installed diarizer (absolute). Missing ⇒ no split.
    split: bool,
    diarizer_bin: PathBuf,
    diarizer_model: PathBuf,
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
        split: file_cfg.speakers.split,
        diarizer_bin: base.join(crate::DIARIZER_BIN_REL),
        diarizer_model: base.join(crate::DIARIZER_MODEL_REL),
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
    if cfg.split {
        let missing: Vec<String> = [&cfg.diarizer_bin, &cfg.diarizer_model]
            .into_iter()
            .filter(|p| !p.exists())
            .map(|p| p.display().to_string())
            .collect();
        if missing.is_empty() {
            log::info!(
                "far-end split: on — diarizer {} + {} ({})",
                cfg.diarizer_bin.display(),
                cfg.diarizer_model.display(),
                crate::diar::provenance(&cfg.diarizer_bin, &cfg.diarizer_model)
            );
        } else {
            log::warn!(
                "far-end split is on but the diarizer is missing ({}) — meetings will be transcribed \
                 WITHOUT splitting until it is installed (`bash models/provision.sh`, \
                 `bash models/build-diarizer.sh`, then `meetscribe install`).",
                missing.join(", ")
            );
        }
    } else {
        log::info!("far-end split: off ([speakers] split = false in {})", config::config_path(&base).display());
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
    let board = Arc::new(StatusBoard::new(cfg.base.clone()));
    board.update(|a| a.paused = was_paused);
    let (jobs, transcriber) = spawn_transcriber(Arc::clone(&board), run_transcribe_job)
        .context("daemon: could not start the transcription worker")?;

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
                log::info!(
                    "paused (flag {} present) — not starting new recordings",
                    status::pause_path(&cfg.base).display()
                );
            } else {
                log::info!("resumed (pause flag cleared)");
            }
            board.update(|a| a.paused = paused_now);
        }
        if paused_now {
            sleep_interruptible(POLL_INTERVAL);
            continue;
        }

        match detector.active_app() {
            Ok(Some(app)) => {
                board.update(|a| a.recording = Some(detect::app_label(&app).to_string()));
                let recorded = record_and_queue(&cfg, &app, &jobs);
                board.update(|a| a.recording = None);
                match recorded {
                    Ok(()) => {
                        capture_fail_streak = 0;
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

    // Closing the queue lets the worker finish the job in hand (it skips queued ones on shutdown —
    // their WAVs stay on disk) and exit. Joining keeps `--once` meaning "recorded AND transcribed".
    drop(jobs);
    if transcriber.join().is_err() {
        log::error!("transcription worker panicked outside a job");
    }
    log::info!("meetscribe daemon shutting down cleanly");
    Ok(())
}

/// What the daemon is doing right now. The capture loop owns `recording` and `paused`; the worker
/// owns `transcribing`. Both can be live at once, and the tray shows one state, so it is derived.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Activity {
    /// App label of the call being recorded.
    recording: Option<String>,
    /// App label of the meeting whisper is working on.
    transcribing: Option<String>,
    paused: bool,
}

impl Activity {
    /// Recording wins (a live call is what the user must see), then Transcribing, then Paused. A
    /// pause set mid-transcription keeps showing `transcribing` until it ends, as it always did.
    fn shown(&self) -> Shown {
        if let Some(app) = &self.recording {
            (status::DaemonState::Recording, Some(app.clone()))
        } else if let Some(app) = &self.transcribing {
            (status::DaemonState::Transcribing, Some(app.clone()))
        } else if self.paused {
            (status::DaemonState::Paused, None)
        } else {
            (status::DaemonState::Idle, None)
        }
    }
}

/// A state as the tray renders it: what `status.json` carries.
type Shown = (status::DaemonState, Option<String>);

/// The [`Activity`] plus what was last written to `status.json`, shared by the capture loop and the
/// worker. The file is rewritten only when the shown state changes, so `since_epoch` keeps meaning
/// "since this state began" and the file does not churn each poll.
struct StatusBoard {
    base: PathBuf,
    inner: Mutex<(Activity, Option<Shown>)>,
}

impl StatusBoard {
    fn new(base: PathBuf) -> Self {
        Self {
            base,
            inner: Mutex::new((Activity::default(), None)),
        }
    }

    fn update(&self, change: impl FnOnce(&mut Activity)) {
        // A poisoned lock only means another thread panicked mid-update; the data is still usable.
        let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let (activity, last) = &mut *guard;
        change(activity);
        let shown = activity.shown();
        if last.as_ref() != Some(&shown) {
            write_status(&self.base, shown.0, shown.1.clone());
            *last = Some(shown);
        }
    }
}

/// Write `status.json` for the tray (best-effort — a write failure is logged, never fatal).
fn write_status(base: &Path, state: status::DaemonState, app: Option<String>) {
    let now = crate::now_epoch();
    let s = status::Status {
        state,
        app,
        since_epoch: now,
        updated_epoch: now,
        pid: std::process::id(),
    };
    if let Err(e) = s.write(base) {
        log::warn!("could not write status file: {e}");
    }
}

/// One finished recording waiting for transcribe+store+export.
struct TranscribeJob {
    dir: PathBuf,
    /// Friendly app label, shown by the tray while this job runs.
    label: String,
    opts: pipeline::PipelineOpts,
}

/// Start the worker that runs `run` on each queued job, one at a time, in arrival order. It marks
/// the board `transcribing` while a job runs and clears it only once the queue is empty (so a
/// backlog does not flicker the tray through `idle`). After [`SHUTDOWN`] it skips queued jobs.
/// The worker exits when every `Sender` is dropped.
fn spawn_transcriber(
    board: Arc<StatusBoard>,
    run: impl Fn(&TranscribeJob) + Send + 'static,
) -> std::io::Result<(mpsc::Sender<TranscribeJob>, JoinHandle<()>)> {
    let (tx, rx) = mpsc::channel::<TranscribeJob>();
    let handle = std::thread::Builder::new()
        .name("transcriber".to_string())
        .spawn(move || {
            let mut next = rx.recv().ok();
            while let Some(job) = next {
                if SHUTDOWN.load(Ordering::SeqCst) {
                    log::info!(
                        "shutdown requested — WAVs kept at {}, skipping transcription",
                        job.dir.display()
                    );
                } else {
                    board.update(|a| a.transcribing = Some(job.label.clone()));
                    run(&job);
                }
                next = match rx.try_recv() {
                    Ok(job) => Some(job),
                    Err(_) => {
                        board.update(|a| a.transcribing = None);
                        rx.recv().ok()
                    }
                };
            }
        })?;
    Ok((tx, handle))
}

/// Transcribe+store+export one session. Every failure, including a panic inside the pipeline, is
/// logged and contained: the WAVs are kept and the worker moves on to the next meeting.
fn run_transcribe_job(job: &TranscribeJob) {
    log::info!("transcribing {} …", job.dir.display());
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pipeline::transcribe_and_store(&job.dir, &job.opts)
    }));
    match outcome {
        Ok(Ok(out)) => log::info!(
            "stored{} → {} ({} segments)",
            out.stored_id
                .map(|id| format!(" meeting id {id}"))
                .unwrap_or_default(),
            out.md_path.display(),
            out.segments.len()
        ),
        Ok(Err(e)) => log::error!(
            "transcription failed for {} — WAVs kept: {e}",
            job.dir.display()
        ),
        Err(_) => log::error!(
            "transcription panicked for {} — WAVs kept",
            job.dir.display()
        ),
    }
}

/// Record one meeting (`app` holds the mic), then gate it and queue it for transcription. Returns
/// as soon as capture ends, so the caller is back to watching for the next call while whisper
/// works. Returns `Err` ONLY if capture failed to start/run — the caller throttles + backs off, so
/// a persistently-denied grant (common on first run) doesn't spin+spam at the poll rate.
fn record_and_queue(
    cfg: &DaemonConfig,
    app: &str,
    jobs: &mpsc::Sender<TranscribeJob>,
) -> Result<()> {
    let started = crate::now_epoch();
    let dir = cfg.sessions_dir.join(export::stamp_compact(started));
    let label = detect::app_label(app);
    log::info!(
        "meeting detected — {label} ({app}) took the mic → recording to {}",
        dir.display()
    );

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
        log::info!(
            "shutdown requested — WAVs kept at {}, skipping transcription",
            dir.display()
        );
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
        split: cfg.split,
        diarizer_bin: cfg.diarizer_bin.clone(),
        diarizer_model: cfg.diarizer_model.clone(),
        diarizer_rttm: None,
    };
    log::info!("queued {} for transcription", dir.display());
    let job = TranscribeJob {
        dir,
        label: label.to_string(),
        opts,
    };
    if let Err(mpsc::SendError(job)) = jobs.send(job) {
        log::error!(
            "transcription worker is gone — WAVs kept at {}",
            job.dir.display()
        );
    }
    Ok(())
}

/// Disk/log hygiene: prune old session dirs + rotate the daemon logs (both config-driven, both
/// best-effort — a failure is logged, never fatal). Runs at startup and once per day.
fn run_maintenance(cfg: &DaemonConfig) {
    match crate::maintenance::prune_sessions(
        &cfg.sessions_dir,
        cfg.sessions_days,
        crate::now_epoch(),
    ) {
        Ok(r) if r.removed > 0 => {
            log::info!(
                "retention: pruned {} old session(s) (kept {})",
                r.removed,
                r.kept
            );
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

#[cfg(test)]
mod tests {
    use super::*;
    use status::DaemonState;

    fn temp_base(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("meetscribe-daemon-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn job(dir: &Path, label: &str) -> TranscribeJob {
        TranscribeJob {
            dir: dir.to_path_buf(),
            label: label.to_string(),
            opts: pipeline::PipelineOpts {
                model: dir.join("ggml-large-v3.bin"),
                lang: "es".to_string(),
                title: None,
                db_path: dir.join("meetscribe.db"),
                export_dir: dir.to_path_buf(),
                no_store: true,
                speaker_model: None,
                split: false,
                diarizer_bin: dir.join("diarizer"),
                diarizer_model: dir.join("diarizer.onnx"),
                diarizer_rttm: None,
            },
        }
    }

    fn shown_on_disk(base: &Path) -> (DaemonState, Option<String>) {
        let s = status::Status::read(base).expect("status.json written");
        (s.state, s.app)
    }

    #[test]
    fn a_live_recording_outranks_a_running_transcription() {
        let mut a = Activity {
            transcribing: Some("Chrome".into()),
            paused: true,
            ..Activity::default()
        };
        assert_eq!(
            a.shown(),
            (DaemonState::Transcribing, Some("Chrome".into()))
        );
        a.recording = Some("Zoom".into());
        assert_eq!(a.shown(), (DaemonState::Recording, Some("Zoom".into())));
        a.recording = None;
        a.transcribing = None;
        assert_eq!(a.shown(), (DaemonState::Paused, None));
        a.paused = false;
        assert_eq!(a.shown(), (DaemonState::Idle, None));
    }

    /// The bug this worker exists for: while a long transcription runs, the capture loop must be
    /// free to record the next call. The job here blocks until the test releases it, standing in for
    /// a 50-minute whisper run; a second recording starts and ends meanwhile.
    #[test]
    fn a_recording_can_start_and_queue_while_a_transcription_runs() {
        let base = temp_base("overlap");
        let board = Arc::new(StatusBoard::new(base.clone()));
        let (started_tx, started_rx) = mpsc::channel::<String>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let release_rx = Mutex::new(release_rx);
        let (jobs, worker) = spawn_transcriber(Arc::clone(&board), move |job| {
            started_tx.send(job.label.clone()).unwrap();
            release_rx.lock().unwrap().recv().unwrap();
        })
        .unwrap();

        jobs.send(job(&base, "Chrome")).unwrap();
        assert_eq!(
            started_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
            "Chrome"
        );
        assert_eq!(
            shown_on_disk(&base),
            (DaemonState::Transcribing, Some("Chrome".into()))
        );

        // The next call arrives mid-transcription: the loop can record it (nothing waits on whisper).
        board.update(|a| a.recording = Some("Meet".into()));
        assert_eq!(
            shown_on_disk(&base),
            (DaemonState::Recording, Some("Meet".into()))
        );
        board.update(|a| a.recording = None);
        jobs.send(job(&base, "Meet")).unwrap();
        assert_eq!(
            shown_on_disk(&base),
            (DaemonState::Transcribing, Some("Chrome".into()))
        );

        // Backlog runs in order, and the tray goes idle only once the queue is empty.
        release_tx.send(()).unwrap();
        assert_eq!(
            started_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
            "Meet"
        );
        assert_eq!(
            shown_on_disk(&base),
            (DaemonState::Transcribing, Some("Meet".into()))
        );
        release_tx.send(()).unwrap();
        drop(jobs);
        worker.join().unwrap();
        assert_eq!(shown_on_disk(&base), (DaemonState::Idle, None));

        std::fs::remove_dir_all(&base).ok();
    }
}
