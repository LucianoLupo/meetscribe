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
use crate::{export, maintenance, pipeline, session, status};

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
    // Transcription runs in a child process of this very binary (`meetscribe transcribe`).
    let exe = std::env::current_exe().context("daemon: cannot resolve its own executable")?;
    let child_base = cfg.base.clone();
    let (jobs, transcriber) =
        spawn_transcriber(Arc::clone(&board), move |job| run_transcribe_job(job, &exe, &child_base))
        .context("daemon: could not start the transcription worker")?;
    for (dir, label, title) in pending_sessions(&cfg.sessions_dir) {
        log::info!("resuming a transcription the last run did not reach: {}", dir.display());
        queue_job(&jobs, job_for(&cfg, dir, label, title));
    }

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

/// One finished recording waiting for transcribe+store+export. The queue itself is in memory, so
/// each queued session dir also carries a `PENDING_MARKER` file (label, then title): it outlives a
/// restart, the next start re-queues it, and retention never prunes the dir while it is there.
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
                    log::warn!(
                        "shutdown requested — {} stays pending; it is transcribed on the next start",
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

fn job_for(cfg: &DaemonConfig, dir: PathBuf, label: String, title: String) -> TranscribeJob {
    let opts = pipeline::PipelineOpts {
        model: cfg.model.clone(),
        lang: cfg.lang.clone(),
        title: Some(title),
        db_path: cfg.db_path.clone(),
        export_dir: dir.clone(),
        no_store: false,
        speaker_model: Some(cfg.speaker_model.clone()),
        split: cfg.split,
        diarizer_bin: cfg.diarizer_bin.clone(),
        diarizer_model: cfg.diarizer_model.clone(),
        diarizer_rttm: None,
    };
    TranscribeJob { dir, label, opts }
}

/// Mark the session dir pending (best-effort: without the marker the job still runs now, it just
/// would not survive a restart), then hand the job to the worker.
fn queue_job(jobs: &mpsc::Sender<TranscribeJob>, job: TranscribeJob) {
    let title = job.opts.title.as_deref().unwrap_or_default();
    let marker = job.dir.join(maintenance::PENDING_MARKER);
    if let Err(e) = std::fs::write(&marker, format!("{}\n{title}\n", job.label)) {
        log::warn!("could not write {} — this meeting will not be retried after a restart: {e}", marker.display());
    }
    log::info!("queued {} for transcription", job.dir.display());
    if let Err(mpsc::SendError(job)) = jobs.send(job) {
        log::error!("transcription worker is gone — WAVs kept at {}", job.dir.display());
    }
}

/// Session dirs still marked pending — queued but never attempted (the daemon stopped first) —
/// oldest first, as `(dir, label, title)`. Dir names are UTC stamps, so name order is time order.
fn pending_sessions(sessions_dir: &Path) -> Vec<(PathBuf, String, String)> {
    let Ok(entries) = std::fs::read_dir(sessions_dir) else { return Vec::new() };
    let mut found: Vec<(PathBuf, String, String)> = entries
        .filter_map(|e| {
            let dir = e.ok()?.path();
            let text = std::fs::read_to_string(dir.join(maintenance::PENDING_MARKER)).ok()?;
            let mut lines = text.lines();
            let label = lines.next().filter(|l| !l.is_empty()).unwrap_or("meeting").to_string();
            let title = lines.next().map(str::to_string).unwrap_or_else(|| label.clone());
            Some((dir, label, title))
        })
        .collect();
    found.sort_by(|a, b| a.0.cmp(&b.0));
    found
}

/// How the transcription child ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChildOutcome {
    Exited(i32),
    Signaled(i32),
    /// Never started, or lost track of it (`try_wait` failed).
    Failed,
}

#[derive(Debug, PartialEq, Eq)]
enum Marker {
    Keep,
    Clear,
}

/// What happens to a session's `pending` marker after its transcription child ends. Success clears
/// it. A child that ended while the daemon was shutting down was (or may have been) stopped by us,
/// so the marker stays and the next start retries. Any other failure clears it — the WAVs stay and
/// `meetscribe transcribe <dir>` re-runs it by hand — so a session that always crashes whisper
/// cannot retry on every start.
fn marker_after(outcome: ChildOutcome, shutting_down: bool) -> Marker {
    match outcome {
        ChildOutcome::Exited(0) => Marker::Clear,
        _ if shutting_down => Marker::Keep,
        _ => Marker::Clear,
    }
}

/// The `meetscribe transcribe` argv (after the subcommand) that reproduces `opts` exactly. Every
/// field is passed, because the CLI's defaults (`es`, split on, repo-relative models) are wrong for
/// the daemon and unknown flags are ignored, not rejected. Pinned by a round-trip test against
/// `parse_transcribe_args`. `--yield-to-calls` makes the child drop to background priority while a
/// call is being recorded (see `status::spawn_call_yield`).
fn child_args(dir: &Path, opts: &pipeline::PipelineOpts, base: &Path) -> Vec<String> {
    let path = |p: &Path| p.display().to_string();
    let mut a = vec![
        path(dir),
        "--model".into(),
        path(&opts.model),
        "--lang".into(),
        opts.lang.clone(),
        "--db".into(),
        path(&opts.db_path),
        "--export-dir".into(),
        path(&opts.export_dir),
        "--diarizer-bin".into(),
        path(&opts.diarizer_bin),
        "--diarizer-model".into(),
        path(&opts.diarizer_model),
        "--yield-to-calls".into(),
        path(base),
    ];
    if let Some(t) = &opts.title {
        a.extend(["--title".into(), t.clone()]);
    }
    if let Some(m) = &opts.speaker_model {
        a.extend(["--speaker-model".into(), path(m)]);
    }
    if let Some(r) = &opts.diarizer_rttm {
        a.extend(["--diarizer-rttm".into(), path(r)]);
    }
    if !opts.split {
        a.push("--no-split".into());
    }
    if opts.no_store {
        a.push("--no-store".into());
    }
    a
}

/// Transcribe+store+export one session in a child process (`meetscribe transcribe`), so a whisper
/// `abort()` can never take down the daemon or a recording in progress. The child's log lines go to
/// the daemon's stderr (one log); its stdout — the full transcript — is discarded.
fn run_transcribe_job(job: &TranscribeJob, exe: &Path, base: &Path) {
    log::info!("transcribing {} …", job.dir.display());
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("transcribe")
        .args(child_args(&job.dir, &job.opts, base))
        .current_dir(&job.dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null());
    let outcome = run_child(cmd);
    let shutting_down = SHUTDOWN.load(Ordering::SeqCst);
    match outcome {
        ChildOutcome::Exited(0) => log::info!("transcribed {}", job.dir.display()),
        other => log::error!("transcription failed for {} ({other:?}) — WAVs kept", job.dir.display()),
    }
    match marker_after(outcome, shutting_down) {
        Marker::Keep => log::warn!(
            "{} stays pending; it is transcribed on the next start",
            job.dir.display()
        ),
        Marker::Clear => {
            if let Err(e) = std::fs::remove_file(job.dir.join(maintenance::PENDING_MARKER)) {
                log::warn!("could not clear the pending marker in {}: {e}", job.dir.display());
            }
        }
    }
}

/// Run `cmd` to completion. On shutdown, send the child SIGTERM rather than wait out a long
/// transcription (the marker keeps the session for the next start). The child stays in the
/// daemon's process group, so if the daemon itself is killed, launchd takes the child down too
/// instead of leaving an orphan that could store a meeting the next start transcribes again.
fn run_child(mut cmd: std::process::Command) -> ChildOutcome {
    use std::os::unix::process::ExitStatusExt;
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            log::error!("could not start the transcription child: {e}");
            return ChildOutcome::Failed;
        }
    };
    let mut terminated = false;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return match (status.code(), status.signal()) {
                    (Some(code), _) => ChildOutcome::Exited(code),
                    (None, Some(sig)) => ChildOutcome::Signaled(sig),
                    (None, None) => ChildOutcome::Failed,
                };
            }
            Ok(None) => {}
            Err(e) => {
                log::error!("lost track of the transcription child: {e}");
                let _ = child.kill();
                let _ = child.wait();
                return ChildOutcome::Failed;
            }
        }
        if !terminated && SHUTDOWN.load(Ordering::SeqCst) {
            // SAFETY: signalling our own, not yet reaped, child.
            unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
            terminated = true;
        }
        std::thread::sleep(Duration::from_millis(250));
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

    // Queued even on shutdown: the marker survives the restart and the next start transcribes it.
    let title = format!("{label} — {}", export::fmt_utc(started));
    queue_job(jobs, job_for(cfg, dir, label.to_string(), title));
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

    /// A restart must not lose queued meetings: whatever the last run marked pending and never
    /// reached is found again, oldest first, with the label and title it was queued under.
    #[test]
    fn pending_sessions_are_found_again_oldest_first() {
        let sessions = temp_base("pending");
        for (name, marker) in [
            ("20261001T170000Z", Some("Meet\nMeet — 2026-10-01 17:00 UTC\n")),
            ("20261001T130000Z", Some("Chrome\nChrome — 2026-10-01 13:00 UTC\n")),
            ("20261001T150000Z", None), // transcribed already: no marker
        ] {
            std::fs::create_dir_all(sessions.join(name)).unwrap();
            if let Some(m) = marker {
                std::fs::write(sessions.join(name).join(maintenance::PENDING_MARKER), m).unwrap();
            }
        }
        let found = pending_sessions(&sessions);
        let got: Vec<(String, &str, &str)> = found
            .iter()
            .map(|(d, l, t)| (d.file_name().unwrap().to_string_lossy().into_owned(), l.as_str(), t.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                ("20261001T130000Z".to_string(), "Chrome", "Chrome — 2026-10-01 13:00 UTC"),
                ("20261001T170000Z".to_string(), "Meet", "Meet — 2026-10-01 17:00 UTC"),
            ]
        );
        std::fs::remove_dir_all(&sessions).ok();
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

    /// The daemon's argv for its child must parse back into exactly the options it holds — the CLI
    /// silently ignores unknown flags and its defaults are wrong for the daemon, so drift here would
    /// quietly drop speaker names, the split or the configured language.
    #[test]
    fn child_args_round_trip_through_the_cli_parser() {
        let base = PathBuf::from("/Users/someone/.meetscribe");
        let dir = base.join("sessions/20261001-190352");
        for split in [true, false] {
            let mut j = job(&dir, "Chrome");
            j.opts.lang = "en".to_string();
            j.opts.split = split;
            j.opts.no_store = false;
            j.opts.title = Some("Chrome — 2026-10-01 19:03 UTC".to_string());
            j.opts.speaker_model = Some(base.join("models/speaker/campplus.onnx"));
            let argv = child_args(&dir, &j.opts, &base);
            let parsed = crate::parse_transcribe_args(&argv).unwrap().expect("not --help");
            assert_eq!(parsed.dir, dir);
            assert_eq!(parsed.opts, j.opts, "split = {split}");
            assert_eq!(parsed.yield_to_calls, Some(base.clone()));
        }
    }

    #[test]
    fn the_marker_survives_only_a_shutdown_interrupted_attempt() {
        use ChildOutcome::*;
        for (outcome, shutting_down, want) in [
            (Exited(0), false, Marker::Clear),
            (Exited(0), true, Marker::Clear),
            (Exited(1), false, Marker::Clear),
            (Signaled(libc::SIGABRT), false, Marker::Clear),
            (Failed, false, Marker::Clear),
            (Signaled(libc::SIGTERM), true, Marker::Keep),
            (Signaled(libc::SIGINT), true, Marker::Keep),
            (Exited(1), true, Marker::Keep),
        ] {
            assert_eq!(marker_after(outcome, shutting_down), want, "{outcome:?} shutting_down={shutting_down}");
        }
    }

    /// The reason transcription is a child process: whisper.cpp has aborted before (`34a8290`).
    /// An aborting child must be reported, not take this process with it.
    #[test]
    fn an_aborting_child_is_reported_and_the_parent_survives() {
        let mut cmd = std::process::Command::new("/bin/sh");
        cmd.args(["-c", "kill -ABRT $$"]);
        assert_eq!(run_child(cmd), ChildOutcome::Signaled(libc::SIGABRT));
        let mut ok = std::process::Command::new("/bin/sh");
        ok.args(["-c", "exit 0"]);
        assert_eq!(run_child(ok), ChildOutcome::Exited(0));
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
