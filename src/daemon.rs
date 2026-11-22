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
//! With `[live] enabled`, each call also gets a live reader (`meetscribe live`, a child of this
//! process, see [`LiveReader`]). It only reads what capture has checkpointed; nothing about a
//! recording waits for it or depends on it.
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

/// How long after hangup a live reader may go on finishing its last windows before it is killed.
const READER_GRACE: Duration = Duration::from_secs(300);

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

#[derive(Clone)]
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
    /// `[live]` as it is in force (`config::LiveSettings::effective`).
    live: config::LiveEffective,
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
                    // Absolute now: the transcription child runs in the session dir, so a relative
                    // path would resolve somewhere else than the startup check saw.
                    model_override = Some(std::path::absolute(v).unwrap_or_else(|_| PathBuf::from(v)));
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
        live: file_cfg.live.effective(),
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
        "live reader: {}; storing the live transcript at hangup: {}; live streams kept {}",
        if cfg.live.enabled { "on" } else { "off" },
        if cfg.live.promote { "on" } else { "off" },
        if cfg.live.retain_days == 0 { "for ever".to_string() } else { format!("{} day(s)", cfg.live.retain_days) }
    );
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
    anyhow::ensure!(board.instance.is_some(), "daemon: cannot verify its process start identity");
    board.update(|a| a.paused = was_paused);
    board.refresh_counts(&cfg.sessions_dir);
    // Transcription runs in a child process of this very binary (`meetscribe transcribe`).
    let exe = std::env::current_exe().context("daemon: cannot resolve its own executable")?;
    // The live reader is this binary too: installed, signed, and not replaced by a `cargo build`.
    let mut reader = LiveReader::new(cfg.live.enabled.then(|| {
        let (exe, cfg) = (exe.clone(), cfg.clone());
        Box::new(move |dir: &Path| {
            let mut command = std::process::Command::new(&exe);
            // stderr is inherited: the reader's log lines join the daemon's, as the transcription
            // child's do. Its stdout would be the transcript.
            command.arg("live").args(reader_args(dir, &cfg, std::process::id()))
                .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null());
            command
        }) as ReaderCommand
    }));
    let child_base = cfg.base.clone();
    let recovery_cfg = cfg.clone();
    let finished_board = Arc::clone(&board);
    let scan_board = Arc::clone(&board);
    let finished_sessions = cfg.sessions_dir.clone();
    let (jobs, transcriber) = spawn_transcriber(
        Arc::clone(&board),
        move |job| { run_transcribe_job(job, &exe, &child_base); finished_board.refresh_counts(&finished_sessions); },
        move || { scan_board.refresh_counts(&recovery_cfg.sessions_dir); pending_sessions(&recovery_cfg.sessions_dir).into_iter()
            .map(|(dir, label, title)| job_for(&recovery_cfg, dir, label, title)).collect() },
    ).context("daemon: could not start the transcription worker")?;
    for (dir, label, title) in pending_sessions(&cfg.sessions_dir) {
        log::info!("resuming a transcription the last run did not reach: {}", dir.display());
        queue_job(&jobs, job_for(&cfg, dir, label, title));
    }

    // Consecutive capture-start failures for the currently-detected app — throttles the error log
    // and backs off, so a persistently-denied grant doesn't spin+spam at the poll rate.
    let mut capture_fail_streak = 0u32;
    while !board.stopping() {
        // First thing on every turn, so that no way round the loop skips it: the paused branch
        // and the back-off after a capture that could not start both `continue` from below.
        log_reader(&reader.tick(Instant::now()));
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
            sleep_interruptible(POLL_INTERVAL, &board);
            continue;
        }

        match detector.active_app() {
            Ok(Some(app)) => {
                board.update(|a| a.recording = Some(detect::app_label(&app).to_string()));
                let recorded = record_and_queue(&cfg, &app, &jobs, &board, &mut reader,
                    |dir, stop, managed| session::run_capture(dir, stop, None, managed));
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
                        sleep_interruptible(backoff, &board);
                        continue;
                    }
                }
            }
            Ok(None) => capture_fail_streak = 0,
            Err(e) => log::warn!("detector poll failed (transient): {e}"),
        }
        // Interruptible idle wait.
        sleep_interruptible(POLL_INTERVAL, &board);
    }

    // Before the join, which can last as long as a transcription: no reader outlives the loop,
    // whether it ended by a signal, while paused, during a back-off or after `--once`.
    log_reader(&reader.stop());
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
    pending: u64,
    attention: u64,
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
type Published = (Shown, u64, u64);
struct StatusBoard {
    base: PathBuf,
    instance: Option<status::Instance>,
    inner: Mutex<(Activity, Option<Published>, i64)>,
}

impl StatusBoard {
    fn new(base: PathBuf) -> Self {
        Self {
            base,
            instance: status::Instance::current(),
            inner: Mutex::new((Activity::default(), None, 0)),
        }
    }

    fn update(&self, change: impl FnOnce(&mut Activity)) {
        // A poisoned lock only means another thread panicked mid-update; the data is still usable.
        let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let (activity, last, since) = &mut *guard;
        change(activity);
        let shown = activity.shown();
        let published = (shown.clone(), activity.pending, activity.attention);
        if last.as_ref() != Some(&published) {
            let now = crate::now_epoch();
            if last.as_ref().is_none_or(|last| last.0 != shown) { *since = now; }
            let snapshot = status::Status { state: shown.0, app: shown.1, since_epoch: *since,
                updated_epoch: now, pid: std::process::id(), instance: self.instance.clone(),
                pending: activity.pending, attention: activity.attention };
            match snapshot.write(&self.base) {
                Ok(()) => {
                    log::info!(target: "meetscribe::event", "{}", serde_json::json!({"component":"daemon", "event":"status_published",
                        "instance_id":self.instance.as_ref().map(|own| own.nonce.as_str()), "daemon_state":snapshot.state,
                        "pending":snapshot.pending, "attention":snapshot.attention, "outcome":"ok"}));
                    *last = Some(published);
                }
                Err(error) => log::warn!("could not write status file: {error}"),
            }
        }
    }

    fn stopping(&self) -> bool {
        if self.instance.as_ref().is_some_and(|own| status::stop_requested(&self.base, own))
            && !SHUTDOWN.swap(true, Ordering::SeqCst) {
            log::info!(target: "meetscribe::event", "{}", serde_json::json!({"component":"control", "event":"stop_received",
                "instance_id":self.instance.as_ref().map(|own| own.nonce.as_str()), "outcome":"ok"}));
        }
        SHUTDOWN.load(Ordering::SeqCst)
    }

    fn refresh_counts(&self, sessions: &Path) { self.refresh_counts_from(std::fs::read_dir(sessions)); }

    fn refresh_counts_from(&self, entries: std::io::Result<impl IntoIterator<Item = std::io::Result<std::fs::DirEntry>>>) {
        let (mut pending, mut attention) = (0u64, 0u64);
        let mut incomplete = false;
        match entries {
            Err(_) => { attention = 1; incomplete = true; }
            Ok(entries) => for entry in entries {
                let entry = match entry { Ok(entry) => entry, Err(_) => { attention += 1; incomplete = true; continue; } };
                if entry.file_name().to_string_lossy().starts_with('.') { continue; }
                match entry.file_type() {
                    Ok(kind) if kind.is_dir() => {},
                    Ok(kind) if !kind.is_symlink() => continue,
                    _ => { attention += 1; incomplete = true; continue; }
                }
                match crate::recovery::State::read(&entry.path()) {
                    Ok(state) if state.phase == crate::recovery::Phase::NeedsAttention || state.unarchived_manual_output => attention += 1,
                    Ok(state) if !matches!(state.phase, crate::recovery::Phase::Complete | crate::recovery::Phase::SkippedShort) => pending += 1,
                    Ok(_) => {}, Err(_) => attention += 1,
                }
            },
        }
        if incomplete {
            log::warn!(target: "meetscribe::event", "{}", serde_json::json!({"component":"daemon", "event":"recovery_scan_failed",
                "outcome":"attention", "error_class":"storage"}));
        }
        self.update(|activity| { activity.pending = pending; activity.attention = attention; });
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
    /// The directory of live streams (`--promote-live`), when the child may store this session's
    /// live transcript instead of running whisper. Beside `opts`, as the pipeline takes it.
    promote_live: Option<PathBuf>,
}

/// Start the worker that runs `run` on each queued job, one at a time, in arrival order. It marks
/// the board `transcribing` while a job runs and clears it only once the queue is empty (so a
/// backlog does not flicker the tray through `idle`). After [`SHUTDOWN`] it skips queued jobs.
/// The worker exits when every `Sender` is dropped.
fn spawn_transcriber(
    board: Arc<StatusBoard>,
    run: impl Fn(&TranscribeJob) + Send + 'static,
    due: impl Fn() -> Vec<TranscribeJob> + Send + 'static,
) -> std::io::Result<(mpsc::Sender<TranscribeJob>, JoinHandle<()>)> {
    spawn_transcriber_loop(board, run, due, crate::now_epoch, Duration::from_secs(5))
}

fn job_identity(job: &TranscribeJob) -> String {
    job.opts.session_key.clone().unwrap_or_else(|| job.dir.display().to_string())
}

fn spawn_transcriber_loop(
    board: Arc<StatusBoard>,
    run: impl Fn(&TranscribeJob) + Send + 'static,
    due: impl Fn() -> Vec<TranscribeJob> + Send + 'static,
    now: impl Fn() -> i64 + Send + 'static,
    poll: Duration,
) -> std::io::Result<(mpsc::Sender<TranscribeJob>, JoinHandle<()>)> {
    use std::collections::{HashMap, HashSet, VecDeque};
    let (tx, rx) = mpsc::channel::<TranscribeJob>();
    let handle = std::thread::Builder::new().name("transcriber".into()).spawn(move || {
        let mut queue = VecDeque::new();
        let mut queued = HashSet::new();
        let mut recheck_after = HashMap::<String, i64>::new();
        loop {
            if queue.is_empty() {
                board.update(|activity| activity.transcribing = None);
                match rx.recv_timeout(poll) {
                    Ok(job) => { queued.insert(job_identity(&job)); queue.push_back(job); }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if !SHUTDOWN.load(Ordering::SeqCst) {
                            for job in due() {
                                let key = job_identity(&job);
                                if recheck_after.get(&key).is_none_or(|deadline| *deadline <= now()) && queued.insert(key) {
                                    queue.push_back(job);
                                }
                            }
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            while let Ok(job) = rx.try_recv() {
                if queued.insert(job_identity(&job)) { queue.push_back(job); }
            }
            if let Some(job) = queue.pop_front() {
                let key = job_identity(&job);
                let eligible = job.opts.session_key.is_none() || crate::recovery::State::read(&job.dir).is_ok_and(|state| state.due(now()));
                if !SHUTDOWN.load(Ordering::SeqCst) && eligible {
                    let attempted_at = now();
                    board.update(|activity| activity.transcribing = Some(job.label.clone()));
                    run(&job);
                    // A busy orphan owns its metadata; do not race it by saving a deadline.
                    // This worker-level delay prevents repeated lock probes/spawns without charging attempts.
                    recheck_after.insert(key.clone(), attempted_at.saturating_add(60));
                }
                queued.remove(&key);
                if crate::recovery::State::read(&job.dir).is_ok_and(|state| matches!(state.phase,
                    crate::recovery::Phase::Complete | crate::recovery::Phase::NeedsAttention | crate::recovery::Phase::SkippedShort)) {
                    recheck_after.remove(&key);
                }
                // Drain arrivals before clearing the board so a known backlog cannot flash idle.
                while let Ok(job) = rx.try_recv() {
                    if queued.insert(job_identity(&job)) { queue.push_back(job); }
                }
            }
        }
        board.update(|activity| activity.transcribing = None);
    })?;
    Ok((tx, handle))
}

fn job_for(cfg: &DaemonConfig, dir: PathBuf, label: String, title: String) -> TranscribeJob {
    let stored = crate::recovery::State::read(&dir).ok();
    let opts = pipeline::PipelineOpts {
        model: cfg.model.clone(),
        lang: cfg.lang.clone(),
        title: Some(stored.as_ref().map(|state| state.title.clone()).unwrap_or(title)),
        db_path: stored.as_ref().map(|state| state.db_path.clone()).unwrap_or_else(|| cfg.db_path.clone()),
        export_dir: stored.as_ref().map(|state| state.export_dir.clone()).unwrap_or_else(|| dir.clone()),
        no_store: false,
        session_key: stored.map(|state| state.key),
        resume: false,
        speaker_model: Some(cfg.speaker_model.clone()),
        split: cfg.split,
        diarizer_bin: cfg.diarizer_bin.clone(),
        diarizer_model: cfg.diarizer_model.clone(),
        diarizer_rttm: None,
    };
    // Only a session with a key has a stream to find and a recorded length to check it against,
    // and the child refuses the flag without one. `live.promote` is already false without
    // `live.enabled`.
    let promote_live = (cfg.live.promote && opts.session_key.is_some()).then(|| crate::live::default_out_dir(&cfg.base));
    TranscribeJob { dir, label, opts, promote_live }
}

/// Mark the session dir pending (best-effort: without the marker the job still runs now, it just
/// would not survive a restart), then hand the job to the worker.
fn queue_job(jobs: &mpsc::Sender<TranscribeJob>, job: TranscribeJob) {
    let title = job.opts.title.as_deref().unwrap_or_default();
    let marker = job.dir.join(maintenance::PENDING_MARKER);
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    match std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&marker) {
        Ok(mut file) => {
            if let Err(error) = write!(file, "{}\n{title}\n", job.label) {
                log::warn!("pending marker write failed; durable session state remains authoritative: {error}");
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            log::info!("existing pending material preserved; durable session state controls recovery");
        }
        Err(error) => log::warn!("pending marker unavailable; durable session state controls recovery: {error}"),
    }
    log::info!("queued {} for transcription", job.dir.display());
    if let Err(mpsc::SendError(job)) = jobs.send(job) {
        log::error!("transcription worker is gone — WAVs kept at {}", job.dir.display());
    }
}

/// Durable managed sessions eligible now, independently of a legacy pending marker.
/// Unverified legacy/corrupt sessions are preserved, never silently adopted or inserted.
fn pending_sessions(sessions_dir: &Path) -> Vec<(PathBuf, String, String)> {
    pending_sessions_at(sessions_dir, crate::now_epoch())
}

fn pending_sessions_at(sessions_dir: &Path, now: i64) -> Vec<(PathBuf, String, String)> {
    let Ok(entries) = std::fs::read_dir(sessions_dir) else { return Vec::new() };
    let mut found: Vec<_> = entries.filter_map(|entry| {
        let dir = entry.ok()?.path();
        let state = crate::recovery::State::read(&dir).ok()?;
        if state.phase == crate::recovery::Phase::Complete
            && std::fs::symlink_metadata(dir.join(maintenance::PENDING_MARKER)).is_ok() {
            let reconciled = (|| -> Result<()> {
                if let Some(session) = crate::recovery::Session::open(&dir)? { session.reconcile_pending()?; }
                Ok(())
            })();
            if reconciled.is_err() { log::warn!("component=recovery event=pending_reconcile_failed"); }
        }
        if !state.due(now) { return None; }
        // A held lock means an owner is at work on this session: the capture loop for the whole
        // of a call, or a transcription started by hand. A capture cut short by a crash holds no
        // lock and stays due. The probe never waits and releases the lock at once.
        if matches!(crate::recovery::Session::open(&dir), Ok(None)) { return None; }
        let label = detect::app_label(&state.app).to_string();
        Some((state.started_at, dir, label, state.title))
    }).collect();
    found.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    found.into_iter().map(|(_, dir, label, title)| (dir, label, title)).collect()
}

/// How the transcription child ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChildOutcome {
    Exited(i32),
    Signaled(i32),
    /// The OS did not start a child, so no child could reserve this attempt.
    LaunchFailed,
    /// A child started, but its final status is unknown; it may have reserved an attempt.
    TrackingLost,
}

#[derive(Debug, PartialEq, Eq)]
enum Marker {
    Keep,
    Clear,
}

/// Exit zero alone is not completion. Retry budgets live in durable session state,
/// not in destructive marker removal after a failed child.
fn marker_after(_outcome: ChildOutcome, archive_proven: bool) -> Marker {
    if archive_proven { Marker::Clear } else { Marker::Keep }
}

/// The `meetscribe transcribe` argv (after the subcommand) that reproduces `opts` exactly. Every
/// field is passed, because the CLI's defaults (`es`, split on, repo-relative models) are wrong for
/// the daemon and unknown flags are ignored, not rejected. Pinned by a round-trip test against
/// `parse_transcribe_args`. `--yield-to-calls` makes the child drop to background priority while a
/// call is being recorded (see `status::spawn_call_yield`). `promote_live` becomes `--promote-live`
/// only for a session with a key: the parser refuses the flag without one, and a child that
/// cannot start transcribes nothing.
fn child_args(dir: &Path, opts: &pipeline::PipelineOpts, base: &Path, promote_live: Option<&Path>) -> Vec<String> {
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
    if let Some(key) = &opts.session_key { a.extend(["--session-key".into(), key.clone()]); }
    if opts.resume { a.push("--resume".into()); }
    if let Some(live) = promote_live.filter(|_| opts.session_key.is_some()) {
        a.extend(["--promote-live".into(), path(live)]);
    }
    a
}

/// Transcribe+store+export one session in a child process (`meetscribe transcribe`), so a whisper
/// `abort()` can never take down the daemon or a recording in progress. The child's log lines go to
/// the daemon's stderr (one log); its stdout — the full transcript — is discarded.
fn run_transcribe_job(job: &TranscribeJob, exe: &Path, base: &Path) {
    run_transcribe_job_at(job, exe, base, crate::now_epoch);
}

fn run_transcribe_job_at(job: &TranscribeJob, exe: &Path, base: &Path, now: impl Fn() -> i64) {
    log::info!("transcribing {} …", job.dir.display());
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("transcribe")
        .args(child_args(&job.dir, &job.opts, base, job.promote_live.as_deref()))
        .current_dir(&job.dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null());
    let outcome = run_child(cmd);
    if outcome == ChildOutcome::LaunchFailed {
        let persist = (|| -> Result<()> {
            let Some(key) = &job.opts.session_key else { return Ok(()); };
            let Some(mut session) = crate::recovery::Session::open(&job.dir)? else { return Ok(()); };
            anyhow::ensure!(session.state.key == *key, "launch failure session identity mismatch");
            let at = now();
            if !session.state.due(at) { return Ok(()); }
            // A previous child may already own the durable reservation. Settle that attempt,
            // rather than charging it again just because this replacement could not launch.
            if session.state.in_flight {
                return session.failed("interrupted_attempt", at);
            }
            session.reserve(at, false)?;
            session.failed("child_launch", at)
        })();
        if persist.is_err() {
            log::error!(target: "meetscribe::event", "{}", serde_json::json!({
                "component":"recovery", "event":"launch_failure_persist_failed", "outcome":"error", "error_class":"storage"
            }));
        }
    }
    let archive_proven = (|| -> Result<bool> {
        let Some(session) = crate::recovery::Session::open(&job.dir)? else { return Ok(false); };
        if job.opts.session_key.as_ref().is_some_and(|key| *key != session.state.key) { return Ok(false); }
        session.reconcile_pending() // Lock remains held through proof and marker removal.
    })().unwrap_or_else(|_| { log::warn!("component=recovery event=pending_reconcile_failed"); false });
    match outcome {
        ChildOutcome::Exited(0) if archive_proven => log::info!("transcribed {}", job.dir.display()),
        ChildOutcome::Exited(0) => log::warn!("component=recovery event=child_success_without_archive_proof"),
        other => log::error!("transcription failed for {} ({other:?}) — WAVs kept", job.dir.display()),
    }
    match marker_after(outcome, archive_proven) {
        Marker::Keep => log::warn!(
            "{} stays protected; durable recovery state controls the next attempt",
            job.dir.display()
        ),
        Marker::Clear => log::info!("component=recovery event=completion_reconciled"),
    }
}

/// Run our child to completion. New keyed children hold the session lock and a durable
/// attempt reservation; retries reconcile the committed mapping rather than inserting twice.
/// Legacy unkeyed children do not inherit this guarantee and require upgrade quiescence.
fn run_child(mut cmd: std::process::Command) -> ChildOutcome {
    use std::os::unix::process::ExitStatusExt;
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            log::error!("could not start the transcription child: {e}");
            return ChildOutcome::LaunchFailed;
        }
    };
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return match (status.code(), status.signal()) {
                    (Some(code), _) => ChildOutcome::Exited(code),
                    (None, Some(sig)) => ChildOutcome::Signaled(sig),
                    (None, None) => ChildOutcome::TrackingLost,
                };
            }
            Ok(None) => {}
            Err(e) => {
                log::error!("lost track of the transcription child: {e}");
                let _ = child.kill();
                let _ = child.wait();
                return ChildOutcome::TrackingLost;
            }
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Builds the live reader's command for a session directory.
type ReaderCommand = Box<dyn Fn(&Path) -> std::process::Command>;

/// Why the daemon ended a live reader itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Killed {
    /// The next call started: one whisper context at a time.
    Replaced,
    /// Still running [`READER_GRACE`] after hangup.
    Deadline,
    /// The capture it followed failed, or the daemon is leaving its loop.
    Stopped,
    /// Its status could not be read.
    TrackingLost,
}

impl Killed {
    /// The `reason` of a `live_child_killed` log line (`diagnostics.rs` allows these four).
    fn as_str(self) -> &'static str {
        match self {
            Self::Replaced => "replaced",
            Self::Deadline => "deadline",
            Self::Stopped => "stopped",
            Self::TrackingLost => "tracking_lost",
        }
    }
}

/// What happened to a live reader in one call of a [`LiveReader`] method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReaderChange {
    Started,
    /// The OS did not start it. The recording goes on without a live transcript.
    SpawnFailed,
    /// It ended by itself: with this exit code, or by a signal (`None`).
    Exited(Option<i32>),
    Killed(Killed),
}

/// A [`ReaderChange`] and the session whose reader it happened to.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ReaderReport {
    key: String,
    change: ReaderChange,
}

struct RunningReader {
    key: String,
    child: std::process::Child,
    /// When to kill it; set at hangup.
    deadline: Option<Instant>,
}

/// The one owner of the daemon's live reader: at most one child at any time.
///
/// A reader outlives its call so it can finish its last windows; it is ended by its own exit, by
/// the deadline set at hangup, by the next call, or when the daemon leaves its loop. Nothing here
/// runs during a capture: [`tick`](Self::tick) is called from the idle loop only.
///
/// Every method returns what happened to a child, once; [`log_reader`] writes that down. A
/// daemon that is killed outright runs none of this: the reader notices by itself
/// (`live --owner-pid`).
struct LiveReader {
    /// `None` when `[live] enabled` is false: nothing is ever started.
    command: Option<ReaderCommand>,
    running: Option<RunningReader>,
}

impl LiveReader {
    fn new(command: Option<ReaderCommand>) -> Self {
        Self { command, running: None }
    }

    /// End the reader still held, if any, and start one for the recording in `dir`. A reader
    /// that cannot be started is reported and leaves nothing behind.
    fn start(&mut self, dir: &Path, key: &str) -> Vec<ReaderReport> {
        let mut reports = self.end(Killed::Replaced);
        let Some(command) = &self.command else { return reports; };
        let change = match command(dir).spawn() {
            Ok(child) => {
                self.running = Some(RunningReader { key: key.to_string(), child, deadline: None });
                ReaderChange::Started
            }
            Err(_) => ReaderChange::SpawnFailed,
        };
        reports.push(ReaderReport { key: key.to_string(), change });
        reports
    }

    /// The call ended: the reader has [`READER_GRACE`] from `now` to finish.
    fn hangup(&mut self, now: Instant) {
        if let Some(running) = &mut self.running { running.deadline = Some(now + READER_GRACE); }
    }

    /// Look at the reader without waiting: report its exit, and kill it past its deadline.
    fn tick(&mut self, now: Instant) -> Vec<ReaderReport> {
        let Some(running) = &mut self.running else { return Vec::new(); };
        match running.child.try_wait() {
            Ok(None) if running.deadline.is_some_and(|deadline| now >= deadline) => self.end(Killed::Deadline),
            Ok(None) => Vec::new(),
            Ok(Some(_)) | Err(_) => self.end(Killed::TrackingLost),
        }
    }

    /// End the reader now, whatever it is doing.
    fn stop(&mut self) -> Vec<ReaderReport> {
        self.end(Killed::Stopped)
    }

    /// Let go of the child held, if any: report the exit of one that has ended by itself,
    /// otherwise kill and reap it and report `why`.
    fn end(&mut self, why: Killed) -> Vec<ReaderReport> {
        let Some(mut running) = self.running.take() else { return Vec::new(); };
        let change = match running.child.try_wait() {
            Ok(Some(status)) => ReaderChange::Exited(status.code()),
            Ok(None) => { let _ = running.child.kill(); let _ = running.child.wait(); ReaderChange::Killed(why) }
            Err(_) => { let _ = running.child.kill(); let _ = running.child.wait(); ReaderChange::Killed(Killed::TrackingLost) }
        };
        vec![ReaderReport { key: running.key, change }]
    }
}

impl Drop for LiveReader {
    /// A panic that unwinds must not leave a reader running, like `diar::Running`.
    fn drop(&mut self) {
        log_reader(&self.stop());
    }
}

/// The `meetscribe live` argv (after the subcommand) for the daemon's reader of `dir`. Every
/// option is passed, as for the transcription child: the reader's own defaults follow a base
/// directory and a config file it must not have to find again. Pinned by a round-trip test
/// against `live::parse_args`.
fn reader_args(dir: &Path, cfg: &DaemonConfig, owner_pid: u32) -> Vec<String> {
    let path = |p: &Path| p.display().to_string();
    vec![
        path(dir),
        "--model".into(),
        path(&cfg.model),
        "--lang".into(),
        cfg.lang.clone(),
        "--out".into(),
        path(&crate::live::default_out_dir(&cfg.base)),
        "--base".into(),
        path(&cfg.base),
        "--max-window-secs".into(),
        cfg.live.max_window_secs.to_string(),
        "--owner-pid".into(),
        owner_pid.to_string(),
    ]
}

/// The log line of one [`ReaderReport`]: an event name, the session, an outcome and, for a
/// reader that exited with one, its exit code; for one the daemon ended, why. Nothing else
/// about the process.
fn reader_event(report: &ReaderReport) -> serde_json::Value {
    let (event, outcome) = match report.change {
        ReaderChange::Started => ("live_child_started", "ok"),
        ReaderChange::SpawnFailed => ("live_child_started", "error"),
        ReaderChange::Exited(Some(0)) => ("live_child_exited", "ok"),
        ReaderChange::Exited(_) => ("live_child_exited", "error"),
        ReaderChange::Killed(_) => ("live_child_killed", "attention"),
    };
    let mut line = serde_json::json!({"component": "daemon", "event": event, "session_id": report.key, "outcome": outcome});
    if report.change == ReaderChange::SpawnFailed { line["error_class"] = serde_json::json!("child_launch"); }
    if let ReaderChange::Exited(Some(code)) = report.change { line["exit_code"] = serde_json::json!(code); }
    if let ReaderChange::Killed(why) = report.change { line["reason"] = serde_json::json!(why.as_str()); }
    line
}

fn log_reader(reports: &[ReaderReport]) {
    for report in reports {
        log::info!(target: "meetscribe::event", "{}", reader_event(report));
    }
}

/// Record one meeting (`app` holds the mic), then gate it and queue it for transcription. Returns
/// as soon as capture ends, so the caller is back to watching for the next call while whisper
/// works. Returns `Err` ONLY if capture failed to start/run — the caller throttles + backs off, so
/// a persistently-denied grant (common on first run) doesn't spin+spam at the poll rate.
///
/// `capture` is `session::run_capture` (without its debug hook); a test passes its own. `reader`
/// follows the recording when `[live] enabled`: started here before capture, stopped at once
/// when the capture fails, and given its deadline when the call ends.
fn record_and_queue(
    cfg: &DaemonConfig,
    app: &str,
    jobs: &mpsc::Sender<TranscribeJob>,
    board: &StatusBoard,
    reader: &mut LiveReader,
    capture: impl FnOnce(&Path, Box<dyn FnMut() -> bool + '_>, Option<&mut crate::recovery::Session>) -> Result<session::CaptureSummary>,
) -> Result<()> {
    let started = crate::now_epoch();
    let suffix = crate::recovery::new_identity();
    let dir = cfg.sessions_dir.join(format!("{}-{}", export::stamp_compact(started), &suffix[..8]));
    let label = detect::app_label(app);
    let title = format!("{label} — {}", export::fmt_utc(started));
    let mut managed = crate::recovery::Session::begin(&dir, &cfg.db_path, &dir, app, &title, started)?;
    log::info!("meeting detected — {label} ({app}) took the mic → recording to {}", dir.display());
    // Before capture starts, and never a reason to fail it. Ending a reader left from a call
    // less than READER_GRACE ago costs one kill and one wait here.
    log_reader(&reader.start(&dir, &managed.state.key));

    // Stop predicate: the app released the mic for MEETING_END_DEBOUNCE, or we were signalled.
    // The detector is polled at most every DETECT_POLL_DURING_CAPTURE (the capture loop ticks far
    // faster). A transient detector error keeps the last state (don't drop a live recording).
    // `released_since` is Some only while the app has been off the mic; it fully encodes the
    // "released?" state (no separate bool needed).
    let detector = MeetingDetector::with_allowlist(cfg.allowlist.clone());
    let mut last_poll = Instant::now();
    let mut released_since: Option<Instant> = None;
    let stop = move || -> bool {
        if board.stopping() {
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

    let summary = match capture(&dir, Box::new(stop), Some(&mut managed)) {
        Ok(summary) => {
            if let Err(error) = managed.finish_capture(summary.duration_secs, cfg.min_secs) {
                log_reader(&reader.stop());
                return Err(error);
            }
            summary
        }
        Err(error) => {
            log_reader(&reader.stop());
            if let Err(state_error) = managed.capture_failed() {
                log::error!("component=recovery event=capture_state_write_failed error={state_error:#}");
            }
            return Err(error.context("managed capture failed; files preserved"));
        }
    };
    drop(managed); // Release capture ownership before the processing child may acquire it.
    // The reader still has its last windows to decode. Set before either return below.
    reader.hangup(Instant::now());
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

    let mut job = job_for(cfg, dir, label.to_string(), title);
    // `--once` stops this call's reader as soon as the loop is left, before it can finish its
    // stream. Offering that stream would only make the child wait for a reader that is gone.
    if cfg.once {
        job.promote_live = None;
    }
    // Queued even on shutdown: the marker survives the restart and the next start transcribes it.
    queue_job(jobs, job);
    Ok(())
}

/// Disk/log hygiene: prune old session dirs and old live streams + rotate the daemon logs (all
/// config-driven, all best-effort — a failure is logged, never fatal). Runs at startup and once
/// per day, never during this daemon's own capture.
fn run_maintenance(cfg: &DaemonConfig) {
    match crate::maintenance::prune_sessions(&cfg.sessions_dir, cfg.sessions_days, crate::now_epoch())
    {
        Ok(r) if r.removed > 0 => {
            log::info!("retention: pruned {} old session(s) (kept {})", r.removed, r.kept);
        }
        Ok(_) => {}
        Err(e) => log::warn!("retention: session prune failed: {e}"),
    }
    // Independent of `live.enabled`: a stream written by a hand-started reader is pruned too.
    match crate::maintenance::prune_live(&crate::live::default_out_dir(&cfg.base), cfg.live.retain_days, crate::now_epoch()) {
        Ok(r) if r.removed > 0 => log::info!("retention: pruned {} old live stream(s) (kept {})", r.removed, r.kept),
        Ok(_) => {}
        Err(e) => log::warn!("retention: live stream prune failed: {e}"),
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
fn sleep_interruptible(total: Duration, board: &StatusBoard) {
    let step = Duration::from_millis(200);
    let mut left = total;
    while left > Duration::ZERO && !board.stopping() {
        let s = step.min(left);
        std::thread::sleep(s);
        left = left.saturating_sub(s);
    }
}

/// A daemon configuration under `base`, with the values `run_daemon` gives a file that sets
/// nothing but `[live]`.
#[cfg(test)]
fn test_config(base: &Path, live: config::LiveEffective) -> DaemonConfig {
    let file = Config::default();
    DaemonConfig {
        base: base.to_path_buf(), sessions_dir: base.join("sessions"), logs_dir: base.join("logs"), db_path: base.join("meetscribe.db"),
        model: base.join("models/ggml-large-v3.bin"), speaker_model: base.join(crate::SPEAKER_MODEL_REL), split: file.speakers.split,
        diarizer_bin: base.join(crate::DIARIZER_BIN_REL), diarizer_model: base.join(crate::DIARIZER_MODEL_REL), lang: file.daemon.lang.clone(),
        min_secs: file.daemon.min_secs, once: false, allowlist: file.effective_allowlist(),
        sessions_days: file.retention.sessions_days, log_max_mb: file.retention.log_max_mb, live,
    }
}

#[cfg(test)]
#[path = "audit_pipeline.rs"]
mod audit_pipeline;

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
                session_key: None,
                resume: false,
                speaker_model: None,
                split: false,
                diarizer_bin: dir.join("diarizer"),
                diarizer_model: dir.join("diarizer.onnx"),
                diarizer_rttm: None,
            },
            promote_live: None,
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
        // Legacy markers alone are quarantined; only versioned metadata supplies identity.
        assert!(pending_sessions(&sessions).is_empty());
        for (name, app, title, started) in [
            ("later-directory", "us.zoom.xos", "Later release sync", 200),
            ("earlier-directory", "com.google.Chrome", "Earlier release sync", 100),
        ] {
            let dir = sessions.join(name);
            let mut state = crate::recovery::Session::begin(&dir, &sessions.join("archive.db"), &dir, app, title, started).unwrap();
            state.start_roll(0, 16_000, 0.0).unwrap();
            state.checkpoint(0, 16_000, 16_000).unwrap();
            state.finish_capture(1.0, 0.1).unwrap();
        }
        let found = pending_sessions(&sessions);
        let got: Vec<(String, &str, &str)> = found
            .iter()
            .map(|(d, l, t)| (d.file_name().unwrap().to_string_lossy().into_owned(), l.as_str(), t.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                ("earlier-directory".to_string(), "Chrome", "Earlier release sync"),
                ("later-directory".to_string(), "Zoom", "Later release sync"),
            ]
        );
        std::fs::remove_dir_all(&sessions).ok();
    }

    /// A recording in progress holds its session lock, so the scan leaves it alone. The same
    /// session with the lock free and the state still `Recording` is a capture cut short by a
    /// crash, and that one is due for recovery.
    #[test]
    fn the_scan_skips_a_session_whose_lock_is_held() {
        let sessions = temp_base("scan-held");
        let dir = sessions.join("recording");
        let mut owner = crate::recovery::Session::begin(&dir, &sessions.join("archive.db"), &dir,
            "com.google.Chrome", "Scan fixture", 100).unwrap();
        owner.start_roll(0, 16_000, 0.0).unwrap();
        owner.checkpoint(0, 16_000, 16_000).unwrap();
        assert_eq!(owner.state.phase, crate::recovery::Phase::Recording);
        assert!(pending_sessions_at(&sessions, 1000).is_empty(), "a recording in progress was offered for transcription");
        drop(owner);
        let found = pending_sessions_at(&sessions, 1000);
        let names: Vec<_> = found.iter().map(|(dir, _, _)| dir.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, ["recording"], "an interrupted recording must stay recoverable");
        std::fs::remove_dir_all(&sessions).ok();
    }

    /// The same rule through the worker: no transcription child is started for a recording in
    /// progress, however often the idle scan runs, and one is started once the lock is free.
    #[test]
    fn the_worker_starts_no_job_for_a_recording_in_progress() {
        let base = temp_base("worker-held");
        let sessions = base.join("sessions");
        let dir = sessions.join("recording");
        let mut owner = crate::recovery::Session::begin(&dir, &base.join("archive.db"), &dir,
            "com.google.Chrome", "Worker fixture", 100).unwrap();
        owner.start_roll(0, 16_000, 0.0).unwrap();
        owner.checkpoint(0, 16_000, 16_000).unwrap();
        let key = owner.state.key.clone();
        let (ran_tx, ran_rx) = mpsc::channel();
        let board = Arc::new(StatusBoard::new(base.clone()));
        let (jobs, worker) = spawn_transcriber_loop(board, move |job| ran_tx.send(job.dir.clone()).unwrap(), move || {
            pending_sessions_at(&sessions, 1000).into_iter().map(|(dir, label, _)| {
                let mut item = job(&dir, &label);
                item.opts.session_key = Some(key.clone()); item.opts.no_store = false; item
            }).collect()
        }, || 1000, Duration::from_millis(5)).unwrap();
        // About thirty idle scans while the recording holds its lock.
        assert!(ran_rx.recv_timeout(Duration::from_millis(150)).is_err(), "a job ran for a recording in progress");
        drop(owner);
        assert_eq!(ran_rx.recv_timeout(Duration::from_secs(3)).unwrap().file_name().unwrap(), "recording");
        assert!(ran_rx.recv_timeout(Duration::from_millis(30)).is_err(), "the same session ran twice inside its recheck delay");
        drop(jobs); worker.join().unwrap();
        std::fs::remove_dir_all(base).unwrap();
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
        let live = base.join("live");
        // A session with a key (what the daemon records) and one without (a legacy directory),
        // each with and without the live transcript on offer.
        for (split, key, promote) in [(true, None, None), (false, None, None), (true, Some(KEY), None),
            (false, Some(KEY), Some(live.as_path())), (true, Some(KEY), Some(live.as_path())), (true, None, Some(live.as_path()))] {
            let mut j = job(&dir, "Chrome");
            j.opts.lang = "en".to_string();
            j.opts.split = split;
            j.opts.no_store = false;
            j.opts.session_key = key.map(String::from);
            j.opts.title = Some("Chrome — 2026-10-01 19:03 UTC".to_string());
            j.opts.speaker_model = Some(base.join("models/speaker/campplus.onnx"));
            let argv = child_args(&dir, &j.opts, &base, promote);
            let parsed = crate::parse_transcribe_args(&argv).unwrap().expect("not --help");
            assert_eq!(parsed.dir, dir);
            assert_eq!(parsed.opts, j.opts, "split = {split}");
            assert_eq!(parsed.yield_to_calls, Some(base.clone()));
            // Without a key the parser refuses the flag, so it is never written for such a job.
            assert_eq!(parsed.promote_live.as_deref(), promote.filter(|_| key.is_some()), "key = {key:?}");
            assert_eq!(argv.iter().any(|arg| arg == "--promote-live"), promote.is_some() && key.is_some());
        }
    }

    const KEY: &str = "a6e79ba0314c4cc2b647f4d25b327011";
    const OTHER_KEY: &str = "b6e79ba0314c4cc2b647f4d25b327011";

    fn live(text: &str) -> config::LiveEffective {
        toml::from_str::<Config>(text).unwrap().live.effective()
    }
    const LIVE_ON: &str = "[live]\nenabled = true\npromote = true\n";

    /// A finished recording of `secs` under `dir`, as the capture loop leaves it.
    fn recorded_session(dir: &Path, db: &Path, secs: f64) -> String {
        let mut session = crate::recovery::Session::begin(dir, db, dir, "com.google.Chrome", "Flag fixture", 1_762_880_000).unwrap();
        session.start_roll(0, 16_000, 0.0).unwrap();
        session.checkpoint(0, 16_000, 16_000).unwrap();
        session.finish_capture(secs, 20.0).unwrap();
        session.state.key.clone()
    }

    /// `promote` needs `enabled`, and a session needs a key: the two rules of `job_for`, from
    /// the config text to the child's parsed arguments.
    #[test]
    fn the_promotion_flag_needs_promotion_in_effect_and_a_session_key() {
        let base = temp_base("promote-flag");
        let sessions = base.join("sessions");
        let keyed = sessions.join("keyed");
        let key = recorded_session(&keyed, &base.join("meetscribe.db"), 30.0);
        for (text, promoted) in [(LIVE_ON, true), ("[live]\npromote = true\n", false), ("[live]\nenabled = true\n", false), ("", false)] {
            let cfg = test_config(&base, live(text));
            let job = job_for(&cfg, keyed.clone(), "Chrome".into(), "a title the stored state overrides".into());
            assert_eq!(job.opts.session_key.as_deref(), Some(key.as_str()));
            assert_eq!(job.promote_live, promoted.then(|| base.join("live")), "config: {text:?}");
            let argv = child_args(&job.dir, &job.opts, &base, job.promote_live.as_deref());
            let parsed = crate::parse_transcribe_args(&argv).unwrap().expect("not --help");
            assert_eq!((parsed.opts, parsed.promote_live), (job.opts, job.promote_live), "config: {text:?}");
        }
        // A directory with no readable state has no key: no flag, whatever the config says.
        let legacy = sessions.join("legacy");
        std::fs::create_dir_all(&legacy).unwrap();
        let job = job_for(&test_config(&base, live(LIVE_ON)), legacy, "Chrome".into(), "Legacy".into());
        assert_eq!((job.opts.session_key, job.promote_live), (None, None));
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// The recovery scan builds its jobs through `job_for` too, and a session keeps the database
    /// it was bound to: such a job carries the flag and that database, not the daemon's.
    #[test]
    fn a_recovered_session_bound_to_another_database_carries_the_flag_and_its_database() {
        let base = temp_base("promote-recovered");
        let sessions = base.join("sessions");
        let other_db = base.join("elsewhere/archive.db");
        let key = recorded_session(&sessions.join("bound"), &other_db, 30.0);
        let cfg = test_config(&base, live(LIVE_ON));
        let jobs: Vec<TranscribeJob> = pending_sessions_at(&sessions, crate::now_epoch()).into_iter()
            .map(|(dir, label, title)| job_for(&cfg, dir, label, title)).collect();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].opts.session_key.as_deref(), Some(key.as_str()));
        assert_eq!((jobs[0].opts.db_path.clone(), jobs[0].promote_live.clone()), (other_db, Some(base.join("live"))));
        assert_ne!(jobs[0].opts.db_path, cfg.db_path);
        std::fs::remove_dir_all(&base).unwrap();
    }

    // ---- The live reader's owner. The child is `/bin/sleep` or `/bin/sh`, never `meetscribe live`.

    fn reader_running(program: &'static str, args: &'static [&'static str]) -> LiveReader {
        LiveReader::new(Some(Box::new(move |_: &Path| {
            let mut command = std::process::Command::new(program);
            command.args(args);
            command
        })))
    }
    fn sleeper() -> LiveReader { reader_running("/bin/sleep", &["60"]) }

    /// A reader that writes its process id into `pid_file` and then sleeps, for a test that must
    /// look at the process after the owner has let go of it.
    fn announcing_sleeper(pid_file: PathBuf) -> LiveReader {
        LiveReader::new(Some(Box::new(move |_: &Path| {
            let mut command = std::process::Command::new("/bin/sh");
            command.args(["-c", "echo $$ > \"$0\"; exec sleep 60"]).arg(&pid_file);
            command
        })))
    }

    fn announced_pid(pid_file: &Path) -> u32 {
        let waiting = Instant::now();
        loop {
            if let Some(pid) = std::fs::read_to_string(pid_file).ok().and_then(|text| text.trim().parse().ok()) { return pid; }
            assert!(waiting.elapsed() < Duration::from_secs(20), "the reader never announced itself");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn unspawnable() -> LiveReader { reader_running("/nonexistent-meetscribe-reader-xyz", &[]) }

    fn pid_of(reader: &LiveReader) -> u32 { reader.running.as_ref().expect("a reader is held").child.id() }

    /// Whether a process still exists. A child that was killed AND reaped does not; one that was
    /// only killed would still be there, as a zombie.
    fn exists(pid: u32) -> bool {
        // SAFETY: signal 0 checks for the process and sends nothing.
        (unsafe { libc::kill(pid as libc::pid_t, 0) }) == 0
    }

    fn report(key: &str, change: ReaderChange) -> ReaderReport { ReaderReport { key: key.to_string(), change } }

    #[test]
    fn the_reader_owner_holds_one_child_and_reports_each_end_once() {
        use ReaderChange::{Killed as Ended, Started};
        let dir = Path::new("/nonexistent-meetscribe-session-xyz");
        let now = Instant::now();
        let mut reader = sleeper();
        assert!(reader.tick(now).is_empty() && reader.stop().is_empty(), "nothing is held yet");
        reader.hangup(now);

        assert_eq!(reader.start(dir, KEY), [report(KEY, Started)]);
        let first = pid_of(&reader);
        assert!(exists(first));
        // The next call: the reader of the previous one is ended first, and reported once.
        assert_eq!(reader.start(dir, OTHER_KEY), [report(KEY, Ended(Killed::Replaced)), report(OTHER_KEY, Started)]);
        assert!(!exists(first), "two readers at once");
        let second = pid_of(&reader);

        // During the call there is no deadline; after hangup it is 300 s away.
        assert!(reader.tick(now + 10 * READER_GRACE).is_empty(), "a reader was ended during its call");
        reader.hangup(now);
        assert!(reader.tick(now + READER_GRACE - Duration::from_millis(1)).is_empty());
        assert!(exists(second));
        assert_eq!(reader.tick(now + READER_GRACE), [report(OTHER_KEY, Ended(Killed::Deadline))]);
        assert!(!exists(second));
        assert!(reader.tick(now + 2 * READER_GRACE).is_empty() && reader.stop().is_empty(), "an end was reported twice");

        assert_eq!(reader.start(dir, KEY), [report(KEY, Started)]);
        let third = pid_of(&reader);
        assert_eq!(reader.stop(), [report(KEY, Ended(Killed::Stopped))]);
        assert!(!exists(third));
        assert!(reader.stop().is_empty());

        // A panic that unwinds drops the owner: no reader is left.
        assert_eq!(reader.start(dir, KEY), [report(KEY, Started)]);
        let fourth = pid_of(&reader);
        drop(reader);
        assert!(!exists(fourth));
    }

    #[test]
    fn a_reader_that_ends_by_itself_is_reported_once_with_how_it_ended() {
        let dir = Path::new("/nonexistent-meetscribe-session-xyz");
        let script: [(&'static [&'static str], ReaderChange); 3] = [(&["-c", "exit 0"], ReaderChange::Exited(Some(0))),
            (&["-c", "exit 3"], ReaderChange::Exited(Some(3))), (&["-c", "kill -TERM $$"], ReaderChange::Exited(None))];
        for (args, change) in script {
            let mut reader = reader_running("/bin/sh", args);
            assert_eq!(reader.start(dir, KEY), [report(KEY, ReaderChange::Started)]);
            let pid = pid_of(&reader);
            let waiting = Instant::now();
            let reports = loop {
                let reports = reader.tick(Instant::now());
                if !reports.is_empty() { break reports; }
                assert!(waiting.elapsed() < Duration::from_secs(10), "the reader's exit was never noticed");
                std::thread::sleep(Duration::from_millis(5));
            };
            assert_eq!(reports, [report(KEY, change)]);
            assert!(!exists(pid), "the reader was not reaped");
            assert!(reader.tick(Instant::now()).is_empty() && reader.stop().is_empty(), "an exit was reported twice");
        }
    }

    #[test]
    fn a_reader_that_cannot_be_started_is_reported_and_leaves_nothing() {
        let dir = Path::new("/nonexistent-meetscribe-session-xyz");
        let mut reader = unspawnable();
        assert_eq!(reader.start(dir, KEY), [report(KEY, ReaderChange::SpawnFailed)]);
        assert!(reader.running.is_none());
        reader.hangup(Instant::now());
        assert!(reader.tick(Instant::now() + READER_GRACE).is_empty() && reader.stop().is_empty());
        // With `[live] enabled = false` there is no command at all: every method does nothing.
        let mut off = LiveReader::new(None);
        assert!(off.start(dir, KEY).is_empty() && off.running.is_none());
        off.hangup(Instant::now());
        assert!(off.tick(Instant::now() + READER_GRACE).is_empty() && off.stop().is_empty());
    }

    /// The reader's argv must parse back into what the daemon holds: the reader's own defaults
    /// (a base directory, a config file, 12 s) are not the daemon's values.
    #[test]
    fn reader_args_round_trip_through_the_readers_parser() {
        let base = PathBuf::from("/Users/someone/.meetscribe");
        let dir = base.join("sessions/20261001-190352");
        let mut cfg = test_config(&base, live("[live]\nenabled = true\nmax_window_secs = 20\n"));
        cfg.lang = "pt".to_string();
        cfg.model = PathBuf::from("/models/other.bin");
        let argv = reader_args(&dir, &cfg, 4242);
        // Parsed against another base directory, so that nothing can come from a default.
        let (opts, session) = crate::live::parse_args(&argv, false, PathBuf::from("/nonexistent-meetscribe-base-xyz")).unwrap().expect("not --help");
        assert_eq!(session, Some(dir));
        assert_eq!((opts.model, opts.lang.as_str(), opts.out_dir, opts.base), (cfg.model.clone(), "pt", base.join("live"), base.clone()));
        assert_eq!((opts.max_window_ms, opts.owner_pid, opts.print), (20_000, Some(4242), false));
        // A window the reader would refuse never reaches it.
        let short = test_config(&base, live("[live]\nenabled = true\nmax_window_secs = 1\n"));
        assert!(crate::live::parse_args(&reader_args(&base, &short, 1), false, base.clone()).is_ok());
    }

    /// Each thing that can happen to a reader, as the log carries it: through the real formatter,
    /// which withholds an event with a name or a value it does not know.
    #[test]
    fn every_reader_report_is_a_log_line_the_formatter_accepts() {
        use ReaderChange::{Exited, SpawnFailed, Started};
        for (change, event, outcome, exit_code, error_class, reason) in [
            (Started, "live_child_started", "ok", None, None, None),
            (SpawnFailed, "live_child_started", "error", None, Some("child_launch"), None),
            (Exited(Some(0)), "live_child_exited", "ok", Some(0), None, None),
            (Exited(Some(3)), "live_child_exited", "error", Some(3), None, None),
            (Exited(Some(255)), "live_child_exited", "error", Some(255), None, None),
            (Exited(None), "live_child_exited", "error", None, None, None),
            (ReaderChange::Killed(Killed::Replaced), "live_child_killed", "attention", None, None, Some("replaced")),
            (ReaderChange::Killed(Killed::Deadline), "live_child_killed", "attention", None, None, Some("deadline")),
            (ReaderChange::Killed(Killed::Stopped), "live_child_killed", "attention", None, None, Some("stopped")),
            (ReaderChange::Killed(Killed::TrackingLost), "live_child_killed", "attention", None, None, Some("tracking_lost")),
        ] {
            let line = crate::diagnostics::formatted_event(&reader_event(&report(KEY, change)));
            assert_eq!((line["component"].as_str(), line["event"].as_str(), line["session_id"].as_str(), line["outcome"].as_str()),
                (Some("daemon"), Some(event), Some(KEY), Some(outcome)), "{change:?}");
            assert_eq!((line["exit_code"].as_u64(), line["error_class"].as_str()), (exit_code, error_class), "{change:?}");
            assert_eq!(line["reason"].as_str(), reason, "{change:?}");
        }
    }

    // ---- `record_and_queue` with a capture of the test's own.

    struct Call { base: PathBuf, cfg: DaemonConfig, board: StatusBoard, jobs: mpsc::Sender<TranscribeJob>, queued: mpsc::Receiver<TranscribeJob> }

    fn call(tag: &str, live_text: &str) -> Call {
        let base = temp_base(tag);
        let (jobs, queued) = mpsc::channel();
        Call { cfg: test_config(&base, live(live_text)), board: StatusBoard::new(base.clone()), base, jobs, queued }
    }

    /// What capture leaves for a call that recorded `secs`: one roll, one checkpoint, a summary.
    fn captured(managed: Option<&mut crate::recovery::Session>, secs: f64) -> Result<session::CaptureSummary> {
        let session = managed.expect("the daemon captures into a managed session");
        session.start_roll(0, 16_000, 0.0)?;
        session.checkpoint(0, 16_000, 16_000)?;
        Ok(session::CaptureSummary { duration_secs: secs, segments: 1, rebuilds: 0, dropped: 0, num_buffers: 2 })
    }

    /// The one session directory a call made.
    fn session_of(call: &Call) -> PathBuf {
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(&call.cfg.sessions_dir).unwrap().map(|entry| entry.unwrap().path())
            .filter(|path| !path.file_name().unwrap().to_string_lossy().starts_with('.')).collect();
        assert_eq!(dirs.len(), 1, "{dirs:?}");
        dirs.remove(0)
    }

    /// A reader that cannot start is never a reason to lose a recording.
    #[test]
    fn a_reader_that_cannot_start_does_not_fail_the_recording() {
        let call = call("rq-spawn", LIVE_ON);
        let mut reader = unspawnable();
        record_and_queue(&call.cfg, "com.google.Chrome", &call.jobs, &call.board, &mut reader, |_, _, managed| captured(managed, 30.0)).unwrap();
        let job = call.queued.try_recv().expect("the recording was not queued");
        let state = crate::recovery::State::read(&job.dir).unwrap();
        assert_eq!((state.phase, state.capture_status), (crate::recovery::Phase::Ready, crate::recovery::CaptureStatus::Complete));
        assert_eq!(job.opts.session_key, Some(state.key));
        assert_eq!(job.promote_live, Some(call.base.join("live")));
        assert!(reader.running.is_none());
        std::fs::remove_dir_all(&call.base).unwrap();
    }

    /// `--once` stops the reader as the loop is left, so the call it recorded is not offered
    /// its live transcript: the child would wait for a reader that is already gone. A session
    /// from the backlog keeps the offer, since no reader of it is being stopped.
    #[test]
    fn a_call_recorded_with_once_is_not_offered_its_live_transcript() {
        let mut call = call("rq-once", LIVE_ON);
        call.cfg.once = true;
        let mut reader = unspawnable();
        record_and_queue(&call.cfg, "com.google.Chrome", &call.jobs, &call.board, &mut reader, |_, _, managed| captured(managed, 30.0)).unwrap();
        let job = call.queued.try_recv().expect("the recording was not queued");
        assert_eq!(job.promote_live, None);
        let backlog = job_for(&call.cfg, job.dir.clone(), "Chrome".into(), "a title the stored state overrides".into());
        assert_eq!(backlog.promote_live, Some(call.base.join("live")));
        std::fs::remove_dir_all(&call.base).unwrap();
    }

    /// The reader is started before capture, for the session capture writes; it is left running
    /// when the call ends well, with its deadline, and the session's lock is free for the child.
    #[test]
    fn a_good_capture_leaves_the_reader_running_with_a_deadline() {
        use std::cell::RefCell;
        use std::rc::Rc;
        let call = call("rq-good", LIVE_ON);
        let started_for = Rc::new(RefCell::new(Vec::<PathBuf>::new()));
        let seen = Rc::clone(&started_for);
        let mut reader = LiveReader::new(Some(Box::new(move |dir: &Path| {
            seen.borrow_mut().push(dir.to_path_buf());
            let mut command = std::process::Command::new("/bin/sleep");
            command.arg("60");
            command
        })));
        let at_capture = Rc::clone(&started_for);
        record_and_queue(&call.cfg, "com.google.Chrome", &call.jobs, &call.board, &mut reader, |dir, _, managed| {
            assert_eq!(*at_capture.borrow(), [dir.to_path_buf()], "the reader was not started before capture, for this session");
            captured(managed, 30.0)
        }).unwrap();
        let job = call.queued.try_recv().expect("the recording was not queued");
        assert_eq!(started_for.borrow().as_slice(), std::slice::from_ref(&job.dir));
        assert!(crate::recovery::Session::open(&job.dir).unwrap().is_some(), "the session is still locked after the call");
        let pid = pid_of(&reader);
        assert!(exists(pid), "the reader must outlive the call to finish its last windows");
        let now = Instant::now();
        assert!(reader.tick(now).is_empty());
        assert_eq!(reader.tick(now + READER_GRACE), [report(&crate::recovery::State::read(&job.dir).unwrap().key, ReaderChange::Killed(Killed::Deadline))]);
        assert!(!exists(pid));
        std::fs::remove_dir_all(&call.base).unwrap();
    }

    /// A call too short to transcribe returns early; its reader must have a deadline all the same.
    #[test]
    fn a_short_capture_leaves_the_reader_a_deadline_too() {
        let call = call("rq-short", LIVE_ON);
        let mut reader = sleeper();
        record_and_queue(&call.cfg, "com.google.Chrome", &call.jobs, &call.board, &mut reader, |_, _, managed| captured(managed, 5.0)).unwrap();
        assert!(call.queued.try_recv().is_err(), "a call under min_secs was queued");
        let state = crate::recovery::State::read(&session_of(&call)).unwrap();
        assert_eq!(state.phase, crate::recovery::Phase::SkippedShort);
        let pid = pid_of(&reader);
        assert_eq!(reader.tick(Instant::now() + READER_GRACE), [report(&state.key, ReaderChange::Killed(Killed::Deadline))],
            "the reader of a short call was left with no deadline");
        assert!(!exists(pid));
        std::fs::remove_dir_all(&call.base).unwrap();
    }

    /// Capture has two ways to fail: `run_capture` itself, and saving the end of the recording.
    /// Either way the reader is stopped at once: nothing more will be recorded for it to follow.
    #[test]
    fn a_failed_capture_stops_the_reader_on_both_error_paths() {
        let failing = call("rq-fail", LIVE_ON);
        let pid_file = failing.base.join("reader.pid");
        let mut reader = announcing_sleeper(pid_file.clone());
        let error = record_and_queue(&failing.cfg, "com.google.Chrome", &failing.jobs, &failing.board, &mut reader, |_, _, _| {
            announced_pid(&pid_file); // the reader is running when capture fails
            anyhow::bail!("synthetic capture failure")
        }).unwrap_err();
        assert!(format!("{error:#}").contains("synthetic capture failure"));
        assert!(reader.running.is_none(), "the reader of a failed capture is still held");
        assert!(!exists(announced_pid(&pid_file)), "the reader of a failed capture was left running");
        assert!(failing.queued.try_recv().is_err());
        assert_eq!(crate::recovery::State::read(&session_of(&failing)).unwrap().phase, crate::recovery::Phase::NeedsAttention);
        std::fs::remove_dir_all(&failing.base).unwrap();

        // The capture ran, and the state that says so cannot be saved (the directory is gone).
        let unsaved = call("rq-unsaved", LIVE_ON);
        let pid_file = unsaved.base.join("reader.pid");
        let mut reader = announcing_sleeper(pid_file.clone());
        let outcome = record_and_queue(&unsaved.cfg, "com.google.Chrome", &unsaved.jobs, &unsaved.board, &mut reader, |dir, _, managed| {
            announced_pid(&pid_file);
            let summary = captured(managed, 30.0)?;
            std::fs::remove_dir_all(dir)?;
            Ok(summary)
        });
        assert!(outcome.is_err(), "the end of the capture was saved into a directory that is gone");
        assert!(reader.running.is_none(), "the reader is still held after the capture's end could not be saved");
        assert!(!exists(announced_pid(&pid_file)), "the reader was left running after the capture's end could not be saved");
        assert!(unsaved.queued.try_recv().is_err());
        std::fs::remove_dir_all(&unsaved.base).unwrap();
    }

    /// With `[live]` left alone the daemon starts nothing and offers no live transcript.
    #[test]
    fn with_live_off_a_call_starts_no_reader_and_passes_no_flag() {
        for text in ["", "[live]\npromote = true\n"] {
            let call = call("rq-off", text);
            let mut reader = LiveReader::new(call.cfg.live.enabled.then(|| -> ReaderCommand { Box::new(|_: &Path| std::process::Command::new("/bin/sleep")) }));
            record_and_queue(&call.cfg, "com.google.Chrome", &call.jobs, &call.board, &mut reader, |_, _, managed| captured(managed, 30.0)).unwrap();
            let job = call.queued.try_recv().expect("the recording was not queued");
            assert!(reader.running.is_none() && job.promote_live.is_none(), "config: {text:?}");
            let argv = child_args(&job.dir, &job.opts, &call.base, job.promote_live.as_deref());
            assert!(!argv.iter().any(|arg| arg == "--promote-live"));
            std::fs::remove_dir_all(&call.base).unwrap();
        }
    }

    /// `live-watch.log` belongs to a hand-started `live watch`: the daemon never truncates it,
    /// whatever `[live]` says. Old live streams go with `retain_days`, on or off.
    #[test]
    fn maintenance_leaves_the_watch_log_alone_and_prunes_streams_by_retain_days() {
        for (text, pruned) in [("", false), ("[live]\nenabled = true\n", false), ("[live]\nretain_days = 7\n", true)] {
            let base = temp_base("maintenance-live");
            let mut cfg = test_config(&base, live(text));
            cfg.log_max_mb = 1;
            std::fs::create_dir_all(&cfg.logs_dir).unwrap();
            let big = vec![b'x'; 1024 * 1024 + 1];
            for name in ["meetscribe.err.log", "live-watch.log"] { std::fs::write(cfg.logs_dir.join(name), &big).unwrap(); }
            let stream = base.join("live").join(format!("{KEY}.jsonl"));
            std::fs::create_dir_all(stream.parent().unwrap()).unwrap();
            std::fs::write(&stream, format!("{}\n", serde_json::json!({"v": 1, "seq": 0, "type": "meeting_started", "key": KEY,
                "app": "Chrome", "started_at": 1_794_443_305, "model": "ggml-fixture"}))).unwrap();
            std::fs::File::options().write(true).open(&stream).unwrap()
                .set_modified(std::time::SystemTime::now() - Duration::from_secs(30 * 86_400)).unwrap();

            run_maintenance(&cfg);
            let len = |name: &str| std::fs::metadata(cfg.logs_dir.join(name)).unwrap().len();
            assert_eq!(len("meetscribe.err.log"), 0, "the daemon's own log is rotated as before");
            assert_eq!(len("live-watch.log"), big.len() as u64, "config: {text:?}");
            assert!(!cfg.logs_dir.join("live-watch.log.1").exists(), "config: {text:?}");
            assert_eq!(!stream.exists(), pruned, "config: {text:?}");
            std::fs::remove_dir_all(&base).unwrap();
        }
    }

    #[test]
    fn retry_deadlines_progress_during_recording_and_pause_without_restart() {
        use std::sync::atomic::AtomicI64;
        for iteration in 0..5 {
            let base = temp_base(&format!("retry-{iteration}"));
            let dir = base.join("recording");
            let mut session = crate::recovery::Session::begin(&dir, &base.join("archive.db"), &dir,
                "com.google.Chrome", "Retry schedule fixture", 1_762_880_000).unwrap();
            session.start_roll(0, 16_000, 0.0).unwrap();
            session.checkpoint(0, 16_000, 16_000).unwrap();
            session.finish_capture(1.0, 0.1).unwrap();
            drop(session);
            let clock = Arc::new(AtomicI64::new(1000));
            let board = Arc::new(StatusBoard::new(base.clone()));
            board.update(|activity| { activity.recording = Some("Zoom".into()); activity.paused = true; });
            let (finished_tx, finished_rx) = mpsc::channel();
            let run_clock = Arc::clone(&clock);
            let scan_clock = Arc::clone(&clock);
            let worker_clock = Arc::clone(&clock);
            let scan_dir = dir.clone();
            let (jobs, worker) = spawn_transcriber_loop(Arc::clone(&board), move |job| {
                let mut session = crate::recovery::Session::open(&job.dir).unwrap().unwrap();
                let now = run_clock.load(Ordering::SeqCst);
                session.reserve(now, false).unwrap();
                session.failed("processing", now).unwrap();
                let status = (session.state.attempts, session.state.next_retry, session.state.phase);
                drop(session);
                finished_tx.send(status).unwrap();
            }, move || {
                let state = crate::recovery::State::read(&scan_dir).unwrap();
                if !state.due(scan_clock.load(Ordering::SeqCst)) { return Vec::new(); }
                (0..2).map(|_| {
                    let mut item = job(&scan_dir, "Chrome");
                    item.opts.session_key = Some(state.key.clone()); item.opts.no_store = false; item
                }).collect()
            }, move || worker_clock.load(Ordering::SeqCst), Duration::from_millis(5)).unwrap();
            for (now, attempt, next) in [(1000, 1, 1060), (1060, 2, 1360), (1360, 3, 3160), (3160, 4, 3160)] {
                clock.store(now, Ordering::SeqCst);
                let status = finished_rx.recv_timeout(Duration::from_secs(3)).unwrap();
                assert_eq!(status.0, attempt);
                if attempt < 4 { assert_eq!(status.1, next); }
                else { assert_eq!(status.2, crate::recovery::Phase::NeedsAttention); }
                assert_eq!(shown_on_disk(&base).0, DaemonState::Recording);
                assert!(finished_rx.recv_timeout(Duration::from_millis(30)).is_err(), "a duplicate job or premature retry ran");
            }
            clock.store(100_000, Ordering::SeqCst);
            assert!(finished_rx.recv_timeout(Duration::from_millis(30)).is_err());
            drop(jobs); worker.join().unwrap();
            std::fs::remove_dir_all(base).unwrap();
        }
        println!("{{\"event\":\"audit_case_complete\",\"case\":\"f2_retry_worker\",\"iterations\":5}}");
    }

    #[test]
    fn inventory_failure_cannot_publish_a_proven_empty_backlog() {
        for iteration in 0..5 {
            let base = temp_base(&format!("inventory-{iteration}"));
            let sessions = base.join("sessions");
            let owner = crate::recovery::Session::begin(&sessions.join("recording"), &base.join("archive.db"),
                &base.join("exports"), "org.example.fixture", "Recovery inventory fixture", 1_762_880_000).unwrap();
            let board = StatusBoard::new(base.clone()); board.refresh_counts(&sessions);
            assert_eq!(status::Status::read(&base).unwrap().pending, 1);
            let not_a_directory = base.join("unavailable-inventory"); std::fs::write(&not_a_directory, b"unavailable").unwrap();
            board.refresh_counts(&not_a_directory);
            assert!(status::Status::read(&base).unwrap().attention > 0, "an unreadable inventory must not look archived/green");
            board.refresh_counts(&sessions);
            let restored = status::Status::read(&base).unwrap();
            assert_eq!((restored.pending, restored.attention), (1, 0));
            board.refresh_counts_from(Ok(vec![Err(std::io::Error::other("injected entry-read failure"))]));
            assert!(status::Status::read(&base).unwrap().attention > 0);
            board.refresh_counts(&sessions);
            assert_eq!(status::Status::read(&base).unwrap().pending, 1);
            drop(owner); std::fs::remove_dir_all(base).unwrap();
        }
        println!("{{\"event\":\"audit_case_complete\",\"case\":\"f6_inventory_failure\",\"iterations\":5}}");
    }

    #[test]
    fn counts_publish_without_a_coarse_state_change() {
        let base = temp_base("counts");
        let board = StatusBoard::new(base.clone());
        board.update(|_| {});
        let first = status::Status::read(&base).unwrap();
        board.update(|activity| { activity.pending = 2; activity.attention = 1; });
        let second = status::Status::read(&base).unwrap();
        assert_eq!(second.state, first.state);
        assert_eq!(second.since_epoch, first.since_epoch);
        assert_eq!((second.pending, second.attention), (2, 1));
        assert!(second.instance.is_some());
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn addressed_request_interrupts_the_idle_stop_poll() {
        if std::env::var_os("MEETSCRIBE_TEST_STOP_POLL").is_some() {
            let base = temp_base("stop-poll-child");
            let board = StatusBoard::new(base.clone());
            board.update(|_| {});
            let own = board.instance.clone().unwrap();
            assert!(!board.stopping());
            let request_base = base.clone();
            let writer = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                status::request_stop(&request_base, Some(&own), status::process_start, status::write_stop).unwrap()
            });
            let start = Instant::now();
            sleep_interruptible(Duration::from_secs(10), &board);
            assert!(start.elapsed() < Duration::from_secs(3));
            assert!(writer.join().unwrap());
            assert!(board.stopping());
            std::fs::remove_dir_all(base).unwrap();
            println!("{{\"event\":\"addressed_stop_poll_ack\"}}");
            return;
        }
        for _ in 0..5 {
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "daemon::tests::addressed_request_interrupts_the_idle_stop_poll", "--nocapture"])
                .env("MEETSCRIBE_TEST_STOP_POLL", "1").output().unwrap();
            assert!(child.status.success(), "stop-poll child failed: {child:?}");
            assert!(String::from_utf8_lossy(&child.stdout).contains("addressed_stop_poll_ack"), "stop-poll child ran no verified test");
        }
        println!("{{\"event\":\"audit_case_complete\",\"case\":\"f6_addressed_poll\",\"iterations\":5}}");
    }

    #[test]
    fn the_marker_clears_only_after_complete_archive_proof_regardless_of_exit() {
        use ChildOutcome::*;
        for (outcome, archive_proven, want) in [
            (Exited(0), false, Marker::Keep),
            (Exited(0), true, Marker::Clear),
            (Exited(1), false, Marker::Keep),
            (Signaled(libc::SIGABRT), false, Marker::Keep),
            (LaunchFailed, false, Marker::Keep),
            (TrackingLost, false, Marker::Keep),
            (Signaled(libc::SIGTERM), true, Marker::Clear),
            (Signaled(libc::SIGINT), true, Marker::Clear),
            (Exited(1), true, Marker::Clear),
            (TrackingLost, true, Marker::Clear),
            (LaunchFailed, true, Marker::Clear),
        ] {
            assert_eq!(marker_after(outcome, archive_proven), want, "{outcome:?} archive_proven={archive_proven}");
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
        }, Vec::new)
        .unwrap();

        jobs.send(job(&base.join("first"), "Chrome")).unwrap();
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
        jobs.send(job(&base.join("second"), "Meet")).unwrap();
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
