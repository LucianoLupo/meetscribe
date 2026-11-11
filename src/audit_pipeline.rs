//! Real child argv/parser/pipeline/outcome regressions. Only model inference is replaced.
use super::*;
use crate::{asr, db, recovery, vad};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::sync::atomic::AtomicI64;
use sqlx::Connection;

const START: i64 = 1_762_880_000;
struct SyntheticEngine { root: PathBuf, db: PathBuf, no_store: bool }
impl pipeline::SpeechEngine for SyntheticEngine {
    fn windows(&mut self, audio: &[f32]) -> Result<Vec<vad::SpeechWindow>> {
        Ok(if audio.is_empty() { Vec::new() } else { vec![vad::SpeechWindow { start_ms: 0, end_ms: audio.len() / 16 }] })
    }
    fn transcribe(&mut self, audio: &[f32], _lang: &str, _words: bool) -> Result<(String, f32, Vec<asr::Word>)> {
        if !self.no_store && self.root.join("storage-fault").exists() {
            crate::new_runtime()?.block_on(async {
                let mut connection = sqlx::SqliteConnection::connect_with(&sqlx::sqlite::SqliteConnectOptions::new()
                    .filename(&self.db).create_if_missing(false)).await?;
                sqlx::query("CREATE TRIGGER IF NOT EXISTS audit_storage_fault BEFORE INSERT ON meetings BEGIN SELECT RAISE(ABORT, 'synthetic archive fault'); END")
                    .execute(&mut connection).await?;
                connection.close().await?; anyhow::Ok(())
            })?;
        }
        let text = if audio.first().is_some_and(|value| *value < 0.0) {
            "We approved the synthetic deployment window"
        } else { "The synthetic release checklist is ready" };
        Ok((text.into(), 0.94, Vec::new()))
    }
    fn embed(&mut self, _audio: &[f32]) -> Result<Option<Vec<f32>>> { Ok(Some(vec![1.0 / 192.0f32.sqrt(); 192])) }
}
fn backend(_model: &str, opts: &pipeline::PipelineOpts) -> Result<Box<dyn pipeline::SpeechEngine>> {
    let root = PathBuf::from(std::env::var_os("MEETSCRIBE_PIPELINE_FIXTURE_ROOT").context("fixture root absent")?);
    anyhow::ensure!(!root.join("forbid-inference").exists(), "inference must not run after a committed ingest");
    Ok(Box::new(SyntheticEngine { root, db: opts.db_path.clone(), no_store: opts.no_store }))
}
fn cutpoint(point: &str) {
    let root = PathBuf::from(std::env::var_os("MEETSCRIBE_PIPELINE_FIXTURE_ROOT").unwrap());
    if fs::read_to_string(root.join("cutpoint")).is_ok_and(|selected| selected.trim() == point) {
        recovery::atomic_json(&root.join("cutpoint-ack.json"), &serde_json::json!({"event":"cutpoint_reached", "point":point})).unwrap();
        std::process::exit(85);
    }
}

#[test]
#[ignore = "sacrificial entry invoked explicitly by mandatory audit parent tests"]
fn pipeline_child() {
    let root = PathBuf::from(std::env::var_os("MEETSCRIBE_PIPELINE_FIXTURE_ROOT").expect("fixture root required"));
    crate::diagnostics::init(); pipeline::test_engine(backend); pipeline::test_cutpoints(cutpoint);
    let argv: Vec<String> = fs::read(root.join("argv")).unwrap().split(|byte| *byte == 0)
        .filter(|part| !part.is_empty()).map(|part| String::from_utf8(part.to_vec()).unwrap()).collect();
    assert_eq!(argv.first().map(String::as_str), Some("transcribe"));
    let now: i64 = fs::read_to_string(root.join("clock")).unwrap().trim().parse().unwrap();
    let result = crate::run_transcribe_using(&argv[1..], |dir, opts| pipeline::transcribe_with_clock(dir, opts, || now));
    if let Err(error) = result { crate::diagnostics::command_failed(&error); std::process::exit(1); }
    std::process::exit(0);
}

#[derive(Clone)]
struct Fixture { root: PathBuf, dir: PathBuf, base: PathBuf, executable: PathBuf, opts: pipeline::PipelineOpts }
impl Fixture {
    fn new() -> (Self, recovery::Session) {
        let root = std::env::temp_dir().join(format!("meetscribe-child-{}", recovery::new_identity()));
        recovery::private_dir(&root).unwrap(); let root = fs::canonicalize(root).unwrap();
        let dir = root.join("sessions/recording"); let base = root.join("control"); recovery::private_dir(&base).unwrap();
        let mut owner = recovery::Session::begin(&dir, &root.join("archive.db"), &dir,
            "com.google.Chrome", "Synthetic release review", START).unwrap();
        owner.start_roll(0, 16_000, 0.0).unwrap();
        for (channel, sample) in [("mic", 0.25f32), ("system", -0.5f32)] {
            let mut wav = hound::WavWriter::create(dir.join(format!("{channel}.wav")), hound::WavSpec {
                channels: 1, sample_rate: 16_000, bits_per_sample: 32, sample_format: hound::SampleFormat::Float }).unwrap();
            for _ in 0..48_000 { wav.write_sample(sample).unwrap(); } wav.finalize().unwrap();
        }
        owner.checkpoint(0, 48_000, 48_000).unwrap(); owner.finish_capture(3.0, 0.1).unwrap();
        let opts = pipeline::PipelineOpts { model: root.join("synthetic-model.bin"), lang: "en".into(), title: Some(owner.state.title.clone()),
            db_path: owner.state.db_path.clone(), export_dir: dir.clone(), no_store: false, session_key: Some(owner.state.key.clone()), resume: false,
            speaker_model: Some(root.join("synthetic-speaker.onnx")), split: false,
            diarizer_bin: root.join("absent-diarizer"), diarizer_model: root.join("absent.gguf"), diarizer_rttm: None };
        fs::write(root.join("clock"), format!("{}\n", START + 10)).unwrap();
        let executable = root.join("child.sh");
        let quote = |path: &Path| format!("'{}'", path.display().to_string().replace('\'', "'\\''"));
        fs::write(&executable, format!("#!/bin/sh\nexport MEETSCRIBE_PIPELINE_FIXTURE_ROOT={}\nprintf '%s\\0' \"$@\" > {}\nexec {} --exact daemon::audit_pipeline::pipeline_child --ignored --nocapture\n",
            quote(&root), quote(&root.join("argv")), quote(&std::env::current_exe().unwrap()))).unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
        (Self { root, dir, base, executable, opts }, owner)
    }
    fn job(&self) -> TranscribeJob { TranscribeJob { dir: self.dir.clone(), label: "Chrome".into(), opts: self.opts.clone() } }
    fn queued_job(&self) -> TranscribeJob {
        let (tx, rx) = mpsc::channel(); queue_job(&tx, self.job()); rx.recv().unwrap()
    }
    fn state(&self) -> recovery::State { recovery::State::read(&self.dir).unwrap() }
    fn meetings(&self) -> Vec<db::MeetingRow> {
        crate::new_runtime().unwrap().block_on(async {
            let mut db = db::Db::open_existing(&self.opts.db_path, self.state().archive_id.as_deref().unwrap()).await.unwrap();
            let rows = db.list_meetings().await.unwrap(); db.close().await.unwrap(); rows
        })
    }
    fn remove_fault(&self) {
        fs::remove_file(self.root.join("storage-fault")).unwrap();
        crate::new_runtime().unwrap().block_on(async {
            let mut connection = sqlx::SqliteConnection::connect_with(&sqlx::sqlite::SqliteConnectOptions::new()
                .filename(&self.opts.db_path).create_if_missing(false)).await.unwrap();
            sqlx::query("DROP TRIGGER audit_storage_fault").execute(&mut connection).await.unwrap(); connection.close().await.unwrap();
        });
    }
}

struct LaunchWorker {
    jobs: mpsc::Sender<TranscribeJob>,
    handle: JoinHandle<()>,
    clock: Arc<AtomicI64>,
    finished: mpsc::Receiver<recovery::State>,
    scanned: mpsc::Receiver<usize>,
}

impl LaunchWorker {
    fn start(fixture: &Fixture, executable: PathBuf, at: i64) -> Self {
        let clock = Arc::new(AtomicI64::new(at));
        let run_clock = Arc::clone(&clock); let scan_clock = Arc::clone(&clock); let scheduler_clock = Arc::clone(&clock);
        let scan_fixture = fixture.clone(); let run_fixture = fixture.clone();
        let (finished_tx, finished) = mpsc::channel(); let (scanned_tx, scanned) = mpsc::channel();
        let board = Arc::new(StatusBoard::new(fixture.base.clone()));
        let (jobs, handle) = spawn_transcriber_loop(board, move |job| {
            fs::write(run_fixture.root.join("clock"), format!("{}\n", run_clock.load(Ordering::SeqCst))).unwrap();
            run_transcribe_job_at(job, &executable, &run_fixture.base, || run_clock.load(Ordering::SeqCst));
            finished_tx.send(run_fixture.state()).unwrap();
        }, move || {
            let found: Vec<_> = pending_sessions_at(&scan_fixture.root.join("sessions"), scan_clock.load(Ordering::SeqCst))
                .into_iter().map(|(dir, label, _)| TranscribeJob { dir, label, opts: scan_fixture.opts.clone() }).collect();
            scanned_tx.send(found.len()).unwrap(); found
        }, move || scheduler_clock.load(Ordering::SeqCst), Duration::from_millis(5)).unwrap();
        Self { jobs, handle, clock, finished, scanned }
    }

    fn idle(&self) {
        for _ in 0..3 {
            assert_eq!(self.scanned.recv_timeout(Duration::from_secs(5)).unwrap(), 0,
                "a non-due or terminal session must not be queued after worker restart");
        }
        assert!(self.finished.try_recv().is_err(), "idle worker unexpectedly ran a child");
    }

    fn completed(&self) -> recovery::State {
        self.finished.recv_timeout(Duration::from_secs(10)).expect("real worker did not finish the launch attempt")
    }

    fn stop(self) {
        drop(self.jobs); self.handle.join().unwrap();
    }
}

#[test]
fn launch_failure_budget_survives_worker_restarts_and_exhausts() {
    for _ in 0..5 {
        let (fixture, owner) = Fixture::new(); drop(owner);
        let missing = fixture.root.join("unspawnable-meetscribe"); assert!(!missing.exists());
        // Literal deadlines: initial attempt at 1000, then waits of 60, 300 and 1800 seconds.
        let first = LaunchWorker::start(&fixture, missing.clone(), 1000);
        queue_job(&first.jobs, fixture.job());
        let state = first.completed();
        assert_eq!(state.attempts, 1, "a failed OS spawn must consume the initial durable attempt");
        assert_eq!(state.next_retry, 1060); assert!(!state.in_flight);
        assert_eq!(state.phase, recovery::Phase::Ready);
        assert_eq!(state.error_class.as_deref(), Some("child_launch")); first.stop();
        for (at, attempts, next, phase) in [
            (1060, 2, Some(1360), recovery::Phase::Ready),
            (1360, 3, Some(3160), recovery::Phase::Ready),
            (3160, 4, None, recovery::Phase::NeedsAttention),
        ] {
            let restarted = LaunchWorker::start(&fixture, missing.clone(), at - 1);
            restarted.idle(); assert_eq!(fixture.state().attempts, attempts - 1);
            restarted.clock.store(at, Ordering::SeqCst);
            let state = restarted.completed();
            assert_eq!(state.attempts, attempts); assert_eq!(state.phase, phase); assert!(!state.in_flight);
            if let Some(deadline) = next { assert_eq!(state.next_retry, deadline); }
            assert_eq!(state.error_class.as_deref(), Some("child_launch"));
            assert!(fixture.dir.join(maintenance::PENDING_MARKER).exists()); restarted.stop();
        }
        let exhausted = LaunchWorker::start(&fixture, missing, 100_000);
        exhausted.idle(); assert_eq!(fixture.state().attempts, 4); exhausted.stop();
        assert!(!fixture.root.join("archive.db").exists());
        assert_eq!(fixture.state().phase, recovery::Phase::NeedsAttention);
        fs::remove_dir_all(fixture.root).unwrap();
    }
    println!(r#"{{"event":"audit_case_complete","case":"f2_launch_failure_budget","iterations":5}}"#);
}

#[test]
fn launch_failure_while_busy_is_free_then_retries_after_release() {
    for _ in 0..5 {
        let (fixture, owner) = Fixture::new();
        let before = fs::read(fixture.dir.join(recovery::STATE_FILE)).unwrap();
        let worker = LaunchWorker::start(&fixture, fixture.root.join("unspawnable-meetscribe"), 1000);
        queue_job(&worker.jobs, fixture.job());
        let state = worker.completed();
        assert_eq!(state.attempts, 0); assert!(!state.in_flight); assert_eq!(state.next_retry, 0);
        assert_eq!(fs::read(fixture.dir.join(recovery::STATE_FILE)).unwrap(), before);
        assert!(fixture.dir.join(maintenance::PENDING_MARKER).exists());
        drop(owner); worker.clock.store(1060, Ordering::SeqCst);
        let state = worker.completed(); assert_eq!(state.attempts, 1); assert_eq!(state.next_retry, 1120);
        assert_eq!(state.error_class.as_deref(), Some("child_launch"));
        worker.stop(); assert!(!fixture.root.join("archive.db").exists());
        fs::remove_dir_all(fixture.root).unwrap();
    }
    println!(r#"{{"event":"audit_case_complete","case":"f2_launch_busy_free","iterations":5}}"#);
}

#[test]
fn launch_failure_does_not_double_charge_a_child_reservation() {
    for _ in 0..5 {
        let (fixture, owner) = Fixture::new(); drop(owner);
        fs::write(fixture.root.join("cutpoint"), "markdown_published\n").unwrap();
        let child = LaunchWorker::start(&fixture, fixture.executable.clone(), START + 10);
        queue_job(&child.jobs, fixture.job());
        let reserved = child.completed(); child.stop();
        let ack: serde_json::Value = serde_json::from_slice(&fs::read(fixture.root.join("cutpoint-ack.json")).unwrap()).unwrap();
        assert_eq!(ack["point"], "markdown_published");
        assert_eq!(reserved.attempts, 1, "a launched child must not receive an extra parent charge");
        assert!(reserved.in_flight); assert_eq!(reserved.next_retry, 0);
        assert_eq!(reserved.phase, recovery::Phase::Archived); assert_eq!(fixture.meetings().len(), 1);
        let missing = fixture.root.join("unspawnable-meetscribe");
        let restarted = LaunchWorker::start(&fixture, missing.clone(), START + 10);
        let settled = restarted.completed(); restarted.stop();
        assert_eq!(settled.attempts, 1, "settling an abandoned reservation must not reserve again");
        assert!(!settled.in_flight); assert_eq!(settled.next_retry, START + 70);
        assert_eq!(settled.error_class.as_deref(), Some("interrupted_attempt"));
        assert_eq!(settled.phase, recovery::Phase::Archived); assert_eq!(settled.meeting_id, reserved.meeting_id);
        let retry = LaunchWorker::start(&fixture, missing, START + 69); retry.idle();
        retry.clock.store(START + 70, Ordering::SeqCst);
        let next = retry.completed(); retry.stop();
        assert_eq!(next.attempts, 2); assert_eq!(next.next_retry, START + 370);
        assert_eq!(next.error_class.as_deref(), Some("child_launch"));
        assert_eq!(fixture.meetings().len(), 1); assert!(fixture.dir.join(maintenance::PENDING_MARKER).exists());
        fs::remove_dir_all(fixture.root).unwrap();
    }
    println!(r#"{{"event":"audit_case_complete","case":"f2_launch_reserved_once","iterations":5}}"#);
}

#[test]
fn launch_failure_preserves_ineligible_unkeyed_and_mismatched_state() {
    for _ in 0..5 {
        for scenario in ["not_due", "terminal", "unkeyed", "wrong_key", "unreadable"] {
            let (fixture, mut owner) = Fixture::new(); let mut job = fixture.job();
            match scenario {
                "not_due" => { owner.reserve(1000, false).unwrap(); owner.failed("processing", 1000).unwrap(); }
                "terminal" => owner.needs_attention("processing").unwrap(),
                "unkeyed" => job.opts.session_key = None,
                "wrong_key" => job.opts.session_key = Some(recovery::new_identity()),
                "unreadable" => fs::write(fixture.dir.join(recovery::STATE_FILE), b"invalid metadata\n").unwrap(),
                _ => unreachable!(),
            }
            drop(owner);
            let before = fs::read(fixture.dir.join(recovery::STATE_FILE)).unwrap();
            run_transcribe_job_at(&job, &fixture.root.join("unspawnable-meetscribe"), &fixture.base, || 1059);
            assert_eq!(fs::read(fixture.dir.join(recovery::STATE_FILE)).unwrap(), before, "scenario={scenario}");
            assert!(!fixture.root.join("archive.db").exists()); fs::remove_dir_all(fixture.root).unwrap();
        }
    }
    println!(r#"{{"event":"audit_case_complete","case":"f2_launch_preserve_ineligible","iterations":5}}"#);
}

#[test]
fn audit_f2_child_fault_retry_and_retention() {
    for _ in 0..5 {
        let (fixture, owner) = Fixture::new(); drop(owner);
        fs::write(fixture.root.join("storage-fault"), b"fault enabled\n").unwrap();
        let clock = Arc::new(AtomicI64::new(START + 10));
        let run_clock = Arc::clone(&clock); let due_clock = Arc::clone(&clock); let poll_clock = Arc::clone(&clock);
        let run_fixture = fixture.clone(); let due_fixture = fixture.clone();
        let (finished_tx, finished_rx) = mpsc::channel();
        let board = Arc::new(StatusBoard::new(fixture.base.clone())); board.update(|activity| activity.paused = true);
        let (jobs, worker) = spawn_transcriber_loop(board, move |job| {
            fs::write(run_fixture.root.join("clock"), format!("{}\n", run_clock.load(Ordering::SeqCst))).unwrap();
            run_transcribe_job(job, &run_fixture.executable, &run_fixture.base);
            finished_tx.send(run_fixture.state()).unwrap();
        }, move || pending_sessions_at(&due_fixture.root.join("sessions"), due_clock.load(Ordering::SeqCst)).into_iter()
            .map(|(dir, label, _)| TranscribeJob { dir, label, opts: due_fixture.opts.clone() }).collect(),
            move || poll_clock.load(Ordering::SeqCst), Duration::from_millis(5)).unwrap();
        queue_job(&jobs, fixture.job());
        let first = finished_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!((first.attempts, first.phase), (1, recovery::Phase::Ready));
        assert_eq!(first.next_retry, START + 70); assert!(fixture.meetings().is_empty());
        let fallback = fs::read(fixture.dir.join("transcript.md")).unwrap();
        assert!(fixture.dir.join(maintenance::PENDING_MARKER).is_file());
        assert_eq!(maintenance::prune_sessions(&fixture.root.join("sessions"), 30, START + 1000 * 86_400).unwrap().removed, 0);
        clock.store(first.next_retry, Ordering::SeqCst);
        let second = finished_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(second.attempts, 2); assert_eq!(second.next_retry, START + 370);
        assert_eq!(fs::read(fixture.dir.join("transcript.md")).unwrap(), fallback);
        clock.store(second.next_retry, Ordering::SeqCst);
        let third = finished_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!((third.attempts, third.phase), (3, recovery::Phase::Ready));
        assert_eq!(third.next_retry, START + 2170);
        assert_eq!(fs::read(fixture.dir.join("transcript.md")).unwrap(), fallback);
        fixture.remove_fault(); clock.store(third.next_retry, Ordering::SeqCst);
        let final_state = finished_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!((final_state.attempts, final_state.phase), (4, recovery::Phase::Complete));
        assert_eq!(final_state.archive_id, first.archive_id); assert_eq!(fixture.meetings().len(), 1);
        assert!(!fixture.dir.join(maintenance::PENDING_MARKER).exists());
        assert_eq!(fs::read(fixture.dir.join("transcript.md")).unwrap(), fallback);
        drop(jobs); worker.join().unwrap(); fs::remove_dir_all(fixture.root).unwrap();
    }
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f2_child_retry_retention\",\"iterations\":5}}");
}

#[test]
fn audit_f3_child_commit_and_publication_cutpoints() {
    for _ in 0..5 {
        let (fixture, owner) = Fixture::new(); drop(owner);
        fs::write(fixture.root.join("cutpoint"), "database_committed\n").unwrap();
        run_transcribe_job(&fixture.queued_job(), &fixture.executable, &fixture.base);
        let ack: serde_json::Value = serde_json::from_slice(&fs::read(fixture.root.join("cutpoint-ack.json")).unwrap()).unwrap();
        assert_eq!(ack["point"], "database_committed");
        let rows = fixture.meetings(); assert_eq!(rows.len(), 1); let id = rows[0].id;
        assert_ne!(fixture.state().phase, recovery::Phase::Complete);
        assert!(fixture.dir.join(maintenance::PENDING_MARKER).is_file());
        let manual_md = b"# Operator revision\n\nRetain the amended deployment decision.\n";
        let manual_json = b"[{\"text\":\"Retain the amended deployment decision\"}]\n";
        fs::write(fixture.dir.join("transcript.md"), manual_md).unwrap(); fs::write(fixture.dir.join("transcript.json"), manual_json).unwrap();
        fs::write(fixture.root.join("forbid-inference"), b"existing ingest only\n").unwrap();
        fs::write(fixture.root.join("cutpoint"), "markdown_published\n").unwrap();
        fs::remove_file(fixture.root.join("cutpoint-ack.json")).unwrap();
        let mut retry = fixture.job(); retry.opts.resume = true;
        run_transcribe_job(&retry, &fixture.executable, &fixture.base);
        let ack: serde_json::Value = serde_json::from_slice(&fs::read(fixture.root.join("cutpoint-ack.json")).unwrap()).unwrap();
        assert_eq!(ack["point"], "markdown_published"); assert_ne!(fixture.state().phase, recovery::Phase::Complete);
        let copies = fixture.dir.join(".recovered").join(&fixture.state().key);
        let incomplete = fs::read_dir(&copies).unwrap().next().unwrap().unwrap().path();
        assert!(incomplete.join("transcript.md").is_file()); assert!(!incomplete.join("transcript.json").exists());
        fs::remove_file(fixture.root.join("cutpoint")).unwrap();
        run_transcribe_job(&retry, &fixture.executable, &fixture.base);
        let state = fixture.state(); assert_eq!(state.phase, recovery::Phase::Complete); assert_eq!(state.meeting_id, Some(id));
        assert_ne!(state.publication.unwrap(), incomplete); assert!(incomplete.join("transcript.md").exists());
        assert_eq!(fs::read(fixture.dir.join("transcript.md")).unwrap(), manual_md);
        assert_eq!(fs::read(fixture.dir.join("transcript.json")).unwrap(), manual_json);
        assert_eq!(fixture.meetings().len(), 1); assert!(!fixture.dir.join(maintenance::PENDING_MARKER).exists());
        fs::remove_dir_all(fixture.root).unwrap();
    }
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f3_child_cutpoints\",\"iterations\":5}}");
}

#[test]
fn keyed_child_ignores_untracked_roll_files() {
    for _ in 0..5 {
        let (fixture, owner) = Fixture::new(); drop(owner);
        let extra = fixture.dir.join("mic.001.wav");
        let mut wav = hound::WavWriter::create(&extra, hound::WavSpec { channels: 1, sample_rate: 16_000,
            bits_per_sample: 32, sample_format: hound::SampleFormat::Float }).unwrap();
        for _ in 0..80_000 { wav.write_sample(-0.75f32).unwrap(); } wav.finalize().unwrap();
        let before = fs::read(&extra).unwrap();
        run_transcribe_job(&fixture.queued_job(), &fixture.executable, &fixture.base);
        let rows = fixture.meetings(); assert_eq!(rows.len(), 1);
        assert!((rows[0].duration_secs - 3.0).abs() < 0.001, "uncheckpointed roll entered the archived duration");
        assert_eq!(rows[0].segment_count, 2);
        assert_eq!(fs::read(&extra).unwrap(), before);
        assert_eq!(maintenance::prune_sessions(&fixture.root.join("sessions"), 30, START + 1000 * 86_400).unwrap().removed, 0);
        fs::remove_dir_all(fixture.root).unwrap();
    }
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f1_trusted_roll_inventory\",\"iterations\":5}}");
}

#[test]
fn ordinary_no_store_child_succeeds_without_a_database() {
    for _ in 0..5 {
        let (mut fixture, owner) = Fixture::new(); drop(owner);
        fs::remove_file(fixture.dir.join(recovery::STATE_FILE)).unwrap();
        let sentinel = fixture.root.join("not-a-database"); fs::create_dir(&sentinel).unwrap();
        fixture.opts.no_store = true; fixture.opts.session_key = None; fixture.opts.db_path = sentinel.clone();
        let child = std::process::Command::new(&fixture.executable).arg("transcribe")
            .args(child_args(&fixture.dir, &fixture.opts, &fixture.base)).current_dir(&fixture.dir).output().unwrap();
        assert!(child.status.success(), "no-store child failed: {child:?}");
        assert!(String::from_utf8_lossy(&child.stdout).contains("synthetic release checklist"));
        assert!(sentinel.is_dir()); assert!(!fixture.root.join("archive.db").exists());
        assert!(fixture.dir.join("transcript.json").is_file()); fs::remove_dir_all(fixture.root).unwrap();
    }
}
