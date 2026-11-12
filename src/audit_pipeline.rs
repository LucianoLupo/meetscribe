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
    cutpoint("engine_loaded");
    Ok(Box::new(SyntheticEngine { root, db: opts.db_path.clone(), no_store: opts.no_store }))
}
fn cutpoint(point: &str) {
    let root = PathBuf::from(std::env::var_os("MEETSCRIBE_PIPELINE_FIXTURE_ROOT").unwrap());
    if fs::read_to_string(root.join("holdpoint")).is_ok_and(|selected| selected.trim() == point) {
        recovery::atomic_json(&root.join("hold-ack.json"), &serde_json::json!({"event":"owner_held", "point":point})).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        while !root.join("release-owner").exists() {
            assert!(std::time::Instant::now() < deadline, "parent never released the held test owner");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
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
    recovery::atomic_json(&root.join(format!("entry-{}.json", std::process::id())),
        &serde_json::json!({"event":"pipeline_child_entered", "pid":std::process::id()})).unwrap();
    println!("pipeline_child_entered");
    let argv: Vec<String> = fs::read(root.join("argv")).unwrap().split(|byte| *byte == 0)
        .filter(|part| !part.is_empty()).map(|part| String::from_utf8(part.to_vec()).unwrap()).collect();
    let now: i64 = fs::read_to_string(root.join("clock")).unwrap().trim().parse().unwrap();
    let result = match argv.first().map(String::as_str) {
        Some("transcribe") => crate::run_transcribe_using(&argv[1..], |dir, opts| pipeline::transcribe_with_clock(dir, opts, || now)),
        Some("rerender") => crate::run_rerender(&argv[1..]),
        Some("export") => crate::run_export(&argv[1..]),
        _ => panic!("unsupported sacrificial CLI command"),
    };
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

fn transcribe_argv(fixture: &Fixture, job: &TranscribeJob) -> Vec<String> {
    let mut argv = vec!["transcribe".into()]; argv.extend(child_args(&job.dir, &job.opts, &fixture.base)); argv
}

fn spawn_cli(fixture: &Fixture, argv: &[String]) -> std::process::Child {
    std::process::Command::new(&fixture.executable).args(argv).current_dir(&fixture.dir).env("HOME", &fixture.root)
        .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap()
}

fn child_output(fixture: &Fixture, child: std::process::Child) -> std::process::Output {
    let pid = child.id(); let output = child.wait_with_output().unwrap();
    let entry: serde_json::Value = serde_json::from_slice(&fs::read(fixture.root.join(format!("entry-{pid}.json")))
        .expect("child selector matched zero tests or skipped its entry")).unwrap();
    assert_eq!(entry["event"], "pipeline_child_entered"); assert_eq!(entry["pid"], pid);
    assert!(String::from_utf8_lossy(&output.stdout).contains("pipeline_child_entered")); output
}

fn wait_for_owner(fixture: &Fixture, point: &str) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !fixture.root.join("hold-ack.json").exists() {
        assert!(std::time::Instant::now() < deadline, "child never acknowledged ownership at {point}");
        std::thread::sleep(Duration::from_millis(2));
    }
    let ack: serde_json::Value = serde_json::from_slice(&fs::read(fixture.root.join("hold-ack.json")).unwrap()).unwrap();
    assert_eq!(ack["point"], point);
}

fn session_snapshot(dir: &Path) -> std::collections::BTreeMap<PathBuf, Vec<u8>> {
    let mut files = std::collections::BTreeMap::new();
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() { files.extend(session_snapshot(&entry.path())); }
        else { files.insert(entry.path(), fs::read(entry.path()).unwrap()); }
    }
    files
}

#[test]
fn manual_transcription_and_recovery_exclude_each_other_without_side_effects() {
    for no_store in [false, true] {
        for _ in 0..5 {
            for manual_first in [false, true] {
                let (fixture, owner) = Fixture::new(); drop(owner);
                let recovery = fixture.queued_job(); let mut manual = fixture.job();
                manual.opts.session_key = None; manual.opts.no_store = no_store;
                let sentinel = fixture.root.join("database-must-not-open");
                if no_store { fs::create_dir(&sentinel).unwrap(); manual.opts.db_path = sentinel.clone(); }
                fs::write(fixture.root.join("holdpoint"), "engine_loaded\n").unwrap();
                let first = if manual_first { &manual } else { &recovery };
                let contender = if manual_first { &recovery } else { &manual };
                let owner = spawn_cli(&fixture, &transcribe_argv(&fixture, first)); wait_for_owner(&fixture, "engine_loaded");
                let before = session_snapshot(&fixture.dir); let database = fs::read(&fixture.opts.db_path).ok();
                let blocked = child_output(&fixture, spawn_cli(&fixture, &transcribe_argv(&fixture, contender)));
                assert_eq!(blocked.status.code(), Some(1));
                assert!(String::from_utf8_lossy(&blocked.stderr).contains("\"error_class\":\"session_busy\""));
                assert_eq!(session_snapshot(&fixture.dir), before); assert_eq!(fs::read(&fixture.opts.db_path).ok(), database);
                fs::write(fixture.root.join("release-owner"), b"release\n").unwrap();
                let first = child_output(&fixture, owner); assert!(first.status.success(), "owner failed: {first:?}");
                let manual_pair = manual_first.then(|| (fs::read(fixture.dir.join("transcript.md")).unwrap(), fs::read(fixture.dir.join("transcript.json")).unwrap()));
                let retried = child_output(&fixture, spawn_cli(&fixture, &transcribe_argv(&fixture, contender)));
                assert!(retried.status.success(), "released contender failed: {retried:?}");
                let state = fixture.state(); assert_eq!(state.phase, recovery::Phase::Complete); assert_eq!(state.attempts, 1);
                assert!(state.unarchived_manual_output); assert_eq!(fixture.meetings().len(), if no_store { 1 } else { 2 });
                if let Some((md, json)) = manual_pair {
                    assert_eq!(fs::read(fixture.dir.join("transcript.md")).unwrap(), md);
                    assert_eq!(fs::read(fixture.dir.join("transcript.json")).unwrap(), json);
                    assert_ne!(state.publication.unwrap(), fixture.dir);
                }
                if no_store { assert!(sentinel.is_dir()); assert_eq!(fs::read_dir(sentinel).unwrap().count(), 0); }
                assert_eq!(maintenance::prune_sessions(&fixture.root.join("sessions"), 30, crate::now_epoch() + 100 * 86_400).unwrap().removed, 0);
                fs::remove_dir_all(fixture.root).unwrap();
            }
        }
        let case = if no_store { "f3_manual_no_store_race" } else { "f3_manual_store_race" };
        println!(r#"{{"event":"audit_case_complete","case":"{case}","iterations":5,"orderings":2}}"#);
    }
}

#[test]
fn manual_managed_transcription_uses_trusted_prefixes_without_keyed_ingest() {
    for _ in 0..5 {
        for no_store in [false, true] {
            let (fixture, owner) = Fixture::new(); drop(owner);
            for (name, frames) in [("mic.wav", 64_000), ("mic.001.wav", 32_000), ("system.001.wav", 32_000)] {
                let mut wav = hound::WavWriter::create(fixture.dir.join(name), hound::WavSpec { channels: 1, sample_rate: 16_000,
                    bits_per_sample: 32, sample_format: hound::SampleFormat::Float }).unwrap();
                for _ in 0..frames { wav.write_sample(0.75f32).unwrap(); } wav.finalize().unwrap();
            }
            let before = ["mic.wav", "system.wav", "mic.001.wav", "system.001.wav"].map(|name| (name, fs::read(fixture.dir.join(name)).unwrap()));
            let mut manual = fixture.job(); manual.opts.session_key = None; manual.opts.no_store = no_store;
            let sentinel = fixture.root.join("database-must-not-open");
            if no_store { fs::create_dir(&sentinel).unwrap(); manual.opts.db_path = sentinel.clone(); }
            let output = child_output(&fixture, spawn_cli(&fixture, &transcribe_argv(&fixture, &manual)));
            assert!(output.status.success(), "manual child failed: {output:?}");
            let rendered: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(fixture.dir.join("transcript.json")).unwrap()).unwrap();
            assert_eq!(rendered.len(), 2, "manual managed run ingested an untracked roll");
            for segment in rendered { assert_eq!(segment["t_start"], 0.0); assert_eq!(segment["t_end"], 3.0, "uncommitted tail entered manual output"); }
            let state = fixture.state(); assert_eq!(state.attempts, 0); assert!(state.unarchived_manual_output);
            if no_store { assert_eq!(fs::read_dir(sentinel).unwrap().count(), 0); assert!(!fixture.opts.db_path.exists()); }
            else {
                crate::new_runtime().unwrap().block_on(async {
                    let mut archive = db::Db::open(&fixture.opts.db_path).await.unwrap();
                    assert_eq!(archive.list_meetings().await.unwrap().len(), 1);
                    assert_eq!(archive.ingest_id(&state.key).await.unwrap(), None); archive.close().await.unwrap();
                });
            }
            for (name, bytes) in before { assert_eq!(fs::read(fixture.dir.join(name)).unwrap(), bytes); }
            fs::remove_dir_all(fixture.root).unwrap();
        }
    }
    println!(r#"{{"event":"audit_case_complete","case":"f3_manual_trusted_prefix","iterations":5}}"#);
}

#[test]
fn manual_publication_and_recovery_hold_the_same_session_lock() {
    for _ in 0..5 {
        for command in ["rerender", "export"] {
            for manual_first in [false, true] {
                let (fixture, owner) = Fixture::new(); drop(owner);
                run_transcribe_job(&fixture.queued_job(), &fixture.executable, &fixture.base);
                let state = fixture.state(); assert_eq!(state.phase, recovery::Phase::Complete); let id = state.meeting_id.unwrap();
                fs::write(fixture.root.join("forbid-inference"), b"existing mapping only\n").unwrap();
                fs::write(fixture.dir.join("transcript.md"), b"# Operator release revision\n").unwrap();
                let mut recovery = fixture.job(); recovery.opts.resume = true;
                let recovery_args = transcribe_argv(&fixture, &recovery);
                let mut manual_args = vec![command.to_string(), id.to_string(), "--db".into(), fixture.opts.db_path.display().to_string()];
                if command == "rerender" { manual_args.push("--write".into()); }
                else { manual_args.extend(["--export-dir".into(), fixture.dir.display().to_string()]); }
                let point = if manual_first { "manual_markdown_published" } else { "markdown_published" };
                fs::write(fixture.root.join("holdpoint"), format!("{point}\n")).unwrap();
                let (first, contender) = if manual_first { (&manual_args, &recovery_args) } else { (&recovery_args, &manual_args) };
                let owner = spawn_cli(&fixture, first); wait_for_owner(&fixture, point);
                let before = session_snapshot(&fixture.dir);
                let blocked = child_output(&fixture, spawn_cli(&fixture, contender));
                assert_eq!(blocked.status.code(), Some(1));
                assert!(String::from_utf8_lossy(&blocked.stderr).contains("\"error_class\":\"session_busy\""));
                assert_eq!(session_snapshot(&fixture.dir), before);
                fs::write(fixture.root.join("release-owner"), b"release\n").unwrap();
                assert!(child_output(&fixture, owner).status.success());
                let publication = if manual_first { None } else { let selected = fixture.state().publication.unwrap();
                    Some((selected.clone(), fs::read(selected.join("transcript.md")).unwrap(), fs::read(selected.join("transcript.json")).unwrap())) };
                let basename = if command == "rerender" { "transcript".to_string() } else { format!("meeting-{id}") };
                let manual_pair = manual_first.then(|| (fs::read(fixture.dir.join(format!("{basename}.md"))).unwrap(), fs::read(fixture.dir.join(format!("{basename}.json"))).unwrap()));
                assert!(child_output(&fixture, spawn_cli(&fixture, contender)).status.success());
                if let Some((md, json)) = manual_pair {
                    assert_eq!(fs::read(fixture.dir.join(format!("{basename}.md"))).unwrap(), md);
                    assert_eq!(fs::read(fixture.dir.join(format!("{basename}.json"))).unwrap(), json);
                }
                if let Some((selected, md, json)) = publication {
                    assert_eq!(fs::read(selected.join("transcript.md")).unwrap(), md); assert_eq!(fs::read(selected.join("transcript.json")).unwrap(), json);
                }
                assert_eq!(fixture.meetings().len(), 1); assert_eq!(fixture.state().meeting_id, Some(id));
                assert!(fixture.state().unarchived_manual_output);
                fs::remove_dir_all(fixture.root).unwrap();
            }
        }
    }
    println!(r#"{{"event":"audit_case_complete","case":"f3_manual_publication_race","iterations":5,"orderings":2,"commands":2}}"#);
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

fn wait_for_marker_reconciliation(worker: &LaunchWorker, marker: &Path) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while marker.exists() {
        assert!(std::time::Instant::now() < deadline, "worker never reconciled proven Complete marker");
        assert_eq!(worker.scanned.recv_timeout(Duration::from_secs(5)).unwrap(), 0);
    }
    assert!(worker.finished.try_recv().is_err(), "Complete reconciliation must never launch another ingest");
}

#[test]
fn completed_child_crash_and_worker_restart_reconcile_pending_without_ingest() {
    for _ in 0..5 {
        let (fixture, owner) = Fixture::new(); drop(owner);
        fs::write(fixture.root.join("cutpoint"), "session_complete\n").unwrap();
        run_transcribe_job(&fixture.queued_job(), &fixture.executable, &fixture.base);
        let ack: serde_json::Value = serde_json::from_slice(&fs::read(fixture.root.join("cutpoint-ack.json")).unwrap()).unwrap();
        assert_eq!(ack["point"], "session_complete");
        let state = fixture.state(); assert_eq!(state.phase, recovery::Phase::Complete); assert_eq!(state.attempts, 1);
        let marker = fixture.dir.join(maintenance::PENDING_MARKER);
        assert!(!marker.exists(), "a nonzero child exit after proven Complete must reconcile its marker");
        let metadata = fs::read(fixture.dir.join(recovery::STATE_FILE)).unwrap();
        let microphone = fs::read(fixture.dir.join("mic.wav")).unwrap();
        fs::write(fixture.root.join("forbid-inference"), b"no further inference\n").unwrap();
        // Simulate parent death before cleanup; startup scanning must reconcile without a job.
        fixture.queued_job();
        let worker = LaunchWorker::start(&fixture, fixture.root.join("unspawnable-meetscribe"), START + 100_000);
        wait_for_marker_reconciliation(&worker, &marker); worker.idle();
        // The same reconciliation must also happen while the worker is already idle.
        fixture.queued_job();
        wait_for_marker_reconciliation(&worker, &marker); worker.stop();
        assert_eq!(fs::read(fixture.dir.join(recovery::STATE_FILE)).unwrap(), metadata);
        assert_eq!(fs::read(fixture.dir.join("mic.wav")).unwrap(), microphone);
        assert_eq!(fixture.meetings().len(), 1); assert_eq!(fixture.state().meeting_id, state.meeting_id);
        assert_eq!(maintenance::prune_sessions(&fixture.root.join("sessions"), 30, crate::now_epoch() + 100 * 86_400).unwrap().removed, 1);
        fs::remove_dir_all(fixture.root).unwrap();
    }
    println!(r#"{{"event":"audit_case_complete","case":"f2_complete_marker_reconcile","iterations":5}}"#);
}

#[test]
fn complete_marker_reconciliation_preserves_busy_and_unproven_material() {
    for _ in 0..5 {
        for scenario in ["busy", "missing_archive", "wrong_archive", "edited_markdown", "missing_json", "invalid_state", "marker_directory", "marker_symlink"] {
            let (fixture, owner) = Fixture::new(); drop(owner);
            run_transcribe_job(&fixture.queued_job(), &fixture.executable, &fixture.base);
            assert_eq!(fixture.state().phase, recovery::Phase::Complete);
            let marker = fixture.dir.join(maintenance::PENDING_MARKER);
            fixture.queued_job();
            let state_path = fixture.dir.join(recovery::STATE_FILE);
            let metadata = fs::read(&state_path).unwrap();
            let markdown = fs::read(fixture.dir.join("transcript.md")).unwrap();
            let json = fs::read(fixture.dir.join("transcript.json")).unwrap();
            let wav = fs::read(fixture.dir.join("mic.wav")).unwrap();
            let saved_archive = fixture.root.join("saved-archive.db");
            let mut busy = None;
            match scenario {
                "busy" => busy = recovery::Session::open(&fixture.dir).unwrap(),
                "missing_archive" | "wrong_archive" => {
                    fs::rename(&fixture.opts.db_path, &saved_archive).unwrap();
                    if scenario == "wrong_archive" {
                        crate::new_runtime().unwrap().block_on(async {
                            let archive = db::Db::open(&fixture.opts.db_path).await.unwrap(); archive.close().await.unwrap();
                        });
                    }
                }
                "edited_markdown" => fs::write(fixture.dir.join("transcript.md"), b"# Operator amended release decision\n").unwrap(),
                "missing_json" => fs::remove_file(fixture.dir.join("transcript.json")).unwrap(),
                "invalid_state" => fs::write(&state_path, b"unverified state\n").unwrap(),
                "marker_directory" => {
                    fs::remove_file(&marker).unwrap(); fs::create_dir(&marker).unwrap();
                    fs::write(marker.join("keep.md"), b"Retain this operator note\n").unwrap();
                }
                "marker_symlink" => {
                    fs::remove_file(&marker).unwrap(); let target = fixture.root.join("keep.md");
                    fs::write(&target, b"Retain this operator note\n").unwrap();
                    std::os::unix::fs::symlink(target, &marker).unwrap();
                }
                _ => unreachable!(),
            }
            let faulted_state = fs::read(&state_path).unwrap();
            let worker = LaunchWorker::start(&fixture, fixture.root.join("unspawnable-meetscribe"), START + 100_000);
            worker.idle(); assert!(marker.exists(), "unproven marker removed: {scenario}");
            assert_eq!(fs::read(&state_path).unwrap(), faulted_state);
            assert_eq!(fs::read(fixture.dir.join("mic.wav")).unwrap(), wav);
            if scenario == "missing_archive" { assert!(!fixture.opts.db_path.exists(), "proof must not initialize missing archive"); }
            match scenario {
                "busy" => drop(busy),
                "missing_archive" | "wrong_archive" => {
                    if scenario == "wrong_archive" { fs::remove_file(&fixture.opts.db_path).unwrap(); }
                    fs::rename(saved_archive, &fixture.opts.db_path).unwrap();
                }
                "edited_markdown" => fs::write(fixture.dir.join("transcript.md"), &markdown).unwrap(),
                "missing_json" => fs::write(fixture.dir.join("transcript.json"), &json).unwrap(),
                "invalid_state" => fs::write(&state_path, &metadata).unwrap(),
                "marker_directory" => {
                    assert_eq!(fs::read(marker.join("keep.md")).unwrap(), b"Retain this operator note\n");
                    fs::remove_dir_all(&marker).unwrap(); fixture.queued_job();
                }
                "marker_symlink" => {
                    assert_eq!(fs::read(fixture.root.join("keep.md")).unwrap(), b"Retain this operator note\n");
                    fs::remove_file(&marker).unwrap(); fixture.queued_job();
                }
                _ => unreachable!(),
            }
            wait_for_marker_reconciliation(&worker, &marker); worker.stop();
            assert_eq!(fs::read(&state_path).unwrap(), metadata); assert_eq!(fixture.meetings().len(), 1);
            fs::remove_dir_all(fixture.root).unwrap();
        }
    }
    println!(r#"{{"event":"audit_case_complete","case":"f2_complete_marker_guards","iterations":5}}"#);
}

#[test]
fn complete_reconciliation_preserves_unknown_regular_marker_contents() {
    for _ in 0..5 {
        let (fixture, owner) = Fixture::new(); drop(owner);
        run_transcribe_job(&fixture.queued_job(), &fixture.executable, &fixture.base);
        let marker = fixture.dir.join(maintenance::PENDING_MARKER);
        let notes = b"# Operator notes\nRetain the amended launch decision.\n";
        fs::write(&marker, notes).unwrap();
        let worker = LaunchWorker::start(&fixture, fixture.root.join("unspawnable-meetscribe"), START + 100_000);
        worker.idle(); assert_eq!(fs::read(&marker).expect("unknown regular marker was deleted"), notes);
        let retained = fixture.root.join("retained-operator-notes.md"); fs::rename(&marker, &retained).unwrap();
        fixture.queued_job(); wait_for_marker_reconciliation(&worker, &marker); worker.stop();
        assert_eq!(fs::read(retained).unwrap(), notes);
        assert_eq!(fixture.meetings().len(), 1); fs::remove_dir_all(fixture.root).unwrap();
    }
    println!(r#"{{"event":"audit_case_complete","case":"f2_pending_marker_identity","iterations":5}}"#);
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
fn managed_roll_timestamps_and_playback_share_the_committed_channel_boundary() {
    for _ in 0..5 {
        for uncommitted_tail in [false, true] {
            let (fixture, mut owner) = Fixture::new();
            let write = |path: PathBuf, rate, frames, value: f32| {
                let mut wav = hound::WavWriter::create(path, hound::WavSpec { channels: 1, sample_rate: rate,
                    bits_per_sample: 32, sample_format: hound::SampleFormat::Float }).unwrap();
                for _ in 0..frames { wav.write_sample(value).unwrap(); } wav.finalize().unwrap();
            };
            write(fixture.dir.join("mic.wav"), 48_000, if uncommitted_tail { 240_000 } else { 192_000 }, 0.25);
            write(fixture.dir.join("system.wav"), 48_000, if uncommitted_tail { 192_000 } else { 144_000 }, -0.5);
            owner.state.phase = recovery::Phase::Recording; owner.state.rolls[0].sample_rate = 48_000;
            owner.checkpoint(0, 192_000, 144_000).unwrap();
            owner.start_roll(1, 16_000, 0.25).unwrap();
            write(fixture.dir.join("mic.001.wav"), 16_000, 16_000, 0.75);
            write(fixture.dir.join("system.001.wav"), 16_000, 16_000, -0.75);
            owner.checkpoint(1, 16_000, 16_000).unwrap(); owner.finish_capture(5.25, 0.1).unwrap(); drop(owner);
            let original: Vec<_> = ["mic.wav", "system.wav", "mic.001.wav", "system.001.wav"].into_iter()
                .map(|name| (name, fs::read(fixture.dir.join(name)).unwrap())).collect();
            run_transcribe_job(&fixture.queued_job(), &fixture.executable, &fixture.base);
            assert_eq!(fixture.state().phase, recovery::Phase::Complete);
            let segments = crate::new_runtime().unwrap().block_on(async {
                let mut archive = db::Db::open_existing(&fixture.opts.db_path, fixture.state().archive_id.as_deref().unwrap()).await.unwrap();
                let stored = archive.load_segments(fixture.state().meeting_id.unwrap()).await.unwrap(); archive.close().await.unwrap(); stored
            });
            let second: Vec<_> = segments.iter().filter(|segment| segment.seg.t_start > 1.0).collect();
            assert_eq!(second.len(), 2);
            for segment in second { assert!((segment.seg.t_start - 4.25).abs() < 1e-9,
                "second roll drifted by channel: {:?}, start={}", segment.seg.speaker, segment.seg.t_start); }
            assert!((fixture.meetings()[0].duration_secs - 5.25).abs() < 1e-9);
            let clips = crate::voices::slice_far_end(&fixture.dir, &[(4.25, 5.25)]).unwrap();
            let clip = clips[0].as_ref().expect("stored roll timestamps must select the same playback audio");
            assert_eq!(clip.len(), 16_000); assert!(clip.iter().all(|sample| (*sample + 0.75).abs() < 1e-4));
            assert!(crate::voices::slice_far_end(&fixture.dir, &[(3.5, 3.75)]).unwrap()[0].is_none(), "playback used an uncommitted system tail");
            fs::rename(fixture.dir.join("system.wav"), fixture.root.join("saved-system.wav")).unwrap();
            assert!(crate::voices::slice_far_end(&fixture.dir, &[(0.0, 0.5)]).is_err(), "missing roll zero must not remap roll one to offset zero");
            fs::rename(fixture.root.join("saved-system.wav"), fixture.dir.join("system.wav")).unwrap();
            for (name, bytes) in original { assert_eq!(fs::read(fixture.dir.join(name)).unwrap(), bytes); }
            fs::remove_dir_all(fixture.root).unwrap();
        }
    }
    println!(r#"{{"event":"audit_case_complete","case":"f1_shared_roll_timeline","iterations":5}}"#);
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
fn retained_legacy_child_does_not_authorize_adoption_or_block_new_capture() {
    use std::io::Write;
    for _ in 0..5 {
        let (fixture, owner) = Fixture::new(); drop(owner);
        let legacy = fixture.root.join("sessions/legacy"); let markerless = fixture.root.join("sessions/legacy-markerless");
        for dir in [&legacy, &markerless] {
            fs::create_dir(dir).unwrap(); fs::copy(fixture.dir.join("mic.wav"), dir.join("mic.wav")).unwrap();
            fs::write(dir.join("transcript.json"), b"[{\"text\":\"Retain the legacy release decision\"}]\n").unwrap();
        }
        fs::write(legacy.join(maintenance::PENDING_MARKER), b"Legacy call\nLegacy release review\n").unwrap();
        let before = session_snapshot(&legacy); let markerless_before = session_snapshot(&markerless);
        let mut child = std::process::Command::new("/bin/sh").args(["-c", "printf ready > \"$1/legacy-ready\"; IFS= read -r release; printf 'Legacy child completed normally\\n' > \"$1/sessions/legacy/transcript.md\"", "legacy-fixture"])
            .arg(&fixture.root).stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !fixture.root.join("legacy-ready").exists() { assert!(std::time::Instant::now() < deadline); std::thread::sleep(Duration::from_millis(2)); }
        assert!(child.try_wait().unwrap().is_none());
        let capture_dir = fixture.root.join("sessions/new-capture");
        let capture = recovery::Session::begin(&capture_dir, &fixture.opts.db_path, &capture_dir,
            "org.example.synthetic-call", "New capture during legacy overlap", START + 20).unwrap();
        assert!(recovery::Session::open(&capture_dir).unwrap().is_none());
        let selected = pending_sessions_at(&fixture.root.join("sessions"), START + 1000);
        assert!(selected.iter().any(|(path, _, _)| path == &fixture.dir));
        assert!(selected.iter().all(|(path, _, _)| path != &legacy && path != &markerless));
        assert_eq!(maintenance::prune_sessions(&fixture.root.join("sessions"), 30, crate::now_epoch() + 100 * 86_400).unwrap().removed, 0);
        assert_eq!(session_snapshot(&legacy), before); assert_eq!(session_snapshot(&markerless), markerless_before);
        assert!(!legacy.join(recovery::STATE_FILE).exists()); assert!(!fixture.opts.db_path.exists());
        child.stdin.take().unwrap().write_all(b"finish\n").unwrap(); assert!(child.wait().unwrap().success());
        assert!(legacy.join("transcript.md").exists()); drop(capture);
        assert!(pending_sessions_at(&fixture.root.join("sessions"), START + 1000).iter().all(|(path, _, _)| path != &legacy));
        assert_eq!(maintenance::prune_sessions(&fixture.root.join("sessions"), 30, crate::now_epoch() + 100 * 86_400).unwrap().removed, 0);
        fs::remove_dir_all(fixture.root).unwrap();
    }
    println!(r#"{{"event":"audit_case_complete","case":"f3_legacy_overlap_quarantine","iterations":5}}"#);
}

#[test]
fn enqueue_preserves_unknown_regular_markers_and_symlink_targets() {
    for _ in 0..5 {
        for symlink in [false, true] {
            let (fixture, owner) = Fixture::new(); drop(owner);
            let marker = fixture.dir.join(maintenance::PENDING_MARKER); let target = fixture.root.join("operator-notes.md");
            let notes = b"# Operator notes\nKeep the signed-off release decision.\n";
            if symlink { fs::write(&target, notes).unwrap(); std::os::unix::fs::symlink(&target, &marker).unwrap(); }
            else { fs::write(&marker, notes).unwrap(); }
            assert!(pending_sessions_at(&fixture.root.join("sessions"), START + 10).iter().any(|(path, _, _)| path == &fixture.dir));
            let job = fixture.queued_job();
            assert_eq!(fs::read(&marker).unwrap(), notes, "startup enqueue overwrote unknown marker material");
            assert_eq!(fs::symlink_metadata(&marker).unwrap().file_type().is_symlink(), symlink);
            run_transcribe_job(&job, &fixture.executable, &fixture.base);
            assert_eq!(fixture.state().phase, recovery::Phase::Complete); assert_eq!(fs::read(&marker).unwrap(), notes);
            if symlink { assert_eq!(fs::read(target).unwrap(), notes); }
            fs::remove_dir_all(fixture.root).unwrap();
        }
    }
    println!(r#"{{"event":"audit_case_complete","case":"f2_queue_preserves_marker_material","iterations":5}}"#);
}

#[test]
fn manual_cross_session_destination_is_locked_and_protected() {
    for _ in 0..5 {
        for no_store in [false, true] {
            let (source, owner) = Fixture::new(); drop(owner);
            let (destination, owner) = Fixture::new(); drop(owner);
            fs::write(destination.root.join("holdpoint"), "engine_loaded\n").unwrap();
            let recovery = destination.queued_job();
            let owner = spawn_cli(&destination, &transcribe_argv(&destination, &recovery)); wait_for_owner(&destination, "engine_loaded");
            let before = session_snapshot(&destination.dir); let source_before = session_snapshot(&source.dir);
            let mut manual = source.job(); manual.opts.session_key = None; manual.opts.no_store = no_store; manual.opts.export_dir = destination.dir.clone();
            let sentinel = source.root.join("database-must-not-open");
            if no_store { fs::create_dir(&sentinel).unwrap(); manual.opts.db_path = sentinel.clone(); }
            let blocked = child_output(&source, spawn_cli(&source, &transcribe_argv(&source, &manual)));
            assert_eq!(blocked.status.code(), Some(1), "manual source wrote a different busy managed destination");
            assert_eq!(session_snapshot(&destination.dir), before); assert_eq!(session_snapshot(&source.dir), source_before);
            assert!(!source.opts.db_path.exists());
            fs::write(destination.root.join("release-owner"), b"release\n").unwrap(); assert!(child_output(&destination, owner).status.success());
            let output = child_output(&source, spawn_cli(&source, &transcribe_argv(&source, &manual))); assert!(output.status.success());
            assert!(destination.state().unarchived_manual_output); assert_eq!(destination.meetings().len(), 1);
            assert_eq!(maintenance::prune_sessions(&destination.root.join("sessions"), 30, crate::now_epoch() + 100 * 86_400).unwrap().removed, 0);
            if no_store { assert_eq!(fs::read_dir(sentinel).unwrap().count(), 0); assert!(!source.opts.db_path.exists()); }
            fs::remove_dir_all(source.root).unwrap(); fs::remove_dir_all(destination.root).unwrap();
        }
    }
    println!(r#"{{"event":"audit_case_complete","case":"f3_cross_session_destination","iterations":5}}"#);
}

#[test]
fn managed_playback_rejects_symlinks_and_reads_recoverable_checkpoint_bytes() {
    for _ in 0..5 {
        let (fixture, owner) = Fixture::new(); drop(owner);
        let system = fixture.dir.join("system.wav"); let saved = fixture.root.join("saved-system.wav");
        fs::rename(&system, &saved).unwrap();
        std::os::unix::fs::symlink(fixture.dir.join("mic.wav"), &system).unwrap();
        assert!(crate::voices::slice_far_end(&fixture.dir, &[(0.1, 0.2)]).is_err(), "managed playback followed a substituted capture symlink");
        fs::remove_file(&system).unwrap(); fs::rename(saved, &system).unwrap();
        let mut bytes = fs::read(&system).unwrap();
        let data = bytes.windows(4).position(|window| window == b"data").unwrap();
        bytes[data + 4..data + 8].copy_from_slice(&0u32.to_le_bytes()); fs::write(&system, &bytes).unwrap();
        let slices = crate::voices::slice_far_end(&fixture.dir, &[(0.1, 0.2)]).expect("trusted RIFF prefix should be readable despite stale declared data length");
        assert_eq!(slices[0].as_ref().unwrap().len(), 1600);
        assert!(slices[0].as_ref().unwrap().iter().all(|sample| *sample == -0.5));
        run_transcribe_job(&fixture.queued_job(), &fixture.executable, &fixture.base);
        assert_eq!(fixture.state().phase, recovery::Phase::Complete); assert_eq!(fixture.meetings().len(), 1);
        assert_eq!(fs::read(system).unwrap(), bytes); fs::remove_dir_all(fixture.root).unwrap();
    }
    println!(r#"{{"event":"audit_case_complete","case":"f1_playback_committed_reader","iterations":5}}"#);
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
        assert!(String::from_utf8_lossy(&child.stdout).contains("pipeline_child_entered"));
        assert!(sentinel.is_dir()); assert!(!fixture.root.join("archive.db").exists());
        assert!(fixture.dir.join("transcript.json").is_file()); fs::remove_dir_all(fixture.root).unwrap();
    }
    println!(r#"{{"event":"audit_case_complete","case":"f3_unmanaged_no_store_child","iterations":5}}"#);
}
