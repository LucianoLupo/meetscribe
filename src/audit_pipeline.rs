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
    fn transcribe(&mut self, audio: &[f32], _lang: &str, words: bool) -> Result<(String, f32, Vec<asr::Word>)> {
        if !self.no_store && self.root.join("storage-fault").exists() {
            crate::new_runtime()?.block_on(async {
                let mut connection = sqlx::SqliteConnection::connect_with(&sqlx::sqlite::SqliteConnectOptions::new()
                    .filename(&self.db).create_if_missing(false)).await?;
                sqlx::query("CREATE TRIGGER IF NOT EXISTS audit_storage_fault BEFORE INSERT ON meetings BEGIN SELECT RAISE(ABORT, 'synthetic archive fault'); END")
                    .execute(&mut connection).await?;
                connection.close().await?; anyhow::Ok(())
            })?;
        }
        // Far-end windows of a run that splits are decoded with word timings, after the diarizer started.
        cutpoint(if words { "decoding_far_end" } else { "decoding" });
        let text = if audio.first().is_some_and(|value| *value < 0.0) {
            "We approved the synthetic deployment window"
        } else { "The synthetic release checklist is ready" };
        // A run that splits far-end chunks asks for word timings: one word in each half of the window.
        let half = audio.len() as f64 / 32_000.0;
        let timed = if words { vec![asr::Word { text: "approved".into(), t_start: 0.2, t_end: half - 0.2 },
            asr::Word { text: "deployed".into(), t_start: half + 0.2, t_end: 2.0 * half - 0.2 }] } else { Vec::new() };
        Ok((text.into(), 0.94, timed))
    }
    /// One voice for every fixture, and a second one, orthogonal to it, for audio below -0.7.
    fn embed(&mut self, audio: &[f32]) -> Result<Option<Vec<f32>>> {
        let second = audio.first().is_some_and(|value| *value < -0.7);
        let sign = |axis: usize| if second && axis % 2 == 1 { -1.0f32 } else { 1.0 };
        Ok(Some((0..192).map(|axis| sign(axis) / 192.0f32.sqrt()).collect()))
    }
}
fn backend(_model: &str, opts: &pipeline::PipelineOpts) -> Result<Box<dyn pipeline::SpeechEngine>> {
    let root = PathBuf::from(std::env::var_os("MEETSCRIBE_PIPELINE_FIXTURE_ROOT").context("fixture root absent")?);
    // Written before the refusal below, so a test can tell "never asked for" from "asked and refused".
    fs::write(root.join("engine-ran"), b"an inference backend was requested\n")?;
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

/// A failure a test switches on by name: the file `fault` in the fixture root holds that name.
fn fault(point: &str) -> bool {
    let root = PathBuf::from(std::env::var_os("MEETSCRIBE_PIPELINE_FIXTURE_ROOT").unwrap());
    fs::read_to_string(root.join("fault")).is_ok_and(|selected| selected.trim() == point)
}

#[test]
#[ignore = "sacrificial entry invoked explicitly by mandatory audit parent tests"]
fn pipeline_child() {
    let root = PathBuf::from(std::env::var_os("MEETSCRIBE_PIPELINE_FIXTURE_ROOT").expect("fixture root required"));
    crate::diagnostics::init(); pipeline::test_engine(backend); pipeline::test_cutpoints(cutpoint); pipeline::test_faults(fault);
    // A promotion test may shorten the wait for the live reader: "<stale ms> <cap ms>".
    if let Ok(limits) = fs::read_to_string(root.join("promote-wait")) {
        let mut ms = limits.split_whitespace().map(|value| Duration::from_millis(value.parse().unwrap()));
        crate::promote::test_wait(crate::promote::Wait { stale: ms.next().unwrap(), cap: ms.next().unwrap(), poll: Duration::from_millis(5) });
    }
    // A finalize test may shorten the wait for a locked database: "<ms>".
    if let Ok(wait) = fs::read_to_string(root.join("busy-timeout-ms")) {
        crate::finalize::test_busy_timeout(Duration::from_millis(wait.trim().parse().unwrap()));
    }
    recovery::atomic_json(&root.join(format!("entry-{}.json", std::process::id())),
        &serde_json::json!({"event":"pipeline_child_entered", "pid":std::process::id()})).unwrap();
    println!("pipeline_child_entered");
    let argv: Vec<String> = fs::read(root.join("argv")).unwrap().split(|byte| *byte == 0)
        .filter(|part| !part.is_empty()).map(|part| String::from_utf8(part.to_vec()).unwrap()).collect();
    let now: i64 = fs::read_to_string(root.join("clock")).unwrap().trim().parse().unwrap();
    let result = match argv.first().map(String::as_str) {
        Some("transcribe") => crate::run_transcribe_using(&argv[1..], |dir, opts, promote_live| pipeline::transcribe_with_clock(dir, opts, promote_live, || now)),
        Some("rerender") => crate::run_rerender(&argv[1..]),
        Some("export") => crate::run_export(&argv[1..]),
        Some("finalize") => crate::finalize::run(&argv[1..]),
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

/// Segments and voice clusters of a stored meeting, as the database has them.
fn stored_voices(fixture: &Fixture, id: i64) -> (Vec<db::StoredSegment>, Vec<db::ClusterRow>) {
    crate::new_runtime().unwrap().block_on(async {
        let mut db = db::Db::open_existing(&fixture.opts.db_path, fixture.state().archive_id.as_deref().unwrap()).await.unwrap();
        let stored = (db.load_segments(id).await.unwrap(), db.list_clusters(id).await.unwrap());
        db.close().await.unwrap(); stored
    })
}

/// The far end of the fixture becomes two voices: 1.5 s of one, then 1.5 s of another, with the
/// diarizer's turns replayed from a file. Returns the fixture set up to split at that change.
fn two_voice_fixture() -> Fixture {
    let (mut fixture, owner) = Fixture::new(); drop(owner);
    let mut wav = hound::WavWriter::create(fixture.dir.join("system.wav"), hound::WavSpec {
        channels: 1, sample_rate: 16_000, bits_per_sample: 32, sample_format: hound::SampleFormat::Float }).unwrap();
    for frame in 0..48_000 { wav.write_sample(if frame < 24_000 { -0.5f32 } else { -0.9f32 }).unwrap(); } wav.finalize().unwrap();
    let turns = fixture.root.join("turns.rttm");
    fs::write(&turns, "SPEAKER far 1 0.000 1.500 <NA> <NA> speaker_0 <NA> <NA>\nSPEAKER far 1 1.500 1.500 <NA> <NA> speaker_1 <NA> <NA>\n").unwrap();
    fixture.opts.split = true; fixture.opts.diarizer_rttm = Some(turns);
    fixture
}

/// The batch pass with the far-end split on, which is what the daemon runs by default. The
/// expected values are frozen from the build before `transcribe_core` was divided in two (f138d42),
/// not regenerated from this code: they pin what that pass stores and writes.
#[test]
fn the_batch_pass_with_the_far_end_split_stores_what_it_did_before_it_was_divided() {
    let fixture = two_voice_fixture();
    let output = child_output(&fixture, spawn_cli(&fixture, &transcribe_argv(&fixture, &fixture.job())));
    assert!(output.status.success(), "split transcription failed: {output:?}");
    let rows = fixture.meetings(); assert_eq!(rows.len(), 1);
    let (segments, clusters) = stored_voices(&fixture, rows[0].id);
    let facts = format!("{}\n{}\n{}",
        segments.iter().map(|s| format!("{:?} {:.3}-{:.3} {:?} {}", s.seg.speaker, s.seg.t_start, s.seg.t_end, s.voice_cluster, s.seg.text)).collect::<Vec<_>>().join("\n"),
        clusters.iter().map(|c| format!("{} windows={} secs={:.3} speaker={:?}", c.cluster, c.n_windows, c.speech_secs, c.speaker_id)).collect::<Vec<_>>().join("\n"),
        fs::read_to_string(fixture.dir.join("transcript.json")).unwrap());
    assert_eq!(facts, SPLIT_FACTS);
    assert_eq!((rows[0].segment_count, rows[0].origin), (segments.len() as i64, db::TranscriptOrigin::Batch));
    assert_eq!(proofs(&fixture), (true, true));
    assert!(session_snapshot(&fixture.dir).keys().all(|path| !path.file_name().unwrap().to_str().unwrap().starts_with(".diar-")));
    fs::remove_dir_all(fixture.root).unwrap();
}
const SPLIT_FACTS: &str = r#"You 0.000-3.000 None The synthetic release checklist is ready
Others 0.000-1.500 Some("A") approved
Others 1.500-3.000 Some("B") deployed
A windows=1 secs=1.500 speaker=None
B windows=1 secs=1.500 speaker=None
[
  {
    "speaker": "you",
    "text": "The synthetic release checklist is ready",
    "t_start": 0.0,
    "t_end": 3.0,
    "confidence": 0.94
  },
  {
    "speaker": "others",
    "text": "approved",
    "t_start": 0.0,
    "t_end": 1.5,
    "confidence": 0.94,
    "voice_cluster": "A"
  },
  {
    "speaker": "others",
    "text": "deployed",
    "t_start": 1.5,
    "t_end": 3.0,
    "confidence": 0.94,
    "voice_cluster": "B"
  }
]"#;

// ---- Promotion: the live stream becomes the stored transcript (`promote.rs`). The stream is
// fictional and written by hand in the reader's format; the child is the real argv, parser and
// pipeline. `child_args` does not pass the flag yet (the daemon is wired later), so it is appended.

fn live_dir(fixture: &Fixture) -> PathBuf { fixture.root.join("live") }

fn stream_events(fixture: &Fixture) -> Vec<serde_json::Value> {
    let segment = |speaker: &str, t_start: f64, t_end: f64, text: &str| serde_json::json!({"type": "segment", "speaker": speaker,
        "t_start": t_start, "t_end": t_end, "text": text, "confidence": 0.9, "lag_ms": 3_400, "decode_ms": 900});
    vec![
        // "pt" on purpose: the child is started with "en" (`Fixture::new`), and the stored meeting
        // must say what the reader decoded in.
        serde_json::json!({"type": "meeting_started", "key": fixture.state().key, "app": "Chrome", "started_at": START, "model": "ggml-live-fixture", "lang": "pt"}),
        segment("you", 0.2, 1.1, "The live release checklist is ready"),
        serde_json::json!({"type": "heartbeat", "audio_secs": 2.0, "decoded_secs": 1.1}),
        segment("others", 1.4, 2.6, "We approved the live deployment window"),
        serde_json::json!({"type": "meeting_ended", "duration_secs": 3.0, "reason": "ended", "segments": 2, "short_windows": 1}),
    ]
}

fn write_stream(fixture: &Fixture, events: &[serde_json::Value]) -> PathBuf {
    recovery::private_dir(&live_dir(fixture)).unwrap();
    let path = live_dir(fixture).join(format!("{}.jsonl", fixture.state().key));
    let lines: String = events.iter().enumerate().map(|(seq, event)| {
        let mut line = event.clone(); line["v"] = serde_json::json!(1); line["seq"] = serde_json::json!(seq); format!("{line}\n")
    }).collect();
    fs::write(&path, lines).unwrap(); path
}

fn promote_argv(fixture: &Fixture, job: &TranscribeJob) -> Vec<String> {
    let mut argv = transcribe_argv(fixture, job);
    argv.extend(["--promote-live".to_string(), live_dir(fixture).display().to_string()]); argv
}

/// The record promotion must leave: the pair in `dir`, each file by the hash of what is on disk.
fn record_of(dir: &Path) -> recovery::Provisional {
    let hash = |name: &str| Some(crate::export::sha256_hex(&fs::read(dir.join(name)).unwrap()));
    recovery::Provisional::new(dir, hash("transcript.md"), hash("transcript.json"))
}

fn proofs(fixture: &Fixture) -> (bool, bool) {
    let session = recovery::Session::open(&fixture.dir).unwrap().unwrap();
    (session.proof(recovery::ProofScope::Archive).unwrap(), session.proof(recovery::ProofScope::Retention).unwrap())
}

#[test]
fn a_complete_live_stream_is_stored_without_inference_and_only_once() {
    for _ in 0..3 {
        let (fixture, owner) = Fixture::new(); drop(owner);
        write_stream(&fixture, &stream_events(&fixture));
        fs::write(fixture.root.join("forbid-inference"), b"the live stream is the transcript\n").unwrap();
        let argv = promote_argv(&fixture, &fixture.job());
        let output = child_output(&fixture, spawn_cli(&fixture, &argv));
        assert!(output.status.success(), "promotion failed: {output:?}");
        assert!(!fixture.root.join("engine-ran").exists(), "an accepted stream must not load the engine");
        let rows = fixture.meetings(); assert_eq!(rows.len(), 1); let id = rows[0].id;
        assert_eq!((rows[0].origin, rows[0].model.as_str(), rows[0].segment_count), (db::TranscriptOrigin::Live, "ggml-live-fixture", 2));
        assert_eq!(rows[0].lang, "pt", "the language is the reader's, not this child's --lang");
        assert_eq!((rows[0].duration_secs, rows[0].capture_status.as_str()), (3.0, "complete"));
        let state = fixture.state();
        assert_eq!((state.phase, state.attempts, state.meeting_id), (recovery::Phase::Complete, 1, Some(id)));
        assert_eq!(state.publication.as_deref(), Some(fixture.dir.as_path()), "a first attempt publishes the root pair");
        let markdown = fs::read_to_string(fixture.dir.join("transcript.md")).unwrap();
        assert!(markdown.contains(&format!("- **Transcript:** live, provisional — `meetscribe finalize {id}`")));
        let rendered: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(fixture.dir.join("transcript.json")).unwrap()).unwrap();
        assert_eq!(rendered.iter().map(|segment| segment["text"].as_str().unwrap()).collect::<Vec<_>>(),
            ["The live release checklist is ready", "We approved the live deployment window"]);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("\"event\":\"live_promoted\"") && stderr.contains("\"merged_segments\":2"), "no promotion event in: {stderr}");
        assert!(!stderr.contains("live deployment window"), "meeting text reached the log");
        assert_eq!(proofs(&fixture), (true, true), "the files are a render of the stored rows, and only owned files exist");
        assert_eq!(state.provisional(), Some(record_of(&fixture.dir)), "promotion records the pair it published");

        // The same command again: a Complete session is not due. It fails and changes nothing.
        let before = session_snapshot(&fixture.dir); let database = fs::read(&fixture.opts.db_path).unwrap();
        let again = child_output(&fixture, spawn_cli(&fixture, &argv));
        assert_eq!(again.status.code(), Some(1));
        assert_eq!(session_snapshot(&fixture.dir), before); assert_eq!(fs::read(&fixture.opts.db_path).unwrap(), database);
        assert_eq!(fixture.meetings().len(), 1);
        // And retention treats the promoted session like any stored one.
        assert_eq!(maintenance::prune_sessions(&fixture.root.join("sessions"), 30, crate::now_epoch() + 100 * 86_400).unwrap().removed, 1);
        fs::remove_dir_all(fixture.root).unwrap();
    }
}

#[test]
fn a_stream_that_cannot_be_promoted_runs_the_batch_pass_in_the_same_attempt() {
    type Arrange = fn(&Fixture);
    let unfinished: Arrange = |fixture| { let events = stream_events(fixture); write_stream(fixture, &events[..4]); };
    let cases: [(&str, Arrange); 9] = [
        ("no_stream", |_| {}),
        ("unreadable", |fixture| { recovery::private_dir(&live_dir(fixture).join(format!("{}.jsonl", fixture.state().key))).unwrap(); }),
        ("torn", |fixture| {
            let path = write_stream(fixture, &stream_events(fixture));
            let lines = fs::read_to_string(&path).unwrap();
            let (first, rest) = lines.split_once('\n').unwrap();
            fs::write(path, format!("{first}\n{{\"v\":1,\"seq\":1,\"type\":\"segm\n{rest}")).unwrap();
        }),
        ("reader_gone", |fixture| {
            let events = stream_events(fixture); let path = write_stream(fixture, &events[..4]);
            fs::File::options().write(true).open(path).unwrap().set_modified(std::time::SystemTime::now() - Duration::from_secs(31)).unwrap();
        }),
        ("timeout", unfinished),
        ("not_ended", |fixture| { let mut events = stream_events(fixture); events[4]["reason"] = serde_json::json!("stalled"); write_stream(fixture, &events); }),
        ("key_mismatch", |fixture| { let mut events = stream_events(fixture); events[0]["key"] = serde_json::json!("b".repeat(32)); write_stream(fixture, &events); }),
        ("duration_mismatch", |fixture| { let mut events = stream_events(fixture); events[4]["duration_secs"] = serde_json::json!(2.5); write_stream(fixture, &events); }),
        ("empty", |fixture| { let events = stream_events(fixture); let mut ended = events[4].clone(); ended["segments"] = serde_json::json!(0);
            write_stream(fixture, &[events[0].clone(), ended]); }),
    ];
    for (reason, arrange) in cases {
        let (fixture, owner) = Fixture::new(); drop(owner);
        arrange(&fixture);
        // Only the timeout case waits: the reader looks alive and the cap is cut to 100 ms.
        if reason == "timeout" { fs::write(fixture.root.join("promote-wait"), "30000 100\n").unwrap(); }
        let output = child_output(&fixture, spawn_cli(&fixture, &promote_argv(&fixture, &fixture.job())));
        assert!(output.status.success(), "{reason}: the fallback failed: {output:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("\"event\":\"live_promotion_rejected\"") && stderr.contains(&format!("\"reason\":\"{reason}\"")),
            "{reason}: not logged as rejected for that reason: {stderr}");
        assert!(!stderr.contains("\"event\":\"live_promoted\""), "{reason}");
        assert!(fixture.root.join("engine-ran").is_file(), "{reason}: the batch pass did not run");
        let rows = fixture.meetings(); assert_eq!(rows.len(), 1, "{reason}");
        assert_eq!((rows[0].origin, rows[0].model.as_str(), rows[0].lang.as_str()), (db::TranscriptOrigin::Batch, "synthetic-model", "en"), "{reason}");
        let state = fixture.state();
        assert_eq!((state.phase, state.attempts), (recovery::Phase::Complete, 1), "{reason}: exactly one attempt is reserved");
        assert_eq!(state.publication.as_deref(), Some(fixture.dir.as_path()), "{reason}");
        assert!(!fs::read_to_string(fixture.dir.join("transcript.md")).unwrap().contains("provisional"), "{reason}");
        assert!(!fs::read_to_string(fixture.dir.join(recovery::STATE_FILE)).unwrap().contains("provisional"), "{reason}: a batch session's state gains no key");
        assert_eq!(proofs(&fixture), (true, true), "{reason}");
        fs::remove_dir_all(fixture.root).unwrap();
    }
}

/// The child dies right after the promotion's insert committed. Nothing is stored twice: the next
/// automatic visit only charges the abandoned attempt, and the visit after it (or an explicit
/// `--resume`) finds the ingest and publishes the stored live meeting, without the engine.
#[test]
fn a_promotion_killed_after_its_commit_is_finished_from_the_stored_meeting() {
    for explicit_resume in [false, true] {
        for _ in 0..3 {
            let (fixture, owner) = Fixture::new(); drop(owner);
            write_stream(&fixture, &stream_events(&fixture));
            fs::write(fixture.root.join("forbid-inference"), b"the live stream is the transcript\n").unwrap();
            fs::write(fixture.root.join("cutpoint"), "live_committed\n").unwrap();
            let argv = promote_argv(&fixture, &fixture.job());
            let killed = child_output(&fixture, spawn_cli(&fixture, &argv));
            assert_eq!(killed.status.code(), Some(85));
            let ack: serde_json::Value = serde_json::from_slice(&fs::read(fixture.root.join("cutpoint-ack.json")).unwrap()).unwrap();
            assert_eq!(ack["point"], "live_committed");
            let rows = fixture.meetings(); assert_eq!(rows.len(), 1); let id = rows[0].id;
            assert_eq!(rows[0].origin, db::TranscriptOrigin::Live);
            let state = fixture.state();
            assert_eq!((state.phase, state.attempts, state.in_flight, state.meeting_id), (recovery::Phase::Ready, 1, true, None));
            assert!(!fixture.dir.join("transcript.md").exists());
            fs::remove_file(fixture.root.join("cutpoint")).unwrap();

            if explicit_resume {
                let mut resume = fixture.job(); resume.opts.resume = true;
                let resumed = child_output(&fixture, spawn_cli(&fixture, &promote_argv(&fixture, &resume)));
                assert!(resumed.status.success(), "resume failed: {resumed:?}");
                assert_eq!(fixture.state().attempts, 1, "an explicit resume starts a new budget");
            } else {
                let charged = child_output(&fixture, spawn_cli(&fixture, &argv));
                assert_eq!(charged.status.code(), Some(1), "the first visit after a kill only charges the abandoned attempt");
                let state = fixture.state();
                assert_eq!((state.phase, state.attempts, state.in_flight, state.next_retry), (recovery::Phase::Ready, 1, false, START + 70));
                assert_eq!(state.error_class.as_deref(), Some("interrupted_attempt"));
                assert!(!fixture.dir.join("transcript.md").exists()); assert_eq!(fixture.meetings().len(), 1);
                fs::write(fixture.root.join("clock"), format!("{}\n", state.next_retry)).unwrap();
                let finished = child_output(&fixture, spawn_cli(&fixture, &argv));
                assert!(finished.status.success(), "the retry failed: {finished:?}");
                assert_eq!(fixture.state().attempts, 2);
            }
            let state = fixture.state();
            assert_eq!((state.phase, state.meeting_id), (recovery::Phase::Complete, Some(id)));
            let rows = fixture.meetings(); assert_eq!(rows.len(), 1, "the retry stored a second meeting");
            assert_eq!((rows[0].id, rows[0].origin), (id, db::TranscriptOrigin::Live));
            assert!(!fixture.root.join("engine-ran").exists(), "a stored live meeting must not be decoded again");
            // A retry publishes beside the root, as a batch retry does.
            let publication = state.publication.unwrap();
            assert!(publication.starts_with(fixture.dir.join(".recovered").join(&state.key)));
            assert!(fs::read_to_string(publication.join("transcript.md")).unwrap().contains(&format!("`meetscribe finalize {id}`")));
            assert!(!fixture.dir.join("transcript.md").exists());
            assert_eq!(proofs(&fixture), (true, false), "the archive proof holds; a .recovered directory is not owned capture material");
            assert_eq!(fixture.state().provisional(), Some(record_of(&publication)), "a retry records the place it published in");
            fs::remove_dir_all(fixture.root).unwrap();
        }
    }
}

/// The child dies after both files are written and before the state that would record them is
/// saved. The retry publishes beside them and records that pair; the first pair stays, unrecorded.
#[test]
fn a_promotion_killed_before_its_record_is_saved_is_retried_and_then_has_one() {
    let (fixture, owner) = Fixture::new(); drop(owner);
    write_stream(&fixture, &stream_events(&fixture));
    fs::write(fixture.root.join("forbid-inference"), b"the live stream is the transcript\n").unwrap();
    fs::write(fixture.root.join("cutpoint"), "published\n").unwrap();
    let argv = promote_argv(&fixture, &fixture.job());
    assert_eq!(child_output(&fixture, spawn_cli(&fixture, &argv)).status.code(), Some(85));
    let state = fixture.state();
    assert_eq!((state.phase, state.in_flight, state.provisional.is_none()), (recovery::Phase::Archived, true, true));
    let root = (fs::read(fixture.dir.join("transcript.md")).unwrap(), fs::read(fixture.dir.join("transcript.json")).unwrap());
    fs::remove_file(fixture.root.join("cutpoint")).unwrap();

    assert_eq!(child_output(&fixture, spawn_cli(&fixture, &argv)).status.code(), Some(1), "the first visit only charges the abandoned attempt");
    fs::write(fixture.root.join("clock"), format!("{}\n", fixture.state().next_retry)).unwrap();
    let finished = child_output(&fixture, spawn_cli(&fixture, &argv));
    assert!(finished.status.success(), "the retry failed: {finished:?}");
    let state = fixture.state(); let publication = state.publication.clone().unwrap();
    assert_eq!(state.phase, recovery::Phase::Complete);
    assert!(publication.starts_with(fixture.dir.join(".recovered").join(&state.key)));
    assert_eq!(state.provisional(), Some(record_of(&publication)));
    assert_eq!((fs::read(fixture.dir.join("transcript.md")).unwrap(), fs::read(fixture.dir.join("transcript.json")).unwrap()), root, "the first pair is left as it was");
    assert_eq!(fixture.meetings().len(), 1);
    fs::remove_dir_all(fixture.root).unwrap();
}

// ---- Finalize: a live-stored meeting is replaced by the batch transcript (`finalize.rs`). The
// meeting is promoted through the real child first, exactly as the promotion tests do it.

const LIVE_WORDS: [&str; 2] = ["live release checklist", "live deployment window"];
const BATCH_WORDS: [&str; 2] = ["synthetic release checklist", "synthetic deployment window"];

/// A fixture whose meeting was stored from its live stream; returns it with the meeting id.
fn promoted() -> (Fixture, i64) {
    let (fixture, owner) = Fixture::new(); drop(owner);
    write_stream(&fixture, &stream_events(&fixture));
    let output = child_output(&fixture, spawn_cli(&fixture, &promote_argv(&fixture, &fixture.job())));
    assert!(output.status.success(), "promotion failed: {output:?}");
    assert!(!fixture.root.join("engine-ran").exists());
    let rows = fixture.meetings(); assert_eq!((rows.len(), rows[0].origin), (1, db::TranscriptOrigin::Live));
    (fixture, rows[0].id)
}

fn finalize_argv(fixture: &Fixture, id: i64, extra: &[&str]) -> Vec<String> {
    let mut argv: Vec<String> = ["finalize", &id.to_string(), "--db", fixture.opts.db_path.to_str().unwrap(),
        "--base", fixture.base.to_str().unwrap(), "--lang", "en", "--model", fixture.opts.model.to_str().unwrap()]
        .into_iter().map(String::from).collect();
    argv.extend(extra.iter().map(|flag| flag.to_string())); argv
}

fn finalize(fixture: &Fixture, id: i64, extra: &[&str]) -> std::process::Output {
    child_output(fixture, spawn_cli(fixture, &finalize_argv(fixture, id, extra)))
}

/// Every entry under a directory with what a rewrite would change even when the bytes stay the
/// same: its inode and its modification time. Directories are listed too, so an added empty one shows.
fn entries(dir: &Path) -> std::collections::BTreeMap<PathBuf, (u64, std::time::SystemTime, u64)> {
    use std::os::unix::fs::MetadataExt;
    let mut found = std::collections::BTreeMap::new();
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap(); let metadata = fs::symlink_metadata(entry.path()).unwrap();
        found.insert(entry.path(), (metadata.ino(), metadata.modified().unwrap(), metadata.len()));
        if metadata.is_dir() { found.extend(entries(&entry.path())); }
    }
    found
}

/// What "unchanged" means for a session and its database: same bytes, same files (not rewritten
/// in place), no entry added or removed, and the database byte for byte.
#[derive(PartialEq, Debug)]
struct Untouched { files: std::collections::BTreeMap<PathBuf, Vec<u8>>, entries: std::collections::BTreeMap<PathBuf, (u64, std::time::SystemTime, u64)>, database: Vec<u8> }
fn untouched(fixture: &Fixture) -> Untouched {
    Untouched { files: session_snapshot(&fixture.dir), entries: entries(&fixture.dir), database: fs::read(&fixture.opts.db_path).unwrap() }
}

/// The end state of a finished `finalize`, whatever road led to it.
fn assert_finalized(fixture: &Fixture, id: i64, before: &recovery::State) {
    let rows = fixture.meetings(); assert_eq!(rows.len(), 1, "finalize must not add a meeting");
    assert_eq!((rows[0].id, rows[0].origin, rows[0].model.as_str(), rows[0].lang.as_str()), (id, db::TranscriptOrigin::Batch, "synthetic-model", "en"));
    let (segments, clusters) = stored_voices(fixture, id);
    assert_eq!(rows[0].segment_count, segments.len() as i64);
    assert!(!clusters.is_empty(), "the batch pass finds the far-end voices the live lane does not");
    assert!(segments.iter().all(|segment| BATCH_WORDS.iter().any(|words| segment.seg.text.contains(words))), "a live segment survived");
    let state = fixture.state();
    assert_eq!((state.phase, state.attempts, state.in_flight, state.meeting_id), (recovery::Phase::Complete, before.attempts, false, Some(id)));
    assert_eq!(state.publication, before.publication, "the pair is replaced where it was published");
    assert!(!fs::read_to_string(fixture.dir.join(recovery::STATE_FILE)).unwrap().contains("provisional"), "the record is removed with the last save");
    let publication = state.publication.unwrap();
    let markdown = fs::read_to_string(publication.join("transcript.md")).unwrap();
    assert!(!markdown.contains("provisional") && BATCH_WORDS.iter().all(|words| markdown.contains(words)));
    let session = recovery::Session::open(&fixture.dir).unwrap().unwrap();
    assert!(session.proof(recovery::ProofScope::Archive).unwrap(), "the published pair is the render of the stored rows");
    drop(session);
    let scratch = fixture.base.join("scratch");
    assert!(!scratch.exists() || fs::read_dir(&scratch).unwrap().next().is_none(), "a scratch directory was left behind");
    assert!(session_snapshot(&fixture.dir).keys().all(|path| !path.file_name().unwrap().to_str().unwrap().starts_with(".diar-")));
}

fn no_meeting_text(output: &std::process::Output) {
    let said = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    for words in LIVE_WORDS.iter().chain(&BATCH_WORDS) { assert!(!said.contains(words), "meeting text reached the terminal or the log: {words}"); }
}

#[test]
fn a_promoted_meeting_is_finalized_under_the_same_id() {
    for _ in 0..3 {
        let (fixture, id) = promoted();
        let before = fixture.state();
        let live_json = fs::read(fixture.dir.join("transcript.json")).unwrap();
        let output = finalize(&fixture, id, &[]);
        assert!(output.status.success(), "finalize failed: {output:?}");
        assert_finalized(&fixture, id, &before);
        assert_eq!(proofs(&fixture), (true, true), "and the session is prunable like any batch session");
        assert!(!fixture.dir.join(".recovered").exists());
        assert_ne!(fs::read(fixture.dir.join("transcript.json")).unwrap(), live_json);
        assert!(fixture.root.join("engine-ran").is_file(), "the batch pass did not run");
        no_meeting_text(&output);
        let (stdout, stderr) = (String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        assert!(stdout.contains(&format!("meeting {id}: finalized")), "{stdout}");
        for event in ["finalize_started", "finalize_committed", "finalize_published"] {
            assert!(stderr.contains(&format!("\"event\":\"{event}\"")), "{event} is not in the log: {stderr}");
        }
        assert!(stderr.contains("\"clusters\":1") && stderr.contains(&format!("\"session_id\":\"{}\"", before.key)), "{stderr}");
        assert!(!stderr.contains("invalid_event"), "an event was withheld by the log allowlist: {stderr}");

        // Again: the meeting is final. It says so, exits 0, loads no engine and changes nothing.
        fs::remove_file(fixture.root.join("engine-ran")).unwrap();
        let unchanged = untouched(&fixture);
        let again = finalize(&fixture, id, &[]);
        assert!(again.status.success(), "{again:?}");
        assert!(String::from_utf8_lossy(&again.stdout).contains("already the full transcript"));
        assert!(!fixture.root.join("engine-ran").exists());
        assert_eq!(untouched(&fixture), unchanged);
        fs::remove_dir_all(fixture.root).unwrap();
    }
}

// ---- Finalize: helpers shared by the refusal, kill and interleaving tests.

fn stderr_of(output: &std::process::Output) -> String { String::from_utf8_lossy(&output.stderr).into_owned() }

/// Rewrite the session's state file through plain JSON, as an older build or a hand edit would.
fn edit_state(fixture: &Fixture, edit: impl FnOnce(&mut serde_json::Value)) {
    let path = fixture.dir.join(recovery::STATE_FILE);
    let mut raw: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    edit(&mut raw);
    fs::write(&path, format!("{raw}\n")).unwrap();
}

/// Run statements against the fixture's archive on a connection of the test's own.
fn sql(fixture: &Fixture, statements: &[&str]) {
    crate::new_runtime().unwrap().block_on(async {
        let mut connection = sqlx::SqliteConnection::connect_with(&sqlx::sqlite::SqliteConnectOptions::new()
            .filename(&fixture.opts.db_path).create_if_missing(false).foreign_keys(true)).await.unwrap();
        for statement in statements { sqlx::query(statement).execute(&mut connection).await.unwrap(); }
        connection.close().await.unwrap();
    });
}

/// Give the meeting one far-end voice cluster, as `speakers cluster` does, optionally named by hand.
fn add_voice(fixture: &Fixture, id: i64, manual: bool) {
    crate::new_runtime().unwrap().block_on(async {
        let mut db = db::Db::open_existing(&fixture.opts.db_path, fixture.state().archive_id.as_deref().unwrap()).await.unwrap();
        let far = db.load_segments(id).await.unwrap().into_iter().find(|segment| segment.seg.speaker == crate::transcript::Speaker::Others).unwrap().id;
        let centroid: Vec<u8> = [1.0f32, 0.0].iter().flat_map(|value| value.to_le_bytes()).collect();
        let cluster = db::NewCluster { cluster: "A".into(), speaker_id: None, assigned_by: None, match_score: None, centroid, dim: 2, n_windows: 1, speech_secs: 1.2 };
        db.replace_meeting_clusters(id, &[cluster], &[db::SegmentVoice { segment: far, cluster: 0, inherited: false, embedding: None }], START).await.unwrap();
        if manual {
            let stored = db.get_cluster(id, "A").await.unwrap().unwrap();
            let person = db.add_speaker("Ada", "Example", START).await.unwrap();
            db.set_cluster_speaker(stored.id, Some(person), Some("manual"), None).await.unwrap();
            db.add_voiceprint(person, &stored, START).await.unwrap();
        }
        db.close().await.unwrap();
    });
}

/// Run `finalize` until the test-only point `point`, where the child exits without unwinding.
fn finalize_killed_at(fixture: &Fixture, id: i64, point: &str, extra: &[&str]) {
    fs::write(fixture.root.join("cutpoint"), format!("{point}\n")).unwrap();
    let killed = finalize(fixture, id, extra);
    assert_eq!(killed.status.code(), Some(85), "finalize did not reach {point}: {killed:?}");
    let ack: serde_json::Value = serde_json::from_slice(&fs::read(fixture.root.join("cutpoint-ack.json")).unwrap()).unwrap();
    assert_eq!(ack["point"], point);
    fs::remove_file(fixture.root.join("cutpoint")).unwrap(); fs::remove_file(fixture.root.join("cutpoint-ack.json")).unwrap();
}

fn resume_argv(fixture: &Fixture) -> Vec<String> {
    let mut job = fixture.job(); job.opts.resume = true; transcribe_argv(fixture, &job)
}

fn origin(fixture: &Fixture) -> db::TranscriptOrigin { fixture.meetings()[0].origin }

/// The run must not have needed the engine: used after a stop that follows the commit.
fn forbid_engine(fixture: &Fixture) {
    let _ = fs::remove_file(fixture.root.join("engine-ran"));
    fs::write(fixture.root.join("forbid-inference"), b"the batch transcript is already stored\n").unwrap();
}

/// A diarizer that only waits: installed where `finalize` looks for one under `--base`. While the
/// file `hold` is beside it, it writes its pid and sleeps; without it, it fails at once and the
/// pass falls back to whole chunks.
fn install_waiting_diarizer(fixture: &Fixture) -> PathBuf {
    let bin_dir = fixture.base.join("models/diarizer/bin"); fs::create_dir_all(&bin_dir).unwrap();
    fs::write(fixture.base.join(crate::DIARIZER_MODEL_REL), b"gguf").unwrap();
    let bin = fixture.base.join(crate::DIARIZER_BIN_REL);
    fs::write(&bin, "#!/bin/sh\nhere=$(dirname \"$0\")\n[ -f \"$here/hold\" ] || exit 1\necho $$ > \"$here/pid\"\nexec sleep 30\n").unwrap();
    fs::set_permissions(&bin, fs::Permissions::from_mode(0o700)).unwrap();
    fs::write(bin_dir.join("hold"), b"wait\n").unwrap();
    bin_dir
}

/// A stop before the transaction, of the hardest kind: SIGKILL from outside, in the middle of a
/// decode, with the diarizer running. Nothing under the session directory has changed and nothing
/// will: the daemon's own visit finds a `Complete` session and leaves it.
#[test]
fn finalize_killed_during_the_pass_has_changed_nothing_and_nothing_follows() {
    let (fixture, id) = promoted();
    let bin_dir = install_waiting_diarizer(&fixture);
    let (before, state, proof) = (untouched(&fixture), fixture.state(), proofs(&fixture));
    fs::write(fixture.root.join("holdpoint"), "decoding_far_end\n").unwrap();
    let mut child = spawn_cli(&fixture, &finalize_argv(&fixture, id, &[]));
    wait_for_owner(&fixture, "decoding_far_end");

    // Mid-pass: the diarizer is running, on a copy of the audio that is NOT in the session.
    let scratch = fixture.base.join("scratch");
    let runs: Vec<PathBuf> = fs::read_dir(&scratch).unwrap().map(|entry| entry.unwrap().path()).collect();
    assert_eq!(runs.len(), 1, "one scratch directory per run");
    assert!(runs[0].file_name().unwrap().to_str().unwrap().starts_with(&format!("finalize-{}-", state.key)));
    let working: Vec<String> = fs::read_dir(&runs[0]).unwrap().map(|entry| entry.unwrap().file_name().into_string().unwrap()).collect();
    assert!(working.iter().any(|name| name.starts_with(".diar-") && name.ends_with(".wav")), "the diarizer was not given the scratch directory: {working:?}");
    // The diarizer was spawned a moment ago; give its first line the time to run.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let diarizer: i32 = loop {
        if let Some(pid) = fs::read_to_string(bin_dir.join("pid")).ok().and_then(|pid| pid.trim().parse().ok()) { break pid; }
        assert!(std::time::Instant::now() < deadline, "the diarizer is not running");
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(untouched(&fixture), before, "the pass wrote under the session directory");

    child.kill().unwrap(); // SIGKILL: no destructor runs, the diarizer is orphaned
    let _ = child.wait().unwrap();
    // SAFETY: a plain signal to the fixture's own orphaned `sleep`; no memory is passed.
    unsafe { libc::kill(diarizer, libc::SIGKILL); }
    fs::remove_file(fixture.root.join("holdpoint")).unwrap(); fs::remove_file(fixture.root.join("hold-ack.json")).unwrap();
    assert_eq!(untouched(&fixture), before, "a killed finalize left a trace in the session or the database");
    assert_eq!(origin(&fixture), db::TranscriptOrigin::Live);

    // The daemon's visit, and retention's question, get the answers they got before.
    let visit = child_output(&fixture, spawn_cli(&fixture, &transcribe_argv(&fixture, &fixture.job())));
    assert_eq!(visit.status.code(), Some(1), "a Complete session is not due");
    assert_eq!(untouched(&fixture), before);
    assert_eq!(proofs(&fixture), proof);
    assert!(!fixture.dir.join(".recovered").exists());

    // A second run clears the first one's scratch directory and finishes.
    fs::remove_file(bin_dir.join("hold")).unwrap();
    let output = finalize(&fixture, id, &[]);
    assert!(output.status.success(), "the second finalize failed: {output:?}");
    assert_finalized(&fixture, id, &state);
    assert!(!runs[0].exists(), "the killed run's scratch directory was left");
    fs::remove_dir_all(fixture.root).unwrap();
}

#[test]
fn every_refusal_of_finalize_leaves_the_session_and_the_database_as_they_were() {
    type Arrange = fn(&Fixture, i64);
    let cases: [(&str, Arrange, &str); 9] = [
        ("a call is being recorded", |fixture, _| {
            fs::write(fixture.base.join("status.json"), format!("{}\n", serde_json::json!({"state": "recording", "app": "Chrome", "since_epoch": START, "updated_epoch": START, "pid": 1}))).unwrap();
            assert!(crate::status::call_is_live(&fixture.base), "the fixture's status file is not read as a recording");
        }, "a call is being recorded"),
        ("a name set by hand", |fixture, id| add_voice(fixture, id, true), "manual name"),
        ("a file changed by hand", |fixture, _| {
            let path = fixture.dir.join("transcript.md"); let edited = format!("{}\nmy own note\n", fs::read_to_string(&path).unwrap()); fs::write(path, edited).unwrap();
        }, "was changed since meetscribe wrote it"),
        ("no record", |fixture, _| edit_state(fixture, |state| { state.as_object_mut().unwrap().remove("provisional"); }), "--overwrite"),
        ("a record for another directory", |fixture, _| edit_state(fixture, |state| state["provisional"]["dir"] = serde_json::json!(fixture.root)), "--overwrite"),
        ("a session that is not complete", |fixture, _| edit_state(fixture, |state| state["phase"] = serde_json::json!("archived")), "is not complete"),
        ("a missing recording file", |fixture, _| fs::remove_file(fixture.dir.join("system.wav")).unwrap(), ""),
        ("a scratch place that points into the session", |fixture, _| std::os::unix::fs::symlink(&fixture.dir, fixture.base.join("scratch")).unwrap(), "not a real directory"),
        ("a relative stored path", |fixture, _| sql(fixture, &["UPDATE meetings SET source_dir = 'recording'"]), "--session"),
    ];
    for (name, arrange, said) in cases {
        let (fixture, id) = promoted();
        arrange(&fixture, id);
        let before = untouched(&fixture);
        let output = finalize(&fixture, id, &[]);
        assert_eq!(output.status.code(), Some(1), "{name}: {output:?}");
        assert_eq!(untouched(&fixture), before, "{name}: a refused finalize changed something");
        assert!(!fixture.root.join("engine-ran").exists(), "{name}: refused only after the pass had started");
        let stderr = stderr_of(&output);
        assert!(stderr.contains(said), "{name}: the reason is not said: {stderr}");
        assert!(stderr.contains("\"event\":\"finalize_failed\"") && !stderr.contains("invalid_event"), "{name}: {stderr}");
        assert_eq!(fixture.state().phase, if name.contains("not complete") { recovery::Phase::Archived } else { recovery::Phase::Complete }, "{name}");
        assert_eq!(origin(&fixture), db::TranscriptOrigin::Live, "{name}");
        no_meeting_text(&output);
        fs::remove_dir_all(fixture.root).unwrap();
    }

    // A meeting nobody stored.
    let (fixture, id) = promoted();
    let before = untouched(&fixture);
    let output = finalize(&fixture, id + 41, &[]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr_of(&output).contains("no meeting with id"));
    assert_eq!(untouched(&fixture), before);

    // The session lock held by someone else (here: this test). Not the daemon's "busy" class,
    // which its log counts as a recording it must leave alone.
    let held = recovery::Session::open(&fixture.dir).unwrap().unwrap();
    let output = finalize(&fixture, id, &[]);
    assert_eq!(output.status.code(), Some(1));
    let stderr = stderr_of(&output);
    assert!(stderr.contains("is in use by another meetscribe command") && !stderr.contains("session_busy"), "{stderr}");
    assert_eq!(untouched(&fixture), before);
    drop(held);

    // The session of another meeting, named with --session: another key, archive and database.
    let (other, _) = promoted();
    let other_before = untouched(&other);
    let output = finalize(&fixture, id, &["--session", other.dir.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr_of(&output).contains("does not hold the session of meeting"), "{}", stderr_of(&output));
    assert_eq!((untouched(&fixture), untouched(&other)), (before, other_before));
    assert!(!fixture.root.join("engine-ran").exists());
    fs::remove_dir_all(fixture.root).unwrap(); fs::remove_dir_all(other.root).unwrap();
}

/// The checks that say "this is that meeting's session" run for a `batch` row too: after a stop
/// that follows the commit, the publish step replaces files, and it must not do that in the
/// directory of another meeting.
#[test]
fn a_replaced_meeting_is_published_only_into_its_own_session() {
    let (fixture, id) = promoted();
    finalize_killed_at(&fixture, id, "finalize_committed", &[]);
    assert_eq!(origin(&fixture), db::TranscriptOrigin::Batch);
    let before = untouched(&fixture);
    forbid_engine(&fixture);

    // Copies of the session with one thing wrong each: they share the real session's lock, so
    // the lock is free and only the identity checks stand between them and the publish step.
    type Wrong = fn(&mut serde_json::Value);
    let wrong: [(&str, Wrong); 3] = [
        ("another meeting id", |state| state["meeting_id"] = serde_json::json!(4242)),
        ("another key", |state| state["key"] = serde_json::json!("c".repeat(32))),
        ("another archive", |state| state["archive_id"] = serde_json::json!("d".repeat(32))),
    ];
    for (name, change) in wrong {
        let copy = fixture.root.join("copy");
        recovery::private_dir(&copy).unwrap();
        for entry in fs::read_dir(&fixture.dir).unwrap() { let entry = entry.unwrap(); fs::copy(entry.path(), copy.join(entry.file_name())).unwrap(); }
        let mut raw: serde_json::Value = serde_json::from_slice(&fs::read(copy.join(recovery::STATE_FILE)).unwrap()).unwrap();
        change(&mut raw); raw["export_dir"] = serde_json::json!(copy); raw["publication"] = serde_json::json!(copy); raw["provisional"]["dir"] = serde_json::json!(copy);
        fs::write(copy.join(recovery::STATE_FILE), format!("{raw}\n")).unwrap();
        let copied = session_snapshot(&copy);
        let output = finalize(&fixture, id, &["--session", copy.to_str().unwrap()]);
        assert_eq!(output.status.code(), Some(1), "{name}: {output:?}");
        assert!(stderr_of(&output).contains("does not hold the session of meeting"), "{name}: {}", stderr_of(&output));
        assert_eq!(session_snapshot(&copy), copied, "{name}: files were replaced in a directory that is not the meeting's");
        assert_eq!(untouched(&fixture), before, "{name}");
        fs::remove_dir_all(copy).unwrap();
    }
    // The real session, without --session, is finished.
    let state = fixture.state();
    let output = finalize(&fixture, id, &[]);
    assert!(output.status.success(), "{output:?}");
    assert_finalized(&fixture, id, &state);
    fs::remove_dir_all(fixture.root).unwrap();
}

/// A kill at each step from inside the transaction to the last file. Whatever the step, the
/// stop is safe for the daemon, for retention and for `transcribe --resume`, and one more
/// `finalize <id>` reaches the same end, without the engine once the commit has happened.
#[test]
fn finalize_killed_at_any_step_is_finished_by_running_it_again() {
    // (point, row afterwards is batch, archive proof afterwards)
    let stops = [
        ("replace_in_transaction", false, true),
        ("finalize_committed", true, false),
        ("finalize_hashes_saved", true, false),
        ("export_staged", true, false),
        ("finalize_markdown_replaced", true, false),
        ("finalize_json_replaced", true, true),
    ];
    for (point, committed, proof) in stops {
        let (fixture, id) = promoted();
        let state = fixture.state();
        finalize_killed_at(&fixture, id, point, &[]);
        assert_eq!(origin(&fixture) == db::TranscriptOrigin::Batch, committed, "{point}");
        assert_eq!(proofs(&fixture).0, proof, "{point}: archive proof");
        assert_eq!(fixture.state().phase, recovery::Phase::Complete, "{point}: the session never becomes due");
        assert!(!fixture.dir.join(".recovered").exists(), "{point}");
        if committed {
            assert!(fixture.state().provisional().is_some(), "{point}: the record must outlive the commit");
            // `transcribe --resume` would reset the session and publish a second copy. It refuses.
            let before = untouched(&fixture);
            let resumed = child_output(&fixture, spawn_cli(&fixture, &resume_argv(&fixture)));
            assert_eq!(resumed.status.code(), Some(1), "{point}: {resumed:?}");
            assert_eq!(untouched(&fixture), before, "{point}: a refused --resume changed something");
            // And so does the daemon's own visit, which finds nothing due.
            let visit = child_output(&fixture, spawn_cli(&fixture, &transcribe_argv(&fixture, &fixture.job())));
            assert_eq!(visit.status.code(), Some(1), "{point}");
            assert_eq!(untouched(&fixture), before, "{point}");
            forbid_engine(&fixture);
        }
        if committed && !proof {
            // Files and database disagree: retention must keep the session, however old.
            let pruned = maintenance::prune_sessions(&fixture.root.join("sessions"), 30, crate::now_epoch() + 100 * 86_400).unwrap();
            assert_eq!((pruned.removed, fixture.dir.is_dir()), (0, true), "{point}: retention removed a half-published session");
        }
        let output = finalize(&fixture, id, &[]);
        assert!(output.status.success(), "{point}: the second run failed: {output:?}");
        assert_eq!(fixture.root.join("engine-ran").exists(), !committed, "{point}: the engine runs again only when nothing was committed");
        assert_finalized(&fixture, id, &state);
        let leftovers: Vec<String> = fs::read_dir(&fixture.dir).unwrap().map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .filter(|name| name.ends_with(".tmp")).collect();
        // A kill between writing a temporary file and renaming it leaves that one file, as on
        // every publishing path; nothing deletes a file by its name alone.
        assert_eq!(leftovers.len(), usize::from(point == "export_staged"), "{point}: {leftovers:?}");
        assert_eq!(proofs(&fixture), (true, point != "export_staged"), "{point}");
        no_meeting_text(&output);
        fs::remove_dir_all(fixture.root).unwrap();
    }
}

/// A storage error inside the replacement: nothing is exported, nothing is marked, every row is
/// what it was. (The first transcription of a meeting does the opposite on purpose: there the
/// transcript exists nowhere else, here it is still stored.)
#[test]
fn a_replacement_the_database_refuses_publishes_nothing_and_marks_nothing() {
    let (fixture, id) = promoted();
    let state = fixture.state();
    sql(&fixture, &["CREATE TRIGGER audit_replace_fault BEFORE UPDATE ON meetings BEGIN SELECT RAISE(ABORT, 'synthetic replace fault'); END"]);
    let before = untouched(&fixture);
    let output = finalize(&fixture, id, &[]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(fixture.root.join("engine-ran").is_file(), "the pass ran; only the storing failed");
    assert_eq!(untouched(&fixture), before, "a failed replacement changed the session or the rows");
    let after = fixture.state();
    assert_eq!((after.phase, after.unarchived_manual_output, after.attempts), (recovery::Phase::Complete, false, state.attempts));
    assert_eq!(origin(&fixture), db::TranscriptOrigin::Live);
    assert_eq!(proofs(&fixture), (true, true));
    sql(&fixture, &["DROP TRIGGER audit_replace_fault"]);
    let output = finalize(&fixture, id, &[]);
    assert!(output.status.success(), "{output:?}");
    assert_finalized(&fixture, id, &state);
    fs::remove_dir_all(fixture.root).unwrap();
}

/// After the commit a file someone edited is still theirs: it is not replaced, the error names
/// it, and the record stays so that `--overwrite` can finish.
#[test]
fn a_file_edited_after_the_commit_is_replaced_only_with_overwrite() {
    let (fixture, id) = promoted();
    let state = fixture.state();
    finalize_killed_at(&fixture, id, "finalize_committed", &[]);
    forbid_engine(&fixture);
    let markdown = fixture.dir.join("transcript.md");
    let edited = format!("{}\nmy own note\n", fs::read_to_string(&markdown).unwrap());
    fs::write(&markdown, &edited).unwrap();
    let json = fs::read(fixture.dir.join("transcript.json")).unwrap();

    let output = finalize(&fixture, id, &[]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(stderr_of(&output).contains("was changed since meetscribe wrote it"), "{}", stderr_of(&output));
    assert_eq!(fs::read_to_string(&markdown).unwrap(), edited, "an edited file was replaced without --overwrite");
    assert_eq!(fs::read(fixture.dir.join("transcript.json")).unwrap(), json, "the Markdown comes first; nothing after it is touched");
    assert!(fixture.state().provisional().is_some());

    let output = finalize(&fixture, id, &["--overwrite"]);
    assert!(output.status.success(), "{output:?}");
    assert_finalized(&fixture, id, &state);
    assert!(!fs::read_to_string(&markdown).unwrap().contains("my own note"));
    fs::remove_dir_all(fixture.root).unwrap();
}

/// A meeting promoted by the build before this one has no record. It is refused, and
/// `--overwrite` adopts its files BEFORE the pass, so that a run killed after its commit can
/// still be told from a meeting nobody touched.
#[test]
fn a_meeting_with_no_record_needs_overwrite_and_is_then_finished_like_any_other() {
    let (fixture, id) = promoted();
    edit_state(&fixture, |state| { state.as_object_mut().unwrap().remove("provisional"); });
    let state = fixture.state();
    let before = untouched(&fixture);
    let refused = finalize(&fixture, id, &[]);
    assert_eq!(refused.status.code(), Some(1));
    assert_eq!(untouched(&fixture), before);

    finalize_killed_at(&fixture, id, "finalize_committed", &["--overwrite"]);
    assert_eq!(origin(&fixture), db::TranscriptOrigin::Batch);
    assert_eq!(fixture.state().provisional(), Some(record_of(&fixture.dir)), "the adopted files are on record before the commit");
    let stopped = untouched(&fixture);
    let resumed = child_output(&fixture, spawn_cli(&fixture, &resume_argv(&fixture)));
    assert_eq!(resumed.status.code(), Some(1), "{resumed:?}");
    assert_eq!(untouched(&fixture), stopped);
    assert!(!fixture.dir.join(".recovered").exists());

    forbid_engine(&fixture);
    let output = finalize(&fixture, id, &[]); // no flag: the record is there now
    assert!(output.status.success(), "{output:?}");
    assert_finalized(&fixture, id, &state);
    fs::remove_dir_all(fixture.root).unwrap();
}

fn add_rule(fixture: &Fixture, pattern: &str, replacement: &str) {
    crate::new_runtime().unwrap().block_on(async {
        let mut db = db::Db::open_existing(&fixture.opts.db_path, fixture.state().archive_id.as_deref().unwrap()).await.unwrap();
        db.add_vocab(pattern, replacement, false, START).await.unwrap(); db.close().await.unwrap();
    });
}

/// A vocabulary rule added between two runs changes the render. The file an interrupted run
/// already replaced then matches neither the live transcript nor the new render; it is still
/// recognised, because that run saved its hashes before it wrote and nothing is ever removed
/// from the record until the end.
#[test]
fn a_rule_added_between_runs_does_not_strand_a_half_published_meeting() {
    for kills in [1, 2] {
        let (fixture, id) = promoted();
        finalize_killed_at(&fixture, id, "finalize_markdown_replaced", &[]);
        forbid_engine(&fixture);
        add_rule(&fixture, "synthetic", "FIRSTRULE");
        if kills == 2 {
            finalize_killed_at(&fixture, id, "finalize_markdown_replaced", &[]);
            add_rule(&fixture, "ready", "SECONDRULE");
        }
        let output = finalize(&fixture, id, &[]);
        assert!(output.status.success(), "{kills} kill(s): {output:?}\n{}", stderr_of(&output));
        let state = fixture.state();
        assert!(state.provisional.is_none(), "{kills} kill(s): the record is still there");
        assert_eq!(proofs(&fixture), (true, true), "{kills} kill(s): both files are the render under the rules as they are now");
        let markdown = fs::read_to_string(fixture.dir.join("transcript.md")).unwrap();
        assert!(markdown.contains("FIRSTRULE") && !markdown.contains("provisional"), "{kills} kill(s)");
        assert_eq!(markdown.contains("SECONDRULE"), kills == 2);
        assert!(fs::read_to_string(fixture.dir.join("transcript.json")).unwrap().contains("FIRSTRULE"), "{kills} kill(s): the JSON was left behind");
        fs::remove_dir_all(fixture.root).unwrap();
    }
}

/// The record cannot grow without end: the state file has a size limit, and past it nothing
/// could read the session at all. A seventeenth version is refused before anything is saved.
#[test]
fn a_full_record_refuses_and_overwrite_starts_it_again() {
    let (fixture, id) = promoted();
    let state = fixture.state();
    finalize_killed_at(&fixture, id, "finalize_committed", &[]);
    forbid_engine(&fixture);
    edit_state(&fixture, |state| {
        let own = state["provisional"]["markdown"][0].clone();
        let mut full: Vec<serde_json::Value> = (1..recovery::PROVISIONAL_HASHES).map(|n| serde_json::json!(format!("{n:064x}"))).collect();
        full.push(own);
        state["provisional"]["markdown"] = serde_json::json!(full);
    });
    assert_eq!(fixture.state().provisional().unwrap().markdown.len(), recovery::PROVISIONAL_HASHES);
    let before = untouched(&fixture);
    let output = finalize(&fixture, id, &[]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(stderr_of(&output).contains("--overwrite"), "{}", stderr_of(&output));
    assert_eq!(untouched(&fixture), before, "a refused publication changed something");
    let output = finalize(&fixture, id, &["--overwrite"]);
    assert!(output.status.success(), "{output:?}");
    assert_finalized(&fixture, id, &state);
    fs::remove_dir_all(fixture.root).unwrap();
}

/// What an owner does after a call without thinking of `finalize`: a vocabulary rule, and
/// `speakers cluster` (whose effect on the rows is repeated here). Both change how the meeting
/// renders and rewrite no file, so the files stop being a fresh render. They are still the files
/// promotion wrote, and that is what is asked.
#[test]
fn ordinary_commands_after_the_call_do_not_block_finalize() {
    let (fixture, id) = promoted();
    let state = fixture.state();
    add_voice(&fixture, id, false);
    add_rule(&fixture, "live", "LIVERULE");
    assert!(!proofs(&fixture).0, "the fixture did not make the files differ from a fresh render");
    let output = finalize(&fixture, id, &[]);
    assert!(output.status.success(), "finalize refused after ordinary commands: {}", stderr_of(&output));
    assert_finalized(&fixture, id, &state);
    fs::remove_dir_all(fixture.root).unwrap();
}

fn connection(fixture: &Fixture) -> (tokio::runtime::Runtime, sqlx::SqliteConnection) {
    let runtime = crate::new_runtime().unwrap();
    let connection = runtime.block_on(sqlx::SqliteConnection::connect_with(&sqlx::sqlite::SqliteConnectOptions::new()
        .filename(&fixture.opts.db_path).create_if_missing(false))).unwrap();
    (runtime, connection)
}

/// Wait until the child's pass has started, then long enough for it to have reached the database.
fn wait_until_stuck_on_the_database(fixture: &Fixture, child: &mut std::process::Child) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !fixture.root.join("engine-ran").exists() {
        assert!(std::time::Instant::now() < deadline, "the pass never started");
        std::thread::sleep(Duration::from_millis(5));
    }
    std::thread::sleep(Duration::from_millis(900));
    assert!(child.try_wait().unwrap().is_none(), "finalize did not wait for the database");
}

/// Another process holds the database. A writer makes the replacement wait at `BEGIN`; a reader
/// lets it begin and makes it wait at `COMMIT`. Either way it waits, commits once, and the run
/// ends as if nobody had been there. Held past the timeout, the run fails and nothing changed.
#[test]
fn the_replacement_waits_for_a_busy_database_and_gives_up_without_a_trace() {
    // (the other process's statements, where finalize waits)
    let others: [(&[&str], &str); 2] = [(&["BEGIN IMMEDIATE"], "BEGIN"), (&["BEGIN", "SELECT COUNT(*) FROM meetings"], "COMMIT")];
    for (statements, at) in others {
        // Released in time.
        let (fixture, id) = promoted();
        let state = fixture.state();
        let (runtime, mut other) = connection(&fixture);
        for statement in statements { runtime.block_on(sqlx::query(statement).fetch_all(&mut other)).unwrap(); }
        let mut child = spawn_cli(&fixture, &finalize_argv(&fixture, id, &[]));
        wait_until_stuck_on_the_database(&fixture, &mut child);
        runtime.block_on(sqlx::query("ROLLBACK").execute(&mut other)).unwrap();
        let output = child_output(&fixture, child);
        assert!(output.status.success(), "waiting at {at}: {output:?}\n{}", stderr_of(&output));
        assert_finalized(&fixture, id, &state);
        runtime.block_on(other.close()).unwrap();
        fs::remove_dir_all(fixture.root).unwrap();

        // Held past the timeout (shortened for the test).
        let (fixture, id) = promoted();
        let state = fixture.state();
        fs::write(fixture.root.join("busy-timeout-ms"), "300\n").unwrap();
        let before = untouched(&fixture);
        let (runtime, mut other) = connection(&fixture);
        for statement in statements { runtime.block_on(sqlx::query(statement).fetch_all(&mut other)).unwrap(); }
        let output = finalize(&fixture, id, &[]);
        assert_eq!(output.status.code(), Some(1), "giving up at {at}: {output:?}");
        runtime.block_on(sqlx::query("ROLLBACK").execute(&mut other)).unwrap(); runtime.block_on(other.close()).unwrap();
        assert_eq!(origin(&fixture), db::TranscriptOrigin::Live, "giving up at {at}");
        assert_eq!(untouched(&fixture), before, "giving up at {at}: a failed replacement left a trace");
        // And with the database free again, the same command finishes.
        fs::remove_file(fixture.root.join("busy-timeout-ms")).unwrap();
        let output = finalize(&fixture, id, &[]);
        assert!(output.status.success(), "after giving up at {at}: {output:?}");
        assert_finalized(&fixture, id, &state);
        fs::remove_dir_all(fixture.root).unwrap();
    }
}

/// A `COMMIT` can report an error after it has committed. The connection that saw the error is
/// never asked: the outcome is read on a new one, and the run goes on to publish.
#[test]
fn an_error_reported_after_a_real_commit_is_read_as_committed() {
    let (fixture, id) = promoted();
    let state = fixture.state();
    fs::write(fixture.root.join("fault"), "replace_reports_an_error\n").unwrap();
    let output = finalize(&fixture, id, &[]);
    assert!(output.status.success(), "{output:?}\n{}", stderr_of(&output));
    assert_finalized(&fixture, id, &state);
    fs::remove_dir_all(fixture.root).unwrap();
}

/// A meeting promoted on a retry: the first attempt is killed at `point`, charged, and retried.
/// Its pair is published under `.recovered/`, never at the session root.
fn promoted_on_a_retry(point: &str) -> (Fixture, i64) {
    let (fixture, owner) = Fixture::new(); drop(owner);
    write_stream(&fixture, &stream_events(&fixture));
    fs::write(fixture.root.join("cutpoint"), format!("{point}\n")).unwrap();
    let argv = promote_argv(&fixture, &fixture.job());
    assert_eq!(child_output(&fixture, spawn_cli(&fixture, &argv)).status.code(), Some(85));
    fs::remove_file(fixture.root.join("cutpoint")).unwrap(); fs::remove_file(fixture.root.join("cutpoint-ack.json")).unwrap();
    assert_eq!(child_output(&fixture, spawn_cli(&fixture, &argv)).status.code(), Some(1));
    fs::write(fixture.root.join("clock"), format!("{}\n", fixture.state().next_retry)).unwrap();
    let finished = child_output(&fixture, spawn_cli(&fixture, &argv));
    assert!(finished.status.success(), "the retry failed: {finished:?}");
    let id = fixture.meetings()[0].id;
    (fixture, id)
}

/// Only the pair in the recorded directory is replaced. With no pair at the root, the root stays
/// empty; with one (the first attempt wrote it and died), it is left as it is and named.
#[test]
fn a_meeting_published_beside_the_root_is_replaced_where_it_was_published() {
    for (point, root_pair) in [("live_committed", false), ("published", true)] {
        let (fixture, id) = promoted_on_a_retry(point);
        let state = fixture.state();
        let publication = state.publication.clone().unwrap();
        assert!(publication.starts_with(fixture.dir.join(".recovered")), "{point}");
        assert_eq!(fixture.dir.join("transcript.md").exists(), root_pair, "{point}");
        let root = root_pair.then(|| (fs::read(fixture.dir.join("transcript.md")).unwrap(), fs::read(fixture.dir.join("transcript.json")).unwrap()));
        let output = finalize(&fixture, id, &[]);
        assert!(output.status.success(), "{point}: {output:?}\n{}", stderr_of(&output));
        assert_finalized(&fixture, id, &state);
        let recovered: Vec<PathBuf> = fs::read_dir(publication.parent().unwrap()).unwrap().map(|entry| entry.unwrap().path()).collect();
        assert_eq!(recovered, std::slice::from_ref(&publication), "{point}: no second copy is published");
        match root {
            Some(pair) => {
                assert_eq!((fs::read(fixture.dir.join("transcript.md")).unwrap(), fs::read(fixture.dir.join("transcript.json")).unwrap()), pair, "{point}: the root pair was touched");
                assert!(stderr_of(&output).contains("are an older copy and are left as they are"), "{point}: the root pair is not named: {}", stderr_of(&output));
            }
            None => assert!(!fixture.dir.join("transcript.md").exists() && !fixture.dir.join("transcript.json").exists(), "{point}: a root pair appeared"),
        }
        fs::remove_dir_all(fixture.root).unwrap();
    }
}

/// `transcribe --resume` refuses only what `finalize` left unfinished. A live-stored meeting, and
/// a session whose archive cannot be read, go the way they always went.
#[test]
fn transcribe_goes_on_as_before_unless_a_finalize_is_unfinished() {
    // A record and a `live` row: an explicit resume publishes a second copy, as before, and the
    // record follows the publication.
    let (fixture, id) = promoted();
    let resumed = child_output(&fixture, spawn_cli(&fixture, &resume_argv(&fixture)));
    assert!(resumed.status.success(), "{resumed:?}");
    let state = fixture.state(); let publication = state.publication.clone().unwrap();
    assert!(publication.starts_with(fixture.dir.join(".recovered")));
    assert_eq!((origin(&fixture), state.provisional()), (db::TranscriptOrigin::Live, Some(record_of(&publication))));
    // ...and `finalize` then replaces that copy.
    let output = finalize(&fixture, id, &[]);
    assert!(output.status.success(), "{output:?}\n{}", stderr_of(&output));
    assert_finalized(&fixture, id, &state);
    fs::remove_dir_all(fixture.root).unwrap();

    // An unfinished finalize whose archive is gone, or is another archive: the question cannot be
    // answered, so nothing is refused here and recovery says what it always said.
    for foreign in [false, true] {
        let (fixture, id) = promoted();
        finalize_killed_at(&fixture, id, "finalize_committed", &[]);
        fs::rename(&fixture.opts.db_path, fixture.root.join("moved.db")).unwrap();
        if foreign { crate::new_runtime().unwrap().block_on(async { db::Db::open(&fixture.opts.db_path).await.unwrap().close().await.unwrap(); }); }
        let resumed = child_output(&fixture, spawn_cli(&fixture, &resume_argv(&fixture)));
        assert_eq!(resumed.status.code(), Some(1), "foreign={foreign}");
        let state = fixture.state();
        assert_eq!((state.phase, state.error_class.as_deref()), (recovery::Phase::NeedsAttention, Some("archive_identity")), "foreign={foreign}: not today's path");
        fs::remove_dir_all(fixture.root).unwrap();
    }
}

fn speakers(fixture: &Fixture, args: &[&str]) -> Result<()> {
    let mut argv: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
    argv.extend(["--db".to_string(), fixture.opts.db_path.display().to_string()]);
    crate::speakers::run_speakers(&argv)
}

/// A second, batch-stored meeting with one voice cluster, in the fixture's archive.
fn batch_meeting_with_a_voice(fixture: &Fixture) -> i64 {
    crate::new_runtime().unwrap().block_on(async {
        let mut db = db::Db::open_existing(&fixture.opts.db_path, fixture.state().archive_id.as_deref().unwrap()).await.unwrap();
        let meta = db::MeetingMeta { title: "Planning".into(), source_dir: "/synthetic/planning".into(), model: "fixture".into(), lang: "en".into(),
            started_at: START - 86_400, duration_secs: 2.0, created_at: START, capture_status: None, origin: db::TranscriptOrigin::Batch };
        let segment = crate::transcript::TranscriptSegment { speaker: crate::transcript::Speaker::Others, text: "The plan is agreed".into(), t_start: 0.0, t_end: 2.0, confidence: 0.9 };
        let centroid: Vec<u8> = [1.0f32, 0.0].iter().flat_map(|value| value.to_le_bytes()).collect();
        let cluster = db::NewCluster { cluster: "A".into(), speaker_id: None, assigned_by: None, match_score: None, centroid, dim: 2, n_windows: 1, speech_secs: 2.0 };
        let id = db.insert_meeting_with_voices(&meta, &[segment], &[cluster], &[db::SegmentVoice { segment: 0, cluster: 0, inherited: false, embedding: None }]).await.unwrap();
        db.close().await.unwrap(); id
    })
}

/// `finalize` replaces every voice cluster of a live meeting, at the end of a pass during which
/// it holds the session lock. Each `speakers` command that writes clusters takes that lock for a
/// live meeting, so it is refused for as long as `finalize` runs and can never write across the
/// replacement. A batch meeting, and `rename`, never meet the lock.
#[test]
fn speakers_commands_cannot_cross_a_finalize_and_leave_other_meetings_alone() {
    let (fixture, id) = promoted();
    add_voice(&fixture, id, false);
    let batch = batch_meeting_with_a_voice(&fixture);
    let (live, other) = (id.to_string(), batch.to_string());
    let clusters = |meeting: i64| stored_voices(&fixture, meeting).1.iter().map(|c| (c.cluster.clone(), c.speaker_id, c.assigned_by.clone(), c.skipped)).collect::<Vec<_>>();

    let held = recovery::Session::open(&fixture.dir).unwrap().unwrap(); // what a running finalize holds
    let database = fs::read(&fixture.opts.db_path).unwrap();
    let writers: [&[&str]; 7] = [&["label", &live, "A", "Ada", "Example"], &["unlabel", &live, "A"], &["skip", &live, "A"],
        &["merge", &live, "A", "B"], &["split", &live, "A"], &["cluster", &live, "--recluster"], &["match", &live]];
    for command in writers {
        let error = speakers(&fixture, command).expect_err(command[0]);
        assert!(format!("{error:#}").contains("is in use by another meetscribe command"), "{}: {error:#}", command[0]);
        assert_eq!(fs::read(&fixture.opts.db_path).unwrap(), database, "{}: a refused command wrote", command[0]);
    }
    // Other meetings and the people table are not finalize's: no lock, as before.
    speakers(&fixture, &["label", &other, "A", "Ada", "Example"]).unwrap();
    speakers(&fixture, &["rename", "1", "Ada", "Sample"]).unwrap();
    assert_eq!(clusters(batch)[0].2.as_deref(), Some("manual"));
    // With --all the live meeting is skipped and the rest are done.
    let live_before = clusters(id);
    speakers(&fixture, &["match", "--all"]).unwrap();
    speakers(&fixture, &["cluster", "--all", "--recluster"]).unwrap();
    assert_eq!(clusters(id), live_before, "--all wrote the locked meeting");
    drop(held);

    // The lock is free: the same commands work on the live meeting.
    speakers(&fixture, &["skip", &live, "A"]).unwrap();
    assert!(clusters(id)[0].3, "skip did not reach the live meeting once the lock was free");
    // A live meeting stored from a relative path cannot be locked by guessing where it is.
    sql(&fixture, &[&format!("UPDATE meetings SET source_dir = 'recording' WHERE id = {id}")]);
    let error = speakers(&fixture, &["unlabel", &live, "A"]).unwrap_err();
    assert!(format!("{error:#}").contains("--session"), "{error:#}");
    assert!(clusters(id)[0].3, "a refused command wrote");
    speakers(&fixture, &["unlabel", &live, "A", "--session", fixture.dir.to_str().unwrap()]).unwrap();
    assert!(!clusters(id)[0].3);
    // A live meeting whose recording was pruned has nothing left to replace: no lock, as before.
    sql(&fixture, &[&format!("UPDATE meetings SET source_dir = '/synthetic/pruned-recording' WHERE id = {id}")]);
    speakers(&fixture, &["skip", &live, "A"]).unwrap();
    fs::remove_dir_all(fixture.root).unwrap();
}

/// `rerender --write` reads every meeting's rows before it locks anything. Started before a
/// `finalize` ended, it used to hold the live rows and write their render over the new files.
/// For a meeting it loaded as live it now locks first and reads again.
#[test]
fn rerender_started_before_a_finalize_ended_never_writes_the_old_live_render() {
    for finalize_dies in [false, true] {
        let (fixture, id) = promoted();
        let state = fixture.state();
        fs::write(fixture.root.join("holdpoint"), "rerender_loaded\n").unwrap();
        let argv: Vec<String> = ["rerender", &id.to_string(), "--db", fixture.opts.db_path.to_str().unwrap(), "--write"].into_iter().map(String::from).collect();
        let rerender = spawn_cli(&fixture, &argv);
        wait_for_owner(&fixture, "rerender_loaded"); // it holds the live rows now

        if finalize_dies { finalize_killed_at(&fixture, id, "finalize_committed", &[]); }
        else { let output = finalize(&fixture, id, &[]); assert!(output.status.success(), "{output:?}"); }
        fs::write(fixture.root.join("release-owner"), b"release\n").unwrap();
        let output = child_output(&fixture, rerender);
        assert!(output.status.success(), "finalize_dies={finalize_dies}: rerender failed: {output:?}");

        let markdown = fs::read_to_string(fixture.dir.join("transcript.md")).unwrap();
        assert!(!markdown.contains("provisional") && BATCH_WORDS.iter().all(|words| markdown.contains(words)),
            "finalize_dies={finalize_dies}: the old live render was written over the meeting");
        assert!(proofs(&fixture).0, "finalize_dies={finalize_dies}: the files are not the render of the stored rows");
        if finalize_dies {
            // rerender did the publishing; finalize finds both files done and only ends the record.
            let files = (fs::read(fixture.dir.join("transcript.md")).unwrap(), fs::read(fixture.dir.join("transcript.json")).unwrap());
            forbid_engine(&fixture);
            let output = finalize(&fixture, id, &[]);
            assert!(output.status.success(), "{output:?}\n{}", stderr_of(&output));
            assert_eq!((fs::read(fixture.dir.join("transcript.md")).unwrap(), fs::read(fixture.dir.join("transcript.json")).unwrap()), files);
        } else {
            assert!(String::from_utf8_lossy(&output.stdout).contains("0 to change · 1 already current"), "{}", String::from_utf8_lossy(&output.stdout));
        }
        assert_finalized(&fixture, id, &state);
        fs::remove_dir_all(fixture.root).unwrap();
    }
}

/// Dividing the batch pass must not move the naming step out of the block whose failure still
/// publishes the transcript unstored: whisper's half hour is not thrown away over a name.
#[test]
fn a_naming_failure_in_the_batch_pass_still_ends_in_the_unstored_export() {
    let (fixture, owner) = Fixture::new(); drop(owner);
    fs::write(fixture.root.join("fault"), "naming_fails\n").unwrap();
    let output = child_output(&fixture, spawn_cli(&fixture, &transcribe_argv(&fixture, &fixture.job())));
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let state = fixture.state();
    assert_eq!((state.meeting_id, state.unarchived_manual_output), (None, true), "an unstored export is the owner's to keep");
    assert!(fixture.meetings().is_empty());
    let markdown = fs::read_to_string(fixture.dir.join("transcript.md")).unwrap();
    assert!(BATCH_WORDS.iter().all(|words| markdown.contains(words)), "the transcript was not exported");
    assert!(serde_json::from_slice::<serde_json::Value>(&fs::read(fixture.dir.join("transcript.json")).unwrap()).unwrap().is_array());
    fs::remove_dir_all(fixture.root).unwrap();
}

/// Another recorded session in the same archive and under the same base as `first`.
fn another_session(first: &Fixture, name: &str) -> Fixture {
    let dir = first.root.join("sessions").join(name);
    let mut owner = recovery::Session::begin(&dir, &first.opts.db_path, &dir, "com.google.Chrome", "Synthetic release review", START).unwrap();
    owner.start_roll(0, 16_000, 0.0).unwrap();
    for (channel, sample) in [("mic", 0.25f32), ("system", -0.5f32)] {
        let mut wav = hound::WavWriter::create(dir.join(format!("{channel}.wav")), hound::WavSpec {
            channels: 1, sample_rate: 16_000, bits_per_sample: 32, sample_format: hound::SampleFormat::Float }).unwrap();
        for _ in 0..48_000 { wav.write_sample(sample).unwrap(); } wav.finalize().unwrap();
    }
    owner.checkpoint(0, 48_000, 48_000).unwrap(); owner.finish_capture(3.0, 0.1).unwrap();
    let mut fixture = first.clone();
    fixture.opts.export_dir = dir.clone(); fixture.opts.session_key = Some(owner.state.key.clone()); fixture.dir = dir;
    fixture
}

fn meeting_of(fixture: &Fixture) -> db::MeetingRow {
    let id = fixture.state().meeting_id.unwrap();
    fixture.meetings().into_iter().find(|row| row.id == id).unwrap()
}

#[test]
fn all_live_finalizes_every_live_meeting_and_one_failure_does_not_stop_the_rest() {
    let (first, _) = promoted();
    let second = another_session(&first, "second");
    write_stream(&second, &stream_events(&second));
    assert!(child_output(&second, spawn_cli(&second, &promote_argv(&second, &second.job()))).status.success());
    let third = another_session(&first, "third"); // an ordinary batch meeting: not finalize's business
    assert!(child_output(&third, spawn_cli(&third, &transcribe_argv(&third, &third.job()))).status.success());
    assert_eq!([meeting_of(&first).origin, meeting_of(&second).origin, meeting_of(&third).origin],
        [db::TranscriptOrigin::Live, db::TranscriptOrigin::Live, db::TranscriptOrigin::Batch]);
    let untouched_third = session_snapshot(&third.dir);
    // The older live meeting cannot be finalized: its Markdown was edited.
    let edited = format!("{}\nmy own note\n", fs::read_to_string(first.dir.join("transcript.md")).unwrap());
    fs::write(first.dir.join("transcript.md"), &edited).unwrap();

    let all_live = |extra: &[&str]| -> std::process::Output {
        let mut argv: Vec<String> = ["finalize", "--all-live", "--db", first.opts.db_path.to_str().unwrap(), "--base", first.base.to_str().unwrap(),
            "--lang", "en", "--model", first.opts.model.to_str().unwrap()].into_iter().map(String::from).collect();
        argv.extend(extra.iter().map(|flag| flag.to_string()));
        child_output(&first, spawn_cli(&first, &argv))
    };
    let output = all_live(&[]);
    assert_eq!(output.status.code(), Some(1), "one meeting failed, so the command did: {output:?}");
    assert_eq!([meeting_of(&first).origin, meeting_of(&second).origin], [db::TranscriptOrigin::Live, db::TranscriptOrigin::Batch], "the failure stopped the other meeting");
    assert_eq!(fs::read_to_string(first.dir.join("transcript.md")).unwrap(), edited);
    assert!(String::from_utf8_lossy(&output.stdout).contains(&format!("meeting {}: finalized", meeting_of(&second).id)));
    assert!(stderr_of(&output).contains(&format!("meeting {}:", meeting_of(&first).id)));
    assert_eq!(session_snapshot(&third.dir), untouched_third, "a batch meeting was touched by --all-live");
    no_meeting_text(&output);

    let output = all_live(&["--overwrite"]);
    assert!(output.status.success(), "{output:?}\n{}", stderr_of(&output));
    assert_eq!(meeting_of(&first).origin, db::TranscriptOrigin::Batch);
    assert_eq!(session_snapshot(&third.dir), untouched_third);
    let output = all_live(&[]);
    assert!(output.status.success() && String::from_utf8_lossy(&output.stdout).contains("no live-stored meeting"), "{output:?}");

    // A session directory that is gone: the live transcript is final. With --all-live that is
    // a skip, not a failure; asked for by id, it is said and counted.
    let fourth = another_session(&first, "fourth");
    write_stream(&fourth, &stream_events(&fourth));
    assert!(child_output(&fourth, spawn_cli(&fourth, &promote_argv(&fourth, &fourth.job()))).status.success());
    let gone = meeting_of(&fourth).id;
    fs::remove_dir_all(&fourth.dir).unwrap();
    let output = all_live(&[]);
    assert!(output.status.success() && String::from_utf8_lossy(&output.stdout).contains("the recording is gone"), "{output:?}");
    let output = finalize(&first, gone, &[]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr_of(&output).contains("the recording is gone; the live transcript is final"));
    fs::remove_dir_all(first.root).unwrap();
}

/// A meeting stored from a relative path is found only when its session is named.
#[test]
fn finalize_finds_a_relatively_stored_meeting_through_session() {
    let (fixture, id) = promoted();
    let state = fixture.state();
    sql(&fixture, &["UPDATE meetings SET source_dir = 'recording'"]);
    let output = finalize(&fixture, id, &["--session", fixture.dir.to_str().unwrap()]);
    assert!(output.status.success(), "{output:?}\n{}", stderr_of(&output));
    assert_finalized(&fixture, id, &state);
    fs::remove_dir_all(fixture.root).unwrap();
}

/// `finalize` must produce what the daemon's own pass would have produced for the session, so it
/// reads its options where the daemon reads them. `job_for` is the daemon's; the config mapping
/// above it is `run_daemon`'s, repeated here because it is not a function.
#[test]
fn finalize_runs_the_pass_with_the_options_the_daemon_would_use() {
    let (fixture, owner) = Fixture::new(); drop(owner);
    let base = fixture.base.clone();
    let argv = |extra: &[&str]| -> Vec<String> { ["7", "--base", base.to_str().unwrap()].iter().chain(extra).map(|arg| arg.to_string()).collect() };
    for config in [None, Some("[daemon]\nlang = \"pt\"\n\n[speakers]\nsplit = false\n")] {
        if let Some(text) = config { fs::write(base.join("config.toml"), text).unwrap(); }
        let file = crate::config::Config::load(&base);
        let daemon = DaemonConfig {
            base: base.clone(), sessions_dir: base.join("sessions"), logs_dir: base.join("logs"), db_path: base.join("meetscribe.db"),
            model: base.join("models/ggml-large-v3.bin"), speaker_model: base.join(crate::SPEAKER_MODEL_REL), split: file.speakers.split,
            diarizer_bin: base.join(crate::DIARIZER_BIN_REL), diarizer_model: base.join(crate::DIARIZER_MODEL_REL), lang: file.daemon.lang.clone(),
            min_secs: file.daemon.min_secs, once: false, allowlist: file.effective_allowlist(),
            sessions_days: file.retention.sessions_days, log_max_mb: file.retention.log_max_mb,
        };
        let job = job_for(&daemon, fixture.dir.clone(), "Chrome".into(), "a title the stored state overrides".into());
        assert_eq!(crate::finalize::options(&argv(&[]), &fixture.state()).unwrap(), job.opts, "config: {config:?}");
        assert_eq!(job.opts.lang == "pt", config.is_some(), "the fixture's config was not read");
    }
    // Flags go on top of the configuration.
    fs::remove_file(base.join("config.toml")).unwrap();
    let flagged = crate::finalize::options(&argv(&["--lang", "en", "--no-split", "--model", "/synthetic/other.bin"]), &fixture.state()).unwrap();
    assert_eq!((flagged.lang.as_str(), flagged.split, flagged.model.as_path()), ("en", false, Path::new("/synthetic/other.bin")));
    fs::remove_dir_all(fixture.root).unwrap();
}
