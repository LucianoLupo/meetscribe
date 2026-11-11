//! Disposable, model-free recovery and archive integration regressions.
use crate::{db, recovery, transcript};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq)]
enum CaptureKind { Complete, Interrupted, Uncheckpointed }

fn fixture(kind: CaptureKind) -> (PathBuf, recovery::Session) {
    let base = std::env::temp_dir().join(format!("meetscribe-recovery-{}", recovery::new_identity()));
    recovery::private_dir(&base).unwrap();
    let dir = base.join("recording");
    let mut session = recovery::Session::begin(&dir, &base.join("archive.db"), &dir,
        "org.example.synthetic-call", "Release readiness sync", 1_762_880_000).unwrap();
    session.start_roll(0, 16_000, 0.0).unwrap();
    let frames = if kind == CaptureKind::Interrupted { 24_000 } else { 16_000 };
    for (channel, sample) in [("mic", 0.25f32), ("system", -0.5f32)] {
        let mut writer = hound::WavWriter::create(dir.join(format!("{channel}.wav")), hound::WavSpec {
            channels: 1, sample_rate: 16_000, bits_per_sample: 32, sample_format: hound::SampleFormat::Float,
        }).unwrap();
        for _ in 0..frames { writer.write_sample(sample).unwrap(); }
        writer.finalize().unwrap();
    }
    if kind != CaptureKind::Uncheckpointed { session.checkpoint(0, 16_000, 16_000).unwrap(); }
    if kind == CaptureKind::Complete { session.finish_capture(1.0, 0.1).unwrap(); }
    (base, session)
}

fn meta(session: &recovery::Session) -> db::MeetingMeta {
    db::MeetingMeta {
        title: session.state.title.clone(), source_dir: session.dir.display().to_string(),
        model: "synthetic-whisper-large-v3".into(), lang: "en".into(),
        started_at: session.state.started_at, duration_secs: 1.0, created_at: 1_762_880_010,
        capture_status: Some("partial".into()),
    }
}

fn segment() -> transcript::TranscriptSegment {
    transcript::TranscriptSegment { speaker: transcript::Speaker::Others,
        text: "The release checklist is ready for review".into(), t_start: 0.1, t_end: 0.9, confidence: 0.94 }
}

fn resume_args(dir: &Path) -> Vec<String> {
    vec![dir.display().to_string(), "--resume".into(), "--no-split".into(),
        "--model".into(), dir.join("does-not-exist.bin").display().to_string()]
}

#[test]
fn audit_f1_metadata_lock_and_committed_prefix() {
    for _ in 0..5 {
        let (base, capture) = fixture(CaptureKind::Interrupted);
        let dir = capture.dir.clone();
        let mic_before = std::fs::read(dir.join("mic.wav")).unwrap();
        let system_before = std::fs::read(dir.join("system.wav")).unwrap();
        assert!(recovery::Session::open(&dir).unwrap().is_none(), "live capture must exclude resume");
        drop(capture);
        let mut session = recovery::Session::open(&dir).unwrap().unwrap();
        let inputs = session.inputs().unwrap();
        assert_ne!(inputs, dir, "an uncommitted tail must not be fed to inference");
        for (channel, sample) in [("mic", 0.25f32), ("system", -0.5f32)] {
            let reader = hound::WavReader::open(inputs.join(format!("{channel}.wav"))).unwrap();
            assert_eq!(reader.len(), 16_000);
            assert!(reader.into_samples::<f32>().all(|value| value.unwrap() == sample));
        }
        assert_eq!(session.state.capture_status, recovery::CaptureStatus::Partial);
        assert_eq!(std::fs::read(dir.join("mic.wav")).unwrap(), mic_before);
        assert_eq!(std::fs::read(dir.join("system.wav")).unwrap(), system_before);
        drop(session);
        std::fs::remove_dir_all(base).unwrap();

        let (base, session) = fixture(CaptureKind::Uncheckpointed);
        let dir = session.dir.clone(); drop(session);
        let args = resume_args(&dir);
        let error = crate::run_transcribe(&args).expect_err("uncheckpointed recovery must fail");
        assert!(format!("{error:#}").contains("trusted nonzero checkpoint"), "unexpected recovery error: {error:#}");
        let state = recovery::State::read(&dir).unwrap();
        assert_eq!(state.phase, recovery::Phase::NeedsAttention);
        assert_eq!(state.error_class.as_deref(), Some("checkpoint_invalid"));
        assert!(dir.join("mic.wav").exists() && dir.join("system.wav").exists());
        std::fs::remove_dir_all(base).unwrap();
    }
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f1_metadata\",\"iterations\":5}}");
}

#[test]
fn audit_f3_idempotent_recovery() {
    for _ in 0..5 {
        let (base, mut session) = fixture(CaptureKind::Interrupted);
        session.reserve(crate::now_epoch(), false).unwrap();
        let rt = crate::new_runtime().unwrap();
        let meeting_meta = meta(&session);
        let key = session.state.key.clone();
        let id = rt.block_on(async {
            let mut archive = session.archive().await.unwrap();
            let id = archive.insert_keyed_meeting_with_voices(&meeting_meta, &[segment()], &[], &[], &key).await.unwrap();
            archive.close().await.unwrap(); id
        });
        let dir = session.dir.clone(); let db_path = session.state.db_path.clone();
        let archive_id = session.state.archive_id.clone().unwrap();
        // Deliberately omit the Archived receipt: the persisted DB mapping must win.
        drop(session);
        let manual_markdown = "# Manual release notes\n\nKeep the revised deployment date.\n";
        let manual_json = "[{\"speaker\":\"others\",\"text\":\"Keep the revised deployment date\",\"t_start\":0.1,\"t_end\":0.9,\"confidence\":0.94}]";
        std::fs::write(dir.join("transcript.md"), manual_markdown).unwrap();
        std::fs::write(dir.join("transcript.json"), manual_json).unwrap();
        let args = resume_args(&dir);
        let parsed = crate::parse_transcribe_args(&args).unwrap().unwrap();
        assert_eq!(parsed.opts.db_path, db_path);
        assert!(parsed.opts.model.is_absolute());
        crate::run_transcribe(&args).unwrap(); // Missing model proves this path skips inference.
        let state = recovery::State::read(&dir).unwrap();
        assert_eq!(state.phase, recovery::Phase::Complete);
        assert_eq!(state.meeting_id, Some(id));
        assert_eq!(state.capture_status, recovery::CaptureStatus::Partial);
        let publication = state.publication.unwrap();
        assert_ne!(publication, dir);
        assert!(std::fs::read_to_string(publication.join("transcript.md")).unwrap().contains("partial"));
        assert!(std::fs::read_to_string(publication.join("transcript.json")).unwrap().contains("release checklist"));
        assert_eq!(std::fs::read_to_string(dir.join("transcript.md")).unwrap(), manual_markdown);
        assert_eq!(std::fs::read_to_string(dir.join("transcript.json")).unwrap(), manual_json);

        let saved = base.join("original.db");
        std::fs::rename(&db_path, &saved).unwrap();
        assert!(crate::run_transcribe(&args).is_err());
        assert!(!db_path.exists(), "bound resume must not create a missing archive");
        rt.block_on(async {
            let mut unrelated = db::Db::open(&db_path).await.unwrap();
            assert_ne!(unrelated.archive_id().await.unwrap(), archive_id);
            unrelated.close().await.unwrap();
        });
        assert!(crate::run_transcribe(&args).is_err(), "unrelated archive at the same path must be rejected");
        std::fs::rename(&db_path, base.join("unrelated.db")).unwrap();
        std::fs::rename(saved, &db_path).unwrap();
        crate::run_transcribe(&args).unwrap();
        rt.block_on(async {
            let mut archive = db::Db::open_existing(&db_path, &archive_id).await.unwrap();
            assert_eq!(archive.ingest_id(&key).await.unwrap(), Some(id));
            assert_eq!(archive.list_meetings().await.unwrap().len(), 1);
            assert_eq!(archive.load_segments(id).await.unwrap()[0].seg.text, segment().text);
            archive.close().await.unwrap();
        });
        std::fs::remove_dir_all(base).unwrap();
    }
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f3_resume_identity\",\"iterations\":5}}");
}

#[test]
fn crash_reserved_attempts_do_not_reset_the_budget() {
    for _ in 0..5 {
        let (base, session) = fixture(CaptureKind::Complete);
        let dir = session.dir.clone(); drop(session);
        let mut now = 1000;
        for expected in 1..=4 {
            let mut owner = recovery::Session::open(&dir).unwrap().unwrap();
            owner.reserve(now, false).unwrap();
            assert_eq!(owner.state.attempts, expected);
            assert!(owner.state.in_flight);
            drop(owner); // Simulate losing the owner without an outcome receipt.
            let mut restarted = recovery::Session::open(&dir).unwrap().unwrap();
            assert!(restarted.state.due(now), "an abandoned final reservation still needs classification");
            assert!(restarted.reserve(now, false).is_err());
            assert_eq!(restarted.state.attempts, expected);
            assert!(!restarted.state.in_flight);
            if expected == 4 {
                assert_eq!(restarted.state.phase, recovery::Phase::NeedsAttention);
                assert!(!restarted.state.due(now + 86_400));
                restarted.reserve(now, true).unwrap();
                assert_eq!(restarted.state.attempts, 1, "only explicit resume resets the budget");
            } else {
                assert!(restarted.state.next_retry > now);
                now = restarted.state.next_retry;
            }
            drop(restarted);
        }
        std::fs::remove_dir_all(base).unwrap();
    }
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f2_crash_budget\",\"iterations\":5}}");
}

#[test]
fn audit_f2_storage_retention() {
    for _ in 0..5 {
        let (base, mut session) = fixture(CaptureKind::Complete);
        let dir = session.dir.clone();
        session.protect_outputs().unwrap();
        std::fs::write(dir.join("transcript.md"), "# Export awaiting archive\n\nThe release checklist is ready.\n").unwrap();
        let before = std::fs::read(dir.join("mic.wav")).unwrap();
        let db_path = session.state.db_path.clone();
        drop(session);
        let report = crate::maintenance::prune_sessions(&base, 30, crate::now_epoch() + 100 * 86_400).unwrap();
        assert_eq!(report.removed, 0, "unarchived recording must not become prunable");
        assert_eq!(std::fs::read(dir.join("mic.wav")).unwrap(), before);
        assert!(dir.join("transcript.md").exists());
        assert!(!db_path.exists(), "retention proof must never create an archive");
        std::fs::remove_dir_all(base).unwrap();
    }
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f2_retention\",\"iterations\":5}}");
}

fn archived_fixture(external_exports: bool) -> (PathBuf, recovery::Session) {
    let (base, mut session) = fixture(CaptureKind::Complete);
    if external_exports { session.state.export_dir = base.join("exports"); }
    session.reserve(crate::now_epoch(), false).unwrap();
    let mut metadata = meta(&session); metadata.capture_status = Some("complete".into());
    let key = session.state.key.clone();
    let runtime = crate::new_runtime().unwrap();
    let (id, row, stored) = runtime.block_on(async {
        let mut archive = session.archive().await.unwrap();
        let id = archive.insert_keyed_meeting_with_voices(&metadata, &[segment()], &[], &[], &key).await.unwrap();
        let row = archive.get_meeting(id).await.unwrap().unwrap();
        let stored = archive.load_segments(id).await.unwrap(); archive.close().await.unwrap();
        (id, row, stored)
    });
    let (vocab, _) = crate::render::Vocab::compile(&[]);
    let rendered = crate::render::render(&stored, &crate::render::IdentityMap::from_db(&[]), &vocab);
    let (md, _) = crate::export::write_managed_exports(&session.state.export_dir, &key,
        crate::export::Publication::Initial, &row, &rendered).unwrap();
    session.committed(id).unwrap(); session.complete(md.parent().unwrap()).unwrap();
    (base, session)
}

#[test]
fn verified_initial_publications_are_prunable_and_locks_stay_stable() {
    for _ in 0..5 {
        let (base, session) = archived_fixture(false); let dir = session.dir.clone();
        let lock_path = session.state.lock_dir.join(format!("{}.lock", session.state.key));
        use std::os::unix::fs::MetadataExt;
        let inode = std::fs::metadata(&lock_path).unwrap().ino();
        let now = crate::now_epoch() + 100 * 86_400;
        assert_eq!(crate::maintenance::prune_sessions(&base, 30, now).unwrap().removed, 0, "active owner must exclude retention");
        drop(session);
        assert_eq!(crate::maintenance::prune_sessions(&base, 30, now).unwrap().removed, 1);
        assert!(!dir.exists()); assert_eq!(std::fs::metadata(lock_path).unwrap().ino(), inode);
        std::fs::remove_dir_all(base).unwrap();
    }
}

#[test]
fn clean_short_capture_pruning_needs_no_archive_and_preserves_extra_outputs() {
    for _ in 0..5 {
        for protect in [false, true] {
            let (base, mut session) = fixture(CaptureKind::Complete); let dir = session.dir.clone();
            session.finish_capture(1.0, 2.0).unwrap();
            if protect { std::fs::write(dir.join("transcript.md"), "# Operator short-call notes\n\nKeep this decision.\n").unwrap(); }
            drop(session);
            let report = crate::maintenance::prune_sessions(&base, 30, crate::now_epoch() + 100 * 86_400).unwrap();
            assert_eq!(report.removed, usize::from(!protect));
            assert_eq!(dir.exists(), protect);
            assert!(!base.join("archive.db").exists(), "short-capture proof must not create a database");
            std::fs::remove_dir_all(base).unwrap();
        }
    }
}

#[test]
fn retention_preserves_unknown_nested_and_skipped_session_material() {
    for _ in 0..5 {
        for nested in [".inputs", ".recovered", "mic.wav"] {
            let (base, session) = archived_fixture(true); let dir = session.dir.clone();
            if nested == "mic.wav" { std::fs::remove_file(dir.join(nested)).unwrap(); }
            std::fs::create_dir_all(dir.join(nested)).unwrap();
            let note = dir.join(nested).join("operator-notes.md");
            std::fs::write(&note, "Retain the revised release decision.\n").unwrap();
            drop(session);
            assert_eq!(crate::maintenance::prune_sessions(&base, 30, crate::now_epoch() + 100 * 86_400).unwrap().removed, 0);
            assert!(note.is_file()); std::fs::remove_dir_all(base).unwrap();
        }
        let (base, mut session) = fixture(CaptureKind::Complete); let dir = session.dir.clone();
        session.finish_capture(1.0, 2.0).unwrap();
        let note = dir.join("operator-notes.md"); std::fs::write(&note, "Keep the short-call decision.\n").unwrap();
        drop(session);
        assert_eq!(crate::maintenance::prune_sessions(&base, 30, crate::now_epoch() + 100 * 86_400).unwrap().removed, 0);
        assert!(note.is_file()); std::fs::remove_dir_all(base).unwrap();
    }
}

#[test]
fn lock_release_does_not_wait_for_an_unrelated_fork_to_exec() {
    use std::os::fd::AsRawFd;
    use std::os::unix::net::UnixStream;
    use std::io::Write;
    for _ in 0..5 {
        let (base, owner) = fixture(CaptureKind::Complete);
        let dir = owner.dir.clone();
        let (mut parent, child_stream) = UnixStream::pair().unwrap();
        let child_fd = child_stream.as_raw_fd();
        // SAFETY: after fork the child uses only read/_exit, no allocator, Rust Drop or locks.
        let child = unsafe { libc::fork() };
        assert!(child >= 0);
        if child == 0 {
            let mut byte = 0u8;
            // SAFETY: descriptor and one-byte stack buffer are valid; _exit skips inherited Drops.
            unsafe { libc::read(child_fd, std::ptr::from_mut(&mut byte).cast(), 1); libc::_exit(0); }
        }
        drop(child_stream);
        drop(owner);
        let reopened = recovery::Session::open(&dir);
        parent.write_all(b"x").unwrap();
        let mut status = 0;
        // SAFETY: wait only for the exact child this test created, no process-control signal.
        assert_eq!(unsafe { libc::waitpid(child, &mut status, 0) }, child);
        assert_eq!(status, 0);
        let reopened = reopened.unwrap();
        assert!(reopened.is_some(), "a fork-inherited descriptor delayed normal ownership release");
        drop(reopened);
        std::fs::remove_dir_all(base).unwrap();
    }
}

#[test]
fn audit_f3_independent_database_connections_converge() {
    use std::sync::{Arc, Barrier};
    for _ in 0..5 {
        let (base, session) = fixture(CaptureKind::Complete);
        let path = session.state.db_path.clone(); let meeting_meta = meta(&session);
        let rt = crate::new_runtime().unwrap();
        let archive_id = rt.block_on(async {
            let mut db = db::Db::open(&path).await.unwrap();
            let id = db.archive_id().await.unwrap(); db.close().await.unwrap(); id
        });
        let key = session.state.key.clone(); drop(session);
        let barrier = Arc::new(Barrier::new(2));
        let mut children = Vec::new();
        for _ in 0..2 {
            let (path, archive_id, meta, key, barrier) = (path.clone(), archive_id.clone(), meeting_meta.clone(), key.clone(), Arc::clone(&barrier));
            children.push(std::thread::spawn(move || crate::new_runtime().unwrap().block_on(async {
                let mut db = db::Db::open_existing(&path, &archive_id).await.unwrap();
                assert_eq!(db.ingest_id(&key).await.unwrap(), None);
                barrier.wait();
                let id = db.insert_keyed_meeting_with_voices(&meta, &[segment()], &[], &[], &key).await.unwrap();
                db.close().await.unwrap(); id
            })));
        }
        let ids: Vec<_> = children.into_iter().map(|child| child.join().unwrap()).collect();
        assert_eq!(ids[0], ids[1]);
        rt.block_on(async {
            let mut db = db::Db::open_existing(&path, &archive_id).await.unwrap();
            assert_eq!(db.list_meetings().await.unwrap().len(), 1);
            assert_eq!(db.load_segments(ids[0]).await.unwrap().len(), 1);
            assert_eq!(db.ingest_id(&key).await.unwrap(), Some(ids[0]));
            db.close().await.unwrap();
        });
        std::fs::remove_dir_all(base).unwrap();
    }
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f3_db_race\",\"iterations\":5}}");
}
