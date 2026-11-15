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
        capture_status: Some("partial".into()), origin: db::TranscriptOrigin::Batch,
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

/// A finished, published session whose archive is still at schema v3: what every session stored
/// before this binary looks like. Built through `open_existing`, which never migrates.
fn stored_in_a_v3_archive() -> (PathBuf, recovery::Session) {
    let (base, mut session) = fixture(CaptureKind::Complete);
    let runtime = crate::new_runtime().unwrap();
    let identity = runtime.block_on(db::Db::create_v3_archive(&session.state.db_path)).unwrap();
    session.state.archive_id = Some(identity.clone());
    session.reserve(crate::now_epoch(), false).unwrap();
    let mut metadata = meta(&session); metadata.capture_status = Some("complete".into());
    let key = session.state.key.clone();
    let (id, row, stored) = runtime.block_on(async {
        let mut archive = db::Db::open_existing(&session.state.db_path, &identity).await.unwrap();
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

fn schema_version(state: &recovery::State) -> i32 {
    crate::new_runtime().unwrap().block_on(async {
        let mut archive = db::Db::open_existing(&state.db_path, state.archive_id.as_deref().unwrap()).await.unwrap();
        let version = archive.schema_version().await.unwrap(); archive.close().await.unwrap(); version
    })
}

/// Retention and the pending-marker reconciliation prove a session with this binary before any
/// command has migrated its archive. The proof must hold on the v3 file and must not migrate it.
#[test]
fn a_session_in_a_v3_archive_is_proven_without_migrating_it() {
    for _ in 0..5 {
        let (base, session) = stored_in_a_v3_archive();
        assert!(session.proof(recovery::ProofScope::Archive).unwrap());
        assert!(session.proof(recovery::ProofScope::Retention).unwrap());
        assert_eq!(schema_version(&session.state), 3, "a read-only proof migrated the archive");
        let dir = session.dir.clone(); let state = session.state.clone(); drop(session);
        assert_eq!(crate::maintenance::prune_sessions(&base, 30, crate::now_epoch() + 100 * 86_400).unwrap().removed, 1);
        assert!(!dir.exists());
        assert_eq!(schema_version(&state), 3);
        std::fs::remove_dir_all(base).unwrap();
    }
}

/// The attempt that next touches a bound v3 archive migrates it: that archive, the one the
/// session is bound to, wherever it is. A second session bound to another v3 file is untouched.
#[test]
fn resuming_a_session_migrates_its_own_bound_archive_and_no_other() {
    for _ in 0..5 {
        let (base, session) = stored_in_a_v3_archive();
        let (other_base, other) = stored_in_a_v3_archive();
        assert_ne!(session.state.db_path, other.state.db_path);
        let dir = session.dir.clone(); let id = session.state.meeting_id; drop(session);
        crate::run_transcribe(&resume_args(&dir)).unwrap(); // Missing model: the stored meeting is rendered, nothing is decoded.
        let state = recovery::State::read(&dir).unwrap();
        assert_eq!((state.phase, state.meeting_id), (recovery::Phase::Complete, id));
        assert_eq!(schema_version(&state), 4, "the bound archive was left at v3");
        assert_eq!(schema_version(&other.state), 3, "an archive no attempt touched was migrated");
        crate::new_runtime().unwrap().block_on(async {
            let mut archive = db::Db::open_existing(&state.db_path, state.archive_id.as_deref().unwrap()).await.unwrap();
            let rows = archive.list_meetings().await.unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].origin, db::TranscriptOrigin::Batch, "a meeting stored before the column existed is a batch transcript");
            archive.close().await.unwrap();
        });
        // Migrated or not, both sessions still prove their publication.
        assert!(recovery::Session::open(&dir).unwrap().unwrap().proof(recovery::ProofScope::Archive).unwrap());
        assert!(other.proof(recovery::ProofScope::Retention).unwrap());
        // The second session's turn: its archive, a different file, is the one migrated now.
        let other_dir = other.dir.clone(); drop(other);
        crate::run_transcribe(&resume_args(&other_dir)).unwrap();
        assert_eq!(schema_version(&recovery::State::read(&other_dir).unwrap()), 4);
        std::fs::remove_dir_all(base).unwrap(); std::fs::remove_dir_all(other_base).unwrap();
    }
}

/// An archive whose identity is verified but whose upgrade fails is not a foreign archive: the
/// attempt fails and is retried, the session does not go to `needs_attention("archive_identity")`.
#[test]
fn a_failed_schema_upgrade_is_a_failed_attempt_not_an_identity_failure() {
    use sqlx::Connection;
    for _ in 0..5 {
        let (base, session) = stored_in_a_v3_archive();
        let dir = session.dir.clone(); let db_path = session.state.db_path.clone(); drop(session);
        // The v4 rung cannot apply: its column is already there under a v3 version number.
        crate::new_runtime().unwrap().block_on(async {
            let mut connection = sqlx::SqliteConnection::connect_with(&sqlx::sqlite::SqliteConnectOptions::new()
                .filename(&db_path).create_if_missing(false)).await.unwrap();
            sqlx::query("ALTER TABLE meetings ADD COLUMN transcript_origin TEXT NULL").execute(&mut connection).await.unwrap();
            connection.close().await.unwrap();
        });
        let error = crate::run_transcribe(&resume_args(&dir)).expect_err("the upgrade cannot succeed");
        assert_ne!(crate::diagnostics::error_class(&error), "archive_identity", "unexpected class for: {error:#}");
        let state = recovery::State::read(&dir).unwrap();
        assert_eq!(state.phase, recovery::Phase::Archived, "the session must stay eligible for a retry");
        assert_eq!((state.attempts, state.in_flight, state.error_class.as_deref()), (1, false, Some("processing")));
        assert!(state.next_retry > 0);
        assert_eq!(schema_version(&state), 3, "a failed rung must leave the version where it was");
        std::fs::remove_dir_all(base).unwrap();
    }
}

/// A session state as the build before `finalize` wrote it (f138d42), byte for byte. Written out by
/// hand in that build's field order and checked against it, not regenerated from this code.
const FROZEN_STATE: &str = r#"{"version":1,"key":"0123456789abcdef0123456789abcdef","started_at":1762880000,"app":"com.google.Chrome","title":"Release readiness sync","db_path":"/synthetic/archive.db","export_dir":"/synthetic/recording","lock_dir":"/synthetic/.locks","archive_id":"fedcba9876543210fedcba9876543210","phase":"complete","capture_status":"complete","rolls":[{"index":0,"sample_rate":16000,"gap_seconds":0.0,"mic_frames":48000,"system_frames":48000}],"duration_seconds":3.0,"attempts":1,"in_flight":false,"next_retry":0,"error_class":null,"meeting_id":7,"publication":"/synthetic/recording","unarchived_manual_output":false}"#;

#[test]
fn a_state_without_a_publication_record_is_written_as_it_always_was() {
    let state: recovery::State = serde_json::from_str(FROZEN_STATE).unwrap();
    assert_eq!(serde_json::to_string(&state).unwrap(), FROZEN_STATE);
    assert_eq!(state.provisional(), None);
}

/// The record is read by one piece of code only. Whatever else is stored under its key, every
/// command that reads the session still can, and the value counts as no record.
#[test]
fn a_publication_record_nobody_can_read_is_no_record_and_breaks_nothing() {
    let hash = "ab".repeat(32);
    let good = serde_json::json!({"dir": "/synthetic/recording", "markdown": [hash], "json": [hash]});
    let unreadable = [
        serde_json::json!(5), serde_json::json!("yes"), serde_json::json!({"dir": "/synthetic/recording"}),
        serde_json::json!({"dir": "relative/recording", "markdown": [hash], "json": [hash]}),
        serde_json::json!({"dir": "/synthetic/recording", "markdown": ["not a hash"], "json": [hash]}),
        serde_json::json!({"dir": "/synthetic/recording", "markdown": [hash.to_uppercase()], "json": [hash]}),
        serde_json::json!({"dir": "/synthetic/recording", "markdown": vec![hash.clone(); recovery::PROVISIONAL_HASHES + 1], "json": [hash]}),
    ];
    let base = std::env::temp_dir().join(format!("meetscribe-record-{}", recovery::new_identity()));
    recovery::private_dir(&base).unwrap();
    let with = |value: &serde_json::Value| -> recovery::State {
        let mut raw: serde_json::Value = serde_json::from_str(FROZEN_STATE).unwrap();
        raw["provisional"] = value.clone();
        std::fs::write(base.join(recovery::STATE_FILE), format!("{raw}\n")).unwrap();
        recovery::State::read(&base).expect("the state stays readable whatever the record holds")
    };
    let record = with(&good).provisional().expect("a well-formed record is read");
    assert_eq!((record.dir.as_path(), record.markdown.len(), record.json.len()), (Path::new("/synthetic/recording"), 1, 1));
    for value in &unreadable {
        let state = with(value);
        assert_eq!(state.provisional(), None, "{value} was read as a record");
        assert_eq!(state.provisional.as_ref(), Some(value), "an unreadable value is carried, not dropped, by a save");
    }
    assert_eq!(with(&serde_json::Value::Null).provisional, None);
    std::fs::remove_dir_all(base).unwrap();
}

#[test]
fn a_publication_record_only_grows_and_stops_at_its_limit() {
    let hash = |n: usize| format!("{n:064x}");
    let mut record = recovery::Provisional::new(Path::new("/synthetic/recording"), Some(hash(0)), Some(hash(0)));
    for n in 1..recovery::PROVISIONAL_HASHES { assert!(record.append(&hash(n), &hash(0)), "hash {n} fits"); }
    assert_eq!((record.markdown.len(), record.json.len()), (recovery::PROVISIONAL_HASHES, 1), "a hash already listed is not added twice");
    assert!(record.append(&hash(3), &hash(0)), "a known pair needs no room");
    let full = record.clone();
    assert!(!record.append(&hash(99), &hash(1)), "a seventeenth hash is refused");
    assert_eq!(record, full, "a refused append changes nothing, in either list");
    assert_eq!(record.markdown[0], hash(0), "the first hash is still there");
}

/// On a schema-v3 file every row reads as `batch` only because nothing marks it. That must never
/// be taken for "a finalize committed here": `transcribe --resume` asks before it reserves, and
/// `finalize` itself must leave such a database exactly as it found it.
#[test]
fn an_archive_that_marks_nothing_is_never_taken_for_an_unfinished_finalize() {
    let (base, mut session) = stored_in_a_v3_archive();
    let id = session.state.meeting_id.unwrap();
    assert_eq!(crate::finalize::unfinished(&session.state), None, "without a record nothing is even asked");
    let hash = "ab".repeat(32);
    session.published_provisional(&recovery::Provisional::new(&session.dir, Some(hash.clone()), Some(hash))).unwrap();
    assert_eq!(crate::finalize::unfinished(&session.state), None, "a v3 row is unmarked, not replaced");
    assert_eq!(schema_version(&session.state), 3);

    // `finalize <id>` on that database: the meeting is a batch transcript, so there is nothing
    // to do, and the file is still v3 afterwards. The lookup never migrates.
    let (database, state_file) = (session.state.db_path.clone(), session.dir.join(recovery::STATE_FILE));
    let dir = session.dir.clone();
    drop(session);
    let before = (std::fs::read(&database).unwrap(), std::fs::read(&state_file).unwrap());
    let argv: Vec<String> = [id.to_string(), "--db".into(), database.display().to_string(), "--base".into(), base.join("control").display().to_string()].into();
    crate::finalize::run(&argv).expect("a batch meeting is already final");
    assert_eq!((std::fs::read(&database).unwrap(), std::fs::read(&state_file).unwrap()), before);
    let session = recovery::Session::open(&dir).unwrap().unwrap();
    assert_eq!(schema_version(&session.state), 3, "finalize migrated a database it had nothing to do in");
    drop(session);
    std::fs::remove_dir_all(base).unwrap();
}
