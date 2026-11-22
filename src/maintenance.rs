//! Disk + log hygiene for the background daemon.
//!
//! Three bounded, config-driven jobs run at daemon startup and once per day:
//!   - **prune session dirs** older than `retention.sessions_days` (0 = keep forever — the safe
//!     default; we never delete recordings unless the user opts in). The meeting + transcript stay
//!     in the DB regardless, so `meetscribe export <id>` still works after a session dir is pruned.
//!   - **prune live streams** older than `live.retain_days` (0 = keep). A stream is a meeting's
//!     text outside its session directory; see [`prune_live`] for what may be deleted and how a
//!     reader that starts the same stream at that instant is kept safe.
//!   - **rotate the daemon log** over `retention.log_max_mb` via COPYTRUNCATE (copy → `<log>.1`,
//!     then truncate the original in place). Rename is WRONG here: launchd holds the log fd open
//!     (StandardErrorPath), so a rename would leave it appending to the rotated file. Truncating the
//!     SAME inode means launchd's O_APPEND fd resumes writing at offset 0 — no sparse gap.
//!
//! Pure helpers (`is_older_than`; byte-threshold rotation) are unit-tested; `now_epoch` is injected
//! so pruning is testable on a fresh temp dir without mangling file mtimes.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

const SECS_PER_DAY: i64 = 86_400;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct PruneReport {
    pub removed: usize,
    pub kept: usize,
}

/// A file's modification time as unix epoch seconds; `None` when it cannot be read.
fn modified_epoch(meta: &fs::Metadata) -> Option<i64> {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
}

/// mtime of `path` as unix epoch seconds (best-effort; 0 if unavailable → treated as very old).
fn mtime_epoch(path: &Path) -> i64 {
    fs::metadata(path)
        .ok()
        .and_then(|m| modified_epoch(&m))
        .unwrap_or(0)
}

/// True if something last modified at `mtime_epoch` is strictly older than `days` relative to
/// `now_epoch`. `days == 0` is never old (keep forever), and neither is anything whose age
/// cannot be established: a `days` too large to count in seconds, or an age that overflows.
#[must_use]
pub fn is_older_than(mtime_epoch: i64, now_epoch: i64, days: u64) -> bool {
    if days == 0 {
        return false;
    }
    let Some(limit) = i64::try_from(days)
        .ok()
        .and_then(|days| days.checked_mul(SECS_PER_DAY))
    else {
        return false;
    };
    now_epoch
        .checked_sub(mtime_epoch)
        .is_some_and(|age| age > limit)
}

/// File the daemon drops in a session dir while its transcription is queued; removed after the
/// attempt. Its presence means "recorded, transcript still owed" (see `daemon`).
pub const PENDING_MARKER: &str = "pending";

/// Delete session subdirs older than `days` (by mtime). `days == 0` is a no-op (keep forever).
/// A verified archive plus unchanged exports is required; unknown/unfinished work is kept.
/// `now_epoch` is injected for testability. A per-dir error is logged, not fatal (best-effort).
pub fn prune_sessions(sessions_dir: &Path, days: u64, now_epoch: i64) -> Result<PruneReport> {
    let mut report = PruneReport::default();
    if days == 0 || !sessions_dir.exists() {
        return Ok(report);
    }
    let entries = fs::read_dir(sessions_dir)
        .with_context(|| format!("read sessions dir {}", sessions_dir.display()))?;
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !path.is_dir() || entry.file_name().to_str().is_some_and(|name| name.starts_with('.')) {
            continue; // The stable .locks registry is never a prunable session.
        }
        let session = match crate::recovery::Session::open(&path) {
            Ok(Some(session)) => session,
            _ => { report.kept += 1; continue; }
        };
        let eligible = if session.state.phase == crate::recovery::Phase::SkippedShort {
            session.prunable_short().unwrap_or(false)
        } else { session.proof(crate::recovery::ProofScope::Retention).unwrap_or(false) };
        let modified = mtime_epoch(&path);
        if !eligible || modified == 0 { report.kept += 1; continue; }
        // session owns the same stable lock through proof, age check and deletion.
        if is_older_than(modified, now_epoch, days) {
            match fs::remove_dir_all(&path) {
                Ok(()) => {
                    report.removed += 1;
                    log::info!("retention: removed old session {}", path.display());
                }
                Err(e) => log::warn!("retention: could not remove {} ({e})", path.display()),
            }
        } else {
            report.kept += 1;
        }
    }
    Ok(report)
}

/// The session key in a live stream's file name (`<32 hex digits>.jsonl`, as `live.rs` names
/// it); `None` for any other name.
fn stream_key(name: &std::ffi::OsStr) -> Option<&str> {
    let key = name.to_str()?.strip_suffix(".jsonl")?;
    (key.len() == 32 && key.bytes().all(|byte| byte.is_ascii_hexdigit())).then_some(key)
}

/// Whether a file named `<key>.jsonl` that the pruner holds locked may be deleted: a regular
/// file, last written more than `days` ago, whose first line says it is the live stream of
/// `key`. Everything that cannot be established keeps the file: a modification time that
/// cannot be read (`modified` is `None`), and a `days` too large to count in seconds.
/// A finished stream is not required: a reader that crashed leaves none, and its text would
/// otherwise stay for ever.
#[must_use]
fn stream_expired(regular: bool, modified: Option<i64>, first_line: Option<&str>, key: &str, now_epoch: i64, days: u64) -> bool {
    use crate::live::{StreamEvent, StreamLine};
    let ours = first_line.and_then(|line| serde_json::from_str::<StreamLine>(line).ok())
        .is_some_and(|line| matches!(line.event, StreamEvent::MeetingStarted { key: written, .. } if written == key));
    regular && modified.is_some_and(|modified| is_older_than(modified, now_epoch, days)) && ours
}

/// Longest first line read from a stream: a `meeting_started` event is a few hundred bytes.
const FIRST_LINE_LIMIT: u64 = 64 * 1024;

/// The complete first line of an open file, without its newline.
fn first_line(file: &fs::File) -> Option<String> {
    use std::io::{BufRead, Read};
    let mut line = String::new();
    std::io::BufReader::new(file.take(FIRST_LINE_LIMIT)).read_line(&mut line).ok()?;
    line.strip_suffix('\n').map(str::to_owned)
}

/// A point inside [`prune_stream`] at which a test may change the file system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PruneStep {
    /// The entry was listed and is a regular file; it is not opened yet.
    Listed,
    /// The file is locked and was judged old; the path has not been compared with it yet.
    Locked,
}

/// Delete one stream if it is old, in the order that makes it safe against a writer
/// (`live.rs`, `StreamWriter::create_with`) starting the same stream at this instant:
///
/// 1. the listed entry is a regular file;
/// 2. open it read-only without following a link, and take its lock without waiting (a writer
///    holds that lock for as long as it writes: its stream is kept);
/// 3. judge the LOCKED file: kind, age and first line. A writer reuses and truncates the same
///    file, so an age read before the lock could condemn a stream written a second ago;
/// 4. the path still names the file that is locked;
/// 5. unlink while holding the lock. A writer that opened the file before this finds, once it
///    has the lock, that its file is not at the path any more, and opens the stream again.
///
/// This is safe against a writer that takes the lock, which every reader of this tool does. It
/// does not exclude something that swaps the file at the path without the lock in the instant
/// between 4 and 5.
///
/// `at` is told when 1 and when 3 are done; tests rewrite or replace the stream there.
fn prune_stream(path: &Path, key: &str, days: u64, now_epoch: i64, at: &mut impl FnMut(PruneStep, &Path)) -> bool {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    if !fs::symlink_metadata(path).is_ok_and(|listed| listed.file_type().is_file()) { return false; }
    at(PruneStep::Listed, path);
    // O_NONBLOCK: should the entry have become a FIFO since step 1, the open must not hang the daemon.
    let Ok(file) = fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(path) else { return false; };
    // SAFETY: `file` owns the valid descriptor; flock does not access Rust memory.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 { return false; }
    let Ok(held) = file.metadata() else { return false; };
    if !stream_expired(held.file_type().is_file(), modified_epoch(&held), first_line(&file).as_deref(), key, now_epoch, days) { return false; }
    at(PruneStep::Locked, path);
    if !fs::symlink_metadata(path).is_ok_and(|at| at.dev() == held.dev() && at.ino() == held.ino()) { return false; }
    // `file` is dropped, and the lock released, only after the unlink.
    fs::remove_file(path).is_ok()
}

/// Delete the live streams in `live_dir` last written more than `days` ago (`days == 0` is a
/// no-op: keep). Only an entry directly in the directory, named like a stream and provably one
/// (see [`stream_expired`]), is ever removed; a stream a reader is writing is kept. The
/// recording, the stored transcript and the database are not touched. `now_epoch` is injected
/// for testability.
pub fn prune_live(live_dir: &Path, days: u64, now_epoch: i64) -> Result<PruneReport> {
    prune_live_with(live_dir, days, now_epoch, |_, _| {})
}

/// [`prune_live`], with `at` called for each stream at every [`PruneStep`].
pub(crate) fn prune_live_with(live_dir: &Path, days: u64, now_epoch: i64, mut at: impl FnMut(PruneStep, &Path)) -> Result<PruneReport> {
    let mut report = PruneReport::default();
    if days == 0 || !live_dir.exists() {
        return Ok(report);
    }
    let entries = fs::read_dir(live_dir).with_context(|| format!("read live dir {}", live_dir.display()))?;
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let name = entry.file_name();
        let Some(key) = stream_key(&name) else { continue };
        if prune_stream(&entry.path(), key, days, now_epoch, &mut at) { report.removed += 1; } else { report.kept += 1; }
    }
    Ok(report)
}

/// `<log>.1` — the single rotation backup (e.g. `meetscribe.err.log` → `meetscribe.err.log.1`).
fn backup_path(log_path: &Path) -> PathBuf {
    let mut s = log_path.as_os_str().to_os_string();
    s.push(".1");
    PathBuf::from(s)
}

/// Rotate `log_path` if it exceeds `max_bytes`: copy → `<log>.1`, then truncate the original in
/// place (COPYTRUNCATE — safe with launchd's held O_APPEND fd). `max_bytes == 0` disables rotation.
/// Returns whether it rotated.
pub fn rotate_log(log_path: &Path, max_bytes: u64) -> Result<bool> {
    if max_bytes == 0 || !log_path.exists() {
        return Ok(false);
    }
    let size = fs::metadata(log_path)
        .with_context(|| format!("stat log {}", log_path.display()))?
        .len();
    if size <= max_bytes {
        return Ok(false);
    }
    let backup = backup_path(log_path);
    fs::copy(log_path, &backup)
        .with_context(|| format!("copy {} → {}", log_path.display(), backup.display()))?;
    // Truncate the SAME inode (do NOT rename — launchd holds this fd open and would keep writing to
    // the renamed file). O_TRUNC on the existing path resets it to 0 without changing the inode.
    fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(log_path)
        .with_context(|| format!("truncate {}", log_path.display()))?;
    log::info!(
        "retention: rotated {} ({size} bytes) → {}",
        log_path.display(),
        backup.display()
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_older_than_semantics() {
        let now = 1_000_000_000i64;
        // days=0 → never old.
        assert!(!is_older_than(0, now, 0));
        // Fresh (age 1 day) with a 30-day threshold → not old.
        assert!(!is_older_than(now - SECS_PER_DAY, now, 30));
        // Age 31 days with a 30-day threshold → old.
        assert!(is_older_than(now - 31 * SECS_PER_DAY, now, 30));
        // Exactly at the threshold is NOT old (strict >).
        assert!(!is_older_than(now - 30 * SECS_PER_DAY, now, 30));
        // An age that cannot be established is not old: too many days to count in seconds…
        for days in [u64::MAX, i64::MAX as u64, i64::MAX as u64 / SECS_PER_DAY as u64 + 1] {
            assert!(!is_older_than(0, now, days), "{days} days");
        }
        // …or a modification time so far back that the age overflows.
        assert!(!is_older_than(i64::MIN, now, 30));
    }

    #[test]
    fn prune_keeps_unproven_old_dirs_and_respects_zero() {
        let tmp = std::env::temp_dir().join(format!("meetscribe-prune-{}", std::process::id()));
        let sessions = tmp.join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        for name in ["a", "b", "c"] {
            fs::create_dir_all(sessions.join(name)).unwrap();
        }
        fs::write(sessions.join("stray.txt"), b"not a dir").unwrap();

        let now = mtime_epoch(&sessions.join("a"));

        // days=0 → no-op even though a future "now" would make them old.
        let r = prune_sessions(&sessions, 0, now + 100 * SECS_PER_DAY).unwrap();
        assert_eq!(r, PruneReport { removed: 0, kept: 0 });
        assert!(sessions.join("a").exists());

        // Present "now" → fresh dirs kept (removed=0, kept=3; the stray file is ignored).
        let r = prune_sessions(&sessions, 30, now).unwrap();
        assert_eq!(r, PruneReport { removed: 0, kept: 3 });
        assert!(sessions.join("stray.txt").exists());

        // Age alone is not archive proof: marker-less legacy directories remain protected.
        let r = prune_sessions(&sessions, 30, now + 100 * SECS_PER_DAY).unwrap();
        assert_eq!(r, PruneReport { removed: 0, kept: 3 });
        assert!(sessions.join("a").exists());
        assert!(sessions.join("stray.txt").exists()); // stray file never touched

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn prune_never_removes_a_session_still_waiting_for_its_transcript() {
        let tmp = std::env::temp_dir().join(format!("meetscribe-prune-pending-{}", std::process::id()));
        let sessions = tmp.join("sessions");
        fs::create_dir_all(sessions.join("owed")).unwrap();
        fs::create_dir_all(sessions.join("done")).unwrap();
        fs::write(sessions.join("owed").join(PENDING_MARKER), "Chrome\nChrome — 2026-10-01\n").unwrap();

        let later = mtime_epoch(&sessions.join("owed")) + 100 * SECS_PER_DAY;
        let r = prune_sessions(&sessions, 30, later).unwrap();
        assert_eq!(r, PruneReport { removed: 0, kept: 2 });
        assert!(sessions.join("owed").exists());
        assert!(sessions.join("done").exists(), "absence of pending is not archive proof");

        fs::remove_dir_all(&tmp).ok();
    }

    const KEY: &str = "a6e79ba0314c4cc2b647f4d25b327011";

    fn started(key: &str) -> String {
        serde_json::json!({"v": 1, "seq": 0, "type": "meeting_started", "key": key, "app": "Chrome",
            "started_at": 1_794_443_305, "model": "ggml-fixture", "lang": "es"}).to_string()
    }

    /// The decision on a locked file, apart from the file system: what cannot be established
    /// keeps the stream.
    #[test]
    fn a_stream_is_expired_only_when_everything_about_it_is_established() {
        let now = 1_800_000_000i64;
        let old = Some(now - 8 * SECS_PER_DAY);
        let line = started(KEY);
        assert!(stream_expired(true, old, Some(&line), KEY, now, 7));
        // Exactly at the limit is not old, and neither is anything younger.
        assert!(!stream_expired(true, Some(now - 7 * SECS_PER_DAY), Some(&line), KEY, now, 7));
        assert!(!stream_expired(true, Some(now), Some(&line), KEY, now, 7));
        assert!(!stream_expired(true, old, Some(&line), KEY, now, 0), "0 days keeps everything");
        assert!(!stream_expired(false, old, Some(&line), KEY, now, 7), "not a regular file");
        // A time that could not be read is not "very old".
        assert!(!stream_expired(true, None, Some(&line), KEY, now, 7));
        // A number of days that cannot be counted in seconds keeps every file, whatever its age.
        for days in [u64::MAX, i64::MAX as u64, i64::MAX as u64 / SECS_PER_DAY as u64 + 1] {
            assert!(!stream_expired(true, Some(0), Some(&line), KEY, now, days), "{days} days");
            assert!(!stream_expired(true, Some(i64::MIN), Some(&line), KEY, now, days), "{days} days");
        }
        assert!(!stream_expired(true, Some(i64::MIN), Some(&line), KEY, now, 7), "an age that cannot be computed");
        // Not provably a stream of this key.
        let other = started("b6e79ba0314c4cc2b647f4d25b327011");
        let heartbeat = serde_json::json!({"v": 1, "seq": 0, "type": "heartbeat", "audio_secs": 2.0, "decoded_secs": 0.0}).to_string();
        for first in [None, Some(""), Some("not json"), Some(other.as_str()), Some(heartbeat.as_str())] {
            assert!(!stream_expired(true, old, first, KEY, now, 7), "{first:?}");
        }
    }

    #[test]
    fn prune_live_removes_only_old_streams_of_this_tool() {
        use std::os::unix::fs::symlink;
        let tmp = std::env::temp_dir().join(format!("meetscribe-prune-live-{}", crate::recovery::new_identity()));
        let live = tmp.join("live");
        fs::create_dir_all(&live).unwrap();
        // The fresh stream below is as old as this test, so "now" is the real one.
        let now = crate::now_epoch();
        let aged = |path: &Path, days: i64| {
            let modified = UNIX_EPOCH + std::time::Duration::from_secs((now - days * SECS_PER_DAY) as u64);
            fs::File::options().write(true).open(path).unwrap().set_modified(modified).unwrap();
        };
        let keys = ["a0000000000000000000000000000001", "a0000000000000000000000000000002", "a0000000000000000000000000000003",
            "a0000000000000000000000000000004", "a0000000000000000000000000000005", "a0000000000000000000000000000006"];
        let stream = |key: &str| live.join(format!("{key}.jsonl"));
        // Old and finished; old and never finished (its reader crashed): both go.
        let ended = serde_json::json!({"v": 1, "seq": 1, "type": "meeting_ended", "duration_secs": 3.0, "reason": "ended", "segments": 0, "short_windows": 0});
        fs::write(stream(keys[0]), format!("{}\n{ended}\n", started(keys[0]))).unwrap();
        fs::write(stream(keys[1]), format!("{}\n", started(keys[1]))).unwrap();
        // Fresh: kept.
        fs::write(stream(keys[2]), format!("{}\n", started(keys[2]))).unwrap();
        // Old, named like a stream, but its first line names another meeting, or is no event.
        fs::write(stream(keys[3]), format!("{}\n", started(keys[0]))).unwrap();
        fs::write(stream(keys[4]), "my own notes\n").unwrap();
        for key in [keys[0], keys[1], keys[3], keys[4]] { aged(&stream(key), 30); }
        // A link named like a stream, to an old real stream kept elsewhere.
        let elsewhere = tmp.join("elsewhere.jsonl");
        fs::write(&elsewhere, format!("{}\n", started(keys[5]))).unwrap();
        aged(&elsewhere, 30);
        symlink(&elsewhere, stream(keys[5])).unwrap();
        // Other old things in the directory: a wrong name, a directory named like a stream, a note.
        fs::write(live.join("notes.jsonl"), format!("{}\n", started(keys[0]))).unwrap();
        aged(&live.join("notes.jsonl"), 30);
        let directory = live.join("b0000000000000000000000000000001.jsonl");
        fs::create_dir(&directory).unwrap();
        fs::write(live.join("README.txt"), "kept").unwrap();
        aged(&live.join("README.txt"), 30);
        let names = || { let mut names: Vec<String> = fs::read_dir(&live).unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned()).collect(); names.sort(); names };
        let everything = names();

        // 0 days keeps everything, however old.
        assert_eq!(prune_live(&live, 0, now + 1000 * SECS_PER_DAY).unwrap(), PruneReport { removed: 0, kept: 0 });
        assert_eq!(names(), everything);
        // A directory that does not exist is nothing to do, not an error.
        assert_eq!(prune_live(&tmp.join("absent"), 7, now).unwrap(), PruneReport::default());

        assert_eq!(prune_live(&live, 7, now).unwrap(), PruneReport { removed: 2, kept: 5 });
        let left: Vec<String> = everything.iter().filter(|name| !name.starts_with(keys[0]) && !name.starts_with(keys[1])).cloned().collect();
        assert_eq!(names(), left);
        assert!(elsewhere.exists(), "the file a link pointed at was removed");
        assert!(fs::symlink_metadata(stream(keys[5])).unwrap().file_type().is_symlink());
        // Again: nothing more to remove.
        assert_eq!(prune_live(&live, 7, now).unwrap(), PruneReport { removed: 0, kept: 5 });
        // The fresh one goes once it is old too.
        aged(&stream(keys[2]), 30);
        assert_eq!(prune_live(&live, 7, now).unwrap(), PruneReport { removed: 1, kept: 4 });
        assert!(!stream(keys[2]).exists());

        fs::remove_dir_all(&tmp).ok();
    }

    /// The path can come to name another file while the pruner holds the one it judged: a file
    /// moved into place by something that takes no lock. What is at the path is then left alone.
    #[test]
    fn prune_live_leaves_a_file_that_replaced_the_one_it_locked() {
        let tmp = std::env::temp_dir().join(format!("meetscribe-prune-replaced-{}", crate::recovery::new_identity()));
        let live = tmp.join("live");
        fs::create_dir_all(&live).unwrap();
        let now = crate::now_epoch();
        let path = live.join(format!("{KEY}.jsonl"));
        fs::write(&path, format!("{}\n", started(KEY))).unwrap();
        fs::File::options().write(true).open(&path).unwrap()
            .set_modified(UNIX_EPOCH + std::time::Duration::from_secs((now - 30 * SECS_PER_DAY) as u64)).unwrap();
        let newcomer = tmp.join("newcomer");
        let report = prune_live_with(&live, 7, now, |step, path| if step == PruneStep::Locked {
            fs::write(&newcomer, "moved into place\n").unwrap();
            fs::rename(&newcomer, path).unwrap();
        }).unwrap();
        assert_eq!(report, PruneReport { removed: 0, kept: 1 });
        assert_eq!(fs::read_to_string(&path).unwrap(), "moved into place\n", "the pruner removed a file it had not judged");
        fs::remove_dir_all(&tmp).ok();
    }

    /// launchd holds the daemon's log open for as long as the daemon runs, in append mode. After
    /// a rotation the holder must go on writing at the new end of the file, not at its old offset.
    #[test]
    fn a_log_held_open_in_append_mode_is_written_at_the_new_end_after_rotation() {
        use std::io::Write;
        let tmp = std::env::temp_dir().join(format!("meetscribe-rotate-append-{}", crate::recovery::new_identity()));
        fs::create_dir_all(&tmp).unwrap();
        let log = tmp.join("meetscribe.err.log");
        let mut held = fs::OpenOptions::new().create(true).append(true).open(&log).unwrap();
        held.write_all(b"0123456789ABCDEF").unwrap();
        assert!(rotate_log(&log, 8).unwrap());
        held.write_all(b"next").unwrap();
        assert_eq!(fs::read(&log).unwrap(), b"next", "the holder wrote past a hole left by the rotation");
        assert_eq!(fs::read(backup_path(&log)).unwrap(), b"0123456789ABCDEF");
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn rotate_copytruncates_over_threshold_only() {
        let tmp = std::env::temp_dir().join(format!("meetscribe-rotate-{}", std::process::id()));
        fs::create_dir_all(&tmp).unwrap();
        let log = tmp.join("meetscribe.err.log");
        let content = b"0123456789ABCDEF"; // 16 bytes
        fs::write(&log, content).unwrap();

        // Under threshold → no rotation.
        assert!(!rotate_log(&log, 100).unwrap());
        assert_eq!(fs::read(&log).unwrap(), content);

        // max_bytes=0 → disabled.
        assert!(!rotate_log(&log, 0).unwrap());

        // Over threshold → rotates: backup has the old content, original truncated to 0.
        assert!(rotate_log(&log, 8).unwrap());
        let backup = backup_path(&log);
        assert_eq!(fs::read(&backup).unwrap(), content);
        assert_eq!(fs::metadata(&log).unwrap().len(), 0);

        // Now empty → no further rotation.
        assert!(!rotate_log(&log, 8).unwrap());

        fs::remove_dir_all(&tmp).ok();
    }
}
