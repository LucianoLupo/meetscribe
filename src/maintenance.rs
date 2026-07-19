//! Disk + log hygiene for the background daemon.
//!
//! Two bounded, config-driven jobs run at daemon startup and once per day:
//!   - **prune session dirs** older than `retention.sessions_days` (0 = keep forever — the safe
//!     default; we never delete recordings unless the user opts in). The meeting + transcript stay
//!     in the DB regardless, so `meetscribe export <id>` still works after a session dir is pruned.
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

/// mtime of `path` as unix epoch seconds (best-effort; 0 if unavailable → treated as very old).
fn mtime_epoch(path: &Path) -> i64 {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// True if something last modified at `mtime_epoch` is strictly older than `days` relative to
/// `now_epoch`. `days == 0` is never old (keep forever).
#[must_use]
pub fn is_older_than(mtime_epoch: i64, now_epoch: i64, days: u64) -> bool {
    if days == 0 {
        return false;
    }
    now_epoch - mtime_epoch > (days as i64) * SECS_PER_DAY
}

/// Delete session subdirs older than `days` (by mtime). `days == 0` is a no-op (keep forever).
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
        if !path.is_dir() {
            continue; // only prune session directories, never stray files
        }
        if is_older_than(mtime_epoch(&path), now_epoch, days) {
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
    }

    #[test]
    fn prune_removes_old_dirs_keeps_files_and_respects_zero() {
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

        // A "now" 100 days in the future makes the fresh dirs count as >30d old → all removed.
        let r = prune_sessions(&sessions, 30, now + 100 * SECS_PER_DAY).unwrap();
        assert_eq!(r, PruneReport { removed: 3, kept: 0 });
        assert!(!sessions.join("a").exists());
        assert!(sessions.join("stray.txt").exists()); // stray file never touched

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
