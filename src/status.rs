//! Shared daemon ↔ tray IPC over two files in `~/.meetscribe/`.
//!
//! `status.json` — the daemon writes it on every state change; the tray polls it.
//! `paused`      — a flag file; while it exists the daemon skips STARTING new recordings (a
//!                 recording already in progress finishes normally). The tray toggles it.
//!
//! File-based IPC is the whole point of the separate-process design: the daemon stays
//! single-threaded with NO GUI run loop competing with its Core Audio listeners, and the tray is a
//! plain reader that never touches audio/capture. `cat status.json` / `touch paused` also make the
//! daemon observable and pausable from a shell with no GUI at all.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const STATUS_FILE: &str = "status.json";
pub const PAUSE_FILE: &str = "paused";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonState {
    Idle,
    Recording,
    /// Capture has ended; whisper is still running. Distinct from `Recording` because transcription
    /// of a long meeting blocks the daemon for many minutes — without this the tray would keep
    /// showing "recording" long after the call ended.
    Transcribing,
    Paused,
}

/// What the daemon last published; what the tray renders.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Status {
    pub state: DaemonState,
    /// Friendly app label while recording (e.g. "Chrome"); `None` otherwise.
    pub app: Option<String>,
    /// Unix epoch when this state began.
    pub since_epoch: i64,
    /// Unix epoch of this write (lets a reader spot a stale file).
    pub updated_epoch: i64,
    /// Daemon pid — the tray checks it for liveness and to signal Quit.
    pub pid: u32,
}

#[must_use]
pub fn status_path(base: &Path) -> PathBuf {
    base.join(STATUS_FILE)
}

#[must_use]
pub fn pause_path(base: &Path) -> PathBuf {
    base.join(PAUSE_FILE)
}

/// True while the pause flag file exists.
#[must_use]
pub fn is_paused(base: &Path) -> bool {
    pause_path(base).exists()
}

impl Status {
    /// Write `status.json` atomically (temp + rename) so a reader never sees a half-written file.
    pub fn write(&self, base: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        let tmp = base.join(".status.json.tmp");
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, status_path(base))
    }

    /// Read `status.json`, or `None` if it is absent/unreadable/corrupt (the tray then shows a
    /// neutral "starting…"/"stopped" rather than crashing).
    #[must_use]
    pub fn read(base: &Path) -> Option<Status> {
        let s = std::fs::read_to_string(status_path(base)).ok()?;
        serde_json::from_str(&s).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_roundtrips_via_files() {
        let tmp = std::env::temp_dir().join(format!("meetscribe-status-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        assert_eq!(Status::read(&tmp), None); // absent → None
        let s = Status {
            state: DaemonState::Recording,
            app: Some("Chrome".to_string()),
            since_epoch: 1_700_000_000,
            updated_epoch: 1_700_000_005,
            pid: 4242,
        };
        s.write(&tmp).unwrap();
        assert_eq!(Status::read(&tmp), Some(s));
        // No temp file left behind after the atomic rename.
        assert!(!tmp.join(".status.json.tmp").exists());

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// The tray parses this file with serde; an unrecognised `state` makes `read` return `None`,
    /// which the tray renders as "daemon stopped". Pin the wire spelling so a rename can't silently
    /// turn a live transcribe into a phantom crash.
    #[test]
    fn transcribing_state_roundtrips_as_snake_case() {
        let s = Status {
            state: DaemonState::Transcribing,
            app: Some("Teams".to_string()),
            since_epoch: 1_700_000_000,
            updated_epoch: 1_700_000_000,
            pid: 4242,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains(r#""state":"transcribing""#), "unexpected wire form: {json}");
        assert_eq!(serde_json::from_str::<Status>(&json).unwrap(), s);
    }

    #[test]
    fn pause_flag_presence() {
        let tmp = std::env::temp_dir().join(format!("meetscribe-pause-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(!is_paused(&tmp));
        std::fs::write(pause_path(&tmp), b"").unwrap();
        assert!(is_paused(&tmp));
        std::fs::remove_file(pause_path(&tmp)).unwrap();
        assert!(!is_paused(&tmp));
        std::fs::remove_dir_all(&tmp).ok();
    }
}
