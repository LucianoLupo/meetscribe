//! Shared daemon ↔ tray IPC over two files in `~/.meetscribe/`.
//!
//! `status.json` — the daemon writes it on every state change; the tray polls it.
//! `paused`      — a flag file; while it exists the daemon skips STARTING new recordings (a
//!                 recording already in progress finishes normally). The tray toggles it.
//!
//! File-based IPC is the whole point of the separate-process design: the daemon has
//! NO GUI run loop competing with its Core Audio listeners, and the tray is a
//! plain reader that never touches audio/capture. `cat status.json` / `touch paused` also make the
//! daemon observable and pausable from a shell with no GUI at all.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const STATUS_FILE: &str = "status.json";
pub const PAUSE_FILE: &str = "paused";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonState {
    Idle,
    Recording,
    /// Whisper is working on a finished meeting and no call is being recorded (a recording outranks
    /// it — the daemon records the next call while the last one transcribes). `app` names the
    /// meeting being transcribed.
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

/// How often a transcription child re-reads `status.json` to follow the call state.
const CALL_YIELD_POLL: Duration = Duration::from_secs(2);

/// True while `status.json` says a call is being recorded. An absent or corrupt file reads as no
/// call: the worst case is one transcription running at normal priority, never a stalled one.
#[must_use]
pub fn call_is_live(base: &Path) -> bool {
    Status::read(base).is_some_and(|s| s.state == DaemonState::Recording)
}

/// For the daemon's transcription child: run at normal priority, and drop to background whenever a
/// call is being recorded so whisper never competes with the call app. Polls `status.json` on a
/// background thread for the life of the process.
///
/// The child does this itself because a parent's `setpriority` on a child can lower it but never
/// raise it again (measured 2026-10-02: a parent-raised child stayed at priority 4). The daemon's
/// agent runs at `ProcessType=Standard`; a `Background` agent would be no use here, since launchd's
/// background spawn policy cannot be lifted from inside either, and whisper runs 2.4× slower under
/// it (RTF 0.62 vs 0.25).
pub fn spawn_call_yield(base: PathBuf) {
    let spawned = std::thread::Builder::new()
        .name("call-yield".to_string())
        .spawn(move || {
            let mut background: Option<bool> = None;
            loop {
                let want = call_is_live(&base);
                if background != Some(want) {
                    set_own_background(want);
                    background = Some(want);
                }
                std::thread::sleep(CALL_YIELD_POLL);
            }
        });
    if let Err(e) = spawned {
        log::warn!("could not start the call-yield thread — transcribing at inherited priority: {e}");
    }
}

fn set_own_background(on: bool) {
    let prio = if on { libc::PRIO_DARWIN_BG } else { 0 };
    // SAFETY: a plain syscall on our own process (`who = 0`); no memory is passed.
    let rc = unsafe { libc::setpriority(libc::PRIO_DARWIN_PROCESS, 0, prio) };
    if rc == 0 {
        log::info!(
            "priority: {}",
            if on { "background (a call is being recorded)" } else { "normal (no call)" }
        );
    } else {
        log::warn!("could not change priority: {}", std::io::Error::last_os_error());
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
    fn only_a_recording_counts_as_a_live_call() {
        let tmp = std::env::temp_dir().join(format!("meetscribe-call-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(!call_is_live(&tmp)); // no file → no call
        for (state, live) in [
            (DaemonState::Recording, true),
            (DaemonState::Transcribing, false),
            (DaemonState::Idle, false),
            (DaemonState::Paused, false),
        ] {
            let s = Status { state, app: None, since_epoch: 1, updated_epoch: 1, pid: 1 };
            s.write(&tmp).unwrap();
            assert_eq!(call_is_live(&tmp), live, "{state:?}");
        }
        std::fs::write(status_path(&tmp), b"{not json").unwrap();
        assert!(!call_is_live(&tmp)); // corrupt → no call
        std::fs::remove_dir_all(&tmp).ok();
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
