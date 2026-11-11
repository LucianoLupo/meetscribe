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
    /// Legacy wire field. Never a signaling authority.
    pub pid: u32,
    #[serde(default)]
    pub instance: Option<Instance>,
    #[serde(default)]
    pub pending: u64,
    #[serde(default)]
    pub attention: u64,
}

/// One daemon lifetime, not merely a recyclable process ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Instance {
    pub pid: u32,
    pub start_seconds: u64,
    pub start_micros: u64,
    pub nonce: String,
}

impl Instance {
    pub fn current() -> Option<Self> {
        let pid = std::process::id();
        let (start_seconds, start_micros) = process_start(pid)?;
        Some(Self { pid, start_seconds, start_micros, nonce: crate::recovery::new_identity() })
    }
    fn valid(&self) -> bool {
        (2..=i32::MAX as u32).contains(&self.pid) && self.start_seconds > 0 && self.start_micros < 1_000_000
            && self.nonce.len() == 32 && self.nonce.bytes().all(|byte| byte.is_ascii_hexdigit())
    }
    pub fn live(&self, read: impl Fn(u32) -> Option<(u64, u64)>) -> bool {
        self.valid() && read(self.pid) == Some((self.start_seconds, self.start_micros))
    }
}

/// A failed/short kernel read is unknown, never evidence that a PID belongs to this daemon.
#[cfg(target_os = "macos")]
pub fn process_start(pid: u32) -> Option<(u64, u64)> {
    if !(2..=i32::MAX as u32).contains(&pid) { return None; }
    // SAFETY: zero is valid for this C integer/array struct; the syscall receives its exact size.
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of_val(&info) as i32;
    // SAFETY: info is writable for size bytes, and pid was range-checked before the signed cast.
    let got = unsafe { libc::proc_pidinfo(pid as i32, libc::PROC_PIDTBSDINFO, 0,
        std::ptr::from_mut(&mut info).cast(), size) };
    if got != size || info.pbi_pid != pid || info.pbi_start_tvsec == 0 || info.pbi_start_tvusec >= 1_000_000 { return None; }
    Some((info.pbi_start_tvsec, info.pbi_start_tvusec))
}
#[cfg(not(target_os = "macos"))]
pub fn process_start(_pid: u32) -> Option<(u64, u64)> { None }

pub fn write_stop(path: &Path, selected: &Instance) -> std::io::Result<()> {
    let parent = path.parent().ok_or_else(|| std::io::Error::other("control target has no parent"))?;
    crate::recovery::private_dir(parent).map_err(std::io::Error::other)?;
    crate::recovery::atomic_json(path, selected).map_err(std::io::Error::other)
}

/// Revalidate the displayed selection; only ever address that selection. No signal/launchctl fallback.
pub fn request_stop(
    base: &Path, selected: Option<&Instance>, read: impl Fn(u32) -> Option<(u64, u64)>,
    publish: impl FnOnce(&Path, &Instance) -> std::io::Result<()>,
) -> std::io::Result<bool> {
    let Some(selected) = selected else { return Ok(false); };
    let Some(current) = Status::read(base) else { return Ok(false); };
    if current.pid != selected.pid || current.instance.as_ref() != Some(selected) || !selected.live(read) { return Ok(false); }
    publish(&base.join("control").join(format!("stop-{}.json", selected.nonce)), selected)?;
    log::info!(target: "meetscribe::event", "{}", serde_json::json!({"component":"control", "event":"stop_requested", "instance_id":selected.nonce, "outcome":"ok"}));
    Ok(true)
}

pub fn stop_requested(base: &Path, own: &Instance) -> bool {
    if !own.valid() { return false; }
    let path = base.join("control").join(format!("stop-{}.json", own.nonce));
    std::fs::read(path).ok().and_then(|bytes| serde_json::from_slice::<Instance>(&bytes).ok()).as_ref() == Some(own)
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
        crate::recovery::atomic_json(&status_path(base), self).map_err(std::io::Error::other)
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
            instance: None, pending: 0, attention: 0,
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
            instance: None, pending: 0, attention: 0,
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
            let s = Status { state, app: None, since_epoch: 1, updated_epoch: 1, pid: 1, instance: None, pending: 0, attention: 0 };
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
