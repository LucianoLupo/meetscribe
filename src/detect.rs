//! Meeting auto-detection: which allowlisted app (if any) currently holds the microphone.
//!
//! A meeting is "active" when a process whose bundle-id matches the allowlist has an active
//! INPUT stream (`kAudioProcessPropertyIsRunningInput`). Music/YouTube use OUTPUT only, so they
//! never trip it. The mic-holding process is often a HELPER with a dotted sub-identifier —
//! verified live under launchd, the mic holder was `com.google.Chrome.helper`, not
//! `com.google.Chrome` — so allowlist matching is by identifier OR dotted-prefix, not equality.
//!
//! `snapshot()` is the only platform-specific part (the cidre process enumeration). The
//! allowlist-matching logic is pure and unit-tested off any real device.

/// One process's audio-activity snapshot. `bundle_id` is `None` for processes without one
/// (system daemons); those are skipped by the matcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcAudio {
    pub bundle_id: Option<String>,
    pub input: bool,
    pub output: bool,
}

/// Default meeting-app bundle-ids. A user override is deferred to the Phase-5 versioned config
/// (single config surface) — Phase 4 ships this hardcoded set only.
pub const DEFAULT_ALLOWLIST: &[&str] = &[
    "us.zoom.xos",                // Zoom
    "com.google.Chrome",          // Chrome (Google Meet) — matches com.google.Chrome.helper
    "com.microsoft.teams2",       // Microsoft Teams (new)
    "com.microsoft.teams",        // Microsoft Teams (legacy / classic)
    "com.tinyspeck.slackmacgap",  // Slack (huddles)
    "com.apple.Safari",           // Safari (Meet)
    "company.thebrowser.Browser", // Arc
    "org.mozilla.firefox",        // Firefox
    "com.brave.Browser",          // Brave
];

/// True if `bundle_id` is the allowlisted `entry` OR a dotted sub-identifier of it:
/// `com.google.Chrome.helper` matches `com.google.Chrome`, but `com.google.ChromeX` does not.
#[must_use]
pub fn matches_entry(bundle_id: &str, entry: &str) -> bool {
    bundle_id == entry
        || bundle_id
            .strip_prefix(entry)
            .is_some_and(|rest| rest.starts_with('.'))
}

/// True if `bundle_id` matches any allowlist entry (identifier or dotted sub-identifier).
#[must_use]
pub fn is_allowlisted(bundle_id: &str, allowlist: &[&str]) -> bool {
    allowlist.iter().any(|e| matches_entry(bundle_id, e))
}

/// A friendly app name for the canonical allowlist entry `active_app_in` returns (meeting title
/// + logs). Falls back to "meeting" for anything unmapped.
#[must_use]
pub fn app_label(entry: &str) -> &'static str {
    match entry {
        "us.zoom.xos" => "Zoom",
        "com.google.Chrome" => "Chrome",
        "com.microsoft.teams2" | "com.microsoft.teams" => "Teams",
        "com.tinyspeck.slackmacgap" => "Slack",
        "com.apple.Safari" => "Safari",
        "company.thebrowser.Browser" => "Arc",
        "org.mozilla.firefox" => "Firefox",
        "com.brave.Browser" => "Brave",
        _ => "meeting",
    }
}

/// The canonical allowlisted app id a snapshot matches on active mic input, if any. Returns the
/// allowlist ENTRY (e.g. `com.google.Chrome`), not the raw helper id — cleaner for naming/logging.
#[must_use]
pub fn active_app_in(snapshot: &[ProcAudio], allowlist: &[&str]) -> Option<String> {
    for p in snapshot {
        if !p.input {
            continue;
        }
        let Some(bid) = p.bundle_id.as_deref() else {
            continue;
        };
        if let Some(entry) = allowlist.iter().find(|e| matches_entry(bid, e)) {
            return Some((*entry).to_string());
        }
    }
    None
}

/// Poll the audio HAL for the allowlisted app currently holding the mic (the meeting signal).
/// Holds the effective allowlist (built-ins ∪ user config extras) so the daemon honors
/// `~/.meetscribe/config.toml`.
pub struct MeetingDetector {
    allowlist: Vec<String>,
}

impl MeetingDetector {
    /// A detector over an explicit (config-derived) allowlist.
    #[must_use]
    pub fn with_allowlist(allowlist: Vec<String>) -> Self {
        Self { allowlist }
    }

    /// The allowlisted app currently holding the mic, or `None`. `Err` only on a HAL failure to
    /// enumerate processes (transient — the caller keeps polling).
    pub fn active_app(&self) -> anyhow::Result<Option<String>> {
        let al: Vec<&str> = self.allowlist.iter().map(String::as_str).collect();
        Ok(active_app_in(&snapshot()?, &al))
    }
}

/// Snapshot every client process's audio input/output activity via the Core Audio HAL.
/// Per-process property errors are tolerated (a helper/sandboxed process may refuse a read)
/// rather than aborting the whole poll — only a failure to list processes is an error.
#[cfg(target_os = "macos")]
pub fn snapshot() -> anyhow::Result<Vec<ProcAudio>> {
    use cidre::core_audio as ca;
    let procs = ca::System::processes().map_err(|e| anyhow::anyhow!("processes(): {e:?}"))?;
    Ok(procs
        .iter()
        .map(|p| ProcAudio {
            bundle_id: p.bundle_id().ok().map(|s| s.to_string()),
            input: p.is_running_input().unwrap_or(false),
            output: p.is_running_output().unwrap_or(false),
        })
        .collect())
}

#[cfg(not(target_os = "macos"))]
pub fn snapshot() -> anyhow::Result<Vec<ProcAudio>> {
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_identifier_and_dotted_helper_but_not_lookalike() {
        // Exact identifier.
        assert!(matches_entry("com.google.Chrome", "com.google.Chrome"));
        // Dotted sub-identifier (the real live case: the mic holder is a helper).
        assert!(matches_entry("com.google.Chrome.helper", "com.google.Chrome"));
        assert!(matches_entry(
            "com.google.Chrome.helper.Renderer",
            "com.google.Chrome"
        ));
        // Look-alike prefix WITHOUT a dot boundary must NOT match (no over-broad matching).
        assert!(!matches_entry("com.google.ChromeX", "com.google.Chrome"));
        assert!(!matches_entry("com.evil.Chrome", "com.google.Chrome"));
    }

    #[test]
    fn active_app_requires_input_and_allowlist_membership() {
        let allow = DEFAULT_ALLOWLIST;
        // Music playing (output only) on an allowlisted app → NOT a meeting.
        let music = vec![ProcAudio {
            bundle_id: Some("com.google.Chrome.helper".into()),
            input: false,
            output: true,
        }];
        assert_eq!(active_app_in(&music, allow), None);
        // The same helper now holding the mic → meeting, reported as the canonical entry.
        let call = vec![ProcAudio {
            bundle_id: Some("com.google.Chrome.helper".into()),
            input: true,
            output: true,
        }];
        assert_eq!(active_app_in(&call, allow), Some("com.google.Chrome".into()));
        // A non-allowlisted app holding the mic (e.g. Photo Booth) → NOT a meeting.
        let other = vec![ProcAudio {
            bundle_id: Some("com.apple.PhotoBooth".into()),
            input: true,
            output: false,
        }];
        assert_eq!(active_app_in(&other, allow), None);
        // A process with no bundle-id holding the mic (a daemon) → skipped, not a crash.
        let daemon = vec![ProcAudio {
            bundle_id: None,
            input: true,
            output: false,
        }];
        assert_eq!(active_app_in(&daemon, allow), None);
    }
}
