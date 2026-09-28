//! Versioned user config at `~/.meetscribe/config.toml`.
//!
//! The launchd LaunchAgent runs `meetscribe daemon` with NO flags, so this file is the only way to
//! customize the background daemon (allowlist, language, min length, retention). It is read once at
//! startup; CLI flags on a manual `daemon` run still override it (flag > config > default).
//!
//! Forgiving by design (a background daemon must not crash-loop on a typo):
//!   - missing file  → write a commented default template, use built-in defaults;
//!   - parse error   → warn loudly, use built-in defaults (never abort);
//!   - unknown keys  → captured via `#[serde(flatten)]` and WARNED (serde ignores them silently
//!     otherwise), then defaulted;
//!   - newer `version` than this binary knows → warn, read best-effort (fields are additive-only).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// `<base>/config.toml` — the single owner of the config-file location (mirrors
/// `status::status_path` / `status::pause_path`).
#[must_use]
pub fn config_path(base: &Path) -> PathBuf {
    base.join("config.toml")
}

/// The config schema version this binary understands. Bump ONLY on a breaking change — additive
/// fields default when absent and do not need a bump.
pub const CONFIG_VERSION: u32 = 1;

/// The commented template written on first run. A unit test asserts it parses back to
/// `Config::default()`, so the comments here can never drift from the real defaults.
pub const DEFAULT_CONFIG_TOML: &str = r#"# meetscribe configuration — read once at daemon startup. Edit, then restart the daemon:
#   launchctl kickstart -k gui/$(id -u)/com.lucianolupo.meetscribe
# CLI flags on `meetscribe daemon ...` override the values here (flag > config > default).

# Config schema version. Additive changes do NOT bump this; a file newer than the binary
# understands is read best-effort (unknown keys are warned and ignored).
version = 1

[detector]
# Extra meeting-app bundle-ids to treat as meeting triggers, IN ADDITION to the built-ins
# (Zoom, Chrome, Teams, Slack, Safari, Arc, Firefox, Brave). Matched by identifier OR dotted
# sub-identifier (e.g. "com.example.App" also matches "com.example.App.helper").
allowlist_extra = []
# Set false to use ONLY allowlist_extra and ignore the built-in list.
use_builtin_allowlist = true

[daemon]
# Whisper language code for transcription (meetings are Spanish by default).
lang = "es"
# Sessions shorter than this many seconds are captured but NOT transcribed (drops mic blips).
min_secs = 20.0

[retention]
# Delete session folders (WAVs + on-disk transcripts) older than this many days.
# 0 = keep forever (default — meetscribe never deletes your recordings unless you opt in).
# The meeting + transcript stay in the DB regardless, so `meetscribe export <id>` still works.
sessions_days = 0
# Rotate the daemon log when it exceeds this many megabytes (keeps one .1 backup). 0 = never.
log_max_mb = 10

[speakers]
# Cut far-end chunks where the voice changes (Nemotron diarizer) before naming the voices.
# false = the pre-split pipeline. Needs the diarizer installed: `bash models/provision.sh`,
# `bash models/build-diarizer.sh`, then `meetscribe install`. Missing diarizer = no split.
split = true
"#;

/// Top-level config. `#[serde(default)]` fills any missing field from `Config::default()`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub version: u32,
    pub detector: DetectorConfig,
    pub daemon: DaemonSettings,
    pub retention: RetentionConfig,
    pub speakers: SpeakerSettings,
    /// Unknown top-level keys — captured so we can WARN instead of silently ignoring them.
    #[serde(flatten)]
    pub extra: BTreeMap<String, toml::Value>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            detector: DetectorConfig::default(),
            daemon: DaemonSettings::default(),
            retention: RetentionConfig::default(),
            speakers: SpeakerSettings::default(),
            extra: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectorConfig {
    /// Extra meeting-app bundle-ids, added to the built-in allowlist (or the whole list when
    /// `use_builtin_allowlist = false`). Matched by identifier or dotted sub-identifier.
    pub allowlist_extra: Vec<String>,
    /// When false, the built-in allowlist is ignored and ONLY `allowlist_extra` is used.
    pub use_builtin_allowlist: bool,
    #[serde(flatten)]
    pub extra: BTreeMap<String, toml::Value>,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            allowlist_extra: Vec::new(),
            use_builtin_allowlist: true,
            extra: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DaemonSettings {
    /// Whisper language code (meetings are Spanish → "es").
    pub lang: String,
    /// Sessions shorter than this are captured but not transcribed (drops sub-blip mic uses).
    pub min_secs: f64,
    #[serde(flatten)]
    pub extra: BTreeMap<String, toml::Value>,
}

impl Default for DaemonSettings {
    fn default() -> Self {
        Self {
            lang: "es".to_string(),
            min_secs: 20.0,
            extra: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RetentionConfig {
    /// Delete session dirs older than this many days. 0 = keep forever (the safe default — we never
    /// delete recordings unless the user opts in).
    pub sessions_days: u64,
    /// Rotate the daemon log over this many MB (keeps one `.1` backup). 0 = never rotate.
    pub log_max_mb: u64,
    #[serde(flatten)]
    pub extra: BTreeMap<String, toml::Value>,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            sessions_days: 0,
            log_max_mb: 10,
            extra: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpeakerSettings {
    /// Split far-end chunks at diarizer voice changes before naming (split-then-name).
    /// Rollback switch: false = the pre-split pipeline. Model paths are not configurable here —
    /// the daemon uses the installed diarizer under `~/.meetscribe/models/diarizer/`.
    pub split: bool,
    #[serde(flatten)]
    pub extra: BTreeMap<String, toml::Value>,
}

impl Default for SpeakerSettings {
    fn default() -> Self {
        Self { split: true, extra: BTreeMap::new() }
    }
}

impl Config {
    /// The effective meeting-app allowlist: built-ins (unless disabled) ∪ user extras, de-duped,
    /// order-preserving (built-ins first).
    #[must_use]
    pub fn effective_allowlist(&self) -> Vec<String> {
        let mut v: Vec<String> = if self.detector.use_builtin_allowlist {
            crate::detect::DEFAULT_ALLOWLIST
                .iter()
                .map(|s| (*s).to_string())
                .collect()
        } else {
            Vec::new()
        };
        v.extend(self.detector.allowlist_extra.iter().cloned());
        let mut seen = std::collections::HashSet::new();
        v.retain(|e| seen.insert(e.clone()));
        v
    }

    /// Human-readable warnings (unknown keys + a too-new version). Pure — the daemon logs these.
    #[must_use]
    pub fn collect_warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if self.version > CONFIG_VERSION {
            w.push(format!(
                "version {} is newer than this binary supports ({CONFIG_VERSION}); newer settings are ignored",
                self.version
            ));
        }
        let push_unknown =
            |w: &mut Vec<String>, prefix: &str, extra: &BTreeMap<String, toml::Value>| {
                for k in extra.keys() {
                    w.push(format!("unknown key '{prefix}{k}' (ignored)"));
                }
            };
        push_unknown(&mut w, "", &self.extra);
        push_unknown(&mut w, "detector.", &self.detector.extra);
        push_unknown(&mut w, "daemon.", &self.daemon.extra);
        push_unknown(&mut w, "retention.", &self.retention.extra);
        push_unknown(&mut w, "speakers.", &self.speakers.extra);
        w
    }

    /// PURE read of `<base>/config.toml` — no disk writes. A missing file returns built-in
    /// defaults; a read/parse error warns and returns defaults (a daemon must not crash-loop on a
    /// bad config). Unknown-key / too-new-version warnings are logged. Use this from read-only
    /// callers (e.g. the `detect` diagnostic) that must not materialize the config dir.
    #[must_use]
    pub fn load(base: &Path) -> Config {
        let path = config_path(base);
        if !path.exists() {
            return Config::default();
        }
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                log::warn!(
                    "config: cannot read {} ({e}); using built-in defaults",
                    path.display()
                );
                return Config::default();
            }
        };
        match toml::from_str::<Config>(&text) {
            Ok(cfg) => {
                for m in cfg.collect_warnings() {
                    log::warn!("config: {m}");
                }
                cfg
            }
            Err(e) => {
                log::warn!(
                    "config: parse error in {} ({e}); using built-in defaults",
                    path.display()
                );
                Config::default()
            }
        }
    }

    /// Like [`load`](Self::load) but, when the file is absent, first writes the commented default
    /// template (creating `<base>`). Only the daemon's first run should own template creation, so
    /// only daemon startup calls this — read-only callers use [`load`](Self::load).
    #[must_use]
    pub fn load_or_init(base: &Path) -> Config {
        let path = config_path(base);
        if !path.exists() {
            match std::fs::create_dir_all(base)
                .and_then(|()| std::fs::write(&path, DEFAULT_CONFIG_TOML))
            {
                Ok(()) => log::info!("config: wrote default {}", path.display()),
                Err(e) => log::warn!(
                    "config: could not write default {} ({e}); using built-in defaults",
                    path.display()
                ),
            }
            return Config::default();
        }
        Self::load(base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_template_parses_to_default() {
        // The comments in DEFAULT_CONFIG_TOML can never drift from the real defaults.
        let parsed: Config = toml::from_str(DEFAULT_CONFIG_TOML).expect("template parses");
        assert_eq!(parsed, Config::default());
        assert!(parsed.collect_warnings().is_empty());
    }

    #[test]
    fn speakers_split_defaults_on_and_is_the_rollback_switch() {
        // Existing config files have no [speakers] section → split is on.
        let cfg: Config = toml::from_str("version = 1\n[daemon]\nlang = \"es\"\n").expect("parses");
        assert!(cfg.speakers.split);
        let cfg: Config = toml::from_str("[speakers]\nsplit = false\nspilt = true\n").expect("parses");
        assert!(!cfg.speakers.split);
        assert!(cfg.collect_warnings().iter().any(|w| w.contains("speakers.spilt")));
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        // An almost-empty file → all defaults, no warnings.
        let cfg: Config = toml::from_str("version = 1\n").expect("parses");
        assert_eq!(cfg, Config::default());

        // A partial file overrides only what it sets; the rest defaults.
        let cfg: Config = toml::from_str("[daemon]\nmin_secs = 45.0\n").expect("parses");
        assert_eq!(cfg.daemon.min_secs, 45.0);
        assert_eq!(cfg.daemon.lang, "es"); // default preserved
        assert_eq!(cfg.retention.log_max_mb, 10); // default preserved
        assert!(cfg.collect_warnings().is_empty());
    }

    #[test]
    fn unknown_keys_are_captured_and_warned() {
        let text = r#"
version = 1
nonsense_top = true
[detector]
allowlist_extra = ["com.example.App"]
typo_here = 3
[daemon]
lang = "en"
"#;
        let cfg: Config = toml::from_str(text).expect("parses despite unknown keys");
        // Known fields still read.
        assert_eq!(cfg.daemon.lang, "en");
        assert_eq!(cfg.detector.allowlist_extra, vec!["com.example.App"]);
        // Unknowns captured (not silently dropped).
        assert!(cfg.extra.contains_key("nonsense_top"));
        assert!(cfg.detector.extra.contains_key("typo_here"));
        let warnings = cfg.collect_warnings();
        assert!(warnings.iter().any(|w| w.contains("nonsense_top")));
        assert!(warnings.iter().any(|w| w.contains("detector.typo_here")));
    }

    #[test]
    fn newer_version_warns() {
        let cfg: Config = toml::from_str("version = 999\n").expect("parses");
        let warnings = cfg.collect_warnings();
        assert!(warnings.iter().any(|w| w.contains("newer than this binary")));
    }

    #[test]
    fn effective_allowlist_builtin_plus_extra_deduped() {
        let mut cfg = Config::default();
        cfg.detector.allowlist_extra =
            vec!["com.example.App".to_string(), "us.zoom.xos".to_string()]; // zoom is already builtin
        let al = cfg.effective_allowlist();
        // Built-ins present, extra appended once, no duplicate for the already-builtin zoom.
        assert!(al.contains(&"us.zoom.xos".to_string()));
        assert!(al.contains(&"com.example.App".to_string()));
        assert_eq!(al.iter().filter(|e| *e == "us.zoom.xos").count(), 1);
        assert_eq!(al.len(), crate::detect::DEFAULT_ALLOWLIST.len() + 1);
    }

    #[test]
    fn effective_allowlist_extra_only_when_builtin_disabled() {
        let mut cfg = Config::default();
        cfg.detector.use_builtin_allowlist = false;
        cfg.detector.allowlist_extra = vec!["com.example.App".to_string()];
        assert_eq!(cfg.effective_allowlist(), vec!["com.example.App".to_string()]);
    }

    #[test]
    fn load_is_pure_when_file_absent() {
        // A read-only caller (e.g. `detect`) must NOT materialize the config dir/file.
        let tmp = std::env::temp_dir().join(format!("meetscribe-load-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let cfg = Config::load(&tmp);
        assert_eq!(cfg, Config::default());
        assert!(!tmp.exists(), "load() must not create the base dir");
    }

    #[test]
    fn load_or_init_writes_template_then_reads_it() {
        let tmp = std::env::temp_dir().join(format!("meetscribe-init-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let cfg = Config::load_or_init(&tmp);
        assert_eq!(cfg, Config::default());
        let path = config_path(&tmp);
        assert!(path.exists(), "load_or_init() must write the template");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), DEFAULT_CONFIG_TOML);
        // Second call finds the file and reads it (no overwrite, still the default).
        assert_eq!(Config::load_or_init(&tmp), Config::default());
        std::fs::remove_dir_all(&tmp).ok();
    }
}
