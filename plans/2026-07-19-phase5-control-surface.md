# Phase 5 — control surface + polish

**Plan date:** 2026-07-19 · **Status:** BUILDING · **Roadmap:** `plans/2026-07-18-meetscribe.md` §5
**Baseline:** master `b90c423` (Phase 4 done, 3 ahead of origin — NOT pushed), 22 tests green.

Right-sized medium phase (brief plan → batches → verify each → `/review-branch`; no full `/audit-plan`).
Decisions locked with the user 2026-07-19:
- **Tray = SEPARATE PROCESS** (not in-daemon). Daemon stays single-threaded + untouched; tray is its
  own AppKit process that reads a status file and pauses/quits via a flag file / signal. This protects
  the hard-won Core Audio capture path (validated with NO competing run loop).
- **CoreML prewarm = DOC-ONLY.** Daemon is metal-only and already runs faster than real time
  (RTF ~0.4×); prewarm only helps the opt-in `--features coreml` manual path → just document it.

## Keystone fact
The launchd plist runs `[bin, "daemon"]` with **no flags**, so the background daemon uses all
defaults. A **config file the daemon reads at startup** is the only control surface for it. Config
is therefore built first; everything else reads from it.

## Batches

### A — Versioned config (`src/config.rs`) ✅ keystone
- `~/.meetscribe/config.toml`, top-level `version` (currently 1). Load-or-create-default.
- **warn-and-default on unknown keys** — serde silently ignores unknowns, so capture them via
  `#[serde(flatten)] extra: BTreeMap<String, toml::Value>` and `log::warn!` if non-empty. Warn (not
  fail) when the file's `version` is newer than the binary knows (additive-only forward-compat).
- Fields: `[detector] allowlist_extra`, `use_builtin_allowlist`; `[daemon] lang`, `min_secs`;
  `[retention] sessions_days`, `log_max_mb`.
- Daemon precedence: **CLI flag > config > default.** New dep: `toml` (pure-Rust).
- Detector: `active_app_in` already takes an allowlist slice — build the effective allowlist
  (builtin ∪ extra, or extra-only) at daemon start and thread it through.
- verify: `cargo build` + unit tests (parse, unknown-key warn, version warn, effective-allowlist).

### B — Disk/log hygiene (`src/maintenance.rs`)
- Prune `~/.meetscribe/sessions/*` dirs older than `retention.sessions_days` (0 = keep forever).
  Never touches the current session (it's newest). Age by dir mtime.
- Rotate `~/.meetscribe/logs/*.log` over `log_max_mb`: copy → `.1`, then truncate in place
  (works with launchd's O_APPEND redirect fd — next append resumes at 0, no sparse gap).
- Run at daemon startup + once/day in the idle loop (`last_maintenance: Instant`).
- verify: build + unit tests (age-prune keeps-new/drops-old; rotate-by-size) + drive on a temp base.

### C — CoreML prewarm (doc-only)
- Add an "Opt-in CoreML (~2×)" note to `models/PROVISIONING.md`: build `--features coreml`, run one
  throwaway transcribe to cache the ANE compile, then manual coreml runs skip the ~23-min compile.
  No new subcommand, no live 23-min verify spent now.

### D — Tray (separate process)
- Daemon writes `~/.meetscribe/status.json` on each state change: `{state: idle|recording|paused,
  app, since_epoch, meetings_today?}`. Small, safe, single-threaded write — the ONLY daemon change.
- Daemon checks `~/.meetscribe/paused` each idle tick; while present, skip starting new recordings
  (a recording already in progress finishes). Reflect `paused` in status.json.
- `meetscribe tray` = standalone process: `tray-icon` menu-bar item; polls status.json (~1s); menu =
  status line · Pause/Resume (toggles the flag file) · Open transcripts folder · Quit daemon
  (`launchctl bootout` / SIGTERM). Runs the AppKit/main-thread run loop in THIS process only.
- Optional: a second LaunchAgent `com.lucianolupo.meetscribe.tray` (RunAtLoad) for auto-start; else
  documented manual launch. Update DAEMON.md.
- verify: VISUAL/INTERACTIVE (user drives) — icon shows idle→recording, pause stops new records,
  quit stops the daemon.

## Frozen invariants (unchanged)
Bundle-id `com.lucianolupo.meetscribe` · identity `155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C` ·
model `ggml-large-v3` metal-only for the daemon · single-aggregate tap · you=mic/others=tap ·
storage sqlx plaintext 0600 · rebuild→re-sign→`install` after any daemon-code change.

## Done-gate
`/review-branch` (fix confirmed findings) → commit each batch with `git commit -F <msg>` (direct to
master, no PRs) → update RESUME.md + brain/memory. Push is the USER's call (do not push).
