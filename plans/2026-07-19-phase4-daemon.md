# meetscribe — Phase 4: meeting auto-detection + background launchd daemon

**Plan date:** 2026-07-19 · **Phase:** 4 of 5 (roadmap `plans/2026-07-18-meetscribe.md` §5)
**Status:** REVISED post-`/audit-plan` (verdict: revise-minor; all 11 planDelta edits folded in) — pending user approval to build
**Prereq state:** Phases 0–3 done + pushed (`master`=`origin/master`=`6655410`, tree clean).

## Goal (from roadmap §5)
A background daemon that **auto-detects a meeting** (an allowlisted app holding the mic),
**auto-records** it via the existing route-hardened `DualCapture`, **finalizes → runs the
existing transcribe pipeline → stores AND exports** (Markdown+JSON) via the Phase-3 layer —
with **zero manual action** — installed as a **launchd LaunchAgent** under the frozen bundle-id,
running at login.

## User decisions (locked 2026-07-19)
1. **End-state = enable at login now.** The phase ends with the LaunchAgent loaded + enabled.
   The actual `launchctl bootstrap`/load is **human-gated at the moment** (shown + confirmed),
   even though the end-state is pre-approved — it is the irreversible/outward-facing action.
2. **Verify = mechanical in-session; real-call deferred to the user.** In-session I prove:
   detector fires on a real allowlisted mic-holding app; launchd runtime-survival (TCC +
   process-enum + path resolution) from the relocated+resigned binary; daemon wiring end-to-end
   against an allowlisted app + `say` audio. The real Zoom/Meet auto-record→transcript quality
   pass is the user's to drive later.

## THE headline risk — de-risk FIRST, before building the daemon (Batch 0)
The Microphone + "record system audio" **TCC grants are keyed to the binary's code-signing
designated requirement** (identifier `com.lucianolupo.meetscribe` + Team `L634X3YJBF`), NOT to
how or from where it is launched. The enabled end-state introduces changes to how the grants
were earned (terminal-launched `target/debug/meetscribe`): **(a) launched by launchd** (a
background agent, no controlling terminal); **(b) a relocated + re-signed binary** at a stable
install path. Both are believed safe (same designated requirement ⇒ same TCC identity), but
"believed" ≠ verified on macOS 26.5.

**Batch 0 empirically settles ALL THREE launchd-runtime surfaces at once and gates the whole
phase** (audit fix — the original gate only exercised capture):
- **(1) system-audio + mic capture** under launchd from the relocated binary (RMS check);
- **(2) process-enumeration** — `System::processes()`/`is_running_input()`/`bundle_id()` called
  with NO controlling TTY, logged to a file (settles R2: does the detector's HAL access
  prompt/deny under a background agent?);
- **(3) path resolution** — assert the daemon-resolved DB path == the CLI's `default_db_path()`,
  and that a row written under launchd appears in a terminal `meetscribe list`.

If any surface fails, we STOP and reassess (`.app` bundle, different launchd domain, or plist
env pinning) BEFORE building the detector/daemon on a broken foundation.

### Absolute-path invariant (audit BLOCKER 1 — the load-bearing correctness fix)
**Under a LaunchAgent `cwd=/` and `~` is NOT expanded.** The daemon must resolve NO runtime path
relative to cwd or an unexpanded tilde:
- **Model:** the daemon passes an **absolute** model path into the pipeline (not the repo-relative
  `models/ggml-large-v3.bin`, main.rs:157). `install` provisions `~/.meetscribe/models/ggml-large-v3.bin`
  as a **symlink** to the repo's absolute model (avoids duplicating 2.9 GB; `--copy` option for a
  repo-independent install; documented). The daemon resolves the model at that absolute dotdir path.
- **Home:** resolved via `$HOME` then **`getpwuid(getuid())`** fallback; the silent *relative*
  fallback in `home_dir`/`default_db_path`/`default_export_dir` (main.rs:351-368) is removed so an
  unresolvable home is a **loud fatal error**, never a wrong-location write. `install` also pins
  `EnvironmentVariables{HOME}` in the plist (belt-and-suspenders).
- **Plist:** `StandardOutPath`/`StandardErrorPath` and `ProgramArguments` are **absolute**
  (no literal `~`).

### Stable install path (prevents a self-inflicted TCC break)
A permanently-enabled agent must NOT point at `target/debug/meetscribe`: the next `cargo build`
overwrites it and zeroes its signature → the running daemon silently loses the grant. So the
daemon runs from a **stable copy**: `~/.meetscribe/bin/meetscribe`, re-signed at that path with
the frozen `--identifier`. Dev rebuilds never disturb the installed daemon.

## Architecture — one single-threaded state machine reusing proven layers

```
 meetscribe daemon  (one thread; DualCapture is !Send — nothing crosses a boundary)
   IDLE  ── poll MeetingDetector every ~1.5s ──────────────────────────────┐
     │  allowlisted bundle-id has is_running_input() == true (1 poll)       │
     ▼                                                                      │
   RECORDING ── session::run_capture(dir, stop) → CaptureSummary{secs,…} ───┘
     │   stop = meeting_ended() OR signalled()   (ONE shutdown path; §Signals)
     │   (the EXISTING route-change/rebuild/watchdog drain loop, unchanged)
     ▼  meeting ended → finalize WAVs
   GATE  ── if summary.secs < MIN_MEETING_SECS: log + skip (no model load) ──▶ IDLE
     │  else
     ▼
   FINALIZE ── pipeline::transcribe_and_store(dir, opts) ── store (Phase-3 db)
     │                                                      + export Markdown+JSON ──▶ IDLE
```

Reuse, do not reinvent: **`DualCapture`** (Phase 0/1, route-hardened) and the **transcribe
pipeline + Phase-3 store + export** are consumed behind two small extractions (Batch 2). The
MIN_MEETING_SECS gate short-circuits **before** the 2.9 GB model load, so junk blips cost nothing.

## ✅ Batch 0 RESULT (2026-07-19) — gate PASSED on all three surfaces (0a terminal + 0b launchd)
- Capture TCC (mic + system-audio) **survived relocation+re-sign AND launchd-launch**: under
  launchd `cwd=/`, system.wav RMS=0.154 peak=1.10, num_buffers=2, dropped=0. Headline risk closed.
- Process-enum + `is_running_input()`/`bundle_id()` work under launchd with **no TTY, no prompt**
  (R2 settled). DB write + path resolution correct under launchd.
- **Finding A — launchd provides `HOME`** for gui-domain agents (`HOME=/Users/lucianolupo` at cwd=/).
  So home resolution is not fragile; still pin HOME + add getpwuid as insurance (cheap), not load-bearing.
- **Finding B (critical for the detector) — the mic-holding process is a HELPER**: live capture
  saw `com.google.Chrome.helper` (input=true), NOT `com.google.Chrome`. → **allowlist matching MUST
  be by identifier OR dotted sub-identifier prefix**, or Chrome/Electron meetings never detect.

## Detector design (`src/detect.rs`)
- `System::processes()` → per `Process`: `bundle_id()` (Err/absent ⇒ skip — daemons/helpers have
  none, tolerate per-process Err without aborting the poll, R3), `is_running_input()`. **Active**
  iff any process whose bundle-id matches an allowlist entry has mic input.
- **Matching = identifier OR dotted sub-identifier** (`matches_entry`): `com.google.Chrome.helper`
  matches `com.google.Chrome`, but `com.google.ChromeX` does NOT (Batch-0 Finding B).
- **Allowlist (hardcoded, the ONLY config surface in Phase 4):** Zoom `us.zoom.xos`; Chrome
  `com.google.Chrome`; Teams `com.microsoft.teams2` + legacy `com.microsoft.teams`; Slack
  `com.tinyspeck.slackmacgap`; browsers people take Meet in: Safari `com.apple.Safari`, Arc
  `company.thebrowser.Browser`, Firefox `org.mozilla.firefox`, Brave `com.brave.Browser`.
  **User override is DEFERRED to the Phase-5 versioned config** (single config surface — audit
  scope fix; no `allowlist.txt` in Phase 4).
- **Why music/YouTube can't false-trigger:** playback uses *output*, never `is_running_input()`.
  A brief Chrome mic blip (voice search) is caught by the MIN_MEETING_SECS guard, not the detector.
- **Debug surface:** `meetscribe detect [--watch]` prints the live process table
  (bundle-id · running_input · running_output · allowlisted?) — the mechanical verify tool.
- **Unit tests:** allowlist membership + the pure debounce/min-duration gating decisions
  (matches the repo's tested-pure-helper convention).

## Daemon design (`src/daemon.rs`, `meetscribe daemon`)
- State machine above. Poll `POLL_INTERVAL = 1.5s` in IDLE. Start requires **1 active poll**
  (mic-held is already a strong signal); stop requires the app to have **released the mic for
  `MEETING_END_DEBOUNCE = 10s`** (survives transient route drops mid-call). Constants are
  **tunable, not load-bearing** — called out as such.
- **Session dir:** `~/.meetscribe/sessions/<YYYYMMDD-HHMMSS>/` (created 0700, absolute). WAVs kept
  after transcription (consistent with the `transcribe` subcommand; disk hygiene is Phase 5).
- **Min-duration guard:** a session whose `CaptureSummary.secs < MIN_MEETING_SECS = 20s` is
  finalized but **not transcribed/stored** (logged + left on disk), gated BEFORE the model load.
- **Model backend = metal-only** (no `coreml` feature): unattended runs must never trigger the
  one-time ~23-min CoreML ANE compile on a live meeting (Phase-1.5 gotcha). `install` enforces
  this as a precondition (refuses/warns on a coreml-compiled source binary).
- **Signals (ONE shutdown path):** a SIGTERM/SIGINT handler (launchd unload sends SIGTERM) sets an
  `AtomicBool` that **composes INTO the capture stop predicate** (`stop = meeting_ended() OR
  signalled()`), so finalize reuses the single clean-shutdown path — no second teardown. hound
  gets its explicit `finalize()`; no partial-file corruption (R4).
- **Failure classification (R5 — KeepAlive{SuccessfulExit:false} restarts on ANY non-zero exit):**
  recoverable (mic gone / route dropped / tap transiently unavailable at capture start) → log +
  back off + retry in-loop, **exit 0** (no crash-loop); genuinely fatal misconfig (model missing at
  the absolute path, home unresolvable) → log a loud error and **exit 0** as well (a crash-loop
  helps nobody), surfaced in the log. Never a bare non-zero bubble-up.
- **Logging:** the existing `log::{info,warn,error}!` macros via `env_logger` (already stderr) — no
  new logging stack. launchd routes stderr→`~/.meetscribe/logs/meetscribe.err.log`, stdout→`.out.log`.
- **Single-instance guard:** a `~/.meetscribe/daemon.lock` (raw `libc` flock) so a manual
  `meetscribe daemon` can't run alongside the launchd one and double-open the tap. **Capped** — no
  PID-staleness / stale-lock recovery machinery (kept minimal per audit).
- **Deps:** add bare `libc` (macOS) for the SIGTERM handler + `getpwuid` + `flock` — already in the
  tree transitively; consistent with the repo's low-level (cidre/raw Core Audio) style. **Do NOT**
  pull `signal-hook`/`ctrlc`/`fs2` (higher-level wrappers, against the minimal-deps ethos).

## launchd design (`src/launchd.rs`, `meetscribe install`/`uninstall`)
- Label = **`com.lucianolupo.meetscribe`** (frozen). Plist at
  `~/Library/LaunchAgents/com.lucianolupo.meetscribe.plist`.
- Keys: `ProgramArguments = [<abs>/.meetscribe/bin/meetscribe, daemon]`; `RunAtLoad = true`;
  `KeepAlive = {SuccessfulExit: false}` (restart on crash, not on clean stop); **absolute**
  `StandardOutPath`/`StandardErrorPath` → `~/.meetscribe/logs/`; `EnvironmentVariables = {HOME: <abs>}`;
  `ProcessType = Background`.
- **`meetscribe install` (idempotent):**
  1. `launchctl bootout gui/$UID/<label> 2>/dev/null || true` (stop any running daemon first — no
     stale inode on the rebuild→reinstall loop);
  2. **precondition:** verify the source binary is metal-only (refuse/warn if `coreml`-compiled);
  3. copy the source binary → `~/.meetscribe/bin/meetscribe`;
  4. **re-sign there, remove-then-sign** (frozen recipe, RESUME.md:59):
     `codesign --remove-signature <bin> && codesign --sign 155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C
     --identifier com.lucianolupo.meetscribe --timestamp=none <bin>`;
  5. provision `~/.meetscribe/models/ggml-large-v3.bin` (symlink→repo, or `--copy`);
  6. write the plist (absolute paths + pinned HOME);
  7. `launchctl bootstrap gui/$UID <plist>` (fallback `launchctl load`).
- **`meetscribe uninstall`:** `launchctl bootout`/unload + remove plist (keeps data).
- `docs/DAEMON.md`: install/uninstall, the rebuild→reinstall loop (never leave a stale inode),
  load/unload, logs, allowlist, and the model symlink-vs-copy choice.

## Two extractions (Batch 2) — pin BOTH contracts against BOTH callers up front
Design each contract against the CLI wrapper AND the daemon now, so Batch 3 is pure wiring with no
signature change (audit HIGH — else Batch 3 widens a just-frozen API and regresses `transcribe`).

- **`src/session.rs`** — lift the capture drain-loop body out of `main.rs` into
  `run_capture(out_dir: &Path, stop: impl FnMut() -> bool) -> Result<CaptureSummary>` (`impl` not
  `dyn`, per rust.md). **`CaptureSummary` carries capture duration (secs)** so the daemon gates
  MIN_MEETING_SECS before the model load. `--rebuild-after` (main.rs:614/663) is a debug hook —
  **thread it through as an optional param** so the "zero behavior change" claim stays literally
  true. Interactive/`--seconds` path calls it with an Enter/deadline predicate; daemon calls it with
  `meeting_ended() OR signalled()`. Mechanical extraction — no change to route-change/rebuild/watchdog
  logic. **Moves into session.rs:** `WavStream`, `segment_path`, `append_manifest`, `report`.
- **`src/pipeline.rs`** — lift the transcribe core out of `run_transcribe` into
  `transcribe_and_store(dir: &Path, opts: &PipelineOpts) -> Result<PipelineOutput>`. **`PipelineOpts`
  = { model: PathBuf (absolute), lang, title: Option, db_path, export_dir, no_store }.** The core
  **stores AND exports** Markdown+JSON via `export::write_exports` (export_dir defaults to the session
  dir, matching `transcribe`) — audit BLOCKER 2. **Preserve the store-failure→synth_row→export-anyway
  fallback** (main.rs:318-344): the core returns `row + store-status` (does NOT bail on a DB error),
  so both the CLI and the daemon still write the transcript we spent compute on. CLI `transcribe`
  wrapper keeps arg-parse + printing. **Moves into pipeline.rs:** `read_wav_any_rate`,
  `discover_channel`, `parse_segment_gaps`, `read_segment_gaps`, `model_name`,
  `earliest_capture_start`, `synth_row`, `new_runtime`.
- **Shared helpers → `pub(crate)`** (one definition of the `.meetscribe` root, reused by
  daemon/launchd): `home_dir` (now getpwuid-backed, no relative fallback), `default_db_path`,
  `default_export_dir`, `now_epoch`. Consider a small `paths` module.

## Batches (each: `cargo build` → **re-sign** → drive a REAL check, not just compile/tests)

- **Batch 0 — launchd runtime-survival gate (TCC + process-enum + path resolution). NO daemon code**
  (a ~20-line throwaway probe). Copy the signed binary → `~/.meetscribe/bin/meetscribe`, **remove-then-sign**
  there; temp plist runs the probe with no TTY; `launchctl bootstrap`; play `say` audio ~20s; `bootout`.
  **verify:** (1) probe `system.wav` RMS > ~1e-3 AND `mic.wav` non-silent; (2) the logged
  `processes()`/`is_running_input()`/`bundle_id()` results are non-empty/non-error under launchd;
  (3) daemon-resolved DB path == CLI `default_db_path()` and a probe-written row shows in terminal
  `meetscribe list`. **GATE: all green ⇒ proceed; any red ⇒ STOP + reassess.** Clean up the probe
  plist; note the stale `~/.meetscribe/bin/meetscribe` that Batch 4's `install` overwrites.
- **Batch 1 — `detect.rs` + `meetscribe detect`.** (Independent of Batch 2 — parallelizable.)
  verify: `detect --watch` shows Active while a real allowlisted app holds the mic, Idle under
  music-only. Unit tests for allowlist + gating. (Driven in-session with a real app.)
- **Batch 2 — extractions (`session.rs`, `pipeline.rs`) + `pub(crate)` path helpers.** verify:
  `--seconds 20` capture identical to pre-refactor; `transcribe capture` identical; **plus a
  forced-store-failure case** (`transcribe capture --db /nonexistent/dir/x.db`) asserting the
  Markdown/JSON exports are STILL written (guards the fallback).
- **Batch 3 — `daemon.rs` + `meetscribe daemon`.** verify (foreground, mechanical): allowlisted app
  holds mic + `say` audio → daemon auto-creates a session, captures non-silent, on release finalizes
  → pipeline runs → meeting stored AND **Markdown+JSON written beside the WAVs** (`list` shows it).
  Sub-20s blip → dropped (no store, no model load). SIGTERM mid-capture → clean finalize, exit 0.
- **Batch 4 — `launchd.rs` + `install`/`uninstall` + `docs/DAEMON.md`, then ENABLE.** Primary gate =
  the daemon **auto-detect → record → finalize → pipeline → store+export wiring under the PERMANENT
  plist** (TCC-survival is now a cheap confirmatory smoke — settled by Batch 0, not re-litigated).
  **Human-gated:** show the plist + exact `install` command, confirm, then load + confirm healthy
  (`launchctl print gui/$UID/<label>`). Verify a daemon-stored meeting appears in a terminal `list`.
- **Batch 5 — review + ship.** `/review-branch` → fix confirmed findings (test guards before commit)
  → docs current → `git commit -F` to `master` (direct, no PR) → persist (RESUME.md + memory
  `project_meetscribe.md` + brain).

## Frozen invariants (do NOT change)
Bundle-id `com.lucianolupo.meetscribe`; identity `155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C`
(Team `L634X3YJBF`); **re-sign after every build** — `--remove-signature` THEN `--sign` with
explicit `--identifier`; model `models/ggml-large-v3.bin` (repo/CLI) resolved absolutely for the
daemon; storage plaintext sqlx 0600; single global aggregate tap; you=mic / others=tap.

## Risks (post-audit status)
- **R1 (headline):** TCC grant survival under launchd + relocation — Batch 0 gates it. ✅ addressed.
- **R2:** process-enumeration TCC under a background agent — **now folded into the Batch-0 gate**.
- **R3:** `bundle_id()` Err on helper/sandboxed processes — detector tolerates per-process Err.
- **R4:** SIGTERM mid-capture clean finalize — via the single stop-predicate shutdown path.
- **R5:** KeepAlive restart storms — failure classification: recoverable → retry+exit0, fatal → loud
  log + exit0; never a bare non-zero exit.
- **R6:** debounce/min-duration constants are tunable, not load-bearing — stated as such.
```
