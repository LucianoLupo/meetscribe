# Resume prompt — paste after /clear

Resume **meetscribe** — local-first, background macOS meeting transcriber (Granola-style, brain todo #30). State is in the auto-loaded memory `project_meetscribe.md`; plans in `plans/` (`2026-07-18-meetscribe.md` = phase roadmap §5; `2026-07-18-phase1-capture-watchdog.md` = the Phase 1 detail plan). Read both first.

## Status: Phase 0 ✅ live-passed · Phase 1 (capture watchdog) — Batches 1–3 DONE + verified, Batch 4 (endurance + commit) REMAINS

Phase 1 makes the proven-but-fragile capture layer survive audio-route changes (headphones plug/unplug). Design: `DualCapture` splits a swappable `Option<DeviceInstance>{started,tap}` over process-stable plumbing (rings, producers in a pinned `Box<AudioContext>`, consumers, `Shared`). `rebuild()` tears down the old device and builds a fresh aggregate against the SAME rings; a Core Audio default-device listener + a mic-dry watchdog set an `Arc<AtomicBool>` that the single-threaded drain loop polls — no `!Send` handle crosses a thread.

### Done & verified (this session)
- **Batch 1 — restructure + `rebuild()`.** VERIFIED on real audio (`--rebuild-after`): capture continues across a forced rebuild, mic.wav/system.wav stay byte-for-byte equal length, the gap is silent in BOTH channels, real audio resumes after.
- **Batch 2 — default-device listeners + auto-rebuild.** VERIFIED with real headphone plug/unplug (3 cycles → 6 rebuilds). **RESOLVED the one open unknown: the proc-form `add_prop_listener` FIRES without a CFRunLoop** — our plain `sleep` drain loop is enough; the `dispatch::Queue` fallback was NOT needed. Fixed two bugs found live: (1) **debounce** (500 ms settle) coalesces the notification burst so one physical action = one rebuild (killed the empty 512-sample micro-segments); (2) **`target_rate` adoption** after each rebuild (killed forever-rolling). Each segment's two channels are byte-identical in length.
- **Batch 3 — mic-dry watchdog + startup self-check.** Unit tests pass (`gap_frames_math`, `start_error_remedies`). Happy-path + no-false-watchdog VERIFIED on a 15 s real run (0 rebuilds, no "mic dry" warnings on a quiet-but-flowing mic). `StartError{MicMissing,SystemAudioTccMissing,RouteDropped,Other}` — Display carries the actionable remedy. **Live `tccutil`-revoke test was NOT driven** (outcome is unreliable from a bash-spawned process — prompt vs error — and it risks the hard-won TCC grant); classification/remedy is unit-tested only (Proxy). The watchdog's fire-on-real-drop is a backstop (the listener preempts it on route changes) — reasoned, not independently triggered.

### KEY behavior learned (important)
A cross-rate route change — **BT headset mic @ 16 kHz ↔ built-in mic @ 48 kHz** — has NO common aggregate rate, so `set_nominal_sample_rate` can't hold it and each change **ROLLS a new segment** (`mic.wav`/`system.wav` → `mic.001.wav`… + a `segments.txt` manifest + gap markers). Channels stay frame-aligned WITHIN each segment. A same-rate rebuild instead **pads** both channels with equal silence and keeps one file. Unifying everything to a single 16 kHz stream via in-app resampling (rubato) is the planned **Phase 2** job.

## Do next — Batch 4 (plan `plans/2026-07-18-phase1-capture-watchdog.md`)
1. **Endurance:** 30+ min continuous capture with ≥2 real plug/unplug cycles → both channels grow, equal per-segment counts, gap markers logged, no panic, `dropped()` bounded, process alive, **and the mic-dry watchdog does NOT false-fire** during long system-silence stretches. (Consider offering the user a compressed ~10-min run — the code is already well-validated.)
2. **Alignment check** on the resulting WAVs (per-segment equal length; silence pad at gaps).
3. **Persist:** update the `project-meetscribe` brain/memory + this RESUME.
4. **Commit + push:** `git commit -F <msg>` (avoid the push/force guard) + push to PRIVATE `LucianoLupo/meetscribe`. Then Phase 1.5 (multilingual model provisioning) → Phase 2 (VAD + whisper-rs).

## How to run / verify
- Build + **RE-SIGN after every build** (frozen, mandatory): `cargo build` then `codesign --remove-signature target/debug/meetscribe && codesign --sign 155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C --identifier com.lucianolupo.meetscribe --timestamp=none target/debug/meetscribe`. The explicit `--identifier` is required (rustc's default id changes every build → breaks TCC).
- Run: `./target/debug/meetscribe [--out <dir>] [--seconds <n>] [--rebuild-after <n>]` (no `--seconds` = press Enter to stop). Output to `capture/` (gitignored). Play system audio (e.g. `say`) so `tap_auto_start` fires.
- `cargo test` = the 2 unit tests. Clippy: 2 PRE-EXISTING style warnings (main.rs:123 `seconds.unwrap`, the original `deadline` nested-if) — not from Phase 1.

## Frozen invariants — do NOT change
- Bundle-id `com.lucianolupo.meetscribe` · signing identity `155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C` (Apple Development, TeamID `L634X3YJBF`).
- Engine whisper-rs; model MUST be MULTILINGUAL (Spanish meetings — `medium`/`large-v3-turbo`, not `.en`); storage sqlx plaintext 0600; capture = global Core Audio tap (SCK fallback only); you-vs-them = channel-based.

## Carry-forward gotchas
- `sub_tap_keys` at `ca::hardware::sub_tap_keys`; `StartedDevice` at `ca::hardware::StartedDevice` (not root-re-exported).
- `tap_auto_start=true` → device waits for first system audio before the IO proc fires (silent Mac = `number_buffers=0`, expected). Once started, the mic (clock master) delivers frames continuously even during system silence — that's what makes the mic-dry watchdog sound.
- cidre `output_addr()` has an INPUT-scope bug — use `global_addr()`. Property listeners have NO RAII (manual add/remove; `Drop for DualCapture` removes them).
- Private aggregate (`is_private=true`) is per-process + coreaudiod-reclaimed on exit (incl. crash) → NO launch-time stale cleanup needed (that plan task was dropped, verified in cidre `hardware.rs:1566`).
- `capture/` + `*.wav` are gitignored — recordings never committed.
