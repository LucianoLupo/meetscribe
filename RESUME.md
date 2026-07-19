# Resume prompt — paste after /clear

Resume **meetscribe** — local-first, background macOS meeting transcriber (Granola-style, brain todo #30). State is in the auto-loaded memory `project_meetscribe.md`; plans in `plans/` (`2026-07-18-meetscribe.md` = phase roadmap §5; `2026-07-18-phase1-capture-watchdog.md` = the Phase 1 detail plan). Read both first.

## Status: Phase 0 ✅ · Phase 1 ✅ · Phase 1.5 ✅ · Phase 2 (VAD + transcription) ✅ DONE. NEXT = Phase 3 (session model + sqlx storage + export).

**Phase 2 shipped 2026-07-19.** Full scope incl. the silero VAD gate (user chose full over core-only). `meetscribe transcribe <dir>` = batch-at-finalize: per channel read native-rate WAV → resample to 16 kHz (`rubato 0.15`, `src/resample.rs`) → silero VAD speech windows (`silero` rev 26a6460, `static-model` embeds the onnx — no model file; `src/vad.rs`, coalesce 800 ms / cap 30 s) → whisper each window (`src/asr.rs`, reuses the rtf_probe seq; confidence = mean token prob) → tag You(mic)/Others(system) → `transcript::merge` by t_start → print + write `transcript.json`. New deps: rubato, silero(+ort ONNX Runtime, build-time binary fetch), serde, serde_json. `TranscriptSegment{speaker,text,t_start,t_end,confidence}` defined in `src/transcript.rs` = the Phase-3 storage/export contract. **VERIFIED on the real 5-min Spanish `capture/`:** coherent, correctly-interleaved You/Others transcript (You opens 0–18 s, Others answers "¿por qué no reutilizaste el otro?" at 18.18 s — matches Phase-0 turn-taking); 28 merged segments; **RTF 0.235× metal-only** (better than Phase-1.5 whole-file 0.399× because VAD skips silence); `transcript.json` written; 10/10 unit tests; Phase-2 code clippy-clean (2 pre-existing capture-flow warnings remain).

### Phase 3 pickup notes
- `TranscriptSegment` (`src/transcript.rs`, serde-derived) is the storage contract → persist to sqlx SQLite (`meetings`, `transcript_segments`, 0600). `transcript::merge` already produces the time-ordered list; export = Markdown + JSON per meeting (JSON already emitted).
- `transcribe` subcommand orchestration is in `main.rs` (`run_transcribe`); reads the capture WAV contract (single-segment first-class; multi-segment rate-roll handled by cumulative offset **+ `segments.txt` `gap_frames` now honored** via `parse_segment_gaps` — review fix; parser unit-tested, multi-segment path not driven live yet).
- CoreML: `transcribe` runs metal-only by default; a `--features coreml` build uses the (now warm) ANE cache for ~2× — wire a runtime toggle in Phase 3/5.
- silero VadConfig = crate defaults (meetily's tuned config wasn't recoverable); tune later if VAD over/under-segments.

**Phase 1.5 shipped 2026-07-19.** Provisioned the biggest/best multilingual model = full **`large-v3`** (Spanish meetings). `models/provision.sh` = deliberate, sha256-verified, idempotent download of `ggml-large-v3.bin` (2.9 GB) + the pre-converted CoreML encoder (1.2 GB) from the official whisper.cpp HF repo — NO silent runtime download (zero-telemetry). `whisper-rs 0.13.2` introduced (`Cargo.toml`, macOS block, features `raw-api,metal` + a `[features] coreml` toggle); `src/bin/rtf_probe.rs` measures RTF through the real engine. **Verified on the real 5-min Spanish `capture/system.wav` (M1 Pro):** whisper-rs metal build clean; **metal-only RTF 0.399×**; **metal+CoreML warm RTF 0.194× (~2×)** with a one-time ~23-min ANE compile that then caches; transcripts equivalent + clearly better than the `small` reference. Details + numbers + the "pre-warm CoreML at provisioning so a live meeting never eats the compile" note = `models/PROVISIONING.md`. Model paths: `models/ggml-large-v3.bin` + `models/ggml-large-v3-encoder.mlmodelc/` (side-by-side, gitignored). Detail plan = `plans/2026-07-18-phase1.5-model-provisioning.md`.

### Phase 2 pickup notes
- whisper-rs API used (0.13.2): `WhisperContext::new_with_params(&str, WhisperContextParameters::default())` → `create_state()` → `state.full(FullParams, &[f32])`; read via `full_n_segments()/full_get_segment_text(i)/full_get_segment_t0|t1(i)` (t0/t1 = centiseconds). `full()` wants **16 kHz mono f32**. `rtf_probe.rs` is the working reference.
- CoreML toggles via `--features coreml` (auto-loads `models/ggml-large-v3-encoder.mlmodelc` beside the `.bin`); v1 runtime default (metal vs coreml) is a Phase-2 decision — see PROVISIONING.md recommendation.
- Capture writes native-rate WAVs; Phase 2 owns the 48 kHz→16 kHz rubato resample before `full()` (the probe rejects non-16 kHz on purpose).

---

## (historical) Phase 1 — capture watchdog ✅ (commits 381ff8d + fce5c3e)

**Phase 1 shipped 2026-07-18.** All 4 batches done; 30-min endurance passed (0 dropped, 0 watchdog false-fires, 0 panics, 25.6 min continuous single-segment capture, all segment pairs byte-aligned). `/review-branch` (13 agents) found + FIXED 2 medium correctness bugs before push: (1) rebuild() failure was fatal — now returns typed `StartError` and the drain loop retries non-fatally (a route-drop can't kill the session it's meant to save); (2) `WavStream::reset_channels()` on a rate-held rebuild so a same-rate input-device swap can't misalign channels. Skipped 2 low/conventions findings deliberately (thiserror; Diag field dedup). History below kept for context.

---

## (historical) Phase 1 build log — Batches 1–4

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
