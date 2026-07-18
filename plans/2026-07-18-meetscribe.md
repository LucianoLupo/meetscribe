# meetscribe — local-first, background meeting transcriber for macOS

**Plan date:** 2026-07-18  ·  **Revised:** 2026-07-18 (post `/audit-plan` = revise-major, + superwhisper stack recon)
**Status:** APPROVED-TO-BUILD (pending signing cert for Phase-0 live test)
**Working name:** `meetscribe` (rename freely)  ·  **Repo:** `~/projects/meetscribe/`

> A Granola-style tool that listens to *all* my meetings across every meeting app
> (Zoom, Google Meet in Chrome, Microsoft Teams, Slack huddles) in the background —
> **no meeting bots** — and produces **speaker-labeled transcripts**, **fully on-device**.

Grounded in: `docs/research-2026-07-18.md` (6-agent deep-research, 96 sources) + three
source-verified prior-art dissections (meetily / Muesli+pasrom / anarlog+FluidAudio)
+ a binary-level teardown of superwhisper v2.16.4 (the commercial leader).

---

## 1. Locked decisions

| Fork | Decision |
|------|----------|
| **V1 scope** | **Transcripts only.** Background capture → clean, timestamped, speaker-labeled transcript per meeting. AI-notes layer is v2. |
| **Transcription** | **Local / on-device.** Nothing leaves the machine. |
| **Stack** | **Rust core + native shim.** Realized as **Rust-top + `cidre` shim** (proven by meetily 25k★ / anarlog 8.8k★ — full capture+ASR in Rust, no Swift audio sidecar). |
| **ASR engine (v1)** | **whisper.cpp via `whisper-rs`** (Metal+CoreML). Confirmed by 4 reference tools incl. superwhisper. **Parakeet (`parakeet-rs`) = the perf upgrade (v1.1+).** Apple SpeechTranscriber rejected for v1 (see §6). |
| **Capture (v1)** | **Global Core Audio process tap** (system mixdown) primary; **ScreenCaptureKit ≥15.0 = named Teams fallback**. |
| **Storage (v1)** | **Plaintext SQLite via `sqlx`**, `0600`/`0700` perms on FileVault disk. Encryption-at-rest deferred to v1.1 with a documented one-way migration (§7). |

### Prior-art stack reference (what the incumbents actually ship)
| Tool | Capture | ASR | Diarization | Notes |
|------|---------|-----|-------------|-------|
| **meetily** (Rust) | Core Audio global tap (`cidre`) | whisper-rs / Parakeet(`ort`) | none (mixes streams — the mistake we avoid) | Ollama/cloud |
| **anarlog** (Rust/Tauri) | `cidre` tap + `cpal` | whisper-rs | pyannote-local | llama (`hypr-cactus`) |
| **superwhisper** (closed) | **ScreenCaptureKit** | Argmax (WhisperKit+Parakeet) | Argmax SpeakerKit (Sortformer/pyannote) | llama.cpp (llama-3-8b) |
| **meetscribe (us)** | **Core Audio tap** + SCK fallback | **whisper-rs** | **channel-based** (free you-vs-them) | v2: local LLM |

## 2. Stated assumptions (challenge any)
1. **Apple Silicon + macOS 26.5+ only** for v1 (your M1 Pro). Cross-platform deferred.
2. **Headphones assumed for v1** → skip neural echo-cancellation (v1.1).
3. **English-first**, model configurable (ggml `base.en`/`small.en` → `small`/`medium`/`large-v3-turbo`).
4. **Personal single-user tool.** NOT App-Store-distributable (process taps + process enumeration) → **Developer-ID / locally-signed only.**
5. **v1 = VAD-segmented batch transcribe at meeting-finalize** (no live-transcript surface in v1). RTF is *measured, not gated*. True live streaming = v1.1.
6. **Fully local, ZERO telemetry, no cloud** (unlike superwhisper's Sentry + optional cloud AI).
7. **You-vs-them = channel-based** (mic="You", system="Others"). Multi-remote-speaker splitting needs neural diarization → v2.

## 3. Architecture — Rust-top + `cidre` shim, SINGLE-aggregate capture

**⚠ Audit blocker-1 fix:** capture is **one aggregate device on one clock**, NOT two independent
streams. Two independent audio clocks drift and smear speaker attribution over a long meeting
(research §2/§7: "the highest-value architectural decision in the whole app").

```
                    ┌──────────────────────────────────────────────────────┐
 MEETING DETECTOR ─▶│ kAudioHardwarePropertyProcessObjectList +             │
 (Rust/cidre)       │ kAudioProcessPropertyIsRunningInput (mic-in-use)      │
                    │ gated by bundle-ID allowlist (Zoom/Chrome/Teams/Slack)│
                    └───────────────┬──────────────────────────────────────┘
                                    ▼  meeting started
   CAPTURE LAYER  (Rust, cidre) — ONE AudioHardwareCreateAggregateDevice, ONE clock:
   ├─ ch0 = built-in MIC as input sub-device            → "You"
   └─ ch1 = GLOBAL process tap (stereoGlobalTapExcludeSelf, mixdown) → "Others"
        kAudioSubDeviceDriftCompensationMaxQuality on the non-clock sub-device
        query kAudioTapPropertyFormat live (Teams = 24kHz) → resample 16kHz mono (rubato)
                                    ▼
   BUILD-FRESH: default-output-device listener → full tap+aggregate rebuild
                = ZERO-SAMPLE / route-drop WATCHDOG (meetily has NONE of this)
                + startup self-check: distinguish "TCC grant missing" from "route dropped"
                                    ▼   ring buffer (ringbuf SPSC)
   VAD  (Rust, silero_rs) per channel — port meetily's tuned config; gates ASR (battery)
                                    ▼
   ASR  (Rust, whisper-rs, metal+coreml) — per channel; tag ch0→"You", ch1→"Others"
                                    ▼
   RUST CORE — session state machine · merge 2 channels' segments by timestamp
                                    ▼
   STORAGE (sqlx SQLite, 0600) — meetings · transcript_segments{speaker,text,t_start,t_end,conf}
                                    ▼
   EXPORT (Markdown + JSON per meeting)  ·  CONTROL (CLI + launchd; tray = fast-follow)
```

### Dependency recipe (pinned to meetily's verified revs)
`cidre` (git `yury/cidre`, **rev `a9587fa`**, features=`["av"]`) · `ringbuf` · `rubato` ·
`silero_rs` (**rev `26a6460`**) · `whisper-rs` (features `raw-api,metal,coreml`) ·
`sqlx` (0.8, runtime-tokio, sqlite) · `tokio` · `serde` · `hound` (spike-only, WAV).
**Fresh (NOT meetily ports):** `hound`, later `tray-icon` (needs a main-thread run loop — cost
noted), later `parakeet-rs`/`ort` (Parakeet upgrade). **Dropped:** `symphonia` (no file-import in v1),
`cpal` (only needed by the two-stream fallback).

**Dependency risk:** `cidre` is pre-1.0/experimental — API churns across revs; the ported
`core_audio.rs` compiles only against rev `a9587fa`. Pin it; bump deliberately. Fallback = raw
`objc2`+`coreaudio-sys` (much more work).

## 4. PORT vs BUILD-FRESH
**Port (MIT):** meetily `core_audio.rs` global-tap+aggregate (extend it to add mic as input
sub-device); meetily `vad.rs` (Silero tuning); meetily `Cargo.toml` macOS block. Muesli/pasrom
Swift saved at `scratchpad/src/{muesli,pasrom}/` for cross-reference (esp. SCK-for-Teams path).
**Build fresh (our value-add):** dual-stream **speaker labeling** (meetily *mixes* — its mistake);
**route-change listener + zero-sample watchdog + tap/aggregate rebuild** (meetily has none — audit-verified);
**meeting auto-detection + background daemon** (meetily is manual-start only); **privacy hardening**
(no telemetry; `0600` perms). Mic path adapts from meetily *legacy* `core-old.rs` (clean `capture/microphone.rs` is a stub) — largely moot under the single-aggregate design.

## 5. Phased build (de-risk hardest first)

### Phase 0 — Capture spike  (split 0a/0b per audit)
**Precondition:** freeze ONE reverse-DNS **bundle-id** + ONE **signing identity**; carry both
UNCHANGED through all phases (TCC grant + any Keychain key bind to them). Recovery:
`tccutil reset SystemAudioCaptureRequests <bundle-id>`. Sign with local **Apple Development** cert
(the "record system audio" prompt never fires for an unsigned binary).
- **0a:** each source (mic; global tap) yields non-zero-RMS 16 kHz mono PCM → `system.wav`, `mic.wav`.
- **0b (the true novel risk):** mic + tap fused in **ONE aggregate**; verify a scripted "you speak,
  then remote speaks" lands in correct time order with **<100 ms cross-channel skew after several
  continuous minutes**. Test on a real **Microsoft Teams** call (the known failure case).
- **Teams exit decision:** if the global tap delivers silent/zero-sample Teams audio → either (a)
  state Teams drops from v1, or (b) drop in the thin **SCK ≥15.0** system path for Teams only.
- **verify:** both WAVs intelligible; 0b skew < 100 ms; Teams outcome recorded. *Capture is de-risked
  only when BOTH 0a and 0b pass — 0b (alignment), not the `core_audio.rs` port, is the real risk.*

### Phase 1 — Capture layer productionized
- Route-change/default-output listener → full tap+aggregate rebuild (the watchdog). Stale-aggregate
  cleanup at launch. Startup self-check separates TCC-missing from route-dropped.
- **verify:** 30+ min continuous capture survives a headphone plug/unplug; channels stay aligned
  (same you/them ordering check); watchdog log clean.

### Phase 1.5 — Model provisioning
- One-time **deliberate** download of the chosen ggml model + generate/verify the CoreML
  `*-encoder.mlmodelc`; record its location. **No silent runtime download** (zero-telemetry).
  Validate RTF **metal-only first**, then enable coreml (so a broken CoreML conversion can't block Phase 2).

### Phase 2 — VAD + transcription
- `silero_rs` VAD per channel (ported config) gates ASR. `whisper-rs` worker per channel; tag You/Others.
- Define the `TranscriptSegment{speaker,text,t_start,t_end,confidence}` struct here = the storage/export contract.
- **verify:** real call → speaker-tagged segments from both channels; **measure & report RTF** (not a gate).

### Phase 3 — Session model + storage + export
- Rust core: session state machine; merge the two channels into one time-ordered transcript.
- `sqlx` SQLite (`0600`): `meetings`, `transcript_segments`. Sync/async: `sqlx` is async — clean.
- Export per meeting: Markdown (human) + JSON (machine) to a configurable folder.
- **verify:** full meeting → one readable speaker-labeled, timestamped transcript file; DB round-trips.

### Phase 4 — Meeting auto-detection + background daemon
- `kAudioHardwarePropertyProcessObjectList` + `kAudioProcessPropertyIsRunningInput`, gated by
  bundle-ID allowlist (Zoom `us.zoom.xos`, Chrome `com.google.Chrome`, Teams, Slack).
- Daemon: mic-open+known-app → auto start; mic-released → finalize. `launchd` LaunchAgent at login
  with a **fixed reverse-DNS label tied to the frozen bundle-id**; document load/unload/uninstall.
- **verify:** real call auto-records with zero manual action → ends → transcript saved; no false trigger
  from music/YouTube; **tap prompt/grant survives the launchd repackaging** (bundle-id unchanged).

### Phase 5 — Control surface + polish (tray = fast-follow)
- v1 = **CLI + launchd + "open transcripts folder"**. Then `tray-icon` menu (status/pause/quit).
- Config file with a top-level **`version`** field (additive-only; warn-and-default on unknown keys).
- **verify:** tray reflects live state; tap grant survives `.app` repackaging.

## 6. Deferred (OUT of v1)
- **AI notes** (summary/actions) via local LLM (ollama / llama.cpp — superwhisper uses llama-3-8b) — the v2 headline.
- **Neural AEC** (CoreML DTLN, Muesli-style) for speaker/open-air use.
- **Neural diarization** (Sortformer/FluidAudio/pyannote — the Argmax path) to split multiple remote speakers.
- **Parakeet-via-`parakeet-rs`/`ort`** as faster/lighter ASR (battery/thermal upgrade).
- **Apple SpeechTranscriber** — rejected for v1: keeps the pipeline in Rust, and 0 of 4 reference tools
  (incl. superwhisper) use it; revisit only if a small Swift ASR sidecar becomes acceptable.
- **Encryption-at-rest** (see §7), **EventKit calendar** trigger, **cross-platform**.

## 7. Cross-cutting risks & non-functionals
- **Signing identity + bundle-id = a hard cross-phase invariant** (a re-sign silently zeroes the TCC
  grant *and* could lock out a future Keychain key). Frozen in Phase 0, verified in Phases 4-5.
- **Zero-sample tap bug** — single-source/beta; the route-change rebuild watchdog + a
  "silence while a known meeting app is audible" heuristic handle it.
- **Encryption deferral migration (has teeth):** v1 ships plaintext SQLite. If v1.1 adds SQLCipher it
  needs an explicit one-way `sqlcipher_export()` migration (ATTACH keyed DB → export → atomic swap →
  delete plaintext) — SQLCipher cannot open a plaintext DB in place. Documented now, not a silent trap.
- **Power/thermal** — VAD-gated ASR now; Parakeet (4× lighter on ANE) later. Measure in Phase 2/4.
- **Consent/legal (one-line flag)** — recording your own meetings; some jurisdictions are two-party-consent.
- **UX bar** — Granola / superwhisper for transcript formatting + unobtrusive background behavior.

## 8. Open questions — RESOLVED
1. Capture-first ordering → **yes** (only novel risk); split into 0a/0b.
2. ASR engine → **whisper-rs for v1** (Parakeet upgrade later); SpeechTranscriber rejected (§6).
3. Encryption in v1 → **no** — plaintext + `0600`, deferred with documented migration (§7).
4. Tray in v1 → **fast-follow**; v1 = CLI + daemon.
5. `rusqlite`+SQLCipher vs `sqlx` → **`sqlx`** (no encryption in v1 ⇒ reuse meetily's proven async setup).
