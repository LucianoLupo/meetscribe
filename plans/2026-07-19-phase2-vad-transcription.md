# meetscribe Phase 2 — VAD + transcription (full, with silero VAD gate)

**Plan date:** 2026-07-19 · **Phase:** 2 of master plan `plans/2026-07-18-meetscribe.md` §5 (lines 127–130)
**Scope decision (user):** FULL Phase 2 — include the silero_rs VAD gate now (not deferred).
**On approval:** copy into repo as `plans/2026-07-19-phase2-vad-transcription.md`, then implement.

## Context

Phases 0/1/1.5 are done + pushed (`master` @ `2b43b7e`). Capture writes finalized **native-rate
mono f32 WAVs** per channel (`mic.wav` = You, `system.wav` = Others; `.NNN.wav` + a sparse
`segments.txt` only on cross-rate route changes). Phase 1.5 proved whisper-rs (metal 0.399× /
CoreML 0.194×) on a real Spanish WAV; `src/bin/rtf_probe.rs` is the working ASR reference.

**Phase 2 turns captured audio into a speaker-labeled transcript.** v1 is **batch at
meeting-finalize** (assumption §5, not live) → Phase 2 bolts on *after* capture stops and
**touches none of the RT drain loop**. Per channel: resample to 16 kHz → silero VAD to find
speech windows (gates ASR = skip silence) → whisper each window → tag You/Others → merge both
channels by timestamp → emit `TranscriptSegment{speaker,text,t_start,t_end,confidence}` (the
storage/export contract, defined here). Delivered as a `meetscribe transcribe <dir>` subcommand.

**Verify criterion (master §5):** real call → speaker-tagged segments from both channels;
measure & report RTF (not a gate).

## Key facts established (this session)
- **Capture contract:** WAV = `channels:1, bits:32, Float, sample_rate = native/aggregate rate`
  (main.rs:365-380). Naming via `segment_path()` (main.rs:442-448): `mic.wav`/`system.wav`, then
  `mic.001.wav`… on rate-roll. `segments.txt` line = `seg <N> mic=<path> system=<path> rate=<r>
  gap_frames=<n> prev_rate=<r> reason=route_change` (main.rs:269-278), present ONLY if a cross-rate
  roll happened. Rate is read from the WAV header (`hound spec.sample_rate`, as rtf_probe does).
- **main.rs** = single capture flow, NO subcommand dispatch yet; `parse_args` (main.rs:36-68) knows
  `--out/-o`, `--seconds/-s`, `--rebuild-after`, `-h`. Only `mod capture;` is declared.
- **whisper-rs sequence** (copy from `rtf_probe.rs:101-140`): `WhisperContext::new_with_params` →
  `create_state` → `FullParams` (greedy, `set_language`, print off) → `state.full(params,&[f32])` →
  `full_n_segments/full_get_segment_text/_t0/_t1` (t0/t1 = centiseconds). Input = 16 kHz mono f32.
- **silero** crate `silero` v0.1.0 (git emotechlab rev `26a6460`): `ort =2.0.0-rc.10` + `ndarray`.
  `VadSession::new(VadConfig)` needs feature **`static-model`** (embeds the 1.7 MB onnx — no model
  file to provision). `process(&[f32]) -> Vec<VadTransition>`; `SpeechEnd{start_timestamp_ms,
  end_timestamp_ms, samples}` = the speech interval. `VadConfig::default()` = 0.5/0.35 thresholds,
  600 ms redemption + pre-pad, 16 kHz, 90 ms min speech. (meetily's tuned config isn't local — use
  defaults, tunable later.)
- **No VAD/TranscriptSegment code exists yet** (greenfield). Deps to add: `rubato`, `silero`,
  `serde`, `serde_json` (all absent today).

## Design

### New deps (`Cargo.toml`, all-platform `[dependencies]`)
```toml
rubato = "0.15"                                  # my own 16 kHz resample (batch FftFixedIn API)
silero = { git = "https://github.com/emotechlab/silero-rs", rev = "26a6460", features = ["static-model"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```
`silero` pulls `ort =2.0.0-rc.10`, whose default `download-binaries` fetches the arm64-macOS
ONNX Runtime **at build time** (a normal build-dep fetch — like whisper-rs-sys compiling
whisper.cpp; NOT a shipped-app runtime download, so zero-telemetry runtime stays intact). Do NOT
enable silero's optional `rubato` feature — I feed it 16 kHz directly, so only my `rubato 0.15` is
in the tree. New modules hang off `main.rs` beside `mod capture;`.

### `src/resample.rs`
`pub fn to_16k_mono(input: &[f32], src_rate: u32) -> Result<Vec<f32>>` — passthrough copy if
`src_rate == 16000`; else `rubato::FftFixedIn::<f32>::new(src_rate, 16000, chunk, 1 sub, 1 ch)`
processed over the buffer (pad final chunk). Unit test: 48 kHz sine → len ≈ input·16000/48000.

### `src/vad.rs`
`pub struct SpeechWindow { pub start_ms: usize, pub end_ms: usize }`
`pub fn speech_windows(audio_16k: &[f32], cfg: VadConfig) -> Result<Vec<SpeechWindow>>` — feed the
16 kHz buffer through `VadSession` in ~30 ms frames, collect `SpeechEnd` intervals, then **coalesce**
windows whose gap < `COALESCE_GAP` (800 ms) and **cap** each at `MAX_WINDOW` (30 s) — keeps whisper
windows large enough for accuracy while still skipping long silence (the "gates ASR" win).

### `src/asr.rs`
`pub struct Asr { ctx: WhisperContext }` built once from the model (`--features coreml` still
toggles CoreML). `pub fn transcribe(&self, audio_16k: &[f32], lang: &str) -> Result<(String, f32)>`
— the rtf_probe whisper-rs sequence over one window; confidence = mean token prob
(`full_n_tokens`/`full_get_token_prob`, fallback `1 - no_speech_prob`).

### `src/transcript.rs`
```rust
#[derive(Serialize, Deserialize, Clone)]
pub enum Speaker { You, Others }
#[derive(Serialize, Deserialize, Clone)]
pub struct TranscriptSegment { pub speaker: Speaker, pub text: String,
    pub t_start: f64, pub t_end: f64, pub confidence: f32 }   // seconds
pub fn merge(mut segs: Vec<TranscriptSegment>) -> Vec<TranscriptSegment> // stable sort by t_start
```

### `src/main.rs` — add a `transcribe` subcommand (back-compatible)
First positional arg dispatch: `transcribe <dir>` → the pipeline below; anything else → existing
capture flow unchanged (so `meetscribe --seconds 300` still records). `transcribe`:
1. Discover channel files: `mic.wav`+`system.wav` (+ any `mic.NNN.wav`/`system.NNN.wav`, offsetting
   timestamps by cumulative segment duration + `gap_frames` from `segments.txt`; single-segment is
   the common path and handled first-class).
2. Per channel: read WAV (any rate) → `resample::to_16k_mono` → `vad::speech_windows` → for each
   window slice `state.full` via `Asr::transcribe` → `TranscriptSegment` tagged `You` (mic) /
   `Others` (system), timestamps offset by window start.
3. `transcript::merge(you ++ others)` → print the speaker-labeled transcript to stdout AND write
   `transcript.json` (serde) in `<dir>`. Report total RTF = wall / audio-seconds.

## Files
- `Cargo.toml` — add rubato/silero/serde/serde_json (MODIFIED).
- `src/resample.rs`, `src/vad.rs`, `src/asr.rs`, `src/transcript.rs` — new modules (NEW).
- `src/bin/vad_probe.rs` — throwaway silero+ort build/behavior spike (NEW, mirrors rtf_probe).
- `src/main.rs` — `mod resample; mod vad; mod asr; mod transcript;` + `transcribe` subcommand (MODIFIED).
- `RESUME.md` + brain memory — mark Phase 2 done (MODIFIED at end).

## Implementation batches (each: build → drive real audio, not just compile)

**Batch 0 — deps + silero/ort build spike.** Add the 4 deps; write `src/bin/vad_probe.rs` (load a
16 kHz WAV → `VadSession` → print #speech windows + total speech seconds). `cargo build --bin
vad_probe` (**de-risks the silero+ort build + ONNX Runtime binary fetch on this toolchain** — the
Phase-2 analogue of the Phase-1.5 whisper-rs spike). **Verify:** builds; ort runtime loads; on the
real `capture/system.wav`, detected speech ≈ the ~145 loud seconds Phase 0 measured.

**Batch 1 — resample module.** `src/resample.rs` + unit test (48 kHz sine → 16 kHz length ratio).
**Verify:** unit test passes; 16 kHz passthrough exercised on the real WAV (48 kHz path unit-tested,
driven live later when capture runs on the built-in mic).

**Batch 2 — vad module.** `src/vad.rs` (windows + coalesce/cap). **Verify:** on `system.wav`, windows
cover the speech regions, total window time ≈ speech duration, no window > 30 s.

**Batch 3 — asr + transcript modules.** `src/asr.rs` (reuse whisper-rs seq; confidence) + `src/transcript.rs`
(struct + merge). **Verify:** transcribe 2–3 windows → coherent Spanish; `merge` orders by t_start.

**Batch 4 — `transcribe` subcommand (headline).** Wire it in main.rs; re-sign the main binary after
build (frozen ritual — capture path's TCC grant must survive; the transcribe path itself needs no
TCC). **Verify (headline):** `meetscribe transcribe capture/` on the real 5-min Spanish capture →
merged **speaker-labeled** transcript from BOTH channels (You=mic, Others=system), correct
interleaving vs the Phase-0 turn-taking, `transcript.json` written, RTF reported.

**Batch 5 — persist + review + push.** Update RESUME + brain memory; run **`/review-branch`**, fix
confirmed findings; `git commit -F` + push to private `LucianoLupo/meetscribe`.

## Verification (real audio, not just build/tests)
- **Verified (must drive):** vad_probe detects speech on the real Spanish WAV; `transcribe capture/`
  produces a coherent, correctly-interleaved You/Others transcript from both channels + writes
  transcript.json + reports RTF.
- **Proxy:** `cargo build` (incl. `--features coreml`); resample unit test; sha of nothing new.
- **Not run:** the 48 kHz→16 kHz path on real capture (unit-tested only until a built-in-mic capture);
  multi-segment rate-roll transcription (handled in code, but the common single-segment path is what
  the 5-min capture exercises); Phase-3 storage/export (this phase writes a minimal transcript.json).

## Assumptions & risks (stated, non-blocking)
- **silero + ort =2.0.0-rc.10 build** on this toolchain is the main de-risk (Batch 0). *Mitigation:*
  spike first; if `download-binaries` fails, fall back to ort `load-dynamic` + a brew `onnxruntime`.
- **VAD-window whisper vs whole-file** — per-window whisper can fragment context; the coalesce (800 ms)
  + 30 s cap keeps windows large. If quality dips vs the Phase-1.5 whole-file run, note it and widen
  coalescing. (whisper-whole already worked, so VAD is a battery optimization layered on top.)
- **silero VadConfig defaults** (not meetily's, which isn't recoverable) — sane Silero values; tune later.
- **ort binary + onnx** add ~10–20 MB to the build; embedded silero model is 1.7 MB. Acceptable for a
  local tool; no runtime download (static-model), so zero-telemetry holds.
- **Session length:** this is a big phase on an already-long session — I'll verify each batch live and
  can checkpoint/`/clear` between batches if context degrades.

*Optional:* `/audit-plan` before Batch 0 — skipping unless you want it (contained, well-scoped phase).
