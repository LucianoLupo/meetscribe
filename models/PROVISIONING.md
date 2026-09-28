# Model provisioning log — meetscribe Phase 1.5

**Purpose.** Record exactly which on-device ASR model is provisioned, from where, its
integrity hashes, on-disk location, and measured real-time factor (RTF). This is the
"record its location" + logged-provenance requirement of Phase 1.5. The shipped app
performs **no runtime model download** (zero-telemetry design goal) — provisioning is
this one deliberate, reproducible step (`models/provision.sh`).

## Chosen model — `large-v3` (full, multilingual)

- **Why:** meetings are in **Spanish** → multilingual is mandatory (English-only `.en`
  models are disqualified). `large-v3` is the biggest/most-accurate whisper model
  (32 decoder layers, ~1.5 B params). v1 transcribes in **batch at meeting-finalize**,
  so faster-than-realtime is sufficient; the extra cost vs `large-v3-turbo` buys accuracy
  on Spanish jargon / proper nouns. `turbo` remains a drop-in swap via the same script.

## Source (official whisper.cpp HuggingFace repo)

Downloaded from `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/`:

| Artifact | On disk | Size | sha256 (verified vs HF git-lfs pointer) |
|---|---|---|---|
| `ggml-large-v3.bin` | `models/ggml-large-v3.bin` | 2.88 GiB | `64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2` |
| `ggml-large-v3-encoder.mlmodelc.zip` | unzipped → `models/ggml-large-v3-encoder.mlmodelc/` | 1.1 GiB (zip) | `47837be7594a29429ec08620043390c4d6d467f8bd362df09e9390ace76a55a4` |

The pre-converted CoreML encoder is the officially-sanctioned alternative to running
`generate-coreml-model.sh` locally (avoids a fragile torch→coremltools conversion — the
brew `openai-whisper` venv lacks `coremltools`/`ane_transformers` and homebrew python is
3.14). It is the same class of deliberate provisioning download as the `.bin` itself.

## CoreML naming contract

whisper.cpp (via whisper-rs `coreml` feature) derives the encoder path from the model
path: `ggml-large-v3.bin` → `ggml-large-v3-encoder.mlmodelc` in the **same directory**.
Both artifacts sit side-by-side in `models/` with exactly those names.

## Re-provision

```
bash models/provision.sh
```

Idempotent + integrity-checked: skips artifacts already present with a matching sha256,
aborts on any hash mismatch. Model binaries are gitignored (`.gitignore`: `/models/*.bin`,
`/models/*.mlmodelc/`, `/models/*.zip`); only `provision.sh` + this log are committed.

## Measured RTF — on the real 5-min Spanish capture (`capture/system.wav`, 16 kHz mono)

Measured with `src/bin/rtf_probe.rs` on Apple **M1 Pro** (macOS 26.5, Rust 1.95, cmake 4.1.1).
RTF = wall-clock(inference only) / audio-seconds; **< 1.0× = faster than realtime**.
Informational, not a gate (v1 is batch-at-finalize). 300.0 s of audio in each case.

| Backend | Build | Warm RTF | Wall | Notes |
|---|---|---|---|---|
| Metal-only | `cargo build --bin rtf_probe` | **0.399×** | 119.6 s | Works immediately — no compile step. |
| Metal + CoreML | `cargo build --bin rtf_probe --features coreml` | **0.194×** | 58.3 s | ~2× faster (encoder on ANE). **One-time ~23-min ANE compile on first run**, then cached (warm start ≈ 2 s). |

### Quality — CoreML output is equivalent to Metal, not degraded
Both backends produce the **same transcript**: 51 segments, 370 words, byte-identical text
covering the full 300 s (verified via `--max-print 200` diff). The `whisper_full_with_state:
... failed due to entropy/avg_logprobs ... temperature = 0.00` lines are whisper's normal
temperature-fallback logs (occur on both backends), not errors. Both are clearly better than
the earlier `small`-model reference (`capture/transcript_5min.txt`) on Spanish technical terms.

### Recommendation — RESOLVED in Phase 5
- **Metal-only is the shipped default**, for both the CLI and the background daemon. It needs zero
  setup, starts instantly, and at 0.399× (300 s of audio in ~120 s) already runs faster than real
  time — so the daemon keeps up with any meeting without CoreML.
- **The daemon is metal-only by design and refuses a coreml-built binary** (`meetscribe install`
  bails on a `--features coreml` build). This is deliberate: an unattended daemon must never eat the
  one-time ~23-min ANE compile on a user's first live meeting.
- **CoreML (~2×) is therefore an opt-in for the MANUAL `transcribe` path only** — useful when you
  batch-transcribe a backlog and want it done in half the wall-clock (or lower power/thermal). See
  the pre-warm step below so the ~23-min compile is paid deliberately, once, not on real work.

### Opt-in CoreML (~2×) — manual path, with pre-warm
CoreML is a Cargo feature; the encoder `*.mlmodelc` is already provisioned beside the `.bin`
(see the CoreML naming contract above), so no extra download is needed.

```
# 1. Build the CLI with CoreML.
cargo build --features coreml

# 2. PRE-WARM once: run a throwaway transcription so the one-time ~23-min ANE compile happens
#    now and caches (into ~/Library/Caches). Use any short capture dir — the output is discarded.
./target/debug/meetscribe transcribe capture --no-store --export-dir /tmp/coreml-prewarm

# 3. From now on, `--features coreml` transcribes run at ~2× (warm start ≈ 2 s), e.g.:
./target/debug/meetscribe transcribe <some-session-dir>
```

The compile cache is keyed to the model + machine; it survives across runs and rebuilds. This is a
**doc-only** step by choice (Phase 5) — there is no `prewarm` subcommand, because the shipped daemon
never uses CoreML, so pre-warm only matters when you deliberately opt a manual run into it.

## Speaker-embedding model — 3D-Speaker CAM++ (speaker identity)

Provisioned by the same script into `models/speaker/` (gitignored: `/models/speaker/`), from the
sherpa-onnx maintainers' bare-ONNX export, sha256 read from the HF git-lfs pointer at fetch time:

| Artifact | On disk | Size | sha256 |
|---|---|---|---|
| `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` (repo `csukuangfj/speaker-embedding-models`) | `models/speaker/…onnx` | 28.3 MB | `aa3cfc16…` (full value printed by `provision.sh` on verification) |

- **Input** is an 80-bin Kaldi fbank `(1, T, 80)` — 25/10 ms, dither 0, per-utterance mean
  subtraction, samples in [-1, 1] — computed by `knf-rs` (kaldi-native-fbank). Output is a 192-d
  embedding, L2-normalised in `src/spk.rs`. The mel high cutoff is the Kaldi default (Nyquist);
  sherpa-onnx's `nyquist − 400` is an ASR convention 3D-Speaker did not train with — do not "fix"
  the Rust path to match it (Batch D verified bit-parity against `kaldi_native_fbank` at 0.9996).
- **Why this model:** it handles the telephone-band audio a Bluetooth headset (HFP) produces well
  enough by ear — six of seven far-end groups clean, six of six cross-meeting pairs right — and it
  shares `ort` with the VAD, so there is one ONNX runtime in the binary.
- **Cost:** embedding runs at ~0.006× real time; clustering a 59-minute meeting from its recording
  takes ~14 s wall clock, dominated by decoding the WAV.
- **Provisioned:** 2026-09-17. Calibration and listening results:
  `plans/2026-09-17-batch-d-speaker-embedding-results.md`.

## Far-end diarizer — NVIDIA Nemotron 3 Diarization (split-then-name)

Two artifacts under `models/diarizer/` (gitignored: `/models/diarizer/`):

| Artifact | Made by | Source | Pin |
|---|---|---|---|
| `Nemotron-3-Diarization.q8_0.gguf` (107 MB) | `provision.sh` | HF `nvidia/Nemotron-3-Diarization` | sha256 `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`, **hard-coded** |
| `bin/nemo-speech-diar` + its ggml / NeMo dylibs | `build-diarizer.sh` | `NVIDIA/NeMo-Speech.cpp`, preset `metal-diar` | commit `97a15af` |

- **Why the sha is hard-coded** (unlike the other models, which read it from the git-lfs pointer at
  fetch time): these are the exact bytes the blind listening evaluation used
  (`plans/2026-09-26-nemotron-split-naming.md`); a pointer read would accept an upstream re-upload.
- **Why a source build:** the only NeMo-Speech.cpp release (v0.1.0) rejects this model
  (`pre_ln transformer variant is not supported`). Builds are not byte-reproducible, so the commit is
  pinned and the build is verified by behaviour — `build-diarizer.sh --verify <wav> <rttm>` re-diarizes
  a known recording and diffs the RTTM.
- **How it runs:** as a subprocess of the transcribe pipeline on each far-end roll (16 kHz mono PCM16
  WAV written under the session dir, deleted afterwards), Metal backend. Measured 23.6× real time
  interactively, ~12.6× under the daemon's Background QoS. Any failure falls back to today's pipeline.
- **License:** the model is under the OpenMDW License Agreement v1.1 (commercial use permitted);
  NeMo-Speech.cpp is Apache-2.0 and ggml MIT — see `THIRD-PARTY-NOTICES.md`.
- `install --diarizer-model <gguf> --diarizer-bin <dir>` places both under `~/.meetscribe/models/diarizer/`
  (the bin dir is copied, never symlinked, so `@executable_path` finds the dylibs).

**Provisioned (ASR):** 2026-07-19.
