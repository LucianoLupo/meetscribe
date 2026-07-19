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

**Provisioned:** 2026-07-19.
