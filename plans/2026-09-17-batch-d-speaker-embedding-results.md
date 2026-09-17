# Batch D results — speaker-embedding calibration spike

**Date:** 2026-09-17 · **Plan:** `2026-08-05-speaker-identity-and-vocabulary.md` §D
**Branch:** `spike/speaker-embeddings` · **Probe:** `src/bin/spk_probe.rs`

## Verdict — NO-GO on the current recordings, GO once the input device changes

The embedding pipeline is correct. The recorded audio is telephone-band, and that alone
explains the miss. Fixing it costs nothing: stop using the Bluetooth headset's microphone.

| Exit criterion | Target | Measured | Result |
|---|---|---|---|
| 1. EER, single speech window vs single window | < 5 % | **26.6 %** | ✗ |
| 1b. EER, enrolled voice (10 meetings) vs a new meeting's average | — | **4.9 %** | borderline |
| 2. Same-speaker similarity across codec change | no collapse | 16↔48 kHz mean 0.33 vs 16↔16 0.41 | inconclusive (5 built-in-mic meetings, ~0 `you` windows) |
| 3. Embedding cost beside whisper | negligible | embed RTF **0.006×**; whole stage excl. whisper 0.004× | ✓ |
| 4. Coalesced-window purity | low blended fraction | 9.6 % of long `others` windows fall below the single-speaker baseline | ✓ (segmentation not required for v1) |

Corpus: 133 session dirs walked (4 empty), 112 with ≥ 5 windows on both channels, first
15 min per channel, 12 windows per channel ≥ 1.5 s, 2,796 windows embedded, 59 h of audio
decoded. VAD dominates wall-clock (635 s), embedding is 256 s for 12 h of windows.

## Why the number is bad — and why it is not the code

**The audio has no energy above 4 kHz.** Measured on both channels of several meetings:
0.1 % of spectral energy in 4–6 kHz, 0.01 % in 6–8 kHz. When a Bluetooth headset's mic is the
input device, macOS puts the whole headset into the HFP call profile, which is narrowband in
*both* directions — the far end (`system.wav`) is degraded too. 131 of 136 meetings were
recorded this way (16 kHz aggregate). The one built-in-mic session with music on the far end
shows 58 % of its energy above 8 kHz, so the A2DP path is full-band as expected.

The speaker model (3D-Speaker CAM++, VoxCeleb-style training, 16 kHz full-band) loses most
of its discrimination on telephone-band speech. Same-speaker windows average cosine 0.42 vs
0.22 for different speakers — separated, but with heavy overlap.

**Pipeline correctness was proven three ways:**

1. **Synthetic voices** (macOS `say`, Spanish sentences): same voice 0.91–0.96, different
   voice down to 0.36 — the expected shape on clean audio.
2. **sherpa-onnx parity** (independent C++ fbank + ORT): on real 5 s mic slices it gives the
   same numbers as our path (you-you within 0.60 vs 0.59; you-others 0.14 vs 0.20).
3. **Front-end bit-parity:** Rust `knf-rs` vs Python `kaldi_native_fbank` with the same options
   → cosine 0.9996 on a 23 s clip.

One trap found on the way: sherpa's `FeatureExtractorConfig` defaults the mel high cutoff to
nyquist − 400 Hz (an lhotse/ASR convention). With that option our first cross-check read 0.91;
with the Kaldi default (nyquist) the two agree at 0.9996. 3D-Speaker trains with the Kaldi
default, so the Rust path is the faithful one — do not "fix" it to match sherpa.

## What averaging buys (same corpus, same embeddings)

| Comparison | EER |
|---|---|
| window vs window | 26.6 % |
| meeting-average vs meeting-average | 11.0 % |
| enrolled on 3 meetings vs new meeting's average | 5.9 % ± 2.1 |
| enrolled on 10 meetings vs new meeting's average | 4.9 % ± 1.5 |
| enrolled on 10 meetings vs a single window | 13.3 % ± 1.4 |

So on narrowband audio, "who is the one far-end voice in this 1:1 call" is borderline
workable after enrollment; "which of three far-end voices said this segment" is not.

## Decisions

- **Model:** `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` (192-d), provisioned
  sha-verified via `models/provision.sh` into `models/speaker/` (gitignored). Input is
  **80-bin Kaldi fbank `(1,T,80)`**, not raw audio; the plan's "16 kHz mono in" referred to
  the sample rate only.
- **Front-end:** `knf-rs 0.3.2` (kaldi-native-fbank via cmake) — 25/10 ms, 80 bins, dither 0,
  povey window, per-utterance mean subtraction, samples in [-1, 1]. Matches the model's
  metadata (`normalize_samples=1`, `feature_normalize_type=global-mean`).
- **Deps:** `ort = "=2.0.0-rc.10"` (exact pin = silero's), `ndarray = "0.16"`, `knf-rs`.
- **Provisioning:** option (a) from the plan — `download_verified` now takes `(repo, name, out)`
  and still reads the expected sha from the git-lfs pointer at fetch time.

## Next

1. **Input device → MacBook built-in mic** (System Settings → Sound → Input), headset stays as
   output. Both channels become full-band. The daemon already handles 48 kHz capture.
2. Record 2–3 meetings that way, then `spk_probe` restricted to those sessions. Gate: window
   EER < 5 % on full-band audio.
3. On green: Batch E (migration 2, `speakers`/`voiceprints`, per-meeting clustering, enrollment)
   **plus** `speakers play <meeting> <cluster>` — extract a few seconds of that cluster to a WAV
   and `afplay` it, so a voice can be heard before it is named. Many voiceprints per speaker,
   never one frozen centroid; low-confidence windows stay unassigned.
4. Old narrowband meetings can still be labelled retroactively at the *meeting* level (one far-end
   voice per 1:1 call) once names exist — the 5 % enrolled-vs-meeting number is what applies.
