# eval/

Harness for the Nemotron split-then-name change (`plans/2026-09-26-nemotron-split-naming.md`).

The data it reads — meeting audio, voiceprints, blind-listening keys — is private and lives outside
the repo in a regression set (default `~/.meetscribe/eval/nemotron-split/`). Nothing here contains it.

- `rulec_ref.py ref <meeting>` — reference names for every Nemotron-split window (rule C).
- `rulec_ref.py parity <meeting> <transcript.json>` — time-weighted agreement of a meetscribe export
  with the reference; exits 1 below the 99 % gate (Step 4).
- `src/bin/split_probe.rs` — the frozen reference probe that produced the regression set's A/B
  outputs (path-includes `src/spk.rs`; never make it include code under test).
- `src/bin/ts_probe.rs` — plain vs DTW Whisper token-timestamp probe (audit, 2026-09-27).
