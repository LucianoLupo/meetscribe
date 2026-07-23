# Concurrent (streaming) transcription — plan

**Goal:** transcribe *during* the call and persist per-chunk, so (a) the transcript is essentially
ready at hangup and (b) a daemon crash mid-call preserves everything up to the last sealed chunk.

**Prize:** crash resilience. Today a crash at minute 80 keeps the WAVs but loses the transcript.
Latency (transcript ready at hangup vs +40 min) is the secondary win.

## Non-goals (explicit scope fence)
- **No live in-call transcript UI.** No new window/panel. Tray gets at most a progress line.
- No re-architecting capture. `run_capture` stays the single-threaded Core-Audio owner.
- No cloud / no streaming ASR model swap. Same local whisper large-v3.

## Constraint that dictates the design
`status.rs` states the intent: the daemon stays single-threaded with **no** run loop competing with
its realtime Core-Audio listeners. Whisper pinning GPU + ~160% CPU *inside* the capture process is
exactly the neighbor that drops frames. Current guarantee = "0 dropped over 85 min." We must not
trade a reliable (unrecoverable) WAV for a faster (always re-derivable) transcript.

→ **Transcription runs in a SEPARATE process at low QoS**, never a thread in the daemon. A whisper
crash then cannot take down capture, and the OS scheduler keeps it off the audio thread's back.

## Design (reuses existing multi-segment merge)
1. **Seal on a timer, snapped to silence.** In `session.rs`, every `chunk_secs` (default 300s / 5min)
   roll the current mic+system pair to a fresh segment — but only at the next VAD silence boundary
   (silero already in `vad.rs`) so we never cut mid-word. Append a `segments.txt` line with `gap=0`
   (contiguous). This reuses the exact rate-roll machinery already at session.rs:169-207.
2. **Worker drains sealed segments.** A new low-QoS subprocess (`meetscribe transcribe-worker <dir>`)
   watches `<dir>` for sealed segment pairs, transcribes each with the existing per-segment path
   (pipeline.rs:86-121), and appends offset-corrected segments to `transcript.partial.jsonl` + upserts
   a DB row. Only *sealed* segments (index < current) are eligible — the live segment is never read,
   so writer and reader never touch the same file.
3. **Finalize at hangup.** `record_and_process` seals the trailing partial, waits for the worker to
   drain it (short — one ≤5-min chunk), then runs the existing merge/export to produce the canonical
   `transcript.md`/`.json`. Near-instant instead of ~40 min.
4. **Crash recovery.** On restart, worker resumes from `transcript.partial.jsonl` (skip segments
   already recorded by max index). Idempotent upsert keyed on (session_dir, segment_index).

## Prerequisite
The `Transcribing` status fix (built, tested, uncommitted) must land first — the daemon state machine
this builds on assumes that transition exists. Commit + install it, THEN build streaming on top.

## Batches (each ends with tsc-equivalent: `cargo build && cargo test && cargo clippy`)
- **B0** — land the `Transcribing` fix: commit, reinstall daemon, restart tray. (prereq)
- **B1** — timer+VAD-boundary segment sealing in `session.rs` (+ unit test on boundary snap).
- **B2** — `transcribe-worker` subcommand: drain sealed segments → `transcript.partial.jsonl`.
- **B3** — daemon spawns/monitors the worker; finalize path waits + merges.
- **B4** — crash-resume + idempotent DB upsert; tray progress line (`◐ transcribing N/M`).

## Open decisions (need Luciano)
- **Chunk size:** 5 min default? Shorter = faster crash-recovery granularity + faster hangup, but
  more whisper spin-ups and slightly worse quality at boundaries. 5 min feels right; confirm.
- **Concurrency:** run the worker *during* the call (max crash resilience, accepts a low-priority GPU
  neighbor) vs only at hangup (zero in-call contention, but then it's not truly "at the same time").
  The user said "at the same time" → run during the call. Confirm we accept the scheduling risk,
  mitigated by low QoS + sealed-only reads.

## Verification harness
- Real-call: 0 dropped frames must hold WITH the worker running concurrently (the actual risk).
- Kill -9 the daemon mid-call → partial transcript up to last sealed chunk survives + resumes.
- Finalized transcript is byte-identical to today's batch path on the same WAVs (merge parity).

---

## AUDIT OUTCOME (2026-07-22, /audit-plan, 4 auditors) — verdict: REVISE-MAJOR
Full JSON: workflow run wf_1afb57e2-f8d. The design DIRECTION is validated (process separation is
right for this codebase). But my "reuses existing X" framing was wrong 4 times — this is really
THREE new subsystems, not mechanical reuse. Blockers to fix before any code:

1. **Merge-parity gate is unachievable.** Chunks are whisper-decoded independently (fresh decode
   context + silero coalescing differ at every seam) → cannot be byte-identical to a single-buffer
   pass. Transcript CONTENT changes slightly vs today. Drop the gate → tolerance/equivalence bar.
2. **Per-chunk DB upsert doesn't exist.** db.rs has ONLY `insert_meeting` + `CREATE TABLE IF NOT
   EXISTS` — no upsert, no UNIQUE key, no migration; a schema add is a no-op on the live populated
   db. Fix: drop per-chunk DB entirely → `transcript.partial.jsonl` is the sole crash artifact, one
   `insert_meeting` at finalize. Zero schema change. (Dissolves the B2→B4 ordering inversion too.)
3. **Silero on the capture thread violates the founding constraint.** `vad.rs::speech_windows` is an
   OFFLINE whole-buffer API needing resample-to-16k + onnx — the exact frame-dropping neighbor lines
   14-22 forbid. Fix: cheap RMS/energy gate from WavStream's existing sum_sq/peak, off the realtime
   thread. (Or: don't seal on the capture thread at all — see redesign note below.)
4. **Writer→reader seal contract races.** `discover_channel` keys on file EXISTENCE, but the live
   WAV exists with a placeholder hound header until `finalize`. Fix: seal via a `segments.txt` line
   appended only AFTER both WAVs finalize; worker trusts only index < live.

Plus: manifest token is `gap_frames=` not `gap=`; gate sealing behind `chunk_secs: Option<Duration>`
(CLI `--seconds`/interactive path must keep single-mic.wav shape); extract `seal_and_roll` + a
`transcribe_segment(&Asr,…)` with ONE long-lived Asr (don't reload 3GB/chunk); worker lifecycle +
single-writer lock + logged-and-continue (launchd KeepAlive crash-loops on non-zero exit) + low-QoS
(setpriority/posix_spawnattr); re-apply `min_secs` gate at finalize+resume; new Status fields must be
`Option<T>` + `#[serde(default)]`; wire `chunk_secs` into `[daemon]` config. B0 is stale (fix is
already committed at ed37c80 + installed onto live daemon pid 8060 — done this session).

### The finding that matters most: scope/ROI was NEVER audited
The scope-roi auditor returned a placeholder ("test"/"a"/"b") — it failed. So the one question that
should gate this — *is it worth it?* — went unanswered. My read: today's daemon already produces the
transcript ~40 min after hangup, unattended, and you are never sitting waiting on it. The prize here
is (a) transcript ~instant at hangup + (b) survives a mid-call daemon crash — insurance against a
rare event on a daemon that's been reliable. Cost: 3 new subsystems + real risk to the ONE guarantee
that matters most (0 dropped frames; the WAV is unrecoverable, the transcript always re-derivable).

### Cleaner alternative surfaced by the audit's own constraint
Instead of sealing on the capture thread, leave capture UNTOUCHED (one WAV, as today) and have the
low-QoS worker **tail the growing WAV's raw PCM** from a tracked byte offset (exactly the raw-PCM
read we already did for liveness this session), applying its energy gate + whisper in its OWN
process. This removes blockers 3 and 4 outright (no capture-thread DSP, no seal contract) and
trivially preserves the 0-dropped guarantee. Worth considering before committing to the sealing design.
