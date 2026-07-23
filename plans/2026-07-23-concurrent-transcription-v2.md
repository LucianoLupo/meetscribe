# Concurrent (real-time) transcription — plan v2

*Supersedes `2026-07-22-concurrent-transcription.md`. Authored 2026-07-23 from a
review → adversarial-verify → plan → judge workflow (`wf_a184fb15-480`, 13 agents, 1.7M tokens).
Every one of the 4 prior audit blockers and the SIGILL root cause was verified against source.
Judge verdict: **REVISE-MAJOR — a scope re-stage, not a redesign.***

## TL;DR — the decision

- **If** we build full streaming, the design is settled: **tail-the-growing-WAV**, a separate
  low-QoS worker *process*. Unanimous, adversarially confirmed.
- **But don't build the full stack yet.** The cheap, high-value core (**B0 + a detached
  at-hangup worker**) captures most of the value *and* fixes a confirmed unnamed bug, at a
  fraction of the cost. Build that, **measure**, then gate the streaming subsystem on the
  measurement.

## B0 measurement outcome (2026-07-23) — latency hypothesis REFUTED

B0 shipped and was measured on a real 180 s two-channel capture (before = pre-B0 installed binary,
after = B0 release binary, identical input):

- `create_state` churn **11 → 1** (one long-lived state) — confirmed.
- Transcript output **byte-for-byte identical** before vs after (no cross-window bleed).
- **Wall time unchanged: 18.7 s → 18.2 s.** Eliminating the per-window state churn did **not** cut
  latency. The cost is whisper's **fixed per-window encoder** (each `full()` runs the ~30 s-mel encoder
  regardless of window length), not state creation.
- **Measured RTF is already ~0.20x** → a 13-min meeting ≈ 2.5 min, a 58-min ≈ 12 min post-hangup.
- **The pad-to-1000 ms idea was found inert and removed** (adversarial `/review-branch`): whisper's
  floor is ~100 mel frames = **16 040 samples**, so padding to 16 000 (99 frames) still trips the
  too-short guard and returns empty — which is exactly why before/after were byte-identical. Short
  windows stay **skipped**, matching the known-good pre-existing behavior. Recovering them would mean
  padding past 16 040 **and** accepting whisper hallucination on trailing silence — a separate feature,
  not B0.

**Implications:**
- The "self-inflicted ~40 min / RTF ~3x" premise is **wrong** — latency is already fine.
- The streaming subsystem's headline prize ("transcript instant at hangup") is therefore **dropped**.
- B0 ships as **state-reuse (churn removal) only**; its value is churn/clarity, **not** speed and
  **not** crash-hardening. The real crash fix is the process boundary (B2), not any pad.
- **Revised v1 = B0 (done) + B2 (detached at-hangup worker)** for the two remaining real prizes:
  SIGILL containment and the back-to-back-meeting deafness fix. **Phase 2 (tail-the-WAV streaming) is
  cut** unless a future need reopens it.

## Ground-truth corrections (things we believed that the review disproved)

1. **"749 whisper_init_state cycles" was a misread.** It's **107** Metal init/free cycles
   (61 mic + 46 system VAD windows = *all* windows); the crash is on the **final** window,
   after all audio is decoded — **not** cumulative init exhaustion. 749 was a log-line count.
2. **The SIGILL is uncatchable in-process.** `exit 132` = SIGILL is a native illegal-instruction
   trap, **not** a Rust `Err` and **not** a ggml `abort()` (that would be SIGABRT/134). So an
   **in-process CPU-fallback retry cannot work** — this reverses my earlier "guard + CPU fallback
   fixes it" claim. A **separate worker process is required** for containment, not optional.
3. **"Crash-loops the live daemon" was an overstatement.** Verified: it's a *single* crash +
   clean launchd auto-relaunch (~5 s), one transcript lost, **no loop** (the app has released the
   mic by transcribe time, so the poison session is never retried).
4. **WAV header offset is 68 bytes, not 44** (32-bit float forces `WaveFormatExtensible`), and the
   `data`-length field stays literally `0` for the *entire* recording (hound backfills only on
   finalize/flush). Any tailer must derive `n = (stat_size − 68) / 4` and **must not trust the header**.
5. **"Reuse the liveness raw-PCM read" was false** (the 5th "reuse" mischaracterization). Liveness is
   an in-memory ring watchdog (`session.rs:98-115`); no WAV file is read during capture. The
   byte-offset tailer is **net-new** code.
6. **"0 dropped frames" is a soft invariant** — ring headroom (~21 s, `capture.rs:129`) + a fast
   non-blocking drain, not a blocking-write guarantee. Anything that stalls the drain past ~21 s drops.
7. **Confirmed, previously-unnamed bug:** the daemon is single-threaded and **blocks** in
   `transcribe_and_store` (~40 min) at `daemon.rs:367`, during which it does **not** poll the
   detector (`daemon.rs:240`) → **a back-to-back meeting is missed.** The detached at-hangup worker
   fixes this for free.

## Blocker status (all verified against source)

| # | Status | Resolution |
|---|--------|------------|
| **B1** merge parity | drop the gate | Byte-identity is the wrong bar — but divergence comes **only** from VAD boundary placement (fresh state + greedy decode ⇒ no cross-window context), not "independent decode context." Gate = fuzzy word-agreement (token-IoU/WER ≥ 0.85) vs the batch baseline + a no-truncated-seam-window structural check. |
| **B2** DB upsert | add **no** DB schema | Confirmed: `db.rs` has only `insert_meeting`; `init_schema` is `CREATE … IF NOT EXISTS` with no migration runner / `user_version` / UNIQUE. Any new column or UNIQUE is a **silent no-op on the live populated DB** (lands only on a fresh DB; in-memory tests hide it). Crash artifact = filesystem `transcript.partial.jsonl`; keep the single existing `insert_meeting` at finalize. Idempotency latch is a **filesystem** marker, never a DB constraint. |
| **B3** VAD on capture thread | neutralized by design | Real hazard (onnx + resample CPU on the drain loop can overflow the ring). Removed outright by running **all** VAD/whisper in the worker process. (Note: silero *streams* — it's not "offline" — but off-process is the point.) |
| **B4** seal race | eliminated (common case) | `discover_channel` keys on `.exists()` and the live WAV exists with a 0-length header throughout — the exact race. Tailer keys on **stat-size + `segments.txt`**, never existence. Single growing WAV ⇒ no seal ⇒ B4 gone; residual rate-roll seal handled via stat-size. APFS read-while-append returns the flushed prefix untorn; finalize's header rewrite (bytes 4 & 64) is disjoint from PCM ≥ 68. |

## SIGILL crash — root cause & fix

- **Root cause (medium confidence):** content-triggered native Metal trap on a **degenerate final
  short window** — not exhaustion, not a version-independent whisper-rs bug. Evidence: 2/2 identical
  crashes after the last of 107 windows; sessions 5–12 ran 453–731 windows on the same binary and
  stored fine; 24 earlier sub-1000 ms windows survived — only the last crashed.
- **REQUIRED fix = process boundary** (worker subprocess). The only real containment.
- **SUPPORTING (exposure-reduction, unproven prevention):** one long-lived `WhisperState`
  (kills the 107-cycle Metal churn; measured to **not** affect latency), and CPU-fallback as a
  **fresh-process** retry of the poison chunk. ⚠️ Padding short windows past whisper's floor
  (≥ 16 040 samples) does **not** help containment and risks hallucination — dropped from B0. The
  NaN/Inf-sanitize idea is untestable (crash WAV deleted) — do **not** report B0 as "fixed."
  If a fresh SIGILL reproduces, preserve the WAV and scan the final tail slice first.

---

## Phase 1 — build now (v1)

### B0 — Reuse one whisper state **and MEASURE** · `asr.rs`, `pipeline.rs` — ✅ DONE
Make `Asr` own **one** long-lived `WhisperState` instead of `create_state()` per window
(whisper-rs 0.13.2: `WhisperState` is `Send`/`Sync`, `full()` takes `&mut self` — this is an `Asr`
API change touching both callers, not a drop-in). Short-window handling is **unchanged** (skip
< 100 ms; the pad idea was found inert — see the measurement outcome above — and removed).
→ **verify (done):** `cargo build && cargo test && cargo clippy` clean; before/after transcripts
byte-for-byte identical; measured RTF ~0.20x, wall unchanged → latency premise refuted.
→ **GATE (the whole point of B0):** time a real ~13-min meeting end-to-end and re-attempt the SIGILL
repro. This one measurement decides whether **any** of Phase 2 is worth building.

### B1 — Worker subcommand (batch mode) + `chunk_secs` scaffolding · `config.rs`, `main.rs`, `worker.rs`
Add `chunk_secs: f64` (default `0.0` = off) to `[daemon]` as an additive `serde(default)` field
(version stays 1 — safe, unlike the DB). New `src/worker.rs::run_worker`; `meetscribe worker <dir>`
runs the existing pipeline path (mirrors `run_transcribe` flags).
→ **verify:** build/test/clippy; config round-trip test stays green; `meetscribe worker <fixture>`
produces a transcript.

### B2 — **Detached** at-hangup worker (containment + deafness fix) · `daemon.rs`
Replace the in-process `transcribe_and_store` at `daemon.rs:367` with a **spawn-and-return** of
`worker <dir>` (`std::env::current_exe()` → the signed `~/.meetscribe/bin/meetscribe`); the daemon
returns to **Idle immediately** (do **not** `.status()`-block), and the worker persists via the
existing `insert_meeting` itself. A worker crash (incl. 132) is invisible to launchd KeepAlive
(tracks only the daemon pid) → logged, contained. Add SQLite `busy_timeout` and a **1-worker
concurrency cap** (two overlapping workers = ~6 GB RSS + `SQLITE_BUSY`).
→ **Delivers:** SIGILL containment · fixes the confirmed back-to-back-meeting deafness · transcript
a few minutes post-hangup.
→ **verify:** build/test/clippy; reap-contract unit test (`/bin/sh -c 'exit 132'` → logged, not
propagated); `daemon --once` against a live short session returns to Idle with the transcript landing.

---

## Measurement gate — decide before Phase 2

After B0+B2, with numbers in hand:
- If B0 drops a 13-min meeting from ~40 min to a few minutes, **and** the detached worker already
  gives containment + the deafness fix + transcript-a-few-min-later, then the entire streaming stack
  buys only **a few minutes of latency** on a background tool the user reviews later, plus
  partial-transcript recovery for a **~1-in-12** poison/crash event.
- **Default expectation: stop at v1.** Build Phase 2 only if a *measured* latency or recovery need
  survives B0.

---

## Phase 2 — DEFERRED, measurement-gated (tail-the-WAV streaming)

*Design is settled and sound; the question is only whether the ROI clears the bar above.*
Two real gaps the original plan under-scoped **must be closed before building B5:**
(a) a **stateful cross-call resampler** (buffer the remainder across `read_new()` calls — a fresh
`FftFixedIn` per pass is not sample-accurate and injects a seam transient); (b) a worker-side
**"WAV stopped growing for N s" backstop** + a daemon-side `wait` timeout, so a missing
`capture.done` sentinel can never deadlock the daemon.

- **B3** periodic `WavStream::flush()` every ~2 s (not per 50 ms tick) — crash-durable WAV.
  ⚠️ runs on the drain thread → **mandatory live 0-dropped re-verify is the gate.**
- **B4** byte-offset raw-PCM tailer (`tail.rs`) — walk to the `data` chunk (offset ~68), derive `n`
  from stat size, floor reads to a 4-byte boundary, key seals on stat-size + `segments.txt`.
- **B5** streaming worker + `partial.jsonl` + spawn-before-capture + promote (feature lands).
  `StreamingVad` = persistent `VadSession`, coalesce gap 800 ms, force-seal at `chunk_secs`.
  Finalize = **promote** `partial.jsonl` (never re-run whisper in the daemon). Rate-rolled sessions
  fall back to batch-at-hangup in the child (out of streaming scope).
- **B6** startup recovery sweep — promote orphan `partial.jsonl` (present ∧ no `.persisted` ∧ no
  matching `source_dir` row) via a `worker --promote-only` child.
- **B7** ~~live tray progress~~ — **CUT** for v1/v2 (pure nice-to-have).

### Resolved open decisions (if Phase 2 proceeds)
- **chunk_secs:** ship `0.0` (off → at-hangup, containment only); flip default to `30.0` **only**
  after the live 0-dropped re-verify. Floor: `(0, 1.0]` → warn + disable.
- **during-call vs at-hangup:** during-call (spawn *before* capture) when `chunk_secs > 0`;
  at-hangup when `0`. **Both out-of-process** — that's the containment.
- **parity bar:** fuzzy word-agreement, not byte-identity. Run the B0 determinism probe first
  (batch twice on one WAV, diff `transcript.json`) to know if the baseline is even stable.

## Verification harness
- **0-dropped must hold after B3** with the worker running concurrently — the actual risk; the gate.
- `kill -9` mid-call → `partial.jsonl` up to the last sealed chunk survives + resumes.
- B0 determinism probe (above) before choosing the parity gate.
- Finalized transcript vs batch baseline: fuzzy word-agreement ≥ 0.85, no truncated seam windows.

## Top risks
- **B3 flush on the drain thread** can stall past the ring's ~21 s headroom → dropped frames. 2 s
  interval + tiny writes; the live re-verify is the gate.
- **B0 guards are exposure-reduction, not proven prevention** — containment (B2) is the guarantee.
- **Incremental resampling seams** (Phase 2) — anchor absolute time to cumulative native samples;
  covered by the fuzzy gate, not byte-identity.
- **current_exe under launchd** → the signed TCC-granted binary; a dev run uses the debug binary
  (fine, just be aware daemon and worker are always the same on-disk binary).
