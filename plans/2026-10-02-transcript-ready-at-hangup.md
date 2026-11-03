# Transcript ready at hangup — plan (rev 2, after /audit-plan)

**Decision:** keep capture untouched. Run transcription in a **child process** whose priority is
**raised when no call is live and set to Background during a call**.
- **Phase 1** alone should take an 82-min call's transcript from **~52 min to ~22 min**, and it fixes
  the 2026-10-01 abort regression.
- **Phases 2–4** (a child that tails the growing WAVs while the call is live) take it to **~4–5 min**.
  They are decided on the numbers measured after Phase 1.
- Parity bar: **byte-identical** stored output against frozen goldens.

Supersedes `2026-07-22-concurrent-transcription.md` and `2026-07-23-concurrent-transcription-v2.md`,
adopting their tail-the-growing-WAV design. v2 dropped the hangup goal based on RTF ≈0.20 measured
from the **CLI**; the daemon runs **2.4× slower** at `ProcessType=Background` (measured below), which
v2 never measured. v2's parity premise ("fresh state + greedy ⇒ no cross-window context") no longer
holds since its own B0 (`edc884e`) reused one `WhisperState` (see 2b). The founding constraint stays:
capture is the single-threaded Core-Audio owner, and nothing heavy shares its process.

Audit: `/audit-plan` run `wf_399e5a73-485`, verdict revise-major, all deltas applied below.

## Why (measured 2026-10-01/02)

| Meeting | Hangup → stored (daemon today) | Diarizer alone |
|---|---|---|
| 20 min | 15 min | 46 s |
| 76 min | 40 min | — |
| 82 min | 52 min | 219 s |

- Same 20-min session copy, same binary:
  - normal priority **304 s (RTF 0.25)**;
  - `taskpolicy -b` (Background) **742 s (RTF 0.62)**, i.e. **2.4× slower**;
  - output byte-identical.
- The LaunchAgent sets `ProcessType = Background` (`launchd.rs:155`); the daemon runs at priority 4.
- Debug == release (304 s each), with identical output. The build profile is not a factor.
- That determinism holds only for the same input **in the same decode order**. whisper.cpp seeds
  decoder 0's RNG once per state (`whisper.cpp:3422`; the per-call reseed starts at `j=1`, `:5482`),
  and greedy temperature fallback (`temperature_inc` 0.2, `:4753`) samples from it. With one shared
  state, a fallback window's text depends on every earlier decode on both channels.
- 33 of 33 logged captures ran at 16 kHz, and 0 of 199 sessions has rolled (`*.001.wav`).

## Is it worth it
- Today transcripts arrive 15–52 min late, all day (2026-10-01: 8 meetings).
- If the hangup→stored numbers after Phase 1 satisfy the user, **stop after Phase 1**.
- Phases 2–4 are the remaining ~22 → ~4 min.

## Non-goals
- No live captions and no new UI.
- No change to `session.rs` / `capture.rs` (the 0-dropped guarantee).
- No ASR model change.
- No mid-call resume of partial work: a restart re-runs batch from the `pending` marker.
- Rate ≠ 16 kHz or rolled sessions: no live mode; batch at hangup.

Every phase ends with `cargo build && cargo test && cargo clippy --all-targets`. No `cargo fmt`:
master is not fmt-clean.

## Phase 0 — Measure (done 2026-10-02; ships nothing)
- Background vs normal is measured above (2.4×).
- The priority change ships in Phase 1, on the child. Shipping `ProcessType=Standard` now would make
  the in-process whisper from `8d390c4` compete with the call app while the abort regression is live.

## Phase 1 — Transcription in a child process
- **Regression being fixed:** `8d390c4` runs whisper on a thread inside the daemon, concurrent with
  capture. A whisper.cpp `abort()` (one already happened: `34a8290`) now kills a live recording.
- **Base:** further commits on `fix/transcribe-off-capture-loop` (`8d390c4` and `0bcd970` are
  unpushed; origin/master = `2271de9`). Merge it as one PR; no stacked PR.
- **Command line:** `fn child_args(&PipelineOpts) -> Vec<OsString>` emits every field:
  - `<dir>`, `--title`, `--db`, `--model`, `--export-dir`, `--lang`, `--speaker-model`,
    `--diarizer-bin`, `--diarizer-model`;
  - `--no-split` when `!split`;
  - absolute paths from `DaemonConfig`, with `current_dir` set to the session dir.

  Extract `parse_transcribe_args(argv) -> (PathBuf, PipelineOpts)` from `run_transcribe`. Reason:
  the CLI defaults are `es`, split on and repo-relative models, and unknown flags are silently ignored.
- **Spawn like `diar.rs`:**
  - stdin and stdout go to `Stdio::null()`, because `run_transcribe` prints the whole transcript.
  - stderr is inherited (launchd's err log).
  - Kill and reap on Drop.
  - Stay in the daemon's process group (no `process_group(0)`): launchd kills the group when the
    job dies, which prevents an orphan from storing while the marker survives (a duplicate meeting).
- **Marker:** a pure function `marker_after(outcome, shutting_down) -> Keep | Clear`:
  - exit 0 → Clear;
  - any non-zero exit or signal while `SHUTDOWN` is set → Keep (the next start retries);
  - crash or non-zero exit otherwise → log, keep the WAVs, Clear.
- **Priority:** the agent stays Background.
  - The child runs **raised** while no call is recording: `setpriority(PRIO_DARWIN_PROCESS, pid, 0)`,
    i.e. `taskpolicy -B`.
  - It runs **Background** while a call records, switched at recording start and end.
  - Lowering is always allowed. Raising a child of a Background agent must be shown under launchd. If
    it cannot be done, fall back to the agent at `ProcessType=Standard` plus the same lowering during calls.
- **Tests:**
  - an aborting fake child injected through `spawn_transcriber`'s `run` → the daemon is unaffected and
    the marker is cleared;
  - the `marker_after` table, including (SIGTERM, shutting_down = true) → Keep;
  - a `child_args` → `parse_transcribe_args` round trip, with split on and off.
- **Verify under launchd** (not the CLI):
  - the raised child reaches RTF ≈0.25 on the 20-min copy;
  - a call recorded while a backlog child runs still logs `0 dropped`;
  - hangup→stored is recorded for the next real meetings, which feeds Open decision 2.

### Phase 1 — as built (2026-10-02), deviations measured under launchd
- **The parent cannot raise a child.** `setpriority(PRIO_DARWIN_PROCESS, child, 0)` from a
  background parent left the child at priority 4 with no throughput gain. Only the process itself
  can raise its priority.
- **A Background agent cannot raise itself either.** A test LaunchAgent at `ProcessType=Background`
  shows `spawn type = background (5)`, and a self-raise left every thread at 4 (voice detection alone
  took over 2 min, versus 7 s at normal priority).
- **So we use the fallback:** the agent runs at `ProcessType=Standard` (`spawn type = daemon`).
  - The child gets `--yield-to-calls <base>` and polls `status.json` every 2 s
    (`status::spawn_call_yield`).
  - It **lowers itself** to Background while the state is `recording`, and goes back to normal otherwise.
  - The daemon never calls `setpriority`.
- **Measured (test LaunchAgent, Standard, 20-min copy):**
  - RTF **0.262** (CLI 0.254), byte-identical transcript.
  - Status switched to recording → `priority: background` within 3 s; switched back → `normal`.
- **Self-lowering throttles CPU about 12×** (priority 31 → 4). A one-minute background window only
  added about 6 s to the run, because whisper's heavy work is on the GPU and the voice-splitter
  subprocess spawned before the switch keeps the priority it inherited. **Known gap:** GPU contention
  with the call app is only partly reduced; the real-call check decides it.

## Phase 2 — Parity foundation + incremental engine (only if Open decision 2 says go)
- **2a Goldens, before any refactor:**
  - Use the pre-refactor binary on ≥5 real sessions (20–82 min).
  - Each is a **stored** run into a fresh copy of the real DB under `~/.meetscribe/eval/live-parity/`.
  - Keep `transcript.json` plus a normalized dump of `segments`, `voice_clusters` and
    `segment_voices`, without ids or `created_at`.
  - Run everything twice. If the two runs differ, the diarizer is non-deterministic: record one RTTM
    per session and pass `--diarizer-rttm` on every later run.
- **2b Per-channel decode state:**
  - `Asr` keeps the loaded context plus one `WhisperState` per channel, in both batch and follow mode.
    That is one extra state, not a second copy of the weights.
  - This is the **only intended output change**.
  - Pre-check: a scratch build that decodes system before mic, diffed against the goldens, proves
    the effect is real.
  - The user accepts the diff (Open decision 3); then re-freeze the goldens.
- **2c Shared steps:**
  - Extract `decode_window` (whisper or `transcribe_words` plus the window embedding), `roll_pieces`
    (`pieces_for_chunk`/`assign_words` plus piece embeddings), and the merge/voices/DB/export tail.
  - Batch keeps its loop for **every** session, including `d.start()` before the roll's whisper loop
    and `p.wait()` after it (`pipeline.rs:138` and `:182`), so the diarizer still overlaps whisper.
  - Gate: batch == golden, byte-identical.
- **2d `pipeline::Incremental`** (not "Transcriber": that collides with `spawn_transcriber`):
  - `feed(channel, samples)` and `finish()`, used only by `--follow`.
  - `feed` owns the exact 16 000-sample VAD chunking, aligned from sample 0.
  - The incremental coalescer emits a window only when it is final (the next raw interval arrived and
    did not merge into it). `finish()` flushes the open window using the batch end-of-stream rule.
  - Results go into per-channel vectors, concatenated mic-then-system before `merge_keyed` (a stable sort).
  - `finish()` runs the diarizer, then the shared steps.
- **Tests:**
  - the incremental coalescer == batch `coalesce` at every split point;
  - `Incremental` fed in 1 s / 7.3 s / whole-file steps gives identical windows (VAD only);
  - `Incremental` fed the whole file == batch windows on the golden sessions.

## Phase 3 — Follow mode: `meetscribe transcribe <dir> --follow`
- **Tail reader:** lives next to `read_wav_any_rate`, with the same mono/f32/32-bit checks, and
  returns `(samples, rate)`.
  - Parse the RIFF chunks to the `data` offset and ignore the header sizes, which hound leaves at 0
    until finalize (`write.rs:286,304`).
  - Floor to whole samples only.
  - Every chunk goes through `resample::to_16k_mono` (it clamps to [-1,1]; silero rejects
    out-of-range buffers) before `feed`.
- **`CAPTURE_DONE_MARKER`:** a `pub const` beside `PENDING_MARKER`. It is an empty file meaning "WAVs
  finalized". The child then feeds the remainder and runs `finish()`.
- **Unsupported audio:** if the WAV is not 16 kHz, or `mic.001.wav` appears, the child waits for
  `capture.done` and exits with a distinct code; the daemon runs batch (Phase 4).
- **Poll interval:** `--poll-ms` (default 5000).
- **Progress log:** about every 60 s, one info line with audio fed (s), audio decoded (s) and lag (s).
- **Replay tool (deliverable):**
  - copies a session's WAVs the way hound writes them (zero-size header, then appended data);
  - grows them in 5-s-of-audio steps, faster than real time;
  - finalizes them, then writes `capture.done`.
- **Gate:** follow (replay) == golden on all golden sessions, using the 2a stored comparison.

## Phase 4 — Daemon wiring
- **Recording start:** spawn the follow child outside the queue, at Background. Background RTF is
  0.62 < 1, and the lag line checks that it keeps up. No marker is written at spawn.
- **Capture end,** in `record_and_queue`:
  - below `min_secs`: kill the child and write nothing, as today;
  - otherwise: write `CAPTURE_DONE_MARKER`, raise the child, and call `queue_job` (which writes
    `pending` as today) with `TranscribeJob` kind `Follow(child)`;
  - use kind `Batch` instead if the child already exited or live mode was skipped.
- **Worker:**
  - for `Follow(child)`, wait for it and apply `marker_after`;
  - if the follow child failed, make one batch attempt in the same job;
  - `Activity.transcribing` stays worker-owned: one writer, no change to the status board.
- **Overlap:** a follow child may run alongside a backlog batch child. The real-call check records
  the RSS of both and `0 dropped`. Only if either check fails: SIGSTOP the batch child at recording
  start and SIGCONT it at capture end.
- **Config `[daemon] live`:**
  - `false` in the `Default` impl while it soaks;
  - documented in the template;
  - a `config.rs` test modelled on `speakers_split_defaults_on_and_is_the_rollback_switch`;
  - flip the default after real calls pass.

## Verification
- **Parity:**
  - 2a goldens: stored runs, a fresh DB copy each time, checked for determinism.
  - Phase 2 gate: batch == golden. Phase 3 gate: follow (replay) == golden.
  - Both are byte-identical on `transcript.json` and on the normalized row dump. Any diff blocks the merge.
  - `--no-store` is not enough: it skips `split::finalize`, voices and names (`pipeline.rs:281–362`).
- **Real call (Phase 4):**
  - `meeting ended` → `stored` ≤ diarizer time + 60 s;
  - `0 dropped` with the follow child running, also with a backlog batch child running (RSS of both);
  - the lag line stays bounded.
- **Isolation (Phase 1):**
  - an aborting child → the daemon keeps recording or stays idle, and the marker is cleared;
  - a child killed by SIGTERM while shutting down → the marker is kept, and the next start retries.

## Open decisions (user)
1. **Child priority.** Recommended: raised when no call is live, Background during calls. Fallback:
   the agent at `ProcessType=Standard`.
2. **Phases 2–4 go/no-go,** decided on the hangup→stored numbers after Phase 1.
3. **Accept the 2b re-baseline of batch output.** Recommended: it is what makes byte-identity hold
   by construction.
