# Nemotron split-then-name for far-end speakers

**Decision:** add NVIDIA Nemotron 3 Diarization to the transcribe pipeline to cut far-end Whisper
chunks at voice changes, then name each piece with the existing CAM++ voiceprints under **rule C**:
pieces ≥ 1.5 s (`spk::MIN_WINDOW_MS`) take their own cluster's name; shorter pieces keep the name the
whole chunk gets today. Whisper large-v3 transcription is unchanged (transcribe-then-split). New
meetings only in v1; past meetings are untouched (their manual labels are the ground truth voiceprints
come from). Runtime: the tested NeMo-Speech.cpp diarizer (commit `97a15af`) called as a subprocess.

**Branch:** `feat/nemotron-split` stacked on `spike/speaker-embeddings` (Batch E: 4 commits ahead of
`master`, not on origin, already running in the live daemon). Base decided 2026-09-27, before
creating the branch: stacked. Step 7 reviews against `spike/speaker-embeddings`; retarget to
`master` **before** merging.

**Why:** blind listening, held-out round (Brain #1312): rule C right on 14/15 clips where it
disagrees with today's labels, today 0/15, 1 unclear. Expected effect: 10–16 % of far-end speech
time relabelled in multi-person meetings; named share unchanged. Diarizer cost measured on the 45-min
meeting: 114 s interactive (≈ +2.5 min per meeting-hour) and 214 s under the daemon's Background QoS
(≈ +4.8 min). DTW and extra-embedding costs are measured in Step 3.

**Evidence trail (Brain, project `meetscribe`):** #1304 #1305 (diarizer: 5/5 on mixed chunks, fails
on identity with 7 speakers) · #1310 (ASR bake-off: Whisper stays) · #1311 (long pieces win 6/7,
short pieces lose 1/8 → rule C) · #1312 (rule C held-out 14/15).
**Audits:** `/audit-plan` 2026-09-27 ×2 — revise-major, then revise-minor; both folded in below (raw
syntheses in the regression set: `planDelta.md`, `planDelta2.md`).

**Regression set:** `~/.meetscribe/eval/nemotron-split/` (private meeting data — never in the repo).
Holds the diarizer binary that produced the evidence (97a15af, self-contained, verified to reproduce
the RTTMs byte-for-byte), the q8_0 GGUF, the 3 meetings' RTTMs, A/B windows + `split_probe` outputs,
blind keys + verdicts, a frozen DB copy, probes. See its README.

---

## Scope

**In**
1. A diarizer module returning far-end speaker turns `(t_start, t_end, local_speaker)` for a 16 kHz
   mono buffer, fail-open (any error or timeout → today's behaviour, logged).
2. Word-level timestamps from Whisper so a chunk's text can be split at a turn boundary.
3. Splitting far-end chunks into pieces + rule C naming, on top of the unchanged
   `voices::assemble` / `voices::apply_matches`.
4. Storage of pieces as ordinary `transcript_segments` rows (no schema change — §4).
5. Provisioning:
   - GGUF → `models/diarizer/Nemotron-3-Diarization.q8_0.gguf` via `provision.sh`, sha256
     **hard-coded** `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1` (reason:
     `provision.sh`'s `expected_sha` reads the LFS pointer on `/raw/main/` at fetch time, so it would
     accept an upstream re-upload — we pin the tested bytes). OpenMDW v1.1 notice in `PROVISIONING.md`.
   - New `models/build-diarizer.sh`: builds NeMo-Speech.cpp at `97a15af` (preset `metal-diar`), stages
     `nemo-speech-diar` + its 6 dylibs in `models/diarizer/bin/` with `rpath=@executable_path`. Pin
     the **commit**, not a binary sha (builds are not byte-reproducible). Verify by behaviour:
     `otool -L` shows only `@rpath` + system libs, `doctor` runs, and locally the build reproduces
     `out/20260925-164927.rttm` byte-for-byte.
   - `.gitignore`: add `/models/diarizer/` (repo is public).
   - `THIRD-PARTY-NOTICES.md`: NeMo-Speech.cpp (Apache-2.0) + the bundled ggml dylibs' licence
     (confirm at 97a15af).
6. Rollback switch + paths, reachable from CLI and daemon:
   - `PipelineOpts.split: bool`, `diarizer_model: Option<PathBuf>`, `diarizer_bin: Option<PathBuf>`;
   - CLI `transcribe --no-split`, `--diarizer-model <gguf>`, `--diarizer-bin <exe>` (flags override
     defaults, like `--speaker-model`);
   - `DIARIZER_MODEL_REL` / `DIARIZER_BIN_REL` beside `SPEAKER_MODEL_REL` (repo-relative for the CLI,
     `~/.meetscribe`-relative for the daemon);
   - daemon `[speakers]` = `SpeakerSettings { split: bool, #[serde(flatten)] extra }`, default
     `split = true`; add `push_unknown(&mut w, "speakers.", &self.speakers.extra)` to
     `collect_warnings` (sections are listed by hand, config.rs:187-190); no path keys in
     `config.toml`; commented `[speakers]` block in `DEFAULT_CONFIG_TOML` (round-trip test enforces it);
   - daemon startup: warn if either diarizer file is missing; one info line with effective `split`,
     both paths, the commit and a GGUF sha prefix;
   - install step **3c** mirroring 3b: `install --diarizer-model <gguf> --diarizer-bin <dir>` →
     `provision_model` on the GGUF; **copy** (not symlink) the bin dir to
     `~/.meetscribe/models/diarizer/bin` so `@executable_path` finds the dylibs; `create_dir_all`,
     warn-never-fail, usage string updated; no re-sign (the ad-hoc-signed copy ran unchanged under a
     Background LaunchAgent);
   - rollback = edit `config.toml` → `launchctl kickstart`. Meetings stored while split was on stay split.

**Out (explicit)**
- Mic channel (one person by construction).
- Re-labelling past meetings (`cluster --all --recluster` would wipe manual labels) — separate decision.
- Retro rule C: `speakers cluster [--recluster]` and `speakers split` on a split meeting re-assign its
  short pieces with plain `inherit_short` (no parent-chunk link is stored). Accepted for v1.
- ONNX via `ort` (former runtime A), deferred 2026-09-27: every public export leaves the mel front
  end and the streaming speaker cache to the caller; the one Rust port (parakeet-rs 0.3.8, ~1.4k
  lines) needs ort rc.13 + ndarray 0.17, our pin is `=2.0.0-rc.10`, and `ort-sys` `links =
  "onnxruntime"` means two versions cannot coexist. Revisit only if the subprocess causes packaging
  pain, or once ort moves to ≥ rc.13.
- Live/streaming diarization; the pipeline stays batch-at-finalize.
- Replacing Whisper (#1310).
- Fixing the name→Others loss (~300 s over 3 meetings from clusters under `MIN_WINDOWS`) — measured,
  accepted, revisit after a week of real meetings.

## Design

### 1. Diarizer runtime — decided: B as a subprocess
The exact binary behind #1304–#1312. The prebuilt v0.1.0 release is **not** an option: it rejects the
model (`pre_ln transformer variant is not supported`); no newer release exists.
Audit evidence 2026-09-27 — a LaunchAgent in the daemon's context (`ProcessType=Background`, HOME-only
env, `cwd=/`, minimal PATH) ran on Metal device 0, exited 0, byte-identical RTTMs:
20260922-190155 (52 min) 338 s at load average 15–28; 20260925-164927 (45 min) 214 s (the same file
interactively: 114 s, 23.6× realtime). B vendored is rejected: not the tested binary, and it would put
a second ggml beside whisper-rs-sys's ggml in one binary (likely symbol clashes; unverified).

Subprocess contract:
- write the roll's `audio16` as a `hound` 16 kHz mono PCM16 WAV **under the session dir** (daemon env
  is HOME-only — never rely on `TMPDIR`); delete it afterwards;
- invoke the binary by absolute path; RTTM to stdout or a file under the session dir;
- on failure, log its stderr;
- timeout = the roll's duration (≥ 9× the slowest measured run); on timeout kill it → `None` turns.

### 2. Pipeline order
1. `load_diarizer()` first (only if `split`), then `Asr::load(dtw = diarizer.is_some())`.
2. Roll loop, while `audio16` is alive:
   - diarize the roll → turns (+ roll offset); error/timeout → this roll gets `None`;
   - each VAD window is decoded **once**: far-end windows with turns via `transcribe_words` (text,
     confidence, words), everything else via `transcribe` (unchanged);
   - chunk embedding exactly as today (path A);
   - `pieces = split(chunk, turns, words)` (pure);
   - embed each piece ≥ `MIN_WINDOW_MS` (path B). Nothing is embedded after the loop.
3. Assemble A from `merged` + A windows (unchanged). Assemble B from the pieces, keyed by piece index.
4. Stored path, inside the existing DB block: `load_enrolled` → `apply_matches` for A and for B →
   rule C → `finalize` → insert(final segments, B clusters, voices). Enrolled voiceprints load only
   here, so `--no-store` never opens the DB.
5. `finalize(merged, pieces with parent-chunk index, assembly A, assembly B, rule-C result) ->
   (Vec<TranscriptSegment>, Assembly)`, in order: §4 collapse + re-join → drop word-less pieces →
   merge with mic segments by `t_start` → renumber every `SegmentVoice.segment` to its final index →
   drop clusters left with no embedded row and re-index `SegmentVoice.cluster` → recompute cluster
   stats (§4). Stored clusters are path B's; path A is only an input to rule C.
6. `--no-store` and DB failure render the unsplit `merged`, exactly as today (these paths never show
   names; both keep identical inputs, pipeline.rs:221-223).

### 3. Splitting text — transcribe-then-split
**Why not split before ASR (measured on the 3 meetings):** short pieces (< 1.5 s) are 7.8 % of
far-end time but 35 % of pieces (546/1538); far-end Whisper calls would rise from 842 to 1538, and
`asr.rs` sets no `audio_ctx`, so each call encodes a full 30 s window (likely breaks the Step 6 budget
— estimate) and exposes Whisper's short-clip hallucinations.

Rules:
- Pieces = diarizer turn ∩ chunk; consecutive same-local-speaker pieces < 1.0 s apart merge — exactly
  `split/prep.py`. Chunks with no turn coverage stay whole.
- `transcribe_words` **replaces** `transcribe` on far-end windows when the diarizer has loaded (one
  decode per window). It builds text and confidence exactly like `transcribe` (trim, space-join, mean
  token probability; asr.rs:59-82) and adds token timestamps. `transcribe` is unchanged for mic
  windows and when split is off.
- **Decided: DTW** (`DtwModelPreset::LargeV3`), set at `Asr::load` only when the diarizer has loaded.
  Audit probe (`ts_probe`, same whisper-rs + `FullParams` + token timestamps, 30 multi-speaker chunks,
  811 words): plain put 70 words (16 chunks) where the diarizer hears no speech vs 47 (5 chunks) for
  DTW; plain pulls first words back to the chunk start; 25 plain words have t0 == t1; 0/30 chunks
  changed text between plain and DTW. Go back to plain only if Step 6 fails because of DTW.
- A word goes to the piece containing its midpoint; midpoint in a gap → nearest piece; midpoint in
  two overlapping pieces (50/811 in the probe) → the piece whose turn covers more of the word, tie →
  the earlier piece.
- Piece `t_start/t_end` = roll offset + whole milliseconds (`voices::slice_far_end` refuses off-grid times).
- Word-less pieces are dropped only at storage, after naming, so clustering matches the spike.

### 4. Storage — no schema change
A piece is a normal `TranscriptSegment` (`speaker = Others`, own times/text, the parent chunk's
confidence). `segment_voices`, `voice_clusters`, enrolment and `speakers` commands keep working per-segment.
- After rule C, a chunk whose pieces all resolve to one cluster (or all unassigned) is stored exactly
  as today: one segment, today's embedding.
- In a chunk resolving to ≥ 2 clusters, adjacent pieces with the same resolved **cluster** are
  re-joined into one segment, which takes the embedding of its longest embedded piece; a run with no
  embedded piece is stored `inherited = true`, `embedding = None`. Nothing is re-embedded. Keyed on
  cluster, not name, so two different unnamed voices never merge. Measured: 447 of the 696 added
  segments (64 %) are same-name within-chunk neighbours — the upper bound of what the re-join removes.
- `finalize` recomputes each stored cluster's centroid, `n_windows` and `speech_secs` from the rows
  it actually stores: move `stats_from` from `speakers.rs` to `voices.rs` as a shared pure function.
  (The whole-chunk collapse and the re-join store different embeddings than assemble B clustered;
  without the recompute `speakers merge`/`split` would compute different stats, and merge would
  replace a manual voiceprint.) The stored `match_score` stays the rule-C-time value.
- `render.rs` / `export.rs` stay untouched (nothing groups segments there today), so no past export changes.

### 5. Rule C naming
Pure function in `voices.rs` (not `spk.rs` — `split_probe` path-includes `../spk.rs`, and the frozen
reference must never compile the code under test). Runs after `assemble` + `apply_matches`.
- **Reassigned case:** path A matched the parent chunk to speaker S *and* some path-B cluster is
  matched to S → the short piece moves to that cluster, stored `inherited = true`, `embedding = None`.
  If more than one path-B cluster matches S: pick the one holding a ≥ 1.5 s piece of the same chunk;
  else the highest `match_score`; still tied → the lowest cluster index.
- **Every other case** (path A said Others, or no path-B cluster matches S): the piece keeps the
  membership `assemble` gave it via `inherit_short`, so labelling that cluster later still names it.
- With no enrolled voiceprints the function is the identity on `assemble`'s output — unit-tested.
- **This approximates what was tested** (the tested rule used path A's name directly) → the Step 4
  parity gate must pass before anything ships.

### 6. Fail-open
- Diarizer missing / model absent / non-zero exit / timeout / split off → exactly today's pipeline.
  Never lose a transcript because the diarizer failed.
- Split, rule C and re-join are pure functions over `(chunks, Option<turns>, words, assemblies)`;
  `None` turns → today's segments exactly.
- `src/diar.rs`: `Diarizer::load(bin, model)`, `turns(&mut self, audio16, session_dir)`; may use
  `crate::` helpers (nothing path-includes it). `pipeline::load_diarizer()` mirrors `load_embedder()`:
  returns an `Option`, logs one warn.
- One info line per meeting: session dir, meeting id once stored, split on/off with the reason (off,
  bin missing, model missing, roll k error, timeout), rolls split / total, the commit, a GGUF sha prefix.

## Steps

−1. **Freeze the regression set** — ✅ done 2026-09-27 (`~/.meetscribe/eval/nemotron-split/`, see its
   README; frozen DB = 37 voiceprints / 36 speakers = `split/*.enrolled.json`). Remaining on the new
   branch: commit `src/bin/split_probe.rs` + `ts_probe.rs` unchanged; add `rulec_ref.py` (each B
   window: ≥ 1.5 s → B.out `speaker_id`, else the A.out `speaker_id` of chunk `key // 1000`). The
   frozen A.out/B.out JSONs are the oracle. README note: Step 4 re-transcribes from
   `~/.meetscribe/sessions/<id>`, whose WAVs are not in the frozen set (`retention.sessions_days = 0`
   keeps them for now).
0. **Runtime check** — ✅ 2026-09-27: 20260925-175645 through the launchd-context job, byte-identical
   RTTM (723 s, contended by a concurrent build) → 3/3.
1. **Provisioning + install** (Scope 5 + 6). Verify: fresh provision on a clean models dir passes the
   checksum, a second run is a no-op, `build-diarizer.sh` output reproduces one frozen RTTM; install
   step 3c unit-tested over a temp base dir. **NEVER run `meetscribe install` before Step 8** — its
   launchd label is fixed and it boots out the live recorder, whatever `HOME` is.
2. **Diarizer module** (§1 contract, §6). Tests: (a) missing binary, missing model, non-zero exit and
   timeout (stub script) each make load or turns fail, and `load_diarizer` returns `None` with one
   warning; (b) `say` two-voice boundary test (±0.3 s), `#[ignore]`, `cargo test diar -- --ignored`.
3. **Word timestamps + switch plumbing.** `Asr::transcribe_words`; DTW at load when the diarizer
   loaded. Tests: text + confidence byte-identical with split on vs off on the 30 `ts/` clips and a
   sample of mic windows; a missing diarizer leaves context params identical to today's; `split =
   false` never calls the diarizer. `concat(pieces) == chunk text` on the 30 clips. Plumbing:
   `PipelineOpts`, CLI flags, `[speakers]`. **Timing:** DTW on vs off over the 45-min meeting's full
   ASR pass + extra piece-embedding count from `split/*.B.json` → write the projected Step 6 total here
   before Step 4.
4. **Split + rule C + re-join + finalize.** Unit tests: `None` turns → segments identical to today's;
   each `voices[i].segment`'s final segment contains that voice's piece; `stats_from(stored rows) ==`
   the stored cluster row for every cluster; unstored-path output == today's; `slice_far_end` on piece
   times (covers `speakers play`); rule C identity with no enrolled voiceprints. **Parity**, per session:
   - copy `meetscribe.frozen.db` to scratch;
   - `meetscribe transcribe <session> --db <copy> --export-dir <scratch>` — **not** `--no-store`
     (never loads voiceprints), **not** without `--export-dir` (would overwrite the session's
     `transcript.json`, which `prep.py` reads), **never** against the live DB (`meetings` has no
     unique `source_dir` → duplicates);
   - compare exported far-end names by time with `rulec_ref.py` — **gate ≥ 99 %**; report dropped
     word-less time and auto-matched clusters whose recomputed `n_windows` < `MIN_WINDOWS`.
   - **Amended 2026-09-27 (measured):** the diarizer is deterministic but chaotic in its input —
     2 584 of 43 M samples differing by 1 LSB (our PCM16 rounding vs ffmpeg's) moved its turns to
     88 % frame agreement, and live-diarizer parity landed at 92.4 / 94.9 / 91.7 % (today's labels:
     82.6 / 90.0 / 85.8 %). A label gate against one frozen run therefore measures diarizer noise, not
     our code. The ≥ 99 % gate now runs with `--diarizer-rttm <frozen rttm>` (evaluation-only replay
     of the evaluated turns), which tests pieces, words, rule C and finalize exactly. Real-world
     quality is instead checked by a **blind listening round on live-diarizer output** (new vs today,
     15 clips where they disagree) before Step 8.
   - **Result 2026-09-27 (replayed turns):** long pieces 99.5 / 99.8 / 99.5 % — the code reproduces
     the tested rule. Short pieces deviate on 94 s of 6 484 s far-end (1.5 %: 54 s today-Others →
     named, 26 s named → Others, 14 s different name) — the audited §5 approximation, which keeps the
     later-labelling loop. **Owner decision 2026-09-27:** keep the audited rule; gate = long-piece
     parity ≥ 99 % (met); short-piece disagreements go into the live blind listening round.
5. **Render.** Spot-read of 20260922-190155 + segment-count diff; no render change.
6. **Performance**, like-for-like: same binary context for baseline and split, quiet machine, N = 2,
   load average recorded. **Gate ≤ +30 %** (owner call, 2026-09-27: +20 % = 130 s, and the diarizer
   alone takes 114 s; note 650 s is a CLI figure — today's daemon run of this meeting took 1326 s).
   One Background-context run for information. Over budget → first overlap the diarizer subprocess
   with the roll's Whisper pass; still over → plain timestamps.
   - **Result 2026-09-27 (45-min meeting, `--no-store`, N = 2, alternating):** serial diarizer
     590 → 787 s (**+33 %, over**). Lever 1 applied — the diarizer now starts in the background and
     Whisper runs the same roll meanwhile (`Diarizer::start` / `Pending::wait`): 605 → 711 s
     (**+17 %, pass**; quiet-machine pair 595 → 684 s, +15 %). Replay output byte-identical to the
     serial build. Background-QoS run not repeated (daemon context was measured for the diarizer
     alone in Step 0).
7. **Review.** `/review-branch spike/speaker-embeddings`; fix verified findings; `cargo clippy`, `cargo test`.
8. **Real path.** Install with `--diarizer-model` + `--diarizer-bin` + reinstall the daemon **(ask
   first — it is the live recorder)**; confirm from the §6 log line that the split path ran (not the
   fallback); next captured multi-person meeting → spot-check 5 far-end pieces by ear.

## Assumptions (stated, override any)
- Branch as in the header (stacked).
- Regression set at `~/.meetscribe/eval/nemotron-split/`, never in the repo.
- `split = true` by default once Step 4 parity passes. Existing `config.toml` files have no
  `[speakers]` section, so split turns on at the first daemon restart on the new binary.
- CLI `transcribe` defaults to split on, same as the daemon (Step 4 relies on this); `--no-split`
  reproduces pre-split output.
- ≥ 8-speaker meetings: the diarizer caps at 8 local speakers; extra voices merge (seen at 7).
  Accepted — rule C still names long pieces by voiceprint, so the cap only affects where cuts land.

## Risks
| Risk | Mitigation |
|---|---|
| Rule-C approximation (§5) drifts from what was tested | Step 4 parity ≥ 99 % (replayed turns) or redesign |
| Diarizer output chaotic in its input (1-LSB changes move ~12 % of frames) | parity replays frozen turns; live quality judged by a blind listening round |
| Whisper word timestamps misplace boundary words | DTW (§3, measured better); midpoint + overlap tie-break; Step 3 check |
| Choppier transcripts (2× segments) | §4 storage re-join (64 % of growth is same-name) |
| Diarizer failure loses a meeting | Fail-open §6 + Step 2 test (a) + Step 4 identity test |
| Diarizer binary or model missing | fail-open, startup warning, install 3c, Step 8 log check |
| Hung diarizer | timeout → `None` turns |
| Background QoS ≈ 1.9× slower diarizer | like-for-like Step 6 |
| Voice keys mis-indexed after re-join/drop | `finalize` renumbers + containment test |
| Cluster stats drift | shared `stats_from` + unit test |
| Short pieces of not-yet-enrolled voices never named after labelling | §5 keeps `inherit_short` membership + identity unit test |
| Regression set lost on reboot | Step −1 (done) |
| Retro recluster/split re-assign short pieces by nearest neighbour | accepted, documented in Out |
| New licences in a public repo | OpenMDW + Apache-2.0/ggml notices, Scope 5 |

## Verification buckets (fill at the end)
- **Verified:**
- **Proxy:**
- **Not run:**
