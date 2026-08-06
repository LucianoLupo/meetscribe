# Speaker identity + vocabulary learning

**Plan date:** 2026-08-05 · **Status:** AUDITED (`/audit-plan` → `revise-major`, all deltas folded)
**Roadmap:** post-v1 feature work
**Baseline:** master `414d1b0`, 47 tests green.
**PREREQUISITE for Batch D:** `fix/clamp-audio-range-vad` is committed and pushed —
`origin/fix/clamp-audio-range-vad` @ `840653f`, `src/resample.rs` only. **Branch the D spike from
that ref, not from `master`.** Silero rejects an entire buffer on one out-of-range sample, so an
unclamped meeting silently drops out of the calibration set and biases the EER *toward a falsely
clean number* — the failure is invisible in the result.

Goal: the transcript gets **better every time you use it**. Two learning loops, both retroactive:

1. **Who** — cluster voices in the `others` channel, ask once, remember forever.
2. **What** — capture term corrections (`Cloud Code` → `Claude Code`) and feed them both forward
   (whisper decode bias) and backward (re-render past meetings).

## Keystone fact

`transcript_segments.text` is currently **final** — the pipeline writes rendered text straight to
the DB and `export <id>` replays it verbatim. Every improvement therefore only helps *future*
meetings, and there are already 36 stored meetings that would never benefit.

So the keystone is a **raw/render split**: the DB holds raw ASR output, immutable; names and
corrections are applied when rendering. One label or one correction then improves every past
meeting on re-export, with **no whisper re-run** (~18 min/meeting at RTF 0.28×).

**Verified on the live DB 2026-08-05:** `PRAGMA user_version` = 0 · 36 meeting rows · 13,015
segments · `SELECT count(*) FROM transcript_segments WHERE text <> trim(text)` = **0**, so the
raw-text invariant is retroactively true and **no backfill is needed**. Also verified:
`~/.meetscribe/exports/` holds two files from 2026-07-19 and the tray's "Open recordings folder"
opens `sessions/` (src/tray.rs:207-209) — so `export <id>` alone does **not** deliver the keystone.
See `rerender` in Batch A.

⚠️ **"36" is both a count and a meeting id in this repo.** Meeting id 36 is the 2026-08-04 session;
ids run 1–39 non-contiguously for 36 rows, and the count drifts. **Verify steps below reference
sessions by stamp (`20260804-202205`), never by id.**

A second structural fact shapes the schema: `transcript_segments.speaker` is
`CHECK (speaker IN ('you','others'))`. That column is the **channel** — physically derived from the
capture layer, always certain, and a design commitment of the project. Identity is *inferred* and
revisable. They are different things, so identity goes in a **new nullable column** and the CHECK is
never touched (SQLite cannot ALTER a CHECK; it would mean a table rebuild).

## Batches

### A — Schema ladder + raw/render split ✅ keystone

**Migration ladder — `PRAGMA user_version`.** Idiomatic SQLite, no dependency, no framework.
`init_schema` is **FROZEN as the v0 baseline and never changes again**; `apply_migrations` owns
everything after. `Db::open_in_memory` routes through the identical path so tests exercise the real
ladder. Keep the existing name `init_schema` — do not introduce a parallel `ensure_schema`.

- **Migration 1 (PR 1)** — `CREATE TABLE vocab_corrections(id, pattern, replacement,
  is_regex INTEGER NOT NULL DEFAULT 0 CHECK (is_regex IN (0,1)), enabled INTEGER NOT NULL DEFAULT 1
  CHECK (enabled IN (0,1)), created_at)`; write `user_version = 1`.
- **Migration 2 (PR 2, only after a green Batch D)** — in this exact order: (a) `CREATE TABLE
  speakers`, (b) `CREATE TABLE voiceprints(… speaker_id REFERENCES speakers(id) ON DELETE CASCADE,
  meeting_id INTEGER NULL REFERENCES meetings(id) ON DELETE SET NULL, embedding BLOB, dim,
  sample_secs, created_at)`, (c) `ALTER TABLE transcript_segments ADD COLUMN speaker_id INTEGER NULL
  REFERENCES speakers(id)`, (d) `ADD COLUMN voice_cluster TEXT NULL`; write `user_version = 2`.

🔴 **CREATEs MUST precede the ALTERs — reproduced locally 2026-08-05.** With `foreign_keys(true)`
(db.rs:67, :83), `ALTER TABLE transcript_segments ADD COLUMN speaker_id INTEGER NULL REFERENCES
speakers(id)` against a not-yet-existing `speakers` **succeeds**, and `SELECT`s keep working — but
every subsequent `INSERT` fails at prepare with `no such table: main.speakers`. On the live DB that
is **the daemon silently losing the ability to store any meeting** while `list` and `export` look
healthy. Confirmed that CREATE-first makes the identical INSERT succeed.

- `speaker_id` stays **NULLABLE with no default** — SQLite rejects `ADD COLUMN … REFERENCES` with a
  non-NULL default while FKs are on.
- Each ladder step runs inside **one `BEGIN IMMEDIATE`** that re-reads `user_version` inside the
  transaction and writes the bump before `COMMIT`. `PRAGMA user_version` is transactional, so a
  half-applied migration cannot leave the version bumped. Without this, an ALTER that commits while
  the version write does not **bricks the tool permanently** with duplicate-column on every open.

**`text` is raw-and-immutable — ALREADY TRUE.** db.rs has exactly one write to
`transcript_segments` (the INSERT at line 116) and no UPDATE/DELETE path in the crate. This batch
*promotes it to a documented invariant* and adds a regression test that fails if an UPDATE on
`transcript_segments.text` is ever introduced. A's real cost is the ladder + `render.rs` + `--raw`
+ `rerender`.

**New `src/render.rs`** — the single place segments become display text:
`render(segs, &IdentityMap, &Vocab) -> Vec<RenderedSegment>`. Pure, no I/O, trivially unit-testable.

- **All three call sites**, named explicitly:
  1. `pipeline.rs`'s `write_exports` — **confirmed at pipeline.rs:168** to pass raw `&merged`
     **after** `database.close()` (line 153). Vocab/speakers must be loaded **inside the existing
     `block_on`, before `close()`**. This is the file produced for every daemon-captured meeting; if
     render is wired into `export.rs` only, the feature **never reaches the primary user surface**.
  2. `main.rs::run_export`.
  3. `main.rs::run_transcribe`'s terminal print (a print, not an artifact).
- The `--no-store` path (pipeline.rs:142) and the store-failure fallback (pipeline.rs:161) have no DB
  handle: **both render with empty Vocab/Identity and log that fact** — identical in both branches,
  or the daemon's on-disk `transcript.md` silently diverges from `export <id>` in exactly the
  store-failure scenario the fallback exists to protect.
- **Carrier type:** `load_segments` returns a widened *internal* row carrying `id`, `speaker_id`,
  `voice_cluster` (all NULL until E populates them); `render` takes that. The **serialized** shape
  keeps `speaker` (channel enum) and `text` verbatim and adds `speaker_name`/`speaker_id`/
  `voice_cluster` as `Option` with `#[serde(skip_serializing_if = "Option::is_none")]` — additive
  fields **and** byte-identical JSON while the tables are empty.
- **Naming:** the identity concept is `SpeakerIdentity`/`IdentityMap`. `transcript::Speaker` already
  means CHANNEL; `render(segs, &Speakers, &Vocab)` would put two unrelated `Speaker*` things in one
  signature. `Speaker::label()` becomes the fallback *consumed by* render, never a parallel path.
- `export <id>` gains `--raw` to bypass rendering (escape hatch + a way to diff what changed).
- **`rerender [--all | <id>…]`** — re-runs render over stored segments and rewrites
  `sessions/<stamp>/transcript.md` + `.json` (`source_dir` is stored per meeting, so the mapping is
  free). **Preview by default** (files that would change + first diff); `--write` commits. This is
  the **acceptance criterion for the keystone** — without it, "improves all past meetings" is
  asserted, not delivered. Preview-by-default mirrors `vocab test`: overwriting 40 real session
  files deserves at least the caution a correction pattern gets.

- **verify:** unit tests — ladder 0→1→2 on a fixture DB **with rows**, idempotent re-run, render is
  pure, and the property that makes the gate meaningful: **render is the identity function when
  identity and vocab are both empty**. Fresh DB and migrated v0 fixture must produce identical
  `PRAGMA table_info(transcript_segments)`. On a **copy** of the real DB: the **default `export <id>`
  path (no flag)** is byte-identical in **both** `.md` and `.json` across several meetings, with
  `--raw` as a secondary assertion. **Post-migration, an INSERT of a new meeting must still succeed**
  — a read-only check passes on a write-bricked DB. Never migrate the live DB in a test.

### B — Vocabulary corrections (render side)

- Applied in `render.rs`, **word-boundary matched** by default; plain patterns escaped and wrapped in
  `\b`, `is_regex` opts into raw. `regex` is already compiled via env_logger → env_filter, so this is
  a **declaration, not new build weight**.
- **Ordering is defined and stable (by id)** — never hash-iteration order; later corrections may
  legitimately depend on earlier ones.
- CLI: `vocab add <pattern> <replacement> [--regex]`, `vocab list`, **`vocab disable <id>` /
  `vocab enable <id>` (not `rm`** — keeps ids stable, so the ordering guarantee holds permanently and
  a bad pattern is reversible), `vocab test <id|--all>` (previews affected segments before
  committing). Unknown flags in the new handlers are **errors**, not silently ignored.
- ⚠️ **Do not blanket-replace ambiguous terms.** In the 2026-08-04 transcript one short token is
  both a mangling of a product name and a genuine personal nickname.
- **ROI is measured, not assumed:** on the stored corpus, the four highest-frequency corrections
  account for roughly 67, 48, ~50 and 13 segments — together about 1.5–2% of stored segments,
  stable enough for word-boundary replacement. (The terms themselves are private meeting
  vocabulary and live only in `~/.meetscribe/meetscribe.db`, never in this repo.)
- verify: unit tests (boundary matching, regex opt-in, ordering, disabled = no-op); apply the real
  seed list to the `20260804-202205` session and eyeball the diff.

### C — Whisper decode bias (forward loop) — SPIKE, own PR, may start in parallel

Shares no code with A/B (touches only `asr.rs`, `PipelineOpts`, `config.rs`), so it can start
immediately — but it is a spike with an empirical bar and **must not** gate PR 1.

- `asr.rs` does not call `FullParams::set_initial_prompt` — confirmed present in whisper-rs 0.13.2
  (`whisper_params.rs:792`). Glossary source: `[vocab] glossary` in `config.toml` (explicit, ordered,
  user-owned). **Not** auto-derived from `vocab_corrections`, which will outgrow the prompt budget
  and dilute the bias.
- **Wire `--glossary` into `transcribe`** and have `run_transcribe`/`run_export` read the pure
  `Config::load(base)` the way `run_detect` does. **They read no config today** (main.rs:78-135
  builds `PipelineOpts` purely from argv), so the A/B below would exercise the no-glossary path and
  produce a **false negative that looks like a real result**. Precedence flag > config > default.
- 🔴 **Safety:** whisper-rs `set_initial_prompt` is
  `CString::new(prompt).expect("Initial prompt contains null byte").into_raw()`, and `FullParams` has
  **no `impl Drop`**. So (a) a config typo containing a NUL **panics** — and a panic is not an `Err`,
  so daemon.rs's error match does not catch it and launchd KeepAlive restarts into the same bad
  config, violating config.rs's stated "must not crash-loop on a typo"; and (b) calling it per VAD
  window **leaks a CString per window** in a process that runs for weeks. → **Validate/strip NULs at
  load** and fall back to no prompt; **set the prompt once on `Asr` at load**, not per window.
- ⚠️ Prompt budget ~224 tokens; over-stuffing makes whisper emit primed words that were never said.
  Cap the glossary and log truncation.
- **verify / ship bar (quantified):** on `20260804-202205`, proper-noun recall improves **AND** there
  are **zero** new occurrences of a glossary term in a window where it was not spoken, hand-checked
  over N windows. Measure C against the **residue B cannot fix** — the unstable tail
  (four unstable spelling variants) and Spanish morphological compounds (verbed and
  prefixed forms, plus an internal hostname) that word-boundary matching structurally cannot reach.

### D — Speaker embeddings: calibration spike (STARTS IMMEDIATELY, in parallel with PR 1)

Dependency-free, owns every external unknown in the plan, and is the only batch that can run in its
own worktree without touching a file PR 1 edits. Produces **a number and a go/no-go, not a feature**.

- **Dependencies:** `ort` appears in `Cargo.lock` only as silero's **transitive** dep — Cargo.toml
  has no direct `ort`, `ndarray`, or `regex` entry. D must declare **`ort = "=2.0.0-rc.10"` (exact
  pin** — a floating pre-release req can resolve to another rc and link a second ONNX Runtime) and
  `ndarray`, each with the one-line rationale comment this file's other deps carry.
- Candidates (verified live on HF `csukuangfj/speaker-embedding-models`, bare `.onnx`, 16 kHz mono in
  — exactly what `resample::to_16k_mono` produces):
  | model | size |
  |---|---|
  | `wespeaker_en_voxceleb_CAM++_LM.onnx` | 29.3 MB |
  | `wespeaker_en_voxceleb_resnet34_LM.onnx` | 26.5 MB |
  | `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | 28.3 MB |
- ⚠️ **All are English/VoxCeleb-trained; these meetings are Spanish.** Embeddings model timbre rather
  than phonetics so this should transfer, but that is a hypothesis. D measures it.
- **Provisioning must be deliberate + sha256-verified** per the "no runtime model download"
  commitment. **Decide and state which:** (a) generalize `download_verified` to take `(repo, name)`
  keeping the fetch-time LFS-pointer check, or (b) add a pinned-constant branch. `provision.sh` is
  hardcoded to `ggerganov/whisper.cpp` and derives sha256 from a git-lfs pointer **at download time**
  — it does **not** implement the pinned constant this plan originally assumed.

**The calibration set is free.** The `you` channel is always Luciano, across many meetings and
several mics and codecs:

- **Same-speaker** = `you` vs `you` across meetings — directly measures the Bluetooth-HFP-vs-built-in
  **codec drift** that is the #1 speaker-ID failure mode. Zero labels.
- **Different-speaker** = `you` vs `others` pairs. Zero labels.
- Together: a full ROC and an **empirical threshold + EER**, not a guessed `0.7`.
- ⚠️ **35 of 36 meetings — MEASURED 2026-08-05, no longer an estimate.** 35 resolve to a directory
  holding both WAVs; **0** directories missing, **0** partial pairs. Only meeting id 1 is lost: its
  `source_dir` is the relative string `capture/`, long since overwritten. Treat `source_dir` as
  **untrusted historical data**: resolve it, **skip-with-a-log** (not error) on relative/missing
  paths, and report coverage as "N of M meetings had recoverable audio, K errored".
- **+4 recordings that have audio but no DB row** — `20260722-210911` (14.4 s) and `20260729-212051`
  (14.1 s), both correctly skipped by `min_secs = 20.0`; `20260730-135249` (21.4 min, cause
  unexplained); `20260801-234127` (47.4 min, the clamp-bug casualty). They cannot be re-rendered —
  there is no stored transcript — but for **embeddings they are ordinary material**, and the last
  two are substantial. ⚠️ **The probe must walk `sessions/` directly rather than driving off the
  `meetings` table**, or it silently discards ~69 minutes of usable calibration audio. Their
  `you`/`others` split is unavailable (no segment rows), so use them for **same-speaker `you`
  pairs only if the channel WAVs are read separately** — `mic.wav` is `you` by construction.

**Exit criteria** (all must hold, else stop and reconsider):

1. EER on the free calibration set is low enough to be useful (target < 5%).
2. Same-speaker similarity does not collapse across a codec change.
3. Per-window embedding cost is negligible beside whisper — measure **total wall-clock for decode +
   resample + VAD over the corpus**, not just the ONNX forward pass; the calibration set is 15.3 GB
   of 32-bit float WAV. `spk_probe` subsamples N windows per meeting; the ROC does not need all
   13,015 segments.
4. 🔴 **Coalesced-window purity.** Embed 1.5–3 s **sub-windows within** single long `others` segments
   and compare **intra-segment** embedding dispersion against **inter-segment** dispersion.
   `COALESCE_GAP_MS = 800` (vad.rs:11) merges speech separated by under 800 ms; the `others` channel
   averages **9.51 s/segment, with 2,396 of 7,124 ≥10 s and 777 ≥25 s**, so sequential remote
   speakers get blended into one embedding. Criteria 1–2 are `you`-side and single-speaker by
   construction, **so they can pass while E still produces garbage.** High intra-segment dispersion
   means `sherpa-onnx-pyannote-segmentation-3-0` (7.0 MB, verified available) is **required scope for
   E**, not the optional upgrade it was originally filed as. Note the 1.5 s floor is not the binding
   constraint — 6,377 of 7,124 `others` segments (89.5%) already clear it; the 30 s ceiling is.

- verify: `src/bin/spk_probe.rs` dumps both distributions and the ROC. **Either commit it as a
  first-class probe matching `vad_probe`/`rtf_probe` (same `//!` header shape, clippy-clean — bins in
  `src/bin` are auto-discovered by `cargo clippy --all-targets -- -D warnings`) or keep it out of the
  tree.** "Throwaway but in src/bin" is neither.

### E — Identity: clustering, enrollment, recall (PR 2, only on a green D)

- `speakers(id, name, created_at)`; `voiceprints(…)` per Migration 2 — **many voiceprints per
  speaker, never one frozen centroid.** Matching is best/mean similarity over a speaker's set, which
  is what survives codec drift.
- Per meeting: embed each VAD window ≥ **1.5 s**; shorter windows inherit the nearest confident
  neighbour's label or stay unassigned. Cluster within `others` (agglomerative, cosine, threshold
  from D) → `voice_cluster` = `A`, `B`, …; match cluster voiceprints against enrolled speakers.
- `you` is **not** clustered — Luciano by construction. `[speakers] your_name` defaults to **empty**
  and render falls back to the literal "You"/"Others", so an un-edited install is byte-identical.
- **`run_install` gains a second `provision_model` call** for the speaker model, following the
  existing warn-don't-fail shape ("daemon will transcribe but not identify speakers…"), plus the
  absolute path on `DaemonConfig` and `/models/*.onnx` in `.gitignore`. Without this, speaker ID
  works from a repo checkout and **silently does nothing in the daemon** — the only place meetings
  are actually captured.
- **E needs an update-by-segment-id DB API.** db.rs exposes only insert/list/get/load/close; the
  retro `label` command cannot write `speaker_id` without one. Scope it here rather than discovering
  it mid-build. State whether clustering runs inline before `insert_meeting` (identity written in the
  existing single transaction) or as a post-pass.
- **Enrollment eagerly embeds and stores voiceprint BLOBs for all existing meetings at label time**,
  so the WAVs stop being load-bearing afterwards.
- CLI: `speakers list`, `speakers label <meeting-id> <cluster> <name>`, `speakers unlabel` — one
  nesting convention matching `vocab`. **`relabel` is cut** (it is exactly unlabel + label; document
  the recipe).
- ⚠️ **The `others` channel is a mixdown.** Simultaneous speech yields a blended embedding; v1 leaves
  low-confidence windows **unassigned rather than guessing** — a wrong name is worse than no name.
- verify: run over ≥ 5 real meetings; a 1:1 with a recurring participant must yield exactly one
  cluster **and recall the enrolled name in a later, unlabelled meeting**. That cross-meeting recall
  is the actual feature — a within-meeting cluster proves nothing about persistence.

## Frozen invariants

Nothing leaves the machine (no runtime model download, no telemetry) · mic and system audio stay on
separate channels · one aggregate device, one clock · bundle-id `com.lucianolupo.meetscribe` frozen ·
identity `155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C` · daemon is metal-only `ggml-large-v3` ·
storage sqlx plaintext 0600 · rebuild → re-sign → `install` after any daemon-code change.

**New here:** `transcript_segments.text` is raw ASR output, never rewritten in place — all
presentation is a pure function of (raw, identity, vocab). `init_schema` is frozen at v0; all schema
evolution goes through the ladder.

**Verified constraint (was an open question):** `retention.sessions_days = 0`, 78 WAVs / 15.3 GB
across 40 session dirs. `maintenance::prune_sessions` does `fs::remove_dir_all` when > 0 and runs at
daemon startup + every 24 h — **it would destroy the enrollment corpus with no error.** Batch E's
eager voiceprint storage removes this dependency; **until then, do not enable retention pruning.**

## Sequencing and shape

- **PR 1** = A + B — ladder rung 1 + raw/render split + `rerender` + vocab.
- **C** = its own PR (spike with an empirical bar); may start in parallel immediately.
- **D** = spike, **starts immediately** in parallel with PR 1; no PR, produces a go/no-go.
- **PR 2** = E + ladder rung 2, only on a green D.

Splitting the ladder along the PR boundary is the whole reason to build a ladder rather than a
one-shot DDL block — and it keeps the FK-bearing ALTER **out of the first migration that touches the
live DB**, which dissolves the reproduced blocker for PR 1 entirely.

## Open questions for the user

1. **Correction scope.** Global list, or scoped per meeting-title/project? Global is simpler and
   probably right for one person; scoping matters if work and personal vocabularies collide.
2. **Unknown-speaker prompting.** Should the daemon surface "meeting N has an unlabelled voice" via
   the tray, or is a manual `speakers` check enough? Tray integration is a bigger change.

## Done-gate

**Ordered rollout** (nothing in the original plan backed up a live 36-meeting DB sitting behind a
running launchd daemon):

1. `cp ~/.meetscribe/meetscribe.db ~/.meetscribe/meetscribe.db.pre-v1`
2. bootout the daemon
3. run the new CLI once manually to apply the ladder
4. verify `PRAGMA user_version` **and that an INSERT still succeeds**
5. rebuild → re-sign → `install`

**Revert** = restore the `.pre-v1` copy and reinstall the old binary. Safe because the old
`init_schema` is `IF NOT EXISTS` and all reads select explicit columns — so **keep the ladder
additive: never add a NOT NULL column, never rename one.**

⚠️ The ladder turns `list`/`export` — today pure readers — into **writers on first run** of the new
binary. The DB uses SQLite's default rollback journal (no WAL), so writers serialize and a `list`
issued while the daemon commits an insert can now fail rather than merely read.

Per `CONTRIBUTING.md`: conventional commits, draft PR until CI green on `macos-latest`,
`/review-branch` and fix confirmed findings before marking ready. Docs: `README.md` (feature list),
`docs/DAEMON.md`, `models/PROVISIONING.md` (speaker model provenance + sha256). New config sections
follow the existing `#[serde(default)]` + flattened `extra` + `push_unknown` + `DEFAULT_CONFIG_TOML`
pattern (pinned by `default_template_parses_to_default`); **CONFIG_VERSION stays 1** (additive), and
`docs/DAEMON.md` must tell **existing** users the new keys must be added by hand — `load_or_init`
only writes the template when the file is absent and never merges.

Push and merge are the USER's call.
