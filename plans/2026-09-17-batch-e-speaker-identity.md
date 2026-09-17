# Batch E — speaker identity: cluster far-end voices, name them once, recall them later

**Plan date:** 2026-09-17 · **Status:** BUILT on `spike/speaker-embeddings` (audited: single auditor over 4 dimensions +
independent skeptic on the cut recommendations; all deltas folded — see "Audit deltas" at the
end). Rollout (daemon reinstall) pending the owner's go.
**Base:** `2026-08-05-speaker-identity-and-vocabulary.md` §E + "Frozen invariants", with the deltas below
**Branch:** `spike/speaker-embeddings` @ `f7cef71` (2 commits ahead of master `d898e92`, not pushed)
**Gate that opened this:** Batch D listening tests (results doc addendum) — 6 of 7 within-meeting
far-end groups were one person each; 6 of 6 cross-meeting pairs judged correctly.

## Decision

Build speaker identity for the **far-end channel only** (`system.wav`), inside the existing
pipeline, with identity stored in **new tables that hang off `transcript_segments.id`** rather than
new columns on `transcript_segments`. The owner hears a cluster, names it once as *first name +
last name*, and every later meeting where that voice recurs gets the name automatically. Voices the
owner does not want to name are marked **unknown** explicitly and stop appearing as pending.

## Requirements (from the owner, 2026-09-17)

1. Put names to the far-end voices.
2. Some voices will deliberately stay unknown — that must be a one-command act, not a nag.
3. A name is **first name + last name**, stored as two fields, displayed as "First Last".
4. Public repo: no colleague names, no employer names in code, tests, docs, or commits. Concretely:
   examples in docs/tests use fictional names; never paste `speakers list`/`play` output into a
   commit, PR body, or doc; `play` clip filenames carry meeting id + letter, never a name.

## Deltas from the August §E design, and why

| August §E | This plan | Why |
|---|---|---|
| Migration 2 adds `speaker_id` + `voice_cluster` columns to `transcript_segments` | **No ALTER at all.** Membership lives in `segment_voices(segment_id → cluster)`; the name lives on `voice_clusters.speaker_id` | `db.rs` has a guard test that fails on any `UPDATE transcript_segments`; retro `label`/`cluster`/`merge` would all need one. Separate tables keep the raw-segment invariant literally true and remove the CREATE-before-ALTER hazard entirely |
| `speakers(id, name)` | `speakers(id, first_name, last_name)` | Owner requirement 3 |
| Voiceprints embedded from WAV at label time | **Per-window embeddings stored at clustering time**; label copies the cluster centroid into `voiceprints` | WAVs stop being load-bearing the moment a meeting is clustered; `merge`/`split`/`play`/`--recluster`/threshold sweeps need no audio re-read |
| Short windows "inherit the nearest confident neighbour's label" | Kept, as a pure post-pass: an `others` segment under 1.5 s takes the cluster of the nearest-in-time embedded `others` segment within 30 s, flagged `inherited`, excluded from centroids and counts | 10.7 % of stored far-end segments are under 1.5 s; without this one line in nine renders "Others" between named lines |
| `speakers list\|label\|unlabel` | adds `play`, `skip`, `merge`, `split`, `cluster`, `match`, `rename` | `play` is how the owner names voices (proved by the listening test); `skip` is requirement 2; `merge`/`split` are the one-command fix for the mixed cluster seen in Batch D; `cluster` is the retro path for the 132 stored meetings; `match` re-runs recall after new enrolments; `rename` fixes typos |
| `[speakers] your_name` | **Deferred to v1.1** | Not an owner requirement; `export`/`rerender`/pipeline read no config today, so it costs config plumbing in three render paths or the daemon file and `export` disagree |
| Auto-match threshold 0.7 guessed | **0.55 cosine**, cluster ≥ 5 windows, margin ≥ 0.05 over the runner-up, else unassigned | From the pairs test: 0.62 was the same person, 0.30 different. Small clusters were where the mix happened |
| Segmentation model (pyannote) possibly required | **Not in v1** | pyannote agreed with our VAD-window clusters at 94–100 % on the test meeting; v1.1 only if mixed clusters recur |
| Tray prompt for unlabelled voices (open question 2) | **Cut.** `speakers list --pending` is the surface | Tray is a bigger change; the CLI is what the owner used in the spike |

## Verified facts this plan rests on (2026-09-17)

- Pipeline emits **one `TranscriptSegment` per VAD window** with `t_start = offset + start_ms/1000`
  (`pipeline.rs` window loop) and `transcript::merge` only sorts by `t_start`. So a stored `others`
  segment's times identify exactly the samples that were transcribed; windows whose ASR text is
  empty produce no segment. Zero duplicate far-end `t_start` values inside any stored meeting.
- Session dirs are not uniform: 126 of 138 have exactly `mic.wav` + `system.wav`; 5 have rate-roll
  segments (`system.001.wav` …); 3 `import-*` dirs have **only `system.wav`** (meetings 68–70,
  Float32 16 kHz mono, `others`-only — the owner's own voice is on the far end there: `skip` or
  label yourself); 4 are empty. The pipeline already walks rolls with offsets from
  `segments.txt` (`discover_channel` + `read_segment_gaps`); E reuses that walk.
- All 132 stored meetings still have `system.wav` at `source_dir`; meeting 1's is relative.
- `Embedder` in `src/bin/spk_probe.rs` (ort + knf-rs, `(1,T,80)` fbank in, 192-d L2-normalised out)
  is correct — proven three ways in Batch D. `ort`, `ndarray`, `knf-rs` are already plain deps.
- **There is no lib target.** Modules are declared on the binary; the probe reaches shared code via
  `#[path = "../x.rs"]` includes. So `src/spk.rs` must have **no `crate::` dependencies** (pure
  functions + `Embedder` only) and the probe path-includes it, dropping its private copy.
- `render::IdentityMap` already resolves `Speaker::Others` by `speaker_id`; `StoredSegment` /
  `RenderedSegment` carry `speaker_id`/`voice_cluster` as `Option`, JSON byte-identical while `None`.
- **The stored pipeline path renders raw segments** (`render_fresh(&merged, IdentityMap::empty(), …)`)
  — it has no row ids, so as written the daemon's `transcript.md` would never show a name. Fixed
  below (D2).
- Live DB: `user_version` 1, 132 meetings, 43,272 segments; the Aug-9 daemon binary already has
  the ladder and warns-and-continues on a newer schema.
- `install` creates `bin`, `models`, `logs`, `sessions` only; `provision_model(src, dest, copy)`
  is warn-don't-fail only around `canonicalize`.
- Cost is negligible: embed RTF 0.006×; a 59-min meeting has ~130 far-end windows. VAD was the
  probe's dominant cost (635 s vs 256 s embedding) — the retro path avoids it entirely.

## Schema — migration 2 (`user_version` 1 → 2), one `BEGIN IMMEDIATE`, CREATEs only

```sql
CREATE TABLE IF NOT EXISTS speakers (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  first_name TEXT    NOT NULL,
  last_name  TEXT    NOT NULL,
  created_at INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS voice_clusters (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  meeting_id   INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
  cluster      TEXT    NOT NULL,   -- 'A','B',… by descending speech time AT CLUSTER TIME;
                                   -- merge/split leave gaps or append letters
  speaker_id   INTEGER NULL REFERENCES speakers(id) ON DELETE SET NULL,
  assigned_by  TEXT    NULL CHECK (assigned_by IN ('auto','manual')),
  match_score  REAL    NULL,                           -- cosine of the auto match, if any
  skipped      INTEGER NOT NULL DEFAULT 0 CHECK (skipped IN (0,1)),  -- owner said "unknown"
  centroid     BLOB    NOT NULL,                       -- f32 LE, L2-normalised
  dim          INTEGER NOT NULL,
  n_windows    INTEGER NOT NULL,                       -- embedded windows only
  speech_secs  REAL    NOT NULL,                       -- embedded windows only
  created_at   INTEGER NOT NULL,
  UNIQUE (meeting_id, cluster));

CREATE TABLE IF NOT EXISTS segment_voices (
  segment_id INTEGER PRIMARY KEY REFERENCES transcript_segments(id) ON DELETE CASCADE,
  cluster_id INTEGER NOT NULL REFERENCES voice_clusters(id) ON DELETE CASCADE,
  inherited  INTEGER NOT NULL DEFAULT 0 CHECK (inherited IN (0,1)),  -- < 1.5 s, took a neighbour's cluster
  embedding  BLOB    NULL);                            -- the window's own embedding; NULL when inherited

CREATE TABLE IF NOT EXISTS voiceprints (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  speaker_id  INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
  meeting_id  INTEGER NULL REFERENCES meetings(id) ON DELETE SET NULL,
  cluster_id  INTEGER NOT NULL REFERENCES voice_clusters(id) ON DELETE CASCADE,
  embedding   BLOB    NOT NULL,
  dim         INTEGER NOT NULL,
  sample_secs REAL    NOT NULL,
  created_at  INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS idx_segment_voices_cluster ON segment_voices(cluster_id);
CREATE INDEX IF NOT EXISTS idx_voice_clusters_meeting ON voice_clusters(meeting_id);
```

- All FKs point at tables that exist before the statement runs (order above). No ALTER. Ladder
  stays additive. `SCHEMA_VERSION` → 2; `MIGRATION_V2` is an ordered `&[&str]` like `MIGRATION_V1`.
- **`voiceprints.cluster_id ON DELETE CASCADE`** (not SET NULL): a voiceprint exists only while
  the cluster that produced it exists. `merge`, `split`, `--recluster` therefore drop their
  voiceprints automatically, and a stale print can never keep feeding `auto_match`.
- `load_segments` gains `LEFT JOIN segment_voices sv ON sv.segment_id = ts.id LEFT JOIN
  voice_clusters vc ON vc.id = sv.cluster_id` and fills `speaker_id = vc.speaker_id`,
  `voice_cluster = vc.cluster`. Reads select explicit columns; an older binary ignores the tables.
- **All SQL stays in `db.rs`** — the guard test's coverage claim depends on it — and no comment in
  `db.rs` may contain the literal phrase the guard greps for.
- Storage: 192 × 4 B per window ≈ 100 KB per hour-long meeting; the whole corpus < 15 MB.
- JSON shape: a clustered-but-unlabelled meeting's `transcript.json` gains `voice_cluster` per
  far-end segment (needed for `play` lookup). `rerender`'s preview-by-default is the guard before
  the 132 stored exports are rewritten.

## Code shape

**`src/spk.rs` (new; in the crate AND path-included by the probe ⇒ no `crate::` imports):**
- `Embedder` lifted verbatim from the probe (`load(path)`, `embed(&[f32]) -> Vec<f32>`).
- `cosine`, `centroid(&[&[f32]]) -> Vec<f32>` (mean, re-normalised), BLOB ↔ `Vec<f32>` (f32 LE).
- `cluster(embeddings, cut) -> Vec<usize>` — average-linkage agglomerative on **cosine distance
  = 1 − cosine**, cut **0.45** (`CLUSTER_CUT`; i.e. merge while average cosine ≥ 0.55, the same
  working point as the match threshold). n is a few hundred, so the naive O(n³) is fine; no dep.
- `label_clusters(assignments, secs) -> Vec<String>` — letters `A`, `B`, … by descending speech
  time (`Z` then `AA`, … past 26).
- `inherit_short(embedded: &[(t_start, cluster)], short: &[t_start]) -> Vec<Option<cluster>>` —
  nearest-in-time within `INHERIT_MAX_SECS = 30`, else `None`. Pure, deterministic.
- `auto_match(centroid, n_windows, enrolled: &[(speaker_id, Vec<Vec<f32>>)]) -> Option<(speaker_id, score)>`
  — score per speaker = **max** cosine over that speaker's voiceprints; assign iff
  `n_windows >= MIN_WINDOWS (5)`, `best >= MATCH_THRESHOLD (0.70, raised from 0.55 after day-one labelling: wrong matches 0.55–0.62, right ones ≥ 0.76)`, and
  `best - second >= MATCH_MARGIN (0.05)` where `second = 0.0` when only one speaker is enrolled.
- Constants are `pub const` with a one-line provenance comment each (from the Batch D addendum).
- Tests: `cluster`, `centroid`, `label_clusters`, `inherit_short`, `auto_match` on synthetic
  vectors; `Embedder` tests are `#[ignore]` with a run-by-hand comment (CONTRIBUTING: no model
  files, no network in the suite). The probe's 7 tests move here or stay path-included.

**`src/pipeline.rs`:**
- `PipelineOpts.speaker_model: Option<PathBuf>` (set by both constructors: `transcribe
  --speaker-model <path>` with the repo-relative default `models/speaker/<file>`, and the daemon's
  absolute `base.join("models/speaker/<file>")`). Missing file → log once ("speaker model not
  found — transcribing without speaker identity") and skip; nothing else changes.
- The far-end window loop also calls `emb.embed(&audio16[a..b])` for windows ≥ **1.5 s**
  (`MIN_WINDOW_MS`) whose ASR text is non-empty; embeddings ride alongside the segment through
  the sort (parallel `Vec<Option<Vec<f32>>>` or an index sort).
- After ASR: cluster → letters → centroids → `inherit_short` → `auto_match` against
  `db.load_enrolled()` → all written **in the same transaction** as the meeting via
  `insert_meeting_with_voices`. The `you` channel is never clustered.
- **Stored path renders from the DB, like `export`:** inside the existing `block_on`, after
  `insert_meeting_with_voices`, call `load_segments(id)` + `list_speakers()` before `close()` and
  render via `render::render(&stored, &IdentityMap::from_db(&speakers), &vocab)`. This is what
  puts names into the daemon's `transcript.md`/`.json` — the primary user surface.
- `--no-store` and the store-failure fallback: no clustering, `render_fresh` + empty identity as
  today; both log lines gain "and without speaker names". Identical in both branches.

**`src/db.rs`:** `insert_meeting_with_voices(meta, segs, voices)`, `load_enrolled() ->
Vec<(speaker_id, Vec<Vec<f32>>)>`, `list_speakers`, `find_speaker(first, last)` (case-insensitive
exact), `add_speaker`, `rename_speaker`, `list_clusters(meeting_id)`, `pending_clusters()`
(unlabelled, unskipped, `n_windows >= MIN_WINDOWS`, across meetings), `set_cluster_speaker(id,
speaker_id, assigned_by, score)`, `set_cluster_skipped`, `add_voiceprint`,
`delete_voiceprints_for_cluster`, `cluster_windows(cluster_id) -> Vec<(segment_id, t_start, t_end,
Option<embedding>)>`, `meeting_windows(meeting_id)` (all embedded far-end windows, for
`--recluster`), `replace_meeting_clusters(meeting_id, …)` (delete + insert, one short tx),
`move_segments(from_cluster, to_cluster)`, `delete_cluster`, `far_end_segments(meeting_id) ->
Vec<(segment_id, t_start, t_end)>` (for the retro path). `insert_meeting` stays as the no-voices
wrapper so existing callers and tests are untouched.

**`src/render.rs`:** `IdentityMap::from_db(&[SpeakerRow])` builds `names: id → "First Last"`.
Unlabelled or skipped clusters render with the channel fallback ("Others"); `voice_cluster`
still appears in JSON. `speaker_name` in JSON is the full name. `your_name` stays a private
field defaulting to `None` (v1.1). Switch the three call sites (`pipeline.rs` stored path,
`main.rs` `load_rendered`, `main.rs` `run_rerender`) from `IdentityMap::empty()` to `from_db`.

**`src/daemon.rs`:** `DaemonConfig.speaker_model` (absolute, under `base`), warn at startup if
missing (same shape as the whisper-model warning), pass through to `PipelineOpts`.

**`src/launchd.rs`:** `run_install` adds `models/speaker` to the created dirs and provisions the
speaker model with a second `provision_model` call inside the warn-don't-fail arm; the warn text
names `bash models/provision.sh` ("daemon will transcribe but not identify speakers until …").
`--speaker-model <path>` override mirrors `--model`.

**`src/main.rs` — `speakers` subcommand** (one nesting convention with `vocab`: unknown flags are
errors; `--db <path>`; never creates a DB — a missing file is an error on every subcommand):

| Command | Effect |
|---|---|
| `speakers list [--pending]` | People with voiceprint counts; with `--pending`, every unlabelled, unskipped cluster ≥ `MIN_WINDOWS` across meetings |
| `speakers list <meeting-id>` | That meeting's clusters: letter, embedded windows, minutes, name or `?`, auto score, `skipped` |
| `speakers play <meeting-id> <cluster> [--clips 3] [--secs 6]` | Cut `--clips` embedded windows closest to the centroid, as far apart in time as possible (prefer ≥ 2 min, but a 5-window cluster inside 90 s still plays), `--secs` each, from the meeting's `system.wav` rolls, into a temp dir named `<meeting>-<letter>`, then `afplay` in order. Needs the WAV; says so if the source dir is gone |
| `speakers label <meeting-id> <cluster> "<first>" "<last>"` | Exactly two shell-quoted positionals (compound names work). Reuse the speaker if first+last match case-insensitively, else create. Sets `speaker_id`, `assigned_by='manual'`, clears `skipped`, deletes any voiceprint this cluster contributed, inserts one voiceprint = the cluster centroid |
| `speakers unlabel <meeting-id> <cluster>` | Clears `speaker_id`/`assigned_by`/`match_score`; deletes this cluster's voiceprint |
| `speakers skip <meeting-id> <cluster>` | `skipped = 1`, speaker cleared, this cluster's voiceprint deleted. Requirement 2 |
| `speakers merge <meeting-id> <A> <B> [<C>…]` | Refuses if any of B… carries a manual label different from A's ("unlabel B first"). Moves B…'s windows into A, recomputes A's centroid/counts from stored embeddings, deletes B… (their voiceprints cascade); A keeps its label |
| `speakers split <meeting-id> <cluster> [--cut 0.30]` | Re-clusters that cluster's stored embeddings at a tighter cut; new letters appended; original label and voiceprint dropped (must be re-listened) |
| `speakers cluster <meeting-id> \| --all [--recluster]` | Retro path. **No VAD re-run:** decode + resample each far-end roll once, slice by the stored `others` segments' `t_start`/`t_end` (in the roll's local time, using the same roll offsets the pipeline used), embed each slice ≥ 1.5 s, cluster, inherit, auto-match, write one short `replace_meeting_clusters` tx per meeting. Never a transaction open across decode/embed; one connection for the whole run. Skips already-clustered meetings unless `--recluster`. Any per-meeting error (relative/missing `source_dir`, unreadable WAV, empty dir) logs and continues; summary `clustered N · skipped M (reasons)` like `rerender` |
| `speakers cluster <meeting-id> --recluster` | **From stored embeddings, no audio:** re-runs `cluster()` on the meeting's embedded windows and re-letters. Drops that meeting's labels **and voiceprints** — says so |
| `speakers match <meeting-id> \| --all` | Re-runs `auto_match` on clusters that are unassigned or `auto`; never touches `manual` or `skipped` |
| `speakers rename <speaker-id> "<first>" "<last>"` | Fix a typo; every past render picks it up |

Recipe documented, not a command: relabel = `unlabel` + `label`.

**Why only manual labels enrol:** an auto match never adds a voiceprint. Otherwise one wrong auto
match seeds the next, and the enrolment set drifts away from the owner's ears.

**The owner's loop for the 132 stored meetings** (documented in `docs/DAEMON.md`):
`speakers cluster --all` (no names yet) → `speakers list --pending` → `play` + `label` a few →
`speakers match --all` → `rerender --all` (preview) → `rerender --all --write`. New meetings after
that get names automatically at transcription time.

## Build order (clippy `-D warnings` + `cargo test` green after every step)

1. `src/spk.rs` pure functions + tests; `Embedder` behind `#[ignore]` tests; probe path-includes it.
2. Migration 2 + `db.rs` API + ladder tests: 1→2 on a populated v1 fixture, idempotent,
   `fresh == migrated` `table_info` for **all seven tables**, and `migrated_db_still_accepts_inserts`
   inserts a meeting **with voices plus one label** (exercises every new table's FKs — SQLite
   accepts `REFERENCES <missing>` at CREATE and only fails at DML).
3. `IdentityMap::from_db` + the three render call sites + render tests.
4. Pipeline: embed in the far-end loop, cluster after merge, inherit, `insert_meeting_with_voices`,
   stored-path render via `load_segments`.
5. CLI, the owner's loop first: `cluster`, `list`, `play`, `label`, `skip`; then `merge`, `split`,
   `unlabel`, `match`, `rename`.
6. `daemon.rs`, `launchd.rs`, docs.
7. Rollout.

## Verify

Unit (fixtures only, fictional names, no model files):
- Ladder tests as in build step 2.
- `cluster`: three synthetic well-separated groups come back as three clusters at 0.45; a fourth
  point at cosine distance 0.44 joins, at 0.46 does not.
- `auto_match`: threshold, margin, and `MIN_WINDOWS` each flip the answer independently; one
  enrolled speaker works (runner-up = 0).
- `inherit_short`: takes the nearest neighbour inside 30 s, `None` beyond.
- `insert_meeting_with_voices` round-trip: `load_segments` returns `voice_cluster`/`speaker_id`;
  a meeting inserted with no voices returns `None` everywhere and the JSON export is byte-identical
  to today's (existing test extended).
- `render`: a labelled cluster shows "First Last"; a skipped or unlabelled one shows "Others".
- `merge` recomputes the centroid; `split` produces ≥ 2 clusters from a bimodal set; `unlabel` and
  `skip` remove exactly the voiceprint that `label` added; `merge` refuses conflicting labels.
- `no_code_path_updates_stored_transcript_text` stays green.
- `cargo clippy --all-targets -- -D warnings` clean.

Real path (on a **copy** of the live DB, never the live one):
1. `speakers cluster <id of 20260916-140022>` → the six clean groups of the listening test
   reappear as the largest clusters; `speakers play` on each; owner names them.
2. `speakers cluster` on two later meetings with the same people, then `speakers match` →
   `speakers list <id>` shows the names auto-assigned with scores ≥ 0.55. That cross-meeting
   recall is the feature.
3. `export <id>` on an unclustered meeting is byte-identical in `.md` and `.json` before/after.
4. `speakers cluster --all` over the corpus in the background; the summary counts skipped dirs
   (relative `source_dir`, empty, unreadable); import dirs cluster (only `system.wav` is needed).

## Rollout (system state — ask before each mutating step)

1. `cp ~/.meetscribe/meetscribe.db ~/.meetscribe/meetscribe.db.pre-v2`
2. `launchctl bootout` the daemon
3. run the new CLI once (`meetscribe list`) → ladder to v2; `PRAGMA user_version` = 2 and a
   test INSERT on a copy succeeds
4. rebuild → re-sign → `install` (creates `models/speaker/`, provisions the speaker model)
5. next captured meeting: `speakers list <id>` shows clusters — this is the CONTRIBUTING
   "what you ran, on what macOS, what you observed" evidence for the PR body.

**Revert = reinstall the old binary only.** An older binary reads a v2 DB (additive tables,
explicit-column reads). Restore `.pre-v2` only if the rung itself failed mid-way, which
`BEGIN IMMEDIATE` + the in-transaction version bump already prevents; restoring it otherwise
discards meetings captured after the backup.

## Out of scope (v1.1 candidates)

- Sub-chunking the ~1 % of far-end windows longer than 30 s before embedding (silero can emit one
  raw window up to ~108 s; a long far-end slice may hold several speakers). Skeptic note on S1.
- `[speakers] your_name` config (rename the mic channel).
- pyannote segmentation for turn boundaries inside long windows (only if mixed clusters recur).
- Tray notification for pending voices.
- Clustering the `you` channel (it is the owner by construction).
- Threshold sweep after ~20 labelled meetings — data exists once `cluster --all` has run.

## Docs

`README.md` feature list, `docs/DAEMON.md` (speaker model path, the owner's loop, imported
meetings, JSON gains `voice_cluster`), `models/PROVISIONING.md` (speaker model provenance + sha),
and fix the Batch D results doc header, which still says NO-GO above an addendum that says GO.

Per `CONTRIBUTING.md`: conventional commits, draft PR until CI green, `/review-branch` before ready.
Push and merge are the owner's call.

## Audit deltas folded (2026-09-17)

Blocker D2 (stored path rendered raw ⇒ no names in the daemon transcript) → render from
`load_segments`. Majors: C1 voiceprint cascade; S1 retro path slices by stored times (skeptic-
verified); S4 short-window inheritance; D1 no lib target ⇒ `spk.rs` has no `crate::` imports;
D5 per-meeting transactions and error shape; D3 build order; C3 migration tests write into every
new table. Minors: S2 `your_name` deferred; S3 `--recluster` from stored embeddings; D7/D8
install dir + both `PipelineOpts` constructors; C4 distance definition + runner-up; C6/C7 quoted
positionals, `play` spacing, never create a DB; B2 revert wording; D6 the owner's loop;
C2 SQL-in-db.rs rules; S5/S6/S7/C5/C9/B6 one line each.
