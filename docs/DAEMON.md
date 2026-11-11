# meetscribe daemon — install, operate, uninstall (Phase 4)

The daemon auto-records meetings with **zero manual action**: when an allowlisted app takes the
microphone, it captures (mic = "You", system tap = "Others"), and when the app releases the mic it
finalizes → transcribes → stores + exports Markdown/JSON. It runs at login as a launchd
**LaunchAgent** under the frozen bundle-id `com.lucianolupo.meetscribe`.

## What triggers a recording
A meeting is "active" when a process whose bundle-id matches the allowlist has an **active mic
input stream** (`kAudioProcessPropertyIsRunningInput`). Music/YouTube use *output* only, so they
never trigger it. The mic-holding process is often a helper (e.g. `com.google.Chrome.helper`), so
matching is by identifier **or dotted sub-identifier**.

**Built-in allowlist:**
Zoom `us.zoom.xos` · Chrome `com.google.Chrome` · Teams `com.microsoft.teams2` / `com.microsoft.teams`
· Slack `com.tinyspeck.slackmacgap` · Safari `com.apple.Safari` · Arc `company.thebrowser.Browser`
· Firefox `org.mozilla.firefox` · Brave `com.brave.Browser`.

Tunables: poll every 1.5s · end-debounce 10s (survives a transient route drop) · skip sessions
under 20s (drops mic blips like voice search). The last two are configurable — see below.

## Configure — `~/.meetscribe/config.toml` (Phase 5)
The LaunchAgent runs `meetscribe daemon` with **no flags**, so this file is the only way to change
the background daemon's behavior. It is written with commented defaults on first run and re-read at
startup — edit it, then restart the daemon:

```
launchctl kickstart -k gui/$(id -u)/com.lucianolupo.meetscribe
```

Keys (all optional; unknown keys are warned and ignored, missing keys default):
- `[detector] allowlist_extra = ["com.example.App"]` — extra meeting apps (identifier or dotted
  sub-identifier), ADDED to the built-ins. `use_builtin_allowlist = false` uses ONLY your extras.
- `[daemon] lang = "es"` · `min_secs = 20.0` — transcription language + minimum session length.
- `[retention] sessions_days = 0` — prune eligible managed session folders older than N days
  (**0 = keep forever**, the default). Eligibility requires stable ownership and either a verified
  matching archive/export pair or a clean short capture; unknown, edited and legacy material stays. ·
  `log_max_mb = 10` — rotate the daemon log over N MB (keeps one `.1` backup; 0 = never).

Disk/log hygiene (session prune + log rotation) runs at daemon startup and once per day.

⚠️ **`retention.sessions_days > 0` permanently deletes eligible audio without a confirmation**.
Prune failures are logged; age alone is not permission to delete unarchived or unknown material.
Verified archived transcripts survive, but later re-transcription or voice sampling needs retained
WAVs. Keep `0` when those operations may still be needed.

## Vocabulary corrections — stored text vs written text

`transcript_segments.text` is **raw ASR output and is never rewritten in place.** Corrections
(`meetscribe vocab …`) are applied when a transcript is rendered — by the daemon after each
meeting, by `export <id>`, and by `rerender`. Three consequences worth knowing as an operator:

- A correction added today fixes **every past meeting** on the next `rerender --all --write`, with
  no whisper re-run (~18 min/meeting).
- Corrections are reversible: `vocab disable <id>` then re-render. Nothing was ever destroyed.
- `--no-store`, and a run where the DB write **fails**, both render with NO corrections (there is
  no database to read them from) and log that they did. Their on-disk transcript is therefore raw,
  and re-running `export <id>` later will legitimately differ. Both paths behave identically on
  purpose — a store failure must not produce a file no later export can reproduce.

The daemon applies whatever rules are enabled at the moment it finishes a meeting; it does not need
a restart after `vocab add`, because the rules are read from the DB on each run.

## Speaker identity — far-end voices, named once

After transcribing, the daemon embeds every far-end (`system.wav`) speech window with the speaker
model at `~/.meetscribe/models/speaker/…onnx`, clusters the windows into voices (`A`, `B`, … by
speech time), and matches each cluster against the voiceprints of people you have named. The
result lives in its own tables (`speakers`, `voice_clusters`, `segment_voices`, `voiceprints`);
`transcript_segments` is never touched. Rendering resolves a segment's cluster to a name through
`load_segments`, so the daemon's `transcript.md`, `export`, and `rerender` all agree.

- **Missing speaker model** ⇒ the daemon logs one warning at startup and one per meeting, and
  transcribes without speaker identity. Provision it with `bash models/provision.sh`, then re-run
  `meetscribe install` (which copies/links it under `~/.meetscribe/models/speaker/`).
- **Only manual labels enrol a voiceprint.** An automatic match never does, so one wrong match
  cannot seed the next. A match needs cosine ≥ 0.70, a 0.05 margin over the runner-up, and at
  least 5 embedded windows; anything less stays unnamed (0.70 was set after the first day of
  labelling: wrong matches scored 0.55–0.62, right ones 0.76+).
- **The owner's loop for meetings stored before this feature:** `meetscribe speakers cluster --all`
  (reads each session's `system.wav` once; skips meetings whose recording is gone, and meetings
  already clustered) → `speakers list --pending` → `play` + `label` a few → `speakers match --all`
  → `rerender --all` (preview) → `rerender --all --write`. `cluster --all` over ~130 meetings takes
  roughly half an hour; run it with the daemon idle.
- **Imported far-end-only sessions** (no `mic.wav`) contain your own voice on the far end: `skip`
  it or label yourself.
- **A clustered meeting's `transcript.json` gains `voice_cluster` per far-end segment** even before
  any name exists (it is how `play` finds the audio). `rerender` previews that change like any other.
- Short windows (< 1.5 s) are not embedded; they take the cluster of the nearest embedded window
  within 30 s, or stay unassigned.

## Schema migrations

The database carries a `PRAGMA user_version` and upgrades itself on first open by a newer binary.
Treat installation as a separate, explicitly approved quiet-window operation. First arrange a
consistent SQLite backup and prove restore/integrity on a disposable copy. Do not copy a live
SQLite file as if that were a verified backup, or migrate while a meeting is being stored.

Migrations remain additive/read-compatible, but **do not blindly resume an old binary as a writer**:
its recovery/retention logic does not understand the new session sidecars. If an upgrade fails,
preserve files, use read-only inspection and prefer a forward fix. Restoring an older backup can
discard newer meetings and requires a separate decision.

Rungs: v1 `vocab_corrections`; v2 speaker-identity tables; v3 archive identity, keyed session ingests
and capture completeness. Existing meeting IDs/text remain; legacy completeness is unknown.

⚠️ Migrating turns `list` and `export` into **writers** on their first run under a new binary. The
database uses SQLite's default rollback journal (no WAL), so writers serialize — run the migration
with the daemon stopped rather than alongside a meeting being stored.

## Menu-bar tray (Phase 5) — `meetscribe tray`
An optional, **separate** menu-bar app (its own process; the daemon is untouched). It reads
`~/.meetscribe/status.json` (which the daemon writes) and shows a coloured dot:
green = idle · red = recording · amber = paused · gray = daemon stopped. Menu:
- **Pause / Resume** — toggles `~/.meetscribe/paused`; while present the daemon skips STARTING new
  recordings (a recording already in progress finishes). `touch`/`rm` that file for the same effect.
- **Open recordings folder** / **Open config file**, and **Stop background daemon**.

Launch it manually (`meetscribe tray &`) or add a login LaunchAgent for it. The daemon and tray talk
through status/pause files and instance-addressed Stop requests. Stop is bound to the selected
process-start identity and nonce, not a reusable PID. The tray itself does not capture audio.
An already-running legacy tray is not upgraded by replacing the daemon binary; retire it separately
in an approved upgrade window before relying on the new control guarantees.

## Install (enable at login)
Run from the repo (so the default `--model models/ggml-large-v3.bin` resolves):

```
cargo build
./target/debug/meetscribe install
```

No manual `codesign` step: `install` re-signs the copy it places in `~/.meetscribe/bin/`,
which is the binary the daemon actually runs. It picks the identity in this order —

1. `meetscribe install --identity <sha1>`
2. `$MEETSCRIBE_SIGN_IDENTITY`
3. the sole identity from `security find-identity -v -p codesigning`
4. otherwise it errors, listing what it found

— and resolves it *before* booting out the running daemon, so a signing problem leaves the
existing install untouched. (Sign `target/debug/meetscribe` by hand only if you want to run
the **manual capture** flow directly from it, which needs its own TCC grant.)

`install` performs these stages; it is **not an all-artifact atomic transaction**:
1. Refuse a `coreml` daemon build, resolve the identity, then copy/sign/verify a staged binary.
2. Restrict logs, reconcile an interrupted runtime swap, and validate/stage supplied models and
   the complete diarizer runtime before the agent-stop seam. Self-sources are no-ops; missing
   optional sources preserve installed artifacts. Required-source failures leave the agent running.
3. Request launchd unload and wait for it before publishing replacement artifacts. A timeout
   prevents publication/bootstrap, but launchd disappearance does not prove old child quiescence.
4. Publish the staged binary and validated assets. Model publication preserves a usable old file;
   the runtime keeps backup/journal evidence for reconciliation. Failures return before bootstrap.
5. Write the plist with absolute paths, pinned HOME, login/crash settings and owner-only logs.
6. Bootstrap the new agent. Real signing/TCC, live installation and rollback behavior require
   separate integration validation; source tests do not establish them.

## Legacy recordings and upgrade boundary

Unkeyed legacy recordings are quarantined from automatic recovery and retention, with or without
a `pending` marker. A stopped parent, unloaded launchd label, elapsed delay or new keyed lock does
not prove its old transcription children are gone. No automatic legacy-row adoption is enabled.
Ambiguity preserves those recordings while unrelated new managed captures continue.

Live retirement and existing-row adoption need independently established child quiescence plus
archive/row identity; neither an operator assertion nor a guessed PID is sufficient. This branch
does not claim that upgrade acceptance has been exercised on an installed legacy daemon.

## Rebuild → reinstall loop
After changing daemon code, **rebuild, re-sign, and reinstall** — do NOT just `cargo build` (that
overwrites `target/debug/meetscribe`, but the daemon runs the stable `~/.meetscribe/bin` copy, so it
keeps running the OLD code until you reinstall):

```
cargo build
./target/debug/meetscribe install     # boots out the old, copies+resigns+reloads the new
```

## Operate
```
launchctl print gui/$(id -u)/com.lucianolupo.meetscribe   # status / last exit
tail -f ~/.meetscribe/logs/meetscribe.err.log             # live daemon log (stderr)
meetscribe list                                           # stored meetings (from the daemon or CLI)
meetscribe detect --watch                                 # what the detector sees, live
```
Auto-recorded sessions land in `~/.meetscribe/sessions/<YYYYMMDD-HHMMSS>/` (mic.wav, system.wav,
transcript.md, transcript.json); meetings are stored in `~/.meetscribe/meetscribe.db` (0600).

Managed automatic transcription has an initial attempt plus three retries, with persisted
1-, 5-, and 30-minute waits. Failure to launch the transcription executable uses that same
budget (`child_launch`), including across worker restarts; exhaustion requires attention.
A busy session lock consumes no attempt. An interrupted child's existing reservation is
settled once, not charged again when a replacement executable cannot launch. Unknown
post-launch outcomes are not treated as fresh launch failures. Session metadata preserves
the retry state, and content-free recovery events appear in the stderr log above.

A persisted Complete session can reconcile its stale `pending` marker after an abnormal child
exit or on the next scan, but only under the session lock with archive/export proof and the exact
owned marker contents. Busy, mismatched, edited, symlinked and otherwise unknown material stays.

Managed roll timing uses one shared committed boundary for both channels. Managed manual
transcription also respects checkpointed prefixes and recorded roll identities; it still appends
a new meeting, while `--no-store` opens no DB. Unmanaged manual input keeps the legacy full-file
behavior. Neither recovery nor playback uses uncommitted tails or unknown managed rolls.

## Manual load / unload (without reinstalling)
```
launchctl bootout   gui/$(id -u)/com.lucianolupo.meetscribe                                   # stop
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.lucianolupo.meetscribe.plist       # start
```

## Uninstall
```
meetscribe uninstall     # unloads + removes the plist; KEEPS all data in ~/.meetscribe
```
To also remove data: `rm -rf ~/.meetscribe` (deletes the DB, sessions, exports, and model symlink).

## Frozen invariants (a re-sign with a different identity zeroes the TCC grants)
- Bundle-id / launchd label: `com.lucianolupo.meetscribe` — **permanently frozen.** It binds
  the TCC grant of every installation; changing it would need a legacy-label migration plus a
  documented re-approval step. See the note above `LABEL` in `src/launchd.rs`.
- Signing identity: **not frozen, and not hardcoded** — resolved per-machine at install time
  (flag → `$MEETSCRIBE_SIGN_IDENTITY` → sole auto-detected identity → error). What must stay
  stable is *your* identity across installs on a given machine: re-signing with a different
  one zeroes that machine's TCC grants. `security find-identity -v -p codesigning` lists them.
- Re-sign after every build: **remove-then-sign** with an explicit `--identifier` — `install`
  does this for you. The explicit identifier is mandatory because rustc's default embeds a
  per-build hash, which would change the signed identifier on every rebuild and break TCC.
- Model: `ggml-large-v3` (multilingual — meetings are in Spanish), metal-only for the daemon.

## Signals
`launchctl bootout` (and system logout) sends **SIGTERM**; the daemon folds it into the capture stop
predicate, so it finalizes the current WAVs cleanly and exits 0 (an in-progress meeting is kept on
disk but not transcribed on shutdown). `KeepAlive{SuccessfulExit:false}` restarts it only on a crash.
