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
- `[retention] sessions_days = 0` — delete session folders older than N days (**0 = keep forever**,
  the default; the meeting + transcript stay in the DB, so `export <id>` still works). ·
  `log_max_mb = 10` — rotate the daemon log over N MB (keeps one `.1` backup; 0 = never).

Disk/log hygiene (session prune + log rotation) runs at daemon startup and once per day.

## Menu-bar tray (Phase 5) — `meetscribe tray`
An optional, **separate** menu-bar app (its own process; the daemon is untouched). It reads
`~/.meetscribe/status.json` (which the daemon writes) and shows a coloured dot:
green = idle · red = recording · amber = paused · gray = daemon stopped. Menu:
- **Pause / Resume** — toggles `~/.meetscribe/paused`; while present the daemon skips STARTING new
  recordings (a recording already in progress finishes). `touch`/`rm` that file for the same effect.
- **Open recordings folder** / **Open config file**, and **Stop background daemon**.

Launch it manually (`meetscribe tray &`) or add a login LaunchAgent for it. The daemon and tray talk
only through the two files, so the tray never touches audio and adds no risk to capture.

## Install (enable at login)
Run from the repo (so the default `--model models/ggml-large-v3.bin` resolves):

```
cargo build
codesign --remove-signature target/debug/meetscribe \
  && codesign --sign 155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C \
     --identifier com.lucianolupo.meetscribe --timestamp=none target/debug/meetscribe
./target/debug/meetscribe install
```

`install` is **idempotent** and does, in order:
1. `launchctl bootout` any running instance (so the rebuild→reinstall loop never leaves a stale inode);
2. **refuse** if the binary was built with the `coreml` feature (an unattended `coreml` daemon would
   eat the one-time ~23-min CoreML ANE compile on a live meeting — metal-only only);
3. copy the running binary → `~/.meetscribe/bin/meetscribe` (a **stable path** so future
   `cargo build`s can't zero the running daemon's signature/TCC grant) and **re-sign** it there;
4. provision the model at `~/.meetscribe/models/ggml-large-v3.bin` — a **symlink** to the repo model
   by default (no 2.9 GB copy), or `install --copy` for a repo-independent copy;
5. write `~/Library/LaunchAgents/com.lucianolupo.meetscribe.plist` (absolute paths, pinned `HOME`,
   `RunAtLoad`, `KeepAlive` on crash);
6. `launchctl bootstrap gui/$UID` it (starts now + at every login).

## Rebuild → reinstall loop
After changing daemon code, **rebuild, re-sign, and reinstall** — do NOT just `cargo build` (that
overwrites `target/debug/meetscribe`, but the daemon runs the stable `~/.meetscribe/bin` copy, so it
keeps running the OLD code until you reinstall):

```
cargo build && codesign --remove-signature target/debug/meetscribe \
  && codesign --sign 155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C \
     --identifier com.lucianolupo.meetscribe --timestamp=none target/debug/meetscribe
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
- Bundle-id / launchd label: `com.lucianolupo.meetscribe`
- Signing identity: `155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C` (Apple Development, Team `L634X3YJBF`)
- Re-sign after every build: **remove-then-sign** with an explicit `--identifier`.
- Model: `ggml-large-v3` (multilingual — meetings are in Spanish), metal-only for the daemon.

## Signals
`launchctl bootout` (and system logout) sends **SIGTERM**; the daemon folds it into the capture stop
predicate, so it finalizes the current WAVs cleanly and exits 0 (an in-progress meeting is kept on
disk but not transcribed on shutdown). `KeepAlive{SuccessfulExit:false}` restarts it only on a crash.
