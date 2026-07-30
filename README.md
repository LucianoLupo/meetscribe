# meetscribe

**A background meeting transcriber for macOS that never sends your meetings anywhere.**

No bots join your calls. No audio leaves the machine. No account, no server, no telemetry,
no runtime downloads. meetscribe notices when a meeting app takes your microphone, records
until the call ends, transcribes on-device, and stores a speaker-labeled transcript.

Written in Rust. Runs as a login agent you install once and then forget.

> **Status:** v1, used daily by its author. Transcripts only — AI summaries are not a goal.
> Tested on Apple Silicon / macOS 26. If it breaks for you, an issue with your macOS
> version and `~/.meetscribe/logs/meetscribe.err.log` is genuinely useful.

## Why it's built this way

Every meeting recorder faces one question: how do you capture *both* sides of a call?
Most answers are bad. Bots join the call and announce themselves. Cloud tools upload your
audio. Local tools mix your microphone and the system output into one track — and the
moment you do that, you've destroyed the information that tells you who was speaking.

meetscribe builds a **single Core Audio aggregate device on one clock**, with your
microphone as an input sub-device and a global system-audio process tap as a sub-tap:

```
                 ┌──────────── one aggregate, ONE clock ────────────┐
   microphone ──▶│ ch0  "You"                                       │──▶ mic.wav
   system tap ──▶│ ch1  "Others"                                    │──▶ system.wav
                 └──────────────────────────────────────────────────┘
                         frame-aligned, no drift, no smearing
```

Because the two sources share a clock, they stay sample-for-sample aligned for the length
of the call. Because they stay on separate channels, **speaker attribution is free** — your
voice is the mic channel, everyone else is the tap channel. No neural diarizer, no
guessing, nothing to get wrong.

Two hard-won details are worth stating, since both are easy to get wrong:

- **One aggregate, not two streams.** Two independent capture streams run on two clocks;
  they drift, and the drift smears attribution across the boundary.
- **A system-wide mixdown tap, not a per-process tap.** Per-process `CATapDescription`
  captures *nothing* from Microsoft Teams — the aggregate builds without error and then
  delivers zero-filled buffers. See
  [pasrom/meeting-transcriber#79](https://github.com/pasrom/meeting-transcriber/issues/79).

## Requirements

Read these before cloning — several are hard requirements, not preferences.

| | |
|---|---|
| **Hardware** | Apple Silicon |
| **macOS** | 14.4+. The Core Audio tap API landed in 14.2, but ship ≥14.4 for the correct TCC category |
| **Rust** | stable ≥ 1.88 — a *dependency* floor, not the edition floor (`cidre`, `home`, and `time` require it) |
| **Xcode CLT** | for `security` and `codesign` (`xcode-select --install`) |
| **A code-signing certificate** | **your own** — see below |
| **Disk** | ~4.1 GiB for the ASR model |

### Why you need your own certificate

macOS binds microphone and system-audio permission (TCC) to a binary's **code signature**.
An unsigned binary, or one whose signature changes between builds, gets re-prompted or
silently denied. So meetscribe re-signs its daemon binary at install time with a stable
identifier.

That means you need an **Apple Development certificate** — free with any Apple ID. Create
one in Xcode: *Settings → Accounts → Manage Certificates → + → Apple Development*.

meetscribe finds it automatically if you have exactly one. Otherwise pass it explicitly:

```bash
meetscribe install --identity <sha1>        # or: MEETSCRIBE_SIGN_IDENTITY=<sha1>
security find-identity -v -p codesigning     # to list what you have
```

## Install

```bash
git clone https://github.com/LucianoLupo/meetscribe && cd meetscribe

./models/provision.sh    # ~4.1 GiB, one time, integrity-checked. NOT a runtime download.
cargo build              # metal-only; do not use --features coreml for the daemon
./target/debug/meetscribe install
```

`install` copies the binary to `~/.meetscribe/bin/`, re-signs it there, writes a LaunchAgent,
and starts it. Rebuilds of your working copy never disturb the running daemon.

**On first capture macOS shows two prompts — Microphone and "record system audio". Approve
both.** Nothing is captured until you do.

```bash
meetscribe list          # meetings recorded so far
meetscribe export <id>   # write a transcript to disk
meetscribe uninstall     # unload the agent; your data in ~/.meetscribe is kept
```

## Configuration

`~/.meetscribe/config.toml` is written on first daemon start. Edit, then restart:

```bash
launchctl kickstart -k gui/$(id -u)/com.lucianolupo.meetscribe
```

The defaults reflect its author's setup, and two of them are probably wrong for you:

```toml
[daemon]
lang = "es"        # ← Spanish. Set your language.
min_secs = 20.0    # sessions shorter than this are captured but not transcribed

[retention]
sessions_days = 0  # 0 = keep recordings forever; meetscribe never deletes unless you opt in
```

Meeting apps are detected by bundle-id, with Zoom, Chrome, Teams, Slack, Safari, Arc,
Firefox and Brave built in. Add your own with `detector.allowlist_extra`.

## Known limits

Stated plainly, because you should know them before trusting it with a real meeting.

- **Transcription is batch, not live.** Audio is transcribed when the meeting *ends*, not
  during it. A long call takes a while to appear. Streaming was designed, audited, and
  deliberately shelved — it costs three new subsystems and puts the zero-dropped-frames
  capture guarantee at risk. See `plans/2026-07-22-concurrent-transcription.md`.
- **Transcripts are stored in plaintext SQLite** (`0600`) under `~/.meetscribe/`.
  Encryption at rest is planned for v1.1 with a documented `sqlcipher_export()` migration.
  If your meetings are sensitive, know this now.
- **Speaker attribution is channel-based, not per-person.** You get "You" vs "Others" — not
  "Ana" vs "Bruno". If several people are on the far end, they share a label.
- **Headphones give the cleanest separation.** On speakers, the far end bleeds into your
  microphone; attribution still works but the mic channel is dirtier.
- **The default model is `large-v3`** — the most accurate and the slowest. Swap it via
  `models/provision.sh` if you'd rather have `large-v3-turbo`.
- **Not on crates.io**, and won't be: two dependencies are pinned git revisions, which
  crates.io forbids. Clone and build.

## Credit

The Core Audio capture layer was ported from
**[Zackriya-Solutions/meetily](https://github.com/Zackriya-Solutions/meetily)** (MIT) and
then extended to add the microphone to the aggregate, which is what makes the two-channel
attribution possible. meetily deserves the credit for the working tap recipe.

Design was also shaped by
[pasrom/meeting-transcriber](https://github.com/pasrom/meeting-transcriber) and
[Muesli-HQ/muesli](https://github.com/Muesli-HQ/muesli). Full notices, including the
vendored reference sources under `docs/`, are in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).

The research behind the design decisions is in `docs/research-2026-07-18.md`, and every
build phase has a plan and an audit under `plans/`.

## License

MIT — see [`LICENSE`](LICENSE). The reference sources under `docs/meetily-ref/` and
`docs/swift-ref/` remain under their own upstream copyrights.
