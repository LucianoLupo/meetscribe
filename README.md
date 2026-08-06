# meetscribe

**Records your meetings and writes down what was said — entirely on your Mac.**

No bot joins your call. Nothing is uploaded. There's no account, no server, and no
subscription. You install it once, and from then on it notices when a meeting starts,
records it, and writes a transcript when the call ends.

It also knows **who said what** — your words are labeled separately from everyone else's.

```
You:     so where did we land on the pricing page
Others:  we're shipping the simpler version first
You:     works for me
```

> **Status:** v1, used daily by its author. Transcripts only — it doesn't write summaries.
> Apple Silicon Macs. If something breaks, please open an issue and include your macOS
> version and `~/.meetscribe/logs/meetscribe.err.log`.

---

## Quick start

```bash
git clone https://github.com/LucianoLupo/meetscribe && cd meetscribe

./models/provision.sh    # downloads the speech model (~4 GB, once)
cargo build
./target/debug/meetscribe install
```

The first time it records, macOS asks for two permissions — **microphone** and **record
system audio**. Say yes to both. Nothing is recorded until you do.

That's it. It now starts automatically when you log in.

```bash
meetscribe list          # what it has recorded
meetscribe export <id>   # save a transcript to a file
meetscribe uninstall     # stop it; your recordings stay
```

## Teaching it your words

Whisper mangles names it has never heard — company names, product names, jargon. Tell it
the right spelling once and every transcript it has ever written can be fixed:

```bash
meetscribe vocab add "Cloud Code" "Claude Code"   # add a correction
meetscribe vocab test --all                       # preview what would change
meetscribe rerender --all                         # preview again, per meeting
meetscribe rerender --all --write                 # apply it to every past transcript
```

Corrections are applied when a transcript is **written**, never to what is stored. The
database keeps exactly what the model heard, so a correction is always reversible
(`meetscribe vocab disable <id>`, then `rerender --all --write`) and you can add one years
later and still fix old meetings — no re-transcribing, which would cost ~18 minutes each.

Matching is on whole words, so `NCP` will not rewrite `NCPX`. Pass `--regex` if you want a
real pattern. Both `vocab test` and `rerender` preview by default and change nothing until
you add `--write`.

## Before you start, you need

| | |
|---|---|
| **A Mac with Apple Silicon** | M1 or newer |
| **macOS 14.4 or newer** | |
| **Rust 1.88+** | [rustup.rs](https://rustup.rs) |
| **Xcode command line tools** | `xcode-select --install` |
| **A free Apple developer certificate** | see below — this one surprises people |
| **~4 GB of disk** | for the speech model |

### The certificate thing

macOS ties microphone permission to *which app is asking*, and it identifies apps by their
code signature. So meetscribe has to sign itself — otherwise macOS would forget your
permission every time you rebuilt it.

You need an **Apple Development certificate**. It's free with any Apple ID:

> Xcode → Settings → Accounts → Manage Certificates → **+** → Apple Development

That's all. meetscribe finds it on its own. If you happen to have several, tell it which:

```bash
security find-identity -v -p codesigning   # list yours
meetscribe install --identity <the-long-hex-string>
```

## How it hears both sides

This is the part most meeting tools get wrong, so it's worth a minute.

A call has two sources of sound: **your microphone**, and **whatever your speakers are
playing** — the other people. Most local tools blend those into one recording. The moment
you do that, you can no longer tell who was talking.

meetscribe keeps them apart. It records both at once, on a shared clock, as two separate
tracks:

```
   your microphone  ──▶  track 1  →  "You"
   your system audio ─▶  track 2  →  "Others"
```

Because they're separate, labeling speakers is free — no AI guessing required. Because they
share a clock, they never drift apart, even on a two-hour call.

Two details that took real work to get right, in case you're building something similar:

- **Both tracks come from one audio device, not two.** Two separate recorders run on two
  separate clocks, drift apart, and smear the labels.
- **It listens to all system audio, not to a specific app.** Listening per-app sounds
  tidier, but it silently captures *nothing* from Microsoft Teams
  ([details](https://github.com/pasrom/meeting-transcriber/issues/79)).

## Settings

Edit `~/.meetscribe/config.toml`, then restart it:

```bash
launchctl kickstart -k gui/$(id -u)/com.lucianolupo.meetscribe
```

**Two defaults are probably wrong for you** — they're the author's:

```toml
[daemon]
lang = "es"        # ← Spanish! Change this to your language.
min_secs = 20.0    # ignore anything shorter than 20 seconds

[retention]
sessions_days = 0  # keep recordings forever; it never deletes unless you ask
```

It recognizes Zoom, Chrome, Teams, Slack, Safari, Arc, Firefox and Brave out of the box.
Add others with `detector.allowlist_extra`.

## Things you should know

Stated plainly, so nothing surprises you later.

- **The transcript appears *after* the call, not during it.** A long meeting takes a while
  to process. Live transcription was designed and deliberately shelved — it's three new
  moving parts and risks dropping audio, which is a bad trade for a recorder.
- **Transcripts are stored unencrypted** on your disk (readable only by your user account).
  Encryption is planned. If your meetings are sensitive, know this now.
- **"Others" is one label, not names.** You get *You* vs *Others* — if three people are on
  the far end, they share a label.
- **Headphones work better than speakers.** On speakers, the other people leak into your
  microphone. It still works, your track is just messier.
- **It uses the largest, most accurate speech model**, which is also the slowest. You can
  swap it in `models/provision.sh`.
- **It's not on crates.io** and won't be — two dependencies are pinned to specific git
  commits, which crates.io doesn't allow. Clone and build.

## Where things live

```
~/.meetscribe/
├── sessions/   your recordings
├── logs/       what the daemon is doing
└── config.toml settings
```

Uninstalling leaves all of it alone. To remove everything: `rm -rf ~/.meetscribe`.

## Thanks

The audio capture came from **[meetily](https://github.com/Zackriya-Solutions/meetily)**
(MIT) — meetscribe extends it to add the microphone alongside the system audio, which is
what makes the two-track labeling possible. meetily deserves the credit for the hard part.

[pasrom/meeting-transcriber](https://github.com/pasrom/meeting-transcriber) and
[Muesli-HQ/muesli](https://github.com/Muesli-HQ/muesli) shaped the design too. Full credits
in [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).

Curious why something works the way it does? Every build phase has a written plan under
`plans/`, and the research behind the audio design is in `docs/research-2026-07-18.md`.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Issues and PRs welcome — including "your design is
wrong, here's why".

## License

MIT — see [`LICENSE`](LICENSE). The reference code under `docs/meetily-ref/` and
`docs/swift-ref/` belongs to its original authors.
