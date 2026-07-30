# Contributing

Thanks for looking. meetscribe is a small, opinionated tool — issues and PRs are welcome,
and so is being told the design is wrong.

## Before you can build

Setup, the certificate requirement, and the install loop are in the
[README](README.md#requirements). The rebuild → re-sign → reinstall cycle and every launchd
operation live in [`docs/DAEMON.md`](docs/DAEMON.md).

Those are the single source for that recipe and this file deliberately does not repeat it —
duplicated instructions drift, and a stale signing command is a broken instruction rather
than a cosmetic one.

## The gate

Both must pass before a PR is ready. CI runs exactly these on `macos-latest`:

```bash
cargo clippy --all-targets -- -D warnings
cargo test
```

A second CI job pins the toolchain to the declared MSRV (`rust-version` in `Cargo.toml`) and
runs `cargo check`. If you add a dependency that raises the floor, raise `rust-version` in
the same PR — `@stable` will not catch it, but that job will.

**Keep the test suite pure.** Nothing in it may require audio hardware, a TCC grant, a
signing certificate, or a network. That is what lets CI run the whole suite unfiltered, and
it is worth protecting. A test that genuinely needs a device belongs behind `#[ignore]` with
a comment saying how to run it by hand.

## Testing what CI cannot

Capture, signing, and TCC are unreachable in CI, so the parts of the system most likely to
break are also the parts a green checkmark says nothing about. If your change touches
`capture.rs`, `daemon.rs`, or `launchd.rs`, exercise it on a real call and say so in the PR:
what you ran, on what macOS version, and what you observed. "Tests pass" is not evidence for
those files.

Two known traps, both cheap to hit:

- **`meetscribe install` boots out by launchd label, not by `$HOME`.** Running it from a
  scratch clone will stop the daemon your main checkout installed. There is no isolated
  install.
- **Re-signing with a different identity or identifier zeroes the TCC grant.** If macOS
  starts re-prompting for microphone or system-audio permission, that is the cause.

## Design commitments

These are settled, and a PR that reverses one needs to argue the case first — ideally in an
issue, before you write the code:

- **Nothing leaves the machine.** No telemetry, no runtime model download, no cloud
  transcription, no analytics. This is the whole point of the project.
- **Mic and system audio stay on separate channels.** Mixing them is simpler and destroys
  speaker attribution.
- **One aggregate device, one clock.** Independent streams drift and smear attribution
  across the boundary.
- **The bundle-id `com.lucianolupo.meetscribe` is frozen.** It binds the TCC grant for every
  existing installation. Changing it would require a legacy-label migration and a documented
  re-approval step. See the note above `LABEL` in `src/launchd.rs`.

## Commits and PRs

Conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `chore:`). Explain **why** in the
body — the diff already shows what.

Open PRs as drafts until CI is green. Small and focused beats large and sweeping.

## Where the reasoning lives

Every build phase has a plan under `plans/`, most with an audit attached, and the research
behind the capture design is in `docs/research-2026-07-18.md`. If you are wondering why
something is the way it is, the answer is usually written down there. Reference sources from
the projects that informed the design are under `docs/meetily-ref/` and `docs/swift-ref/` —
reference only, never compiled; see [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).
