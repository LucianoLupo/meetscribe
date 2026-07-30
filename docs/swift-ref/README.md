# Swift reference source (not a build input)

Unmodified copies of Swift files from two upstream projects, kept as readable prior art for
meetscribe's capture design. **Nothing here is compiled, linked, or executed** — meetscribe
is pure Rust and contains no Swift.

Neither directory is covered by meetscribe's own MIT grant in
[`../../LICENSE`](../../LICENSE); each remains under its upstream copyright, whose full
license text is reproduced in
[`../../THIRD-PARTY-NOTICES.md`](../../THIRD-PARTY-NOTICES.md).

## `muesli/` — [Muesli-HQ/muesli](https://github.com/Muesli-HQ/muesli)

MIT, `Copyright (c) 2026 Pranav Hari`. Four files.

The **ScreenCaptureKit** reference. SCK is the fallback capture path for apps a Core Audio
process tap cannot reach; muesli is cited (via pasrom's issue #79) as an SCK-based
transcriber that captures Microsoft Teams correctly. `MeetingNeuralAec.swift` is its neural
acoustic-echo-cancellation implementation — unnecessary for meetscribe v1, whose
mic/system channel separation removes the need for AEC.

## `pasrom/` — [pasrom/meeting-transcriber](https://github.com/pasrom/meeting-transcriber)

MIT, `Copyright (c) 2025 pasrom`. Four files.

Kept for a **negative result** that directly shaped meetscribe.
[Issue #79](https://github.com/pasrom/meeting-transcriber/issues/79) is the evidence that a
*per-process* `CATapDescription` captures nothing from Microsoft Teams: the aggregate builds
without error and then delivers zero-filled buffers, hypothesised to be Teams' non-standard
WebRTC audio routing evading the tap. **This is why meetscribe uses a system-wide mixdown
tap rather than a per-process tap** — a constraint worth not re-discovering the hard way.

## Provenance caveat

These copies were taken around **2026-07-18**. The upstream commit SHAs were not recorded,
so treat them as a dated snapshot, not pinned revisions. For current code, go upstream.
