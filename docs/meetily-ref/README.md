# meetily — reference source (not a build input)

Unmodified copies of five files from
**[Zackriya-Solutions/meetily](https://github.com/Zackriya-Solutions/meetily)**, MIT
licensed, `Copyright (c) 2024 Zackriya Solutions`.

**Nothing here is compiled, linked, or executed.** These files are kept as readable prior
art for meetscribe's capture layer. They are **not** covered by meetscribe's own MIT grant
in [`../../LICENSE`](../../LICENSE) — they remain under meetily's copyright, whose full
license text is reproduced in [`../../THIRD-PARTY-NOTICES.md`](../../THIRD-PARTY-NOTICES.md).

## Why they are here

`core_audio.rs` is the origin of meetscribe's `src/capture.rs`: its global process tap +
`AudioHardwareCreateAggregateDevice` recipe was ported and then extended to add the
microphone as an input sub-device of the same aggregate, putting mic and system audio on
one clock as two separately-addressable channels. meetily mixes the two into a single
stream and thereby discards speaker labels; keeping them apart is meetscribe's core
divergence. meetily also has no route-change or zero-sample watchdog — meetscribe's was
built fresh.

`Cargo.toml` is retained because meetscribe pins `cidre` to rev `a9587fa`, the revision
meetily verified the recipe against; the ported code compiles only against that API surface.

## Provenance caveat

These copies were taken around **2026-07-18**. The upstream commit SHAs were not recorded,
so treat this as a dated snapshot, not a pinned revision. For current code, go upstream.
