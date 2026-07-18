# Resume prompt — paste after /clear

Resume **meetscribe** — the local-first, background macOS meeting transcriber (Granola-style, brain todo #30). Full state is in the auto-loaded memory `project_meetscribe.md`; the plan is at `~/projects/meetscribe/plans/2026-07-18-meetscribe.md` (§5 = phases). Read both first, plus the ported reference at `~/projects/meetscribe/docs/meetily-ref/core_audio.rs`.

**Status: fully unblocked.** `cidre` (rev a9587fa) builds + runs on this toolchain; **signing works** — Apple Development identity, TeamID `L634X3YJBF`, hash `155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C` (WWDR G3 installed, chain valid). Repo `~/projects/meetscribe/` (Rust, edition 2024). Nothing captured yet.

**Do Phase 0 (capture spike):**

1. **Port the capture layer** from `docs/meetily-ref/core_audio.rs` (+ `system.rs`) into `src/`. **Adapt to the audit-mandated SINGLE-aggregate design** (do NOT copy meetily verbatim — it's tap-only and it MIXES streams, which is the mistake we avoid): ONE `AudioHardwareCreateAggregateDevice` on ONE clock containing the built-in **mic as an input sub-device → ch0 "You"** AND the **global process tap as a sub-tap → ch1 "Others"**, with `kAudioSubDeviceDriftCompensationMaxQuality` on the non-clock sub-device. Keep the two channels SEPARATE (that's our free you-vs-them labeling). Add deps: `ringbuf`, `rubato`, `hound`. Query `kAudioTapPropertyFormat` live (Teams delivers 24 kHz, not 48) → resample to 16 kHz mono.

2. **Build + sign** the binary and add `Info.plist` keys `NSMicrophoneUsageDescription` + `NSAudioCaptureUsageDescription`. Sign with the frozen identity: `codesign --force --sign 155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C --timestamp=none <binary>`. Freeze a reverse-DNS bundle-id + THIS identity as a cross-phase invariant (a re-sign zeroes the TCC grant).

3. **Phase 0a** — run during a REAL Teams (or Meet) call → write `system.wav` + `mic.wav`; verify both non-zero RMS + intelligible (remote voices in system.wav, my voice in mic.wav). Teams is the known failure case: if the global tap is silent on Teams, fall back to the ScreenCaptureKit path (plan §5 "Teams exit"; Muesli reference in `docs/swift-ref/muesli`).

4. **Phase 0b (the real novel risk)** — with mic+tap in the ONE aggregate, verify a scripted "you speak, then remote speaks" lands in correct time order with **<100 ms cross-channel skew** after several continuous minutes.

I'll be in a real meeting to test capture. Engine = **whisper-rs**, storage = **sqlx plaintext (0600)**, capture = **global Core Audio tap** (SCK = Teams fallback), you-vs-them = **channel-based** — all already decided in the plan; don't re-litigate. After Phase 0 passes → Phases 1–5 in plan §5.
