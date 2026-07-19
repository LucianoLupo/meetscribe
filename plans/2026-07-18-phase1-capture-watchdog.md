# meetscribe Phase 1 — Capture layer productionized (route-change watchdog)

**Plan date:** 2026-07-18 · **Phase:** 1 of plan `~/projects/meetscribe/plans/2026-07-18-meetscribe.md` §5
**On approval:** copy this file to `~/projects/meetscribe/plans/2026-07-18-phase1-capture-watchdog.md` (project convention), then implement.

## Context

Phase 0 proved the capture layer on a live 5-min Google Meet call: ONE Core Audio aggregate on ONE clock, mic (`"You"`) + global system tap (`"Others"`) as two separate, frame-aligned channels (`number_buffers=2`, aligned over 5 min). But it is **proven-but-fragile**: `DualCapture` (`src/capture.rs`) is RAII drop-to-stop with **no reaction to the audio route changing**. If you unplug/plug headphones mid-meeting, the default input/output device changes, the aggregate silently dies or produces garbage, and you get a half-transcript with **no signal that anything broke**. meetily (our port source) has none of this — the watchdog is meetscribe's actual value-add (plan §4).

**Goal (plan §5 Phase 1 verify criterion):** 30+ min of continuous capture survives a headphone plug/unplug, the two channels stay aligned, and the watchdog log is clean. Plus a startup self-check that tells "TCC grant missing" apart from "route dropped."

## What the research settled (verified against the cidre `a9587fa` source)

- **Listener API:** `ca::System::OBJ.add_prop_listener(&addr, listener_fn, ctx_ptr)` where `listener_fn: extern "C-unwind" fn(ca::Obj, u32, *const ca::PropAddr, *mut T) -> os::Status` (`hardware.rs:19,147`). Address = `ca::PropSelector::HW_DEFAULT_OUTPUT_DEVICE.global_addr()` and `…HW_DEFAULT_INPUT_DEVICE.global_addr()` (`hardware.rs:395,399`; use `global_addr()` — `output_addr()` has an INPUT-scope bug). No RAII → must `remove_prop_listener` (same fn+addr+ctx) in `Drop`. Template: cidre `examples/av-route-changes/main.rs`.
- **Callback thread:** fires on a HAL thread → **only signal, never do HAL work there.** Callback sets an `Arc<AtomicBool>` and returns.
- **Rebuild = fresh aggregate.** cidre RAII: dropping `StartedDevice` does `AudioDeviceStop` then destroys the aggregate; dropping `TapGuard` destroys the tap (`hardware.rs:1413,1684`; `hardware_tapping.rs:27`). IO-proc ids are never destroyed by cidre, but a full aggregate destroy reclaims them, so per-rebuild fresh aggregates don't leak. Keep struct field order (`started` before `tap`).
- **Rate control:** `Device::set_nominal_sample_rate(&mut self, f64)` (`hardware.rs:519`) — lets us force the rebuilt aggregate back to the original rate so the fixed-header WAV stays single-rate.
- **SCOPE REDUCTION 🎯 — drop "stale-aggregate cleanup at launch".** Our aggregate is `is_private=true` (`capture.rs:187`). cidre `hardware.rs:1566-1576` (Apple's documented semantic): a private aggregate is *not published system-wide* and *not persistent across launches* — coreaudiod reclaims it when the creating process exits, **including crash/SIGKILL**. So it cannot leak into a later run, isn't even enumerable from a new process, and cidre exposes no destroy-by-id anyway. Replace that task with a one-line launch log noting the invariant. (If we ever switch to a *published* aggregate this returns — noted, not built.)

## Design

### `src/capture.rs` — split the swappable device from the stable plumbing

Keep **stable for the whole process**: the two `HeapRb` rings, their producers (inside a pinned `Box<AudioContext>`), the consumers `main` drains, the `Arc<Shared>` counters. Make **swappable**: only the physical device instance.

```rust
struct DeviceInstance {                       // !Send; dropping it does HAL teardown
    started: ca::hardware::StartedDevice<ca::AggregateDevice>,
    tap: ca::TapGuard,                         // declared AFTER started → destroy order
}

pub struct DualCapture {
    device: Option<DeviceInstance>,            // None only during rebuild()
    ctx: Box<AudioContext>,                    // STABLE — holds mic_prod/tap_prod; never reallocated
    mic_cons: HeapCons<f32>,                   // STABLE — main drains these
    tap_cons: HeapCons<f32>,                   // STABLE
    shared: Arc<Shared>,                       // STABLE — `dropped` is cumulative
    rebuild_req: Arc<AtomicBool>,              // set by listener + watchdog, polled by drain loop
    listener_ctx: *mut AtomicBool,             // == Arc::as_ptr(&rebuild_req); for remove in Drop
    target_rate: u32,                          // first aggregate's rate; forced on every rebuild
    // diagnostics refreshed each (re)build: mic_uid, mic_native_rate, … (existing pub fields)
}
```

- **Extract `build_device(ctx: &mut AudioContext, target_rate: Option<u32>) -> Result<(DeviceInstance, Diag, u32)>`** from the current `start()` body (`capture.rs:132-241`): read default mic → create global tap → sub-dicts → `AggregateDevice::with_desc` → if `Some(rate)` `agg.set_nominal_sample_rate(rate as f64)` (log+continue on err) → read back actual rate → `agg.create_io_proc_id(audio_proc, Some(ctx))` (**same box → same producers → same rings**) → `device_start`. Returns the instance, refreshed diagnostics, and the actual rate.
- **`start()`** = create rings + `Box<AudioContext>` + `Arc<Shared>` + `Arc<AtomicBool>`, call `build_device(ctx.as_mut(), None)` (first actual rate becomes `target_rate`), register the listeners, assemble `Self`.
- **`rebuild(&mut self) -> Result<RebuildOutcome>`**: `let t0 = Instant::now(); self.device = None;` (RAII stop→destroy-agg→destroy-tap, so the old RT proc stops touching `ctx` before we reuse it) → `build_device(self.ctx.as_mut(), Some(self.target_rate))` → store new `DeviceInstance`, refresh diagnostics, re-seed `shared.agg_rate`. Return `RebuildOutcome { gap_frames: (t0.elapsed() * target_rate) as usize, actual_rate, rate_changed: actual_rate != target_rate }`.
- **`rebuild_requested(&self) -> bool`** = `self.rebuild_req.swap(false, AcqRel)` (cheap; called each drain tick).
- **Listener:** one `extern "C-unwind" fn on_route_change(_obj, n, addrs, ctx: *mut AtomicBool)` — if any address's selector is default in/out, `(*ctx).store(true, Release)`. Register on **both** selectors on `ca::System::OBJ` with `client_data = Arc::as_ptr(&rebuild_req) as *mut AtomicBool` (Arc keeps the `AtomicBool` alive; store-through-shared-ref is sound). `Drop for DualCapture` removes both, then default field drop tears down the device.
- **Startup self-check:** classify the failure points in `build_device` into `enum StartError { MicMissing, SystemAudioTccMissing, RouteDropped, Other(anyhow::Error) }` — mic path failing → `MicMissing`/`RouteDropped`; `create_process_tap()` failing with the unauthorized status → `SystemAudioTccMissing` (exact cidre status determined by driving the real revoked-grant path during Batch 3). `main` prints the actionable remedy (e.g. "approve System Settings → Privacy → Screen & System Audio Recording, or `tccutil reset SystemAudioCaptureRequests com.lucianolupo.meetscribe`").

### `src/main.rs` — spike harness reacts to rebuilds and preserves alignment

In the drain loop (`main.rs:116-131`), after `drain_into` + `write`:
```
if cap.rebuild_requested() {
    // 1. final synchronized drain+write of BOTH channels (they end at equal frame counts)
    // 2. let out = cap.rebuild()?;
    // 3. pad BOTH mic_w and sys_w with `out.gap_frames` zero mono frames (equal padding =
    //    channels stay aligned to each other; ~wall-clock-consistent timeline)
    // 4. log a GAP MARKER: {written_frames, gap_frames, reason: route_change, actual_rate}
    // 5. if out.rate_changed → WARN + roll to a NEW pair of segment WAVs (mic.NNN.wav) at the
    //    new rate + append a manifest line; else keep appending to the same files.
}
```
Add a zero-sample **watchdog** in the same loop (net-new vs meetily): once the device has started (`first_num_buffers() > 0`), if the **mic** ring yields zero samples for > ~3 s → set `rebuild_req` (the mic is the always-on clock master, so mic-count-silence = route drop, distinct from audio silence). The **tap** ring going dry is logged only, never auto-rebuilds — with `tap_auto_start=true` a quiet system genuinely produces no tap buffers, so tap-silence is ambiguous (the documented zero-sample-tap bug's full mitigation, plan §7's "audible-app" heuristic, stays deferred — stated, not silently skipped).

## Files
- `src/capture.rs` — the restructure above (main deliverable).
- `src/main.rs` — drain-loop rebuild handling, silence padding, gap markers, segment rollover, self-check reporting.
- No new deps (uses existing `cidre` + `ringbuf`; `std::time::Instant` for the gap). `Cargo.toml` unchanged.

## Implementation batches (each: `cargo build` → re-sign → drive real audio)
1. **Restructure + `rebuild()`** (no listener yet). Add a temporary `--rebuild-after <s>` spike flag to trigger `rebuild()` manually. Verify: capture continues across a forced rebuild, both WAVs keep growing, per-channel sample counts stay equal, gap padding present. De-risks the restructure independently.
2. **Property listeners → auto-rebuild.** Register on both selectors; `Drop` removes. **First checkpoint = the run-loop question:** plug/unplug headphones, confirm the callback fires while the drain (`sleep`) loop runs. If it does NOT fire → switch to `add_prop_listener_block` + a dedicated `dispatch::Queue` (needs no CFRunLoop; confirmed available). Verify: unplug/plug → auto rebuild logged.
3. **mic-dry watchdog + startup self-check.** Verify self-check via `tccutil reset SystemAudioCaptureRequests com.lucianolupo.meetscribe` (reversible) → expect the TCC-missing message, then re-grant.
4. **Endurance + continuity.** 30+ min run, ≥2 plug/unplug cycles. Then update `RESUME.md` + the `project-meetscribe` memory, `git commit -F` + push.

## Verification (real audio, not just build/tests)
- **Verified (must drive):** Batch-1 forced rebuild continues capture; Batch-2 real headphone plug/unplug auto-triggers rebuild (settles the run-loop unknown); Batch-3 `tccutil reset` yields the TCC-missing message; Batch-4 30-min ≥2-cycle run — both channels grow, equal sample counts per segment, gap markers logged, no panic, `dropped()` bounded, process alive throughout.
- **Alignment check:** after a rebuild, `mic.wav`/`system.wav` (or rolled segments) have equal sample counts and the silence pad sits at the gap (inspect with the existing per-second RMS envelope tooling in `capture/`).
- **Proxy:** `cargo build` clean + re-sign (`codesign --remove-signature` then `--sign 155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C --identifier com.lucianolupo.meetscribe --timestamp=none`); small unit tests for the gap→frames math and the `StartError` classification.

## Assumptions & risks (stated, non-blocking)
- **Rate continuity:** hold the *first* aggregate's rate across rebuilds via `set_nominal_sample_rate`; on the rare "can't hold the rate" case, roll to a new segment file rather than corrupt the header. Pulling the fixed-16 kHz resampler forward stays a Phase 2 concern.
- **Gap length is approximate** (rebuild wall-clock). Cross-channel alignment is exact (equal padding both sides); absolute-timestamp exactness is secondary in v1.
- **Run-loop delivery** of the proc-form listener is the one real unknown — Batch 2 verifies it first, with the block+`dispatch::Queue` fallback ready.
- **Frozen invariants unchanged:** bundle-id `com.lucianolupo.meetscribe`, signing identity `155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C`; re-sign after every build with the explicit `--identifier`.

*Optional:* run `/audit-plan` on this before Batch 1 if you want the multi-agent pre-flight; otherwise the ExitPlanMode approval is the gate.
