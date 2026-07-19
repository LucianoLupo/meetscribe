//! Single-aggregate mic + system-tap capture (macOS), with route-change rebuild.
//!
//! ONE `AudioHardwareCreateAggregateDevice` on ONE clock:
//!   - the built-in MIC as an INPUT sub-device  -> "You"     (clock master)
//!   - a GLOBAL process tap as a sub-tap         -> "Others"  (drift-compensated)
//!
//! Ported/adapted from meetily's `core_audio.rs` (MIT). meetily runs the tap ONLY and
//! deliberately removed `sub_device_list` (to avoid echo tapping the device it also
//! output to). We ADD the mic sub-device back and keep the two sources SEPARATE — that
//! separation is our free channel-based you-vs-them labeling, and one shared clock is
//! what keeps them aligned (the audit's blocker-1 fix vs two drifting streams).
//!
//! The aggregate presents mic and tap as SEPARATE `AudioBuffer`s within the one IO-proc
//! input list (one buffer per sub-source), so the proc uses `IN = 2` and reads the raw
//! `AudioBufList` directly — `av::AudioPcmBuf` only reaches buffer 0 and fails past 2ch.
//!
//! Phase 1: the physical device instance (aggregate + tap + IO proc) is SWAPPABLE while
//! the rings, their producers (in a pinned `Box<AudioContext>`), the consumers `main`
//! drains, and the `Arc<Shared>` counters stay STABLE for the whole process. `rebuild()`
//! tears down the old instance and builds a fresh aggregate that writes into the SAME
//! rings, so a default-device change (headphones plugged/unplugged) never breaks the
//! drain loop or the output files. Private aggregates are per-process and reclaimed by
//! coreaudiod on exit (incl. crash), so there is nothing to clean up at launch.

/// Result of a `DualCapture::rebuild()` — the caller uses this to keep the two output
/// channels aligned across the unavoidable audio gap a rebuild introduces.
#[derive(Debug, Clone, Copy)]
pub struct RebuildOutcome {
    /// Approx samples-per-channel lost during the rebuild (wall-clock × rate). Pad BOTH
    /// channels with exactly this many silence frames so they stay aligned to each other.
    pub gap_frames: usize,
    /// The new aggregate's nominal rate. Equals `target_rate` unless the device refused it.
    pub actual_rate: u32,
    /// True if the new aggregate could NOT be held at the original rate — the fixed-header
    /// WAV must roll to a new segment file rather than be written at the wrong rate.
    pub rate_changed: bool,
}

/// Why capture couldn't start — the startup self-check. Its `Display` is the actionable
/// remedy shown to the user, distinguishing a missing TCC grant from a dropped route.
#[derive(Debug)]
pub enum StartError {
    /// No usable default input device — the mic route is gone (headset pulled, no input).
    MicMissing(String),
    /// Creating the system-audio process tap failed — almost always a missing/denied TCC
    /// grant (the "record system audio" permission).
    SystemAudioTccMissing(String),
    /// The audio route changed out from under us while assembling/starting the aggregate.
    RouteDropped(String),
    /// Anything else.
    Other(String),
}

impl std::fmt::Display for StartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartError::MicMissing(d) => write!(
                f,
                "no input device (mic route dropped?) — connect a microphone / reconnect \
                 your headset, then retry. [{d}]"
            ),
            StartError::SystemAudioTccMissing(d) => write!(
                f,
                "system-audio recording permission is missing — approve meetscribe under \
                 System Settings → Privacy & Security → Screen & System Audio Recording, or run \
                 `tccutil reset SystemAudioCaptureRequests com.lucianolupo.meetscribe`. [{d}]"
            ),
            StartError::RouteDropped(d) => write!(
                f,
                "the audio route changed while starting capture; retry. [{d}]"
            ),
            StartError::Other(d) => write!(f, "{d}"),
        }
    }
}

impl std::error::Error for StartError {}

/// Convert an elapsed rebuild gap to a per-channel silence-pad frame count at `rate`.
/// Extracted (and pure) so the frame math is unit-testable off a real device.
#[must_use]
pub fn gap_to_frames(elapsed: std::time::Duration, rate: u32) -> usize {
    (elapsed.as_secs_f64() * rate as f64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn gap_frames_math() {
        assert_eq!(gap_to_frames(Duration::from_millis(0), 16000), 0);
        assert_eq!(gap_to_frames(Duration::from_millis(50), 16000), 800);
        assert_eq!(gap_to_frames(Duration::from_secs(1), 48000), 48000);
        // Truncates sub-frame remainders rather than rounding up.
        assert_eq!(gap_to_frames(Duration::from_micros(1), 16000), 0);
    }

    #[test]
    fn start_error_remedies() {
        // The TCC variant must point the user at the exact fix.
        let e = StartError::SystemAudioTccMissing("raw".into()).to_string();
        assert!(e.contains("tccutil reset SystemAudioCaptureRequests com.lucianolupo.meetscribe"));
        assert!(e.contains("Screen & System Audio Recording"));
        // The mic variant is about a missing input, not permissions.
        let m = StartError::MicMissing("raw".into()).to_string();
        assert!(m.contains("input device") && !m.contains("tccutil"));
        // Raw detail is always carried through for debugging.
        assert!(StartError::Other("boom".into()).to_string().contains("boom"));
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use anyhow::Result;
    use ringbuf::{
        HeapCons, HeapProd, HeapRb,
        traits::{Consumer, Producer, Split},
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Instant;

    use cidre::{cat, cf, core_audio as ca, ns, os};

    use super::{RebuildOutcome, StartError, gap_to_frames};

    const RING_CAPACITY: usize = 1 << 20; // 1,048,576 f32 per source (~21s @ 48kHz mono)

    /// Shared metadata written by the RT IO-proc, read by the drain thread.
    /// Kept in `Arc`s (NOT read back through the boxed `AudioContext`) so the RT proc's
    /// `&mut AudioContext` is never aliased from another thread.
    struct Shared {
        agg_rate: AtomicU32,
        mic_ch: AtomicU32,
        tap_ch: AtomicU32,
        first_num_buffers: AtomicU32,
        dropped: AtomicU32,
        layout_logged: AtomicBool,
    }

    impl Shared {
        /// Reset the per-instance observations before a (re)build. `dropped` is cumulative
        /// across rebuilds and is deliberately preserved.
        fn reset_instance_observations(&self) {
            self.mic_ch.store(0, Ordering::Release);
            self.tap_ch.store(0, Ordering::Release);
            self.first_num_buffers.store(0, Ordering::Release);
            self.layout_logged.store(false, Ordering::Release);
        }
    }

    /// Owned exclusively by the RT IO-proc (via a stable-heap `Box`). Lives for the whole
    /// process; a rebuild registers a fresh IO proc against this SAME box (same producers
    /// → same rings), so its heap address must never change.
    struct AudioContext {
        mic_prod: HeapProd<f32>,
        tap_prod: HeapProd<f32>,
        shared: Arc<Shared>,
    }

    /// Setup-time diagnostics (native, pre-aggregation), refreshed on each (re)build.
    struct Diag {
        mic_uid: String,
        mic_native_rate: u32,
        mic_native_channels: u32,
        tap_native_rate: u32,
        tap_native_channels: u32,
    }

    /// Everything bound to ONE physical aggregate instance. `!Send`. Dropping it performs
    /// the HAL teardown in the correct order: `started` (stop device → destroy aggregate)
    /// then `tap` (destroy process tap). Field order is load-bearing — do not reorder.
    /// The fields are never read; they are held purely so their `Drop` runs (RAII guards).
    #[allow(dead_code)]
    struct DeviceInstance {
        started: ca::hardware::StartedDevice<ca::AggregateDevice>,
        tap: ca::TapGuard,
    }

    /// The RT audio callback. Mic buffer + tap buffer arrive as separate `AudioBuffer`s.
    /// We push each source's raw interleaved f32 into its own ring and de-interleave/downmix
    /// later off the RT thread (keeps this proc allocation- and syscall-free — no HAL reads).
    ///
    /// The typed views are `AudioBufList<0>` (an 8-byte header only) so the reference is
    /// never wider than CoreAudio's real variable-length `AudioBufferList` allocation, whatever
    /// `number_buffers` it delivers. Individual buffers are read via a raw offset from the
    /// contiguous `buffers` array — the correct idiom for a variable-length buffer list.
    extern "C" fn audio_proc(
        _device: ca::Device,
        _now: &cat::AudioTimeStamp,
        input: &cat::AudioBufList<0>,
        _in_time: &cat::AudioTimeStamp,
        _out: &mut cat::AudioBufList<0>,
        _out_time: &cat::AudioTimeStamp,
        ctx: Option<&mut AudioContext>,
    ) -> os::Status {
        let ctx = match ctx {
            Some(c) => c,
            None => return os::Status::NO_ERR,
        };

        let num_buffers = input.number_buffers as usize;
        if !ctx.shared.layout_logged.swap(true, Ordering::AcqRel) {
            ctx.shared
                .first_num_buffers
                .store(num_buffers as u32, Ordering::Release);
        }

        // buffers[0] -> mic ("You"), buffers[1] -> tap ("Others"). Ordering follows the
        // composition (sub_device_list before tap_list); human-verified by listening in 0a.
        let n = num_buffers.min(2);
        let base = input.buffers.as_ptr(); // *const Buf at the buffers[] field offset
        for i in 0..n {
            // SAFETY: buffers[0..number_buffers] are contiguous `#[repr(C)]` Buf entries in
            // CoreAudio's allocation; `i < min(number_buffers, 2)` keeps this in bounds even
            // though our typed view declares a zero-length array.
            let buf = unsafe { &*base.add(i) };
            let floats = buf.data_bytes_size as usize / std::mem::size_of::<f32>();
            if buf.data.is_null() || floats == 0 {
                continue;
            }
            let data = unsafe { std::slice::from_raw_parts(buf.data as *const f32, floats) };
            let ch = buf.number_channels.max(1);
            let pushed = if i == 0 {
                ctx.shared.mic_ch.store(ch, Ordering::Release);
                ctx.mic_prod.push_slice(data)
            } else {
                ctx.shared.tap_ch.store(ch, Ordering::Release);
                ctx.tap_prod.push_slice(data)
            };
            if pushed < data.len() {
                ctx.shared
                    .dropped
                    .fetch_add((data.len() - pushed) as u32, Ordering::Relaxed);
            }
        }

        os::Status::NO_ERR
    }

    /// The two hardware-object properties whose change means the audio route moved
    /// (headphones plugged/unplugged, a device appearing/disappearing): the system default
    /// output and default input. Reconstructed identically for add and remove.
    fn listener_addrs() -> [ca::PropAddr; 2] {
        [
            ca::PropSelector::HW_DEFAULT_OUTPUT_DEVICE.global_addr(),
            ca::PropSelector::HW_DEFAULT_INPUT_DEVICE.global_addr(),
        ]
    }

    /// Fires on a HAL notification thread when the default output/input device changes. Does
    /// the minimum: flip the rebuild flag and return. NO Core Audio / HAL calls here (that
    /// would risk re-entrancy on the notification thread) — the drain loop does the rebuild.
    extern "C-unwind" fn on_route_change(
        _obj: ca::Obj,
        n: u32,
        addrs: *const ca::PropAddr,
        ctx: *mut AtomicBool,
    ) -> os::Status {
        if ctx.is_null() {
            return os::Status::NO_ERR;
        }
        let addrs = unsafe { std::slice::from_raw_parts(addrs, n as usize) };
        let route_moved = addrs.iter().any(|a| {
            a.selector == ca::PropSelector::HW_DEFAULT_OUTPUT_DEVICE
                || a.selector == ca::PropSelector::HW_DEFAULT_INPUT_DEVICE
        });
        if route_moved {
            // SAFETY: `ctx` is `Arc::as_ptr(&rebuild_req)` — the `AtomicBool` inside an Arc
            // that `DualCapture` keeps alive and whose listener it removes (in `Drop`) before
            // the Arc is released, so this pointer is always valid here. `store` is `&self`.
            unsafe { (*ctx).store(true, Ordering::Release) };
        }
        os::Status::NO_ERR
    }

    /// Build ONE physical aggregate instance (mic input sub-device + global system tap) and
    /// register the RT IO proc against `ctx` (the pinned box). Used by both `start()` (with
    /// `target_rate = None` → adopt whatever rate the device picks) and `rebuild()` (with
    /// `target_rate = Some(rate)` → force the original rate so the fixed-header WAV stays
    /// single-rate). Returns the running instance, refreshed diagnostics, and the actual rate.
    fn build_device(
        ctx: &mut AudioContext,
        target_rate: Option<u32>,
    ) -> Result<(DeviceInstance, Diag, u32), StartError> {
        // A rebuild reuses the same rings/counters; wipe the per-instance observations so the
        // new device re-establishes its layout (cumulative `dropped` is preserved).
        ctx.shared.reset_instance_observations();

        // 1. Mic = default input device = clock master.
        let mic = ca::System::default_input_device()
            .map_err(|e| StartError::MicMissing(format!("no default input device: {e:?}")))?;
        let mic_uid = mic
            .uid()
            .map_err(|e| StartError::MicMissing(format!("mic uid: {e:?}")))?;
        let mic_asbd = mic
            .input_asbd()
            .map_err(|e| StartError::MicMissing(format!("mic input format: {e:?}")))?;
        let mic_uid_str = mic_uid.to_string();

        // 2. Global mono system tap. Empty exclude list = capture ALL system output
        //    (auto-includes Electron/Teams renderer children). Creating the tap is
        //    what triggers the system-audio TCC prompt (needs NSAudioCaptureUsageDescription);
        //    its failure almost always means that grant is missing/denied.
        let excludes = ns::Array::new();
        let tap_desc = ca::TapDesc::with_mono_global_tap_excluding_processes(&excludes);
        let tap = tap_desc.create_process_tap().map_err(|e| {
            StartError::SystemAudioTccMissing(format!("create process tap: {e:?}"))
        })?;
        let tap_uid = tap
            .uid()
            .map_err(|e| StartError::Other(format!("tap uid: {e:?}")))?;
        let tap_asbd = tap
            .asbd()
            .map_err(|e| StartError::Other(format!("tap format: {e:?}")))?;

        // 3. Mic sub-device dict — just the UID (mic is the clock; no drift comp on the master).
        let mic_sub = cf::DictionaryOf::with_keys_values(
            &[ca::sub_device_keys::uid()],
            &[mic_uid.as_type_ref()],
        );

        // 4. Sub-tap dict — UID + drift compensation (Bool true) since the tap is the
        //    non-clock member.
        let sub_tap = cf::DictionaryOf::with_keys_values(
            &[
                ca::hardware::sub_tap_keys::uid(),
                ca::hardware::sub_tap_keys::drift_compensation(),
            ],
            &[tap_uid.as_type_ref(), cf::Boolean::value_true().as_type_ref()],
        );

        // 5. Aggregate composition = meetily's dict + sub_device_list (mic) restored, with
        //    mic as main_sub_device (clock master). Fresh private UID each build.
        let agg_uid = cf::Uuid::new().to_cf_string();
        let agg_desc = cf::DictionaryOf::with_keys_values(
            &[
                ca::aggregate_device_keys::is_private(),
                ca::aggregate_device_keys::is_stacked(),
                ca::aggregate_device_keys::tap_auto_start(),
                ca::aggregate_device_keys::name(),
                ca::aggregate_device_keys::main_sub_device(),
                ca::aggregate_device_keys::uid(),
                ca::aggregate_device_keys::sub_device_list(),
                ca::aggregate_device_keys::tap_list(),
            ],
            &[
                cf::Boolean::value_true().as_type_ref(),
                cf::Boolean::value_false(),
                cf::Boolean::value_true(),
                cf::str!(c"meetscribe-agg").as_type_ref(),
                &mic_uid,
                &agg_uid,
                &cf::ArrayOf::from_slice(&[mic_sub.as_ref()]),
                &cf::ArrayOf::from_slice(&[sub_tap.as_ref()]),
            ],
        );

        let mut agg = ca::AggregateDevice::with_desc(&agg_desc)
            .map_err(|e| StartError::RouteDropped(format!("create aggregate device: {e:?}")))?;

        // Force the original rate on a rebuild so mic.wav/system.wav stay single-rate. If the
        // new sub-device composition can't hold it, log and fall through to the device's rate
        // (the caller sees rate_changed and rolls to a new segment file).
        // Try to hold the previous segment's rate so the fixed-header WAV can continue. This
        // only succeeds when the new sub-device composition supports that rate (e.g. the same
        // device); a genuinely different-rate device (16k headset → 48k built-in mic) rejects
        // it, and the caller rolls to a new segment. Not an error — the expected fallback.
        if let Some(rate) = target_rate
            && agg.set_nominal_sample_rate(rate as f64).is_err()
        {
            log::info!(
                "aggregate can't hold {rate} Hz with the new device (rates differ) \
                 → output will roll to a new segment at the device's rate"
            );
        }
        let actual_rate = agg.nominal_sample_rate().unwrap_or(mic_asbd.sample_rate) as u32;
        ctx.shared.agg_rate.store(actual_rate.max(1), Ordering::Release);

        // Register the RT proc (raw ptr into the boxed ctx heap) against the SAME box, then
        // start. Starting the device with a mic sub-device triggers the microphone TCC prompt.
        let proc_id = agg
            .create_io_proc_id(audio_proc, Some(ctx))
            .map_err(|e| StartError::RouteDropped(format!("create IO proc: {e:?}")))?;
        let started = ca::device_start(agg, Some(proc_id))
            .map_err(|e| StartError::RouteDropped(format!("start aggregate device: {e:?}")))?;

        let diag = Diag {
            mic_uid: mic_uid_str,
            mic_native_rate: mic_asbd.sample_rate as u32,
            mic_native_channels: mic_asbd.channels_per_frame,
            tap_native_rate: tap_asbd.sample_rate as u32,
            tap_native_channels: tap_asbd.channels_per_frame,
        };
        Ok((DeviceInstance { started, tap }, diag, actual_rate))
    }

    /// A running single-aggregate capture. The physical device is swappable (`device`); the
    /// rings/producers (`ctx`), consumers, and `shared` counters are process-stable so a
    /// rebuild never invalidates the drain loop. Holds raw Core Audio handles — not `Send`.
    pub struct DualCapture {
        // Swappable: `None` only for the brief window inside `rebuild()`.
        device: Option<DeviceInstance>,

        // Stable for the whole process lifetime.
        ctx: Box<AudioContext>,
        mic_cons: HeapCons<f32>,
        tap_cons: HeapCons<f32>,
        shared: Arc<Shared>,
        rebuild_req: Arc<AtomicBool>,
        /// Raw pointer to the `AtomicBool` inside `rebuild_req`, as handed to the HAL
        /// listeners; kept so `Drop` can remove them with the identical (fn, addr, ctx).
        listener_ctx: *mut AtomicBool,
        target_rate: u32,

        // Diagnostics — refreshed on each (re)build.
        pub mic_uid: String,
        pub mic_native_rate: u32,
        pub mic_native_channels: u32,
        pub tap_native_rate: u32,
        pub tap_native_channels: u32,
    }

    impl DualCapture {
        pub fn start() -> Result<Self, StartError> {
            // Rings + shared metadata + the rebuild flag, all created ONCE.
            let (mic_prod, mic_cons) = HeapRb::<f32>::new(RING_CAPACITY).split();
            let (tap_prod, tap_cons) = HeapRb::<f32>::new(RING_CAPACITY).split();
            let shared = Arc::new(Shared {
                agg_rate: AtomicU32::new(1),
                mic_ch: AtomicU32::new(0),
                tap_ch: AtomicU32::new(0),
                first_num_buffers: AtomicU32::new(0),
                dropped: AtomicU32::new(0),
                layout_logged: AtomicBool::new(false),
            });
            let mut ctx = Box::new(AudioContext {
                mic_prod,
                tap_prod,
                shared: shared.clone(),
            });

            // First build adopts whatever rate the device picks; that becomes target_rate.
            let (device, diag, actual_rate) = build_device(ctx.as_mut(), None)?;

            // Register the default-device listeners. `Arc::as_ptr` gives a stable pointer to
            // the AtomicBool inside the Arc (the allocation doesn't move when the Arc handle
            // moves into Self); the listener only ever does an atomic store through it. If a
            // registration fails, `?` returns and the local `device`/`ctx` drop cleanly (RAII
            // stop→destroy) — reverse-declaration order drops `device` before `ctx`.
            let rebuild_req = Arc::new(AtomicBool::new(false));
            let listener_ctx = Arc::as_ptr(&rebuild_req) as *mut AtomicBool;
            for addr in listener_addrs() {
                ca::System::OBJ
                    .add_prop_listener(&addr, on_route_change, listener_ctx)
                    .map_err(|e| StartError::Other(format!("add default-device route listener: {e:?}")))?;
            }

            Ok(Self {
                device: Some(device),
                ctx,
                mic_cons,
                tap_cons,
                shared,
                rebuild_req,
                listener_ctx,
                target_rate: actual_rate,
                mic_uid: diag.mic_uid,
                mic_native_rate: diag.mic_native_rate,
                mic_native_channels: diag.mic_native_channels,
                tap_native_rate: diag.tap_native_rate,
                tap_native_channels: diag.tap_native_channels,
            })
        }

        /// Tear down the current aggregate/tap/IO-proc and build a fresh aggregate that writes
        /// into the SAME rings (the same producers in `self.ctx`). `self.mic_cons`/`tap_cons`
        /// are untouched, so the drain loop + WAV writers continue. MUST be called on the
        /// thread that owns `self` (Core Audio handles are `!Send`; none crosses a boundary).
        pub fn rebuild(&mut self) -> Result<RebuildOutcome> {
            let t0 = Instant::now();
            // RAII teardown FIRST — dropping the old instance stops the IO proc (so the RT
            // thread stops touching self.ctx) BEFORE we reuse ctx for the new proc.
            self.device = None;

            let (device, diag, actual_rate) = build_device(self.ctx.as_mut(), Some(self.target_rate))?;
            let gap = t0.elapsed();

            self.device = Some(device);
            self.mic_uid = diag.mic_uid;
            self.mic_native_rate = diag.mic_native_rate;
            self.mic_native_channels = diag.mic_native_channels;
            self.tap_native_rate = diag.tap_native_rate;
            self.tap_native_channels = diag.tap_native_channels;

            // `rate_changed` is relative to the PREVIOUS segment's rate. Then ADOPT the
            // achieved rate as the new target, so a device that simply runs at a different
            // rate (e.g. built-in mic @ 48k after unplugging a 16k headset) rolls ONCE and
            // subsequent rebuilds at that same rate pad instead of rolling forever.
            let rate_changed = actual_rate != self.target_rate;
            self.target_rate = actual_rate;
            let gap_frames = gap_to_frames(gap, actual_rate);
            Ok(RebuildOutcome {
                gap_frames,
                actual_rate,
                rate_changed,
            })
        }

        /// True (once) if a rebuild has been requested since the last check. Poll each drain tick.
        #[must_use]
        pub fn rebuild_requested(&self) -> bool {
            self.rebuild_req.swap(false, Ordering::AcqRel)
        }

        /// Request a rebuild on the next drain tick. Called by the route-change listener
        /// (Phase 1 Batch 2) and the mic-dry watchdog (Batch 3); also drivable manually.
        pub fn request_rebuild(&self) {
            self.rebuild_req.store(true, Ordering::Release);
        }

        /// Pop everything currently buffered from both rings, appending raw interleaved
        /// f32 to the caller's accumulators. Call repeatedly so the rings never overflow.
        pub fn drain_into(&mut self, mic_raw: &mut Vec<f32>, tap_raw: &mut Vec<f32>) {
            drain_ring(&mut self.mic_cons, mic_raw);
            drain_ring(&mut self.tap_cons, tap_raw);
        }

        /// Aggregate nominal rate — the sample rate of BOTH mic and tap in the IO proc.
        pub fn aggregate_rate(&self) -> u32 {
            self.shared.agg_rate.load(Ordering::Acquire)
        }
        /// Channels observed in the mic buffer at runtime (0 until first callback).
        pub fn observed_mic_channels(&self) -> u32 {
            self.shared.mic_ch.load(Ordering::Acquire)
        }
        /// Channels observed in the tap buffer at runtime (0 until first callback).
        pub fn observed_tap_channels(&self) -> u32 {
            self.shared.tap_ch.load(Ordering::Acquire)
        }
        /// `number_buffers` seen on the first callback — 2 = the expected mic+tap layout.
        /// Reset to 0 by a rebuild until the new device fires its first callback.
        pub fn first_num_buffers(&self) -> u32 {
            self.shared.first_num_buffers.load(Ordering::Acquire)
        }
        /// Total f32 samples the RT proc dropped because a ring was full (drain fell behind).
        /// Cumulative across rebuilds.
        pub fn dropped(&self) -> u32 {
            self.shared.dropped.load(Ordering::Acquire)
        }
    }

    impl Drop for DualCapture {
        fn drop(&mut self) {
            // Remove the HAL listeners FIRST (identical fn+addr+ctx), so no route callback can
            // fire mid-teardown. The device (`Option<DeviceInstance>`) and `ctx` then drop via
            // field order — `device` before `ctx` — stopping the IO proc before its ctx frees.
            // `rebuild_req` (the Arc) drops last, so `listener_ctx` stays valid through removal.
            for addr in listener_addrs() {
                let _ = ca::System::OBJ.remove_prop_listener(&addr, on_route_change, self.listener_ctx);
            }
        }
    }

    fn drain_ring(cons: &mut HeapCons<f32>, out: &mut Vec<f32>) {
        let mut tmp = [0f32; 8192];
        loop {
            let n = cons.pop_slice(&mut tmp);
            if n == 0 {
                break;
            }
            out.extend_from_slice(&tmp[..n]);
        }
    }
}

#[cfg(target_os = "macos")]
pub use imp::DualCapture;

#[cfg(not(target_os = "macos"))]
mod stub {
    use super::{RebuildOutcome, StartError};
    use anyhow::{Result, bail};

    // Field parity with the macOS impl so main.rs compiles cross-platform.
    pub struct DualCapture {
        pub mic_uid: String,
        pub mic_native_rate: u32,
        pub mic_native_channels: u32,
        pub tap_native_rate: u32,
        pub tap_native_channels: u32,
    }

    impl DualCapture {
        pub fn start() -> Result<Self, StartError> {
            Err(StartError::Other("meetscribe capture is macOS-only".into()))
        }
        pub fn rebuild(&mut self) -> Result<RebuildOutcome> {
            bail!("meetscribe capture is macOS-only")
        }
        #[must_use]
        pub fn rebuild_requested(&self) -> bool {
            false
        }
        pub fn request_rebuild(&self) {}
        pub fn drain_into(&mut self, _mic: &mut Vec<f32>, _tap: &mut Vec<f32>) {}
        pub fn aggregate_rate(&self) -> u32 {
            0
        }
        pub fn observed_mic_channels(&self) -> u32 {
            0
        }
        pub fn observed_tap_channels(&self) -> u32 {
            0
        }
        pub fn first_num_buffers(&self) -> u32 {
            0
        }
        pub fn dropped(&self) -> u32 {
            0
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use stub::DualCapture;
