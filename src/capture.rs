//! Single-aggregate mic + system-tap capture (macOS).
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
//! The exact buffer ordering + count is a Core Audio HAL runtime property; Phase 0 exists
//! to confirm it, so the proc records the observed layout for the summary.

#[cfg(target_os = "macos")]
mod imp {
    use anyhow::{Result, anyhow};
    use ringbuf::{
        HeapCons, HeapProd, HeapRb,
        traits::{Consumer, Producer, Split},
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    use cidre::{cat, cf, core_audio as ca, ns, os};

    const RING_CAPACITY: usize = 1 << 20; // 1,048,576 f32 per source (~21s @ 48kHz mono)

    /// Shared metadata written by the RT IO-proc, read by the drain thread.
    /// Kept in `Arc`s (NOT read back through the boxed `AudioContext`) so the RT proc's
    /// `&mut AudioContext` is never aliased from another thread.
    struct Shared {
        agg_rate: AtomicU32,
        mic_ch: AtomicU32,
        tap_ch: AtomicU32,
        first_num_buffers: AtomicU32,
        layout_logged: AtomicBool,
    }

    /// Owned exclusively by the RT IO-proc (via a stable-heap `Box`).
    struct AudioContext {
        mic_prod: HeapProd<f32>,
        tap_prod: HeapProd<f32>,
        shared: Arc<Shared>,
    }

    /// The RT audio callback. `IN = 2`: mic buffer + tap buffer arrive as separate
    /// `AudioBuffer`s. We push each source's raw interleaved f32 into its own ring and
    /// de-interleave/downmix later off the RT thread (keeps this proc allocation-free).
    extern "C" fn audio_proc(
        device: ca::Device,
        _now: &cat::AudioTimeStamp,
        input: &cat::AudioBufList<2>,
        _in_time: &cat::AudioTimeStamp,
        _out: &mut cat::AudioBufList<1>,
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

        // buffers[0] -> mic ("You"), buffers[1] -> tap ("Others").
        // Ordering follows the composition (sub_device_list before tap_list); it is
        // human-verified by listening to the two WAVs in Phase 0a.
        let n = num_buffers.min(2);
        for i in 0..n {
            let buf = &input.buffers[i];
            let floats = buf.data_bytes_size as usize / std::mem::size_of::<f32>();
            if buf.data.is_null() || floats == 0 {
                continue;
            }
            let data = unsafe { std::slice::from_raw_parts(buf.data as *const f32, floats) };
            let ch = buf.number_channels.max(1);
            if i == 0 {
                ctx.shared.mic_ch.store(ch, Ordering::Release);
                let _ = ctx.mic_prod.push_slice(data);
            } else {
                ctx.shared.tap_ch.store(ch, Ordering::Release);
                let _ = ctx.tap_prod.push_slice(data);
            }
        }

        // All sub-sources run at the aggregate's (mic clock's) nominal rate; the tap's
        // native rate is drift-resampled into it. So one rate governs both WAVs.
        if let Ok(rate) = device.nominal_sample_rate() {
            ctx.shared.agg_rate.store(rate as u32, Ordering::Release);
        }

        os::Status::NO_ERR
    }

    /// A running single-aggregate capture. RAII: dropping stops the device, destroys the
    /// aggregate, and destroys the tap. Holds raw Core Audio handles — not `Send`.
    pub struct DualCapture {
        _started: ca::hardware::StartedDevice<ca::AggregateDevice>,
        _tap: ca::TapGuard,
        _ctx: Box<AudioContext>,
        mic_cons: HeapCons<f32>,
        tap_cons: HeapCons<f32>,
        shared: Arc<Shared>,

        // Setup-time diagnostics (native, pre-aggregation).
        pub mic_uid: String,
        pub mic_native_rate: u32,
        pub mic_native_channels: u32,
        pub tap_native_rate: u32,
        pub tap_native_channels: u32,
    }

    impl DualCapture {
        pub fn start() -> Result<Self> {
            // 1. Mic = default input device = clock master.
            let mic = ca::System::default_input_device()
                .map_err(|e| anyhow!("no default input device (mic): {e:?}"))?;
            let mic_uid = mic
                .uid()
                .map_err(|e| anyhow!("mic uid: {e:?}"))?;
            let mic_asbd = mic
                .input_asbd()
                .map_err(|e| anyhow!("mic input format: {e:?}"))?;
            let mic_uid_str = mic_uid.to_string();

            // 2. Global mono system tap. Empty exclude list = capture ALL system output
            //    (auto-includes Electron/Teams renderer children). Creating the tap is
            //    what triggers the system-audio TCC prompt (needs NSAudioCaptureUsageDescription).
            let excludes = ns::Array::new();
            let tap_desc = ca::TapDesc::with_mono_global_tap_excluding_processes(&excludes);
            let tap = tap_desc
                .create_process_tap()
                .map_err(|e| anyhow!("create process tap (system audio): {e:?}"))?;
            let tap_uid = tap.uid().map_err(|e| anyhow!("tap uid: {e:?}"))?;
            let tap_asbd = tap.asbd().map_err(|e| anyhow!("tap format: {e:?}"))?;

            // 3. Mic sub-device dict — just the UID (mic is the clock; no drift comp on the master).
            let mic_sub = cf::DictionaryOf::with_keys_values(
                &[ca::sub_device_keys::uid()],
                &[mic_uid.as_type_ref()],
            );

            // 4. Sub-tap dict — UID + drift compensation (Bool true) since the tap is the
            //    non-clock member. (kAudioSubDeviceDriftCompensationMaxQuality is an available
            //    tuning knob for Phase 1; both proven refs use the bare Bool.)
            let sub_tap = cf::DictionaryOf::with_keys_values(
                &[
                    ca::hardware::sub_tap_keys::uid(),
                    ca::hardware::sub_tap_keys::drift_compensation(),
                ],
                &[tap_uid.as_type_ref(), cf::Boolean::value_true().as_type_ref()],
            );

            // 5. Aggregate composition = meetily's dict + sub_device_list (mic) restored,
            //    with mic as main_sub_device (clock master).
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

            let agg = ca::AggregateDevice::with_desc(&agg_desc)
                .map_err(|e| anyhow!("create aggregate device: {e:?}"))?;

            // 6. Rings + shared metadata.
            let (mic_prod, mic_cons) = HeapRb::<f32>::new(RING_CAPACITY).split();
            let (tap_prod, tap_cons) = HeapRb::<f32>::new(RING_CAPACITY).split();

            let seed_rate = agg.nominal_sample_rate().unwrap_or(mic_asbd.sample_rate) as u32;
            let shared = Arc::new(Shared {
                agg_rate: AtomicU32::new(seed_rate),
                mic_ch: AtomicU32::new(0),
                tap_ch: AtomicU32::new(0),
                first_num_buffers: AtomicU32::new(0),
                layout_logged: AtomicBool::new(false),
            });

            let mut ctx = Box::new(AudioContext {
                mic_prod,
                tap_prod,
                shared: shared.clone(),
            });

            // 7. Register the RT proc (raw ptr into the boxed ctx heap) and start.
            //    Starting the device with a mic sub-device triggers the microphone TCC prompt.
            let proc_id = agg
                .create_io_proc_id(audio_proc, Some(ctx.as_mut()))
                .map_err(|e| anyhow!("create IO proc: {e:?}"))?;
            let started = ca::device_start(agg, Some(proc_id))
                .map_err(|e| anyhow!("start aggregate device: {e:?}"))?;

            Ok(Self {
                _started: started,
                _tap: tap,
                _ctx: ctx,
                mic_cons,
                tap_cons,
                shared,
                mic_uid: mic_uid_str,
                mic_native_rate: mic_asbd.sample_rate as u32,
                mic_native_channels: mic_asbd.channels_per_frame,
                tap_native_rate: tap_asbd.sample_rate as u32,
                tap_native_channels: tap_asbd.channels_per_frame,
            })
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
        pub fn first_num_buffers(&self) -> u32 {
            self.shared.first_num_buffers.load(Ordering::Acquire)
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
        pub fn start() -> Result<Self> {
            bail!("meetscribe capture is macOS-only")
        }
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
    }
}

#[cfg(not(target_os = "macos"))]
pub use stub::DualCapture;
