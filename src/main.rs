//! meetscribe — Phase 0 capture spike.
//!
//! Proves the SINGLE-aggregate design: one Core Audio aggregate device on one clock
//! capturing the mic ("You") and a global system tap ("Others") as two SEPARATE,
//! time-aligned channels. Writes `mic.wav` + `system.wav` and reports levels + the
//! observed buffer layout.
//!
//! Phase 0a: run during a REAL call → both WAVs non-zero + intelligible (your voice in
//!           mic.wav, the remote voices in system.wav).
//! Phase 0b: your voice then the remote voice must land in the right time order across
//!           the two files (single clock ⇒ they should be aligned frame-for-frame).
//!
//! Usage:
//!   meetscribe [--out <dir>] [--seconds <n>]
//!   (no --seconds ⇒ records until you press Enter)

mod capture;

use anyhow::{Context, Result};
use capture::DualCapture;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

struct Args {
    out_dir: PathBuf,
    seconds: Option<u64>,
    /// Phase 1 Batch 1: force a rebuild this many seconds in, to exercise the
    /// request→poll→rebuild path without a real route change.
    rebuild_after: Option<u64>,
}

fn parse_args() -> Args {
    let mut out_dir = PathBuf::from("capture");
    let mut seconds = None;
    let mut rebuild_after = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" | "-o" => {
                if let Some(v) = it.next() {
                    out_dir = PathBuf::from(v);
                }
            }
            "--seconds" | "-s" => {
                seconds = it.next().and_then(|v| v.parse().ok());
            }
            "--rebuild-after" => {
                rebuild_after = it.next().and_then(|v| v.parse().ok());
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: meetscribe [--out <dir>] [--seconds <n>] [--rebuild-after <n>]"
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }
    Args {
        out_dir,
        seconds,
        rebuild_after,
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = parse_args();
    std::fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("create output dir {}", args.out_dir.display()))?;

    log::info!("meetscribe capture spike — starting single-aggregate mic+tap capture");
    log::info!("(first run triggers TWO prompts: Microphone and \"record system audio\" — approve both)");

    // Startup self-check: on failure, print the classified, actionable remedy (TCC grant
    // missing vs mic/route dropped) rather than a generic error.
    let mut cap = match DualCapture::start() {
        Ok(c) => c,
        Err(e) => {
            log::error!("cannot start capture — {e}");
            return Err(anyhow::anyhow!("capture startup failed"));
        }
    };

    log::info!(
        "mic (\"You\"):   uid={} native={} Hz, {} ch",
        cap.mic_uid,
        cap.mic_native_rate,
        cap.mic_native_channels
    );
    log::info!(
        "tap (\"Others\"): native={} Hz, {} ch  (Teams typically 24000)",
        cap.tap_native_rate,
        cap.tap_native_channels
    );

    // The aggregate runs at the mic clock's rate; the tap's native rate is drift-resampled
    // into it, so one rate governs both WAVs. It's fixed for the device's lifetime, so we
    // read the seed once here rather than polling inside the RT proc.
    // `rate` and the output files can change if a rebuild can't hold the original rate —
    // then we roll to a new segment. In the common case (rate held) both stay put.
    let mut rate = cap.aggregate_rate().max(1);
    log::info!("aggregate rate = {rate} Hz (both channels)");

    let mut segment: u32 = 0;
    let mut mic_path = segment_path(&args.out_dir, "mic", segment);
    let mut sys_path = segment_path(&args.out_dir, "system", segment);
    let mut mic_w = WavStream::create(&mic_path, rate).context("open mic.wav")?;
    let mut sys_w = WavStream::create(&sys_path, rate).context("open system.wav")?;

    // Stop control: a --seconds deadline, else press Enter. Only wire the stdin thread
    // when there's no deadline — a non-interactive stdin hits EOF instantly and would
    // otherwise stop the capture on the first loop iteration.
    let stop = Arc::new(AtomicBool::new(false));
    if args.seconds.is_none() {
        let stop = stop.clone();
        std::thread::spawn(move || {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
            stop.store(true, Ordering::Release);
        });
        log::info!("recording… press Enter to stop.");
    } else {
        log::info!("recording for {}s…", args.seconds.unwrap());
    }

    let deadline = args.seconds.map(|s| Instant::now() + Duration::from_secs(s));
    let started_at = Instant::now();
    // One-shot forced-rebuild trigger (Batch 1 exercise). None once it has fired.
    let mut rebuild_deadline = args
        .rebuild_after
        .map(|s| started_at + Duration::from_secs(s));
    let mut rebuilds = 0u32;
    // Consecutive failed rebuild attempts (transient route-transition failures). A rebuild
    // failure is NEVER fatal — we re-arm and retry rather than tear down a live recording.
    let mut rebuild_failures = 0u32;
    // A single physical route change emits a BURST of HAL notifications; wait for the route
    // to settle before rebuilding once, so we don't spawn empty micro-segments.
    const REBUILD_SETTLE: Duration = Duration::from_millis(500);
    let mut rebuild_pending_since: Option<Instant> = None;
    // Mic-dry watchdog: the mic is the always-on clock master, so once the device has fired
    // its first callback, a mic that stops delivering FRAMES (not just goes quiet) for this
    // long means the input route dropped — a case a default-device listener can miss.
    const MIC_DRY_TIMEOUT: Duration = Duration::from_secs(3);
    let mut mic_last_seen = Instant::now();
    // Small per-drain scratch buffers (reused) — only a ring's-worth is ever resident.
    let mut mic_raw: Vec<f32> = Vec::new();
    let mut tap_raw: Vec<f32> = Vec::new();

    loop {
        mic_raw.clear();
        tap_raw.clear();
        cap.drain_into(&mut mic_raw, &mut tap_raw);
        mic_w.write(&mic_raw, cap.observed_mic_channels());
        sys_w.write(&tap_raw, cap.observed_tap_channels());

        // Mic-dry watchdog. The mic delivers frames continuously once the device has started,
        // even during system silence, so a dry mic = a dropped input route. Gated on the
        // device having started, and paused while a rebuild is already pending/settling.
        if !mic_raw.is_empty() {
            mic_last_seen = Instant::now();
        }
        if cap.first_num_buffers() > 0
            && rebuild_pending_since.is_none()
            && mic_last_seen.elapsed() > MIC_DRY_TIMEOUT
        {
            log::warn!(
                "mic delivered no frames for >{}s while the device is running → requesting \
                 rebuild (input route dropped?)",
                MIC_DRY_TIMEOUT.as_secs()
            );
            cap.request_rebuild();
            mic_last_seen = Instant::now();
        }

        // Fire the one-shot forced rebuild once its deadline passes.
        if let Some(d) = rebuild_deadline
            && Instant::now() >= d
        {
            log::info!("--rebuild-after fired → requesting rebuild");
            cap.request_rebuild();
            rebuild_deadline = None;
        }

        // A requested rebuild (forced trigger, the route-change listener, or the mic-dry
        // watchdog) starts a settle timer; we only rebuild once the burst has quieted.
        if cap.rebuild_requested() {
            rebuild_pending_since.get_or_insert_with(Instant::now);
        }
        let do_rebuild = rebuild_pending_since
            .map(|since| since.elapsed() >= REBUILD_SETTLE)
            .unwrap_or(false);
        if do_rebuild {
            rebuild_pending_since = None;
            // Synchronized final flush of BOTH channels so they end at equal frame counts
            // before the old device is torn down.
            mic_raw.clear();
            tap_raw.clear();
            cap.drain_into(&mut mic_raw, &mut tap_raw);
            mic_w.write(&mic_raw, cap.observed_mic_channels());
            sys_w.write(&tap_raw, cap.observed_tap_channels());

            let mic_before = mic_w.written();
            let sys_before = sys_w.written();

            // A rebuild failure is NEVER fatal. `rebuild()` fails on exactly the transient
            // operations (default input / process tap / aggregate assembly) that are most
            // likely to hiccup DURING a route change — the moment we rebuild. Killing the
            // session here would make the recovery machinery destroy what it exists to save.
            // So: log the classified remedy, re-arm the settle timer, and retry on a later
            // tick. `cap.device` is left None by a failed rebuild and recovers on a retry.
            match cap.rebuild() {
                Err(e) => {
                    rebuild_failures += 1;
                    if rebuild_failures == 1 || rebuild_failures.is_multiple_of(20) {
                        log::warn!(
                            "rebuild failed — {e} (capture paused; retry #{rebuild_failures})"
                        );
                    }
                    rebuild_pending_since = Some(Instant::now());
                    mic_last_seen = Instant::now();
                }
                Ok(out) => {
                    rebuilds += 1;
                    rebuild_failures = 0;
                    mic_last_seen = Instant::now(); // fresh watchdog window for the new device

                    if out.rate_changed {
                        // The new device couldn't hold the original rate → the fixed-header WAV
                        // can't continue. Finalize the current segment, roll to a fresh pair at
                        // the new rate, and record the boundary + gap in the manifest.
                        let (mn, mr, mp) = std::mem::replace(
                            &mut mic_w,
                            WavStream::create(
                                &segment_path(&args.out_dir, "mic", segment + 1),
                                out.actual_rate,
                            )
                            .context("open rolled mic segment")?,
                        )
                        .finalize()
                        .context("finalize rolled mic segment")?;
                        let (sn, sr, sp) = std::mem::replace(
                            &mut sys_w,
                            WavStream::create(
                                &segment_path(&args.out_dir, "system", segment + 1),
                                out.actual_rate,
                            )
                            .context("open rolled system segment")?,
                        )
                        .finalize()
                        .context("finalize rolled system segment")?;
                        report("mic seg (rolled)   ", &mic_path, mn, mr, mp, rate);
                        report("system seg (rolled)", &sys_path, sn, sr, sp, rate);

                        segment += 1;
                        mic_path = segment_path(&args.out_dir, "mic", segment);
                        sys_path = segment_path(&args.out_dir, "system", segment);
                        let old_rate = rate;
                        rate = out.actual_rate;
                        log::warn!(
                            "GAP MARKER rebuild #{rebuilds}: rate {old_rate}→{rate} Hz, \
                             ~{} frame gap → ROLLED to segment {segment}",
                            out.gap_frames
                        );
                        append_manifest(
                            &args.out_dir,
                            &format!(
                                "seg {segment} mic={} system={} rate={rate} gap_frames={} \
                                 prev_rate={old_rate} reason=route_change",
                                mic_path.display(),
                                sys_path.display(),
                                out.gap_frames,
                            ),
                        );
                    } else {
                        // Rate held → the same files continue. A rate-held rebuild can still
                        // have SWAPPED the input device, so re-latch each WavStream's channel
                        // count before padding — otherwise a device with a different channel
                        // count would be downmixed with the stale divisor and misalign the
                        // channels. Then pad BOTH channels with equal silence for the gap.
                        mic_w.reset_channels();
                        sys_w.reset_channels();
                        mic_w.write_silence(out.gap_frames);
                        sys_w.write_silence(out.gap_frames);
                        log::info!(
                            "GAP MARKER rebuild #{rebuilds}: padded {} silence frames into both \
                             channels at ~{rate} Hz (mic {mic_before}→{}, system {sys_before}→{})",
                            out.gap_frames,
                            mic_w.written(),
                            sys_w.written(),
                        );
                    }
                }
            }
        }

        if stop.load(Ordering::Acquire) {
            break;
        }
        if let Some(d) = deadline {
            if Instant::now() >= d {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    // Final flush of whatever is still buffered in the rings.
    mic_raw.clear();
    tap_raw.clear();
    cap.drain_into(&mut mic_raw, &mut tap_raw);
    mic_w.write(&mic_raw, cap.observed_mic_channels());
    sys_w.write(&tap_raw, cap.observed_tap_channels());

    let elapsed = started_at.elapsed().as_secs_f32();
    let mic_ch = cap.observed_mic_channels();
    let tap_ch = cap.observed_tap_channels();
    let num_buffers = cap.first_num_buffers();
    let dropped = cap.dropped();

    log::info!("stopped after {elapsed:.1}s ({rebuilds} rebuild(s))");
    log::info!(
        "OBSERVED IO-proc layout: number_buffers={num_buffers} (expect 2), \
         mic_buffer_channels={mic_ch}, tap_buffer_channels={tap_ch}"
    );
    if num_buffers != 2 {
        log::warn!(
            "number_buffers={num_buffers} != 2 — mic and tap did NOT arrive as two separate \
             buffers (or the last device hadn't fired yet post-rebuild). Report this number."
        );
    }
    if dropped > 0 {
        log::warn!("{dropped} samples dropped (ring full — drain fell behind); increase RING_CAPACITY or drain interval");
    }

    let (mic_n, mic_rms, mic_peak) = mic_w.finalize().context("finalize mic.wav")?;
    let (sys_n, sys_rms, sys_peak) = sys_w.finalize().context("finalize system.wav")?;
    report("mic.wav    (You)   ", &mic_path, mic_n, mic_rms, mic_peak, rate);
    report("system.wav (Others)", &sys_path, sys_n, sys_rms, sys_peak, rate);

    log::info!(
        "Phase 0a check: both files should be non-zero AND intelligible — YOUR voice in \
         mic.wav, the REMOTE voice(s) in system.wav. If they're swapped, buffer order is \
         tap-then-mic (a one-line fix)."
    );
    Ok(())
}

/// Incremental mono WAV writer: downmixes interleaved frames to mono and streams them to
/// disk as they're drained, so only a ring's-worth of audio is ever in memory. Carries a
/// per-source frame remainder so a drain that splits mid-frame never misaligns channels.
struct WavStream {
    writer: hound::WavWriter<BufWriter<File>>,
    channels: usize,
    pending: Vec<f32>,
    written: usize,
    sum_sq: f64,
    peak: f32,
}

impl WavStream {
    fn create(path: &Path, sample_rate: u32) -> Result<Self> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        Ok(Self {
            writer: hound::WavWriter::create(path, spec)?,
            channels: 0,
            pending: Vec::new(),
            written: 0,
            sum_sq: 0.0,
            peak: 0.0,
        })
    }

    /// Append a freshly-drained interleaved chunk; write every complete mono frame, keep the
    /// remainder. `observed_channels` latches on the first non-zero value (fixed per device).
    fn write(&mut self, interleaved: &[f32], observed_channels: u32) {
        if interleaved.is_empty() {
            return;
        }
        if self.channels == 0 && observed_channels > 0 {
            self.channels = observed_channels as usize;
        }
        let ch = self.channels.max(1);
        self.pending.extend_from_slice(interleaved);
        let full = self.pending.len() - self.pending.len() % ch;
        for frame in self.pending[..full].chunks_exact(ch) {
            let mono = frame.iter().sum::<f32>() / ch as f32;
            let _ = self.writer.write_sample(mono);
            self.written += 1;
            self.sum_sq += mono as f64 * mono as f64;
            self.peak = self.peak.max(mono.abs());
        }
        self.pending.drain(..full);
    }

    /// Forget the latched channel count so the next `write` re-latches from the (possibly
    /// new) device. Called on a rate-held rebuild in case the input device swapped to a
    /// different channel count. Drops any sub-frame remainder (< old channels) to avoid
    /// mixing the old and new channel layouts across the boundary.
    fn reset_channels(&mut self) {
        self.channels = 0;
        self.pending.clear();
    }

    /// Write `frames` mono zero-samples — used to pad a rebuild gap equally into both
    /// channels so they stay aligned to each other.
    fn write_silence(&mut self, frames: usize) {
        for _ in 0..frames {
            let _ = self.writer.write_sample(0.0f32);
            self.written += 1;
        }
    }

    /// Mono samples written to this segment so far.
    fn written(&self) -> usize {
        self.written
    }

    /// Returns (mono samples written, RMS, peak). Any trailing partial frame (< channels)
    /// is discarded — at most `channels - 1` samples.
    fn finalize(self) -> Result<(usize, f32, f32)> {
        let rms = if self.written > 0 {
            (self.sum_sq / self.written as f64).sqrt() as f32
        } else {
            0.0
        };
        self.writer.finalize()?;
        Ok((self.written, rms, self.peak))
    }
}

/// Segment 0 keeps the plain `mic.wav`/`system.wav` names; later segments (only produced
/// when a rebuild can't hold the rate) get a zero-padded suffix, e.g. `mic.001.wav`.
fn segment_path(dir: &Path, base: &str, segment: u32) -> PathBuf {
    if segment == 0 {
        dir.join(format!("{base}.wav"))
    } else {
        dir.join(format!("{base}.{segment:03}.wav"))
    }
}

/// Append one line to the session's segment manifest, recording each rebuild-induced
/// segment boundary + the gap between segments. Best-effort (a manifest write failure must
/// not abort capture).
fn append_manifest(dir: &Path, line: &str) {
    let path = dir.join("segments.txt");
    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{line}") {
                log::warn!("could not append to manifest {}: {e}", path.display());
            }
        }
        Err(e) => log::warn!("could not open manifest {}: {e}", path.display()),
    }
}

fn report(label: &str, path: &Path, n: usize, rms: f32, peak: f32, rate: u32) {
    let secs = if rate > 0 { n as f32 / rate as f32 } else { 0.0 };
    let verdict = if peak < 1e-5 {
        "SILENT — check permissions / source"
    } else {
        "signal present"
    };
    log::info!(
        "{label}: {n} samples, {secs:.1}s, RMS={rms:.5}, peak={peak:.5} → {verdict}  [{}]",
        path.display()
    );
}
