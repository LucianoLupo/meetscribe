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
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

struct Args {
    out_dir: PathBuf,
    seconds: Option<u64>,
}

fn parse_args() -> Args {
    let mut out_dir = PathBuf::from("capture");
    let mut seconds = None;
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
            "-h" | "--help" => {
                eprintln!("usage: meetscribe [--out <dir>] [--seconds <n>]");
                std::process::exit(0);
            }
            _ => {}
        }
    }
    Args { out_dir, seconds }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = parse_args();
    std::fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("create output dir {}", args.out_dir.display()))?;

    log::info!("meetscribe capture spike — starting single-aggregate mic+tap capture");
    log::info!("(first run triggers TWO prompts: Microphone and \"record system audio\" — approve both)");

    let mut cap = DualCapture::start().context(
        "failed to start capture — if you were never prompted, the binary may be unsigned \
         (codesign it) or lacks the Info.plist usage strings",
    )?;

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
    let rate = cap.aggregate_rate().max(1);
    log::info!("aggregate rate = {rate} Hz (both channels)");

    let mic_path = args.out_dir.join("mic.wav");
    let sys_path = args.out_dir.join("system.wav");
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
    // Small per-drain scratch buffers (reused) — only a ring's-worth is ever resident.
    let mut mic_raw: Vec<f32> = Vec::new();
    let mut tap_raw: Vec<f32> = Vec::new();

    loop {
        mic_raw.clear();
        tap_raw.clear();
        cap.drain_into(&mut mic_raw, &mut tap_raw);
        mic_w.write(&mic_raw, cap.observed_mic_channels());
        sys_w.write(&tap_raw, cap.observed_tap_channels());
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

    log::info!("stopped after {elapsed:.1}s");
    log::info!(
        "OBSERVED IO-proc layout: number_buffers={num_buffers} (expect 2), \
         mic_buffer_channels={mic_ch}, tap_buffer_channels={tap_ch}"
    );
    if num_buffers != 2 {
        log::warn!(
            "number_buffers={num_buffers} != 2 — mic and tap did NOT arrive as two separate \
             buffers. system.wav may be silent/wrong; the channel-split adaptation is needed. \
             Report this number."
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
