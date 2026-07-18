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
    let mut mic_raw: Vec<f32> = Vec::new();
    let mut tap_raw: Vec<f32> = Vec::new();

    loop {
        cap.drain_into(&mut mic_raw, &mut tap_raw);
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
    cap.drain_into(&mut mic_raw, &mut tap_raw); // final flush

    let elapsed = started_at.elapsed().as_secs_f32();
    let rate = cap.aggregate_rate();
    let mic_ch = cap.observed_mic_channels();
    let tap_ch = cap.observed_tap_channels();
    let num_buffers = cap.first_num_buffers();

    log::info!("stopped after {elapsed:.1}s — aggregate rate = {rate} Hz");
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

    let mic_mono = downmix(&mic_raw, mic_ch);
    let tap_mono = downmix(&tap_raw, tap_ch);

    let mic_path = args.out_dir.join("mic.wav");
    let sys_path = args.out_dir.join("system.wav");
    write_wav(&mic_path, &mic_mono, rate.max(1)).context("write mic.wav")?;
    write_wav(&sys_path, &tap_mono, rate.max(1)).context("write system.wav")?;

    report("mic.wav    (You)   ", &mic_path, &mic_mono, rate);
    report("system.wav (Others)", &sys_path, &tap_mono, rate);

    log::info!(
        "Phase 0a check: both files should be non-zero AND intelligible — YOUR voice in \
         mic.wav, the REMOTE voice(s) in system.wav. If they're swapped, buffer order is \
         tap-then-mic (a one-line fix)."
    );
    Ok(())
}

/// Average interleaved channels down to mono. `channels == 0` (no callback fired) ⇒ passthrough.
fn downmix(interleaved: &[f32], channels: u32) -> Vec<f32> {
    let ch = channels.max(1) as usize;
    if ch == 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

fn write_wav(path: &Path, mono: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in mono {
        w.write_sample(s)?;
    }
    w.finalize()?;
    Ok(())
}

fn report(label: &str, path: &Path, mono: &[f32], rate: u32) {
    let n = mono.len();
    let secs = if rate > 0 { n as f32 / rate as f32 } else { 0.0 };
    let rms = if n > 0 {
        (mono.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt()
    } else {
        0.0
    };
    let peak = mono.iter().fold(0f32, |a, &s| a.max(s.abs()));
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
