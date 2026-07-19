//! Phase 2 Batch 0 — silero VAD build/behavior spike (throwaway).
//!
//! De-risks the silero + ort (ONNX Runtime) build on this toolchain and confirms
//! the VAD detects speech on the real captured audio before wiring it into the
//! transcription pipeline.
//!
//!   cargo build --bin vad_probe && ./target/debug/vad_probe [--wav <16k-mono-f32.wav>]
//!
//! Capture-free → no TCC, no code-signing required.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use silero::{VadConfig, VadSession, VadTransition};

/// silero (and whisper) want 16 kHz mono f32 in [-1, 1] — what Phase-0 capture writes.
fn load_wav_16k_mono_f32(path: &Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 {
        bail!("{}: sample_rate is {} Hz, expected 16000", path.display(), spec.sample_rate);
    }
    if spec.channels != 1 {
        bail!("{}: {} channels, expected mono", path.display(), spec.channels);
    }
    if spec.sample_format != hound::SampleFormat::Float || spec.bits_per_sample != 32 {
        bail!("{}: expected 32-bit float samples", path.display());
    }
    reader
        .into_samples::<f32>()
        .collect::<std::result::Result<Vec<f32>, _>>()
        .context("read samples")
}

fn main() -> Result<()> {
    let mut wav = PathBuf::from("capture/system.wav");
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--wav" | "-w" => {
                if let Some(v) = it.next() {
                    wav = PathBuf::from(v);
                }
            }
            "-h" | "--help" => {
                eprintln!("usage: vad_probe [--wav <16k-mono-f32.wav>]");
                return Ok(());
            }
            _ => {}
        }
    }

    let audio = load_wav_16k_mono_f32(&wav)?;
    let total_secs = audio.len() as f64 / 16_000.0;
    eprintln!("[vad_probe] {} — {} samples = {:.1}s", wav.display(), audio.len(), total_secs);

    let mut vad = VadSession::new(VadConfig::default()).context("init silero VAD (static-model)")?;

    // process() buffers internally, so chunk size is our choice — feed 1 s at a time.
    let mut windows: Vec<(usize, usize)> = Vec::new();
    for chunk in audio.chunks(16_000) {
        for t in vad.process(chunk).context("vad.process")? {
            if let VadTransition::SpeechEnd { start_timestamp_ms, end_timestamp_ms, .. } = t {
                windows.push((start_timestamp_ms, end_timestamp_ms));
            }
        }
    }
    // Trailing speech still open at end-of-stream → close it at the audio end.
    if vad.is_speaking() {
        let end = (total_secs * 1000.0) as usize;
        let dur = vad.current_speech_duration().as_millis() as usize;
        windows.push((end.saturating_sub(dur), end));
    }

    let speech_ms: usize = windows.iter().map(|(a, b)| b.saturating_sub(*a)).sum();
    println!("\n===== VAD PROBE =====");
    println!("audio          : {total_secs:.1}s");
    println!("speech windows : {}", windows.len());
    println!(
        "total speech   : {:.1}s ({:.0}% of audio)",
        speech_ms as f64 / 1000.0,
        100.0 * speech_ms as f64 / 1000.0 / total_secs
    );
    println!("----- first 20 windows -----");
    for (a, b) in windows.iter().take(20) {
        println!(
            "  [{:7.2}-{:7.2}]  {:.2}s",
            *a as f64 / 1000.0,
            *b as f64 / 1000.0,
            (b - a) as f64 / 1000.0
        );
    }
    Ok(())
}
