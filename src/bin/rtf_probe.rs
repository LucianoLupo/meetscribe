//! Phase 1.5 RTF + accuracy probe (throwaway).
//!
//! Transcribes a 16 kHz mono f32 WAV through whisper-rs — the real Phase-2 engine —
//! and reports the real-time factor (wall / audio) plus the first decoded segments,
//! so the metal-only build can be validated before CoreML is toggled on.
//!
//!   metal-only : cargo build --bin rtf_probe            && ./target/debug/rtf_probe
//!   metal+coreml: cargo build --bin rtf_probe --features coreml && ./target/debug/rtf_probe
//!
//! Capture-free (no mic / system-audio) → no TCC, no code-signing required.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

struct Args {
    model: PathBuf,
    wav: PathBuf,
    lang: String,
    max_print: usize,
}

fn parse_args() -> Args {
    let mut model = PathBuf::from("models/ggml-large-v3.bin");
    let mut wav = PathBuf::from("capture/system.wav");
    let mut lang = "es".to_string();
    let mut max_print = 12usize;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" | "-m" => model = it.next().map(PathBuf::from).unwrap_or(model),
            "--wav" | "-w" => wav = it.next().map(PathBuf::from).unwrap_or(wav),
            "--lang" | "-l" => lang = it.next().unwrap_or(lang),
            "--max-print" => max_print = it.next().and_then(|s| s.parse().ok()).unwrap_or(max_print),
            "-h" | "--help" => {
                eprintln!(
                    "usage: rtf_probe [--model <ggml.bin>] [--wav <16k-mono-f32.wav>] [--lang <code>] [--max-print <n>]"
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }
    Args { model, wav, lang, max_print }
}

/// whisper wants 16 kHz mono f32 in [-1, 1] — exactly what the Phase-0 capture writes.
/// Reject anything else loudly (48 kHz → 16 kHz resample via rubato is a Phase-2 job).
fn load_wav_16k_mono_f32(path: &Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 {
        bail!(
            "{}: sample_rate is {} Hz, expected 16000 (resample is a Phase-2 job)",
            path.display(),
            spec.sample_rate
        );
    }
    if spec.channels != 1 {
        bail!("{}: {} channels, expected mono", path.display(), spec.channels);
    }
    if spec.sample_format != hound::SampleFormat::Float || spec.bits_per_sample != 32 {
        bail!(
            "{}: expected 32-bit float samples (got {:?}/{}-bit)",
            path.display(),
            spec.sample_format,
            spec.bits_per_sample
        );
    }
    reader
        .into_samples::<f32>()
        .collect::<std::result::Result<Vec<f32>, _>>()
        .context("read samples")
}

fn main() -> Result<()> {
    let args = parse_args();
    let coreml = cfg!(feature = "coreml");

    eprintln!(
        "[rtf_probe] backend = {} | model = {} | wav = {} | lang = {}",
        if coreml { "metal+coreml" } else { "metal-only" },
        args.model.display(),
        args.wav.display(),
        args.lang
    );

    let audio = load_wav_16k_mono_f32(&args.wav)?;
    let audio_secs = audio.len() as f64 / 16_000.0;
    eprintln!("[rtf_probe] loaded {} samples = {:.1}s audio", audio.len(), audio_secs);

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8) as i32;

    // On a --features coreml build, whisper.cpp auto-loads
    // <model>-encoder.mlmodelc here and prints "loading Core ML model ..." to stderr.
    let ctx = WhisperContext::new_with_params(
        args.model.to_str().context("model path is not valid UTF-8")?,
        WhisperContextParameters::default(),
    )
    .context("load model (on --features coreml, a failure here = CoreML encoder didn't load)")?;
    let mut state = ctx.create_state().context("create whisper state")?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some(&args.lang));
    params.set_n_threads(threads);
    params.set_translate(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    let t0 = Instant::now();
    state.full(params, &audio).context("whisper full()")?;
    let wall = t0.elapsed().as_secs_f64();

    let rtf = wall / audio_secs;
    let n = state.full_n_segments().context("n_segments")?;

    println!("\n===== RTF PROBE RESULT =====");
    println!("backend      : {}", if coreml { "metal + CoreML" } else { "metal-only" });
    println!("model        : {}", args.model.display());
    println!("threads      : {threads}");
    println!("audio        : {audio_secs:.1}s");
    println!("wall-clock   : {wall:.1}s");
    println!(
        "RTF (wall/audio) : {:.3}x  ({})",
        rtf,
        if rtf < 1.0 { "faster than realtime" } else { "SLOWER than realtime" }
    );
    println!("segments     : {n}");
    println!("----- first {} segments -----", args.max_print);
    for i in 0..n.min(args.max_print as i32) {
        let text = state.full_get_segment_text(i).unwrap_or_default();
        let a = state.full_get_segment_t0(i).unwrap_or(0) as f64 / 100.0;
        let b = state.full_get_segment_t1(i).unwrap_or(0) as f64 / 100.0;
        println!("[{a:7.2}-{b:7.2}] {}", text.trim());
    }
    Ok(())
}
