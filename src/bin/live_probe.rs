//! Live-lane measurement (throwaway; ships nothing to the product).
//!
//! Answers one question before any live code is written: if a reader saw a recording grow in
//! checkpoint-sized steps, cut it into speech windows as they close and ran whisper on each, how
//! far behind the speaker would the text be?
//!
//! The file is replayed on a VIRTUAL clock, so a 45-minute recording is measured in a few minutes:
//! a window becomes available when the checkpoint that contains its end would have been read, it
//! is decoded when the (single) decoder is free, and its lag is decode-end minus speech-end. Decode
//! durations are real wall time on this machine. What this cannot show is contention with a call
//! app on the GPU; that needs a real call.
//!
//!   cargo build --bin live_probe
//!   ./target/debug/live_probe --stereo capture/combined_stereo.wav --model models/ggml-large-v3.bin
//!   ./target/debug/live_probe --mic <dir>/mic.wav --system <dir>/system.wav --compare-ctx
//!
//! Prints timings and counts only. Decoded text is never printed unless `--print-text` is given,
//! so it is safe to run on a real meeting and share the output.
//!
//! Capture-free: no TCC, no code-signing required.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde_json::json;
use silero::{VadConfig, VadSession, VadTransition};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

const RATE: usize = 16_000;
/// whisper returns nothing below ~100 mel frames (see `asr.rs`); such windows are counted, not decoded.
const WHISPER_FLOOR_SAMPLES: usize = 16_040;
/// Encoder frames per second of audio (1500 frames cover whisper's fixed 30 s context).
const CTX_FRAMES_PER_SEC: f64 = 50.0;
const CTX_FULL: i32 = 1500;

struct Args {
    model: PathBuf,
    mic: Option<PathBuf>,
    system: Option<PathBuf>,
    stereo: Option<PathBuf>,
    lang: String,
    max_window_secs: f64,
    checkpoint_secs: f64,
    poll_ms: usize,
    /// Decode with the encoder context fitted to the window instead of the full 30 s.
    fit_ctx: bool,
    /// Decode every window both ways and report timing for each plus how much the text agrees.
    compare_ctx: bool,
    print_text: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: PathBuf::from("models/ggml-large-v3.bin"),
        mic: None,
        system: None,
        stereo: None,
        lang: "es".to_string(),
        max_window_secs: 12.0,
        checkpoint_secs: 2.0,
        poll_ms: 500,
        fit_ctx: false,
        compare_ctx: false,
        print_text: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().with_context(|| format!("{name} needs a value"));
        match flag.as_str() {
            "--model" | "-m" => a.model = PathBuf::from(value("--model")?),
            "--mic" => a.mic = Some(PathBuf::from(value("--mic")?)),
            "--system" => a.system = Some(PathBuf::from(value("--system")?)),
            "--stereo" => a.stereo = Some(PathBuf::from(value("--stereo")?)),
            "--lang" | "-l" => a.lang = value("--lang")?,
            "--max-window-secs" => a.max_window_secs = value("--max-window-secs")?.parse().context("--max-window-secs")?,
            "--checkpoint-secs" => a.checkpoint_secs = value("--checkpoint-secs")?.parse().context("--checkpoint-secs")?,
            "--poll-ms" => a.poll_ms = value("--poll-ms")?.parse().context("--poll-ms")?,
            "--fit-ctx" => a.fit_ctx = true,
            "--compare-ctx" => a.compare_ctx = true,
            "--print-text" => a.print_text = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: live_probe (--stereo <wav> | --mic <wav> --system <wav>) [--model <ggml.bin>] \
                     [--lang <code>] [--max-window-secs 12] [--checkpoint-secs 2] [--poll-ms 500] \
                     [--fit-ctx | --compare-ctx] [--print-text]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument {other} (see --help)"),
        }
    }
    if a.stereo.is_none() && (a.mic.is_none() || a.system.is_none()) {
        bail!("give --stereo <wav>, or both --mic <wav> and --system <wav>");
    }
    if a.max_window_secs < 2.0 || a.checkpoint_secs <= 0.0 {
        bail!("--max-window-secs must be at least 2 and --checkpoint-secs positive");
    }
    Ok(a)
}

/// Read a 16 kHz WAV of any sample format into one f32 vector per channel.
fn load_16k(path: &Path) -> Result<Vec<Vec<f32>>> {
    let reader = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate as usize != RATE {
        bail!("{}: {} Hz, expected 16000 (this probe does not resample)", path.display(), spec.sample_rate);
    }
    let channels = spec.channels as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().collect::<std::result::Result<_, _>>(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.into_samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect()
        }
    }
    .with_context(|| format!("read samples from {}", path.display()))?;
    let mut out = vec![Vec::with_capacity(interleaved.len() / channels); channels];
    for frame in interleaved.chunks_exact(channels) {
        for (channel, sample) in out.iter_mut().zip(frame) {
            channel.push(sample.clamp(-1.0, 1.0));
        }
    }
    Ok(out)
}

fn load_channels(args: &Args) -> Result<[(&'static str, Vec<f32>); 2]> {
    if let Some(stereo) = &args.stereo {
        let mut channels = load_16k(stereo)?;
        if channels.len() != 2 {
            bail!("{}: {} channel(s), --stereo needs exactly 2", stereo.display(), channels.len());
        }
        let system = channels.pop().context("second channel")?;
        let mic = channels.pop().context("first channel")?;
        return Ok([("mic", mic), ("system", system)]);
    }
    let mono = |path: &Path| -> Result<Vec<f32>> {
        let mut channels = load_16k(path)?;
        if channels.len() != 1 {
            bail!("{}: {} channels, expected mono", path.display(), channels.len());
        }
        channels.pop().context("mono channel")
    };
    let mic = mono(args.mic.as_deref().context("--mic")?)?;
    let system = mono(args.system.as_deref().context("--system")?)?;
    Ok([("mic", mic), ("system", system)])
}

/// One speech window as the live reader would have produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Window {
    channel: usize,
    start_ms: usize,
    end_ms: usize,
    /// Virtual time at which the reader holds every sample of this window.
    avail_ms: usize,
}

/// Feed `audio` to a streaming VAD in checkpoint-sized steps and return the windows it closes:
/// at the end of speech, or forced once speech has run for `max_ms`.
fn windows_for(channel: usize, audio: &[f32], step: usize, max_ms: usize, poll_ms: usize) -> Result<Vec<Window>> {
    let mut vad = VadSession::new(VadConfig::default()).context("init silero VAD")?;
    let mut out = Vec::new();
    // Start of speech not yet emitted, and whether that start is a forced cut (not a VAD start).
    let mut open: Option<(usize, bool)> = None;
    let mut fed = 0usize;
    for chunk in audio.chunks(step) {
        fed += chunk.len();
        let now_ms = fed * 1000 / RATE;
        let avail_ms = now_ms + poll_ms;
        for transition in vad.process(chunk).context("vad.process")? {
            match transition {
                VadTransition::SpeechStart { timestamp_ms } => open = Some((timestamp_ms, false)),
                VadTransition::SpeechEnd { start_timestamp_ms, end_timestamp_ms, .. } => {
                    let start_ms = match open.take() {
                        Some((cut, true)) => cut,
                        _ => start_timestamp_ms,
                    };
                    if end_timestamp_ms > start_ms {
                        out.push(Window { channel, start_ms, end_ms: end_timestamp_ms, avail_ms });
                    }
                }
            }
        }
        if let Some((start_ms, _)) = open
            && now_ms.saturating_sub(start_ms) >= max_ms
        {
            out.push(Window { channel, start_ms, end_ms: now_ms, avail_ms });
            open = Some((now_ms, true));
        }
    }
    let total_ms = audio.len() * 1000 / RATE;
    if let Some((start_ms, _)) = open
        && total_ms > start_ms
    {
        out.push(Window { channel, start_ms, end_ms: total_ms, avail_ms: total_ms + poll_ms });
    }
    Ok(out)
}

/// Encoder frames that cover `samples` plus a small margin, capped at the full context.
///
/// Rounded up to a multiple of 8: an odd context (517, for a 9.7 s window) makes whisper.cpp's
/// Metal backend fail `GGML_ASSERT(nb01 % 8 == 0)` and abort the whole process (seen 2026-11-12).
fn fitted_ctx(samples: usize) -> i32 {
    let secs = samples as f64 / RATE as f64;
    let frames = (secs * CTX_FRAMES_PER_SEC).ceil() as i32 + 32;
    (frames.div_euclid(8) + i32::from(frames.rem_euclid(8) != 0)).saturating_mul(8).clamp(128, CTX_FULL)
}

/// whisper.cpp's own log lines include decoded tokens in debug builds. Drop them all, so this
/// probe's output stays free of transcript text (the product does the same in `asr.rs`).
unsafe extern "C" fn drop_native_log(
    _level: whisper_rs::whisper_rs_sys::ggml_log_level,
    _text: *const std::ffi::c_char,
    _user_data: *mut std::ffi::c_void,
) {
}

fn silence_native_logs() {
    // SAFETY: the callback ignores every argument and touches no memory; both setters keep the
    // function pointer for the life of the process, and it is a plain `fn` that never unloads.
    unsafe {
        whisper_rs::set_log_callback(Some(drop_native_log), std::ptr::null_mut());
        whisper_rs::whisper_rs_sys::ggml_log_set(Some(drop_native_log), std::ptr::null_mut());
    }
}

/// Decode one window. Returns (text, wall seconds). Parameters match `asr.rs`, plus the context.
fn decode(state: &mut WhisperState, audio: &[f32], lang: &str, threads: i32, ctx: i32) -> Result<(String, f64)> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some(lang));
    params.set_n_threads(threads);
    params.set_translate(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_audio_ctx(ctx);
    let started = Instant::now();
    state.full(params, audio).context("whisper full()")?;
    let wall = started.elapsed().as_secs_f64();
    let mut text = String::new();
    for i in 0..state.full_n_segments().context("n_segments")? {
        if let Ok(segment) = state.full_get_segment_text(i) {
            let segment = segment.trim();
            if !segment.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(segment);
            }
        }
    }
    Ok((text, wall))
}

fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect::<String>())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Longest common subsequence of two word lists (windows are at most a few dozen words).
fn lcs(a: &[String], b: &[String]) -> usize {
    let mut row = vec![0usize; b.len() + 1];
    for x in a {
        let mut diagonal = 0;
        for (j, y) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = if x == y { diagonal + 1 } else { above.max(row[j]) };
            diagonal = above;
        }
    }
    row[b.len()]
}

/// Replay decode durations on the virtual clock: one decoder, windows taken in arrival order.
/// Returns each window's lag in seconds (decode end minus speech end).
fn lags(windows: &[Window], decode_secs: &[f64]) -> Vec<f64> {
    let mut free_at = 0.0f64;
    windows
        .iter()
        .zip(decode_secs)
        .map(|(w, d)| {
            let start = (w.avail_ms as f64 / 1000.0).max(free_at);
            free_at = start + d;
            free_at - w.end_ms as f64 / 1000.0
        })
        .collect()
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn spread(values: &[f64]) -> serde_json::Value {
    let round = |v: f64| (v * 1000.0).round() / 1000.0;
    json!({
        "p50": round(percentile(values, 0.5)),
        "p90": round(percentile(values, 0.9)),
        "max": round(percentile(values, 1.0)),
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let run_id = format!("{:x}{:x}", std::process::id(), std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0));
    let log = |event: &str, mut fields: serde_json::Value| {
        fields["component"] = json!("live_probe");
        fields["event"] = json!(event);
        fields["run_id"] = json!(run_id);
        eprintln!("{fields}");
    };

    let channels = load_channels(&args)?;
    let audio_secs = channels.iter().map(|(_, a)| a.len()).max().unwrap_or(0) as f64 / RATE as f64;
    let step = ((args.checkpoint_secs * RATE as f64) as usize).max(RATE / 100);
    let max_ms = (args.max_window_secs * 1000.0) as usize;

    let vad_started = Instant::now();
    let mut windows = Vec::new();
    for (index, (_, audio)) in channels.iter().enumerate() {
        windows.extend(windows_for(index, audio, step, max_ms, args.poll_ms)?);
    }
    let vad_secs = vad_started.elapsed().as_secs_f64();
    windows.sort_by_key(|w| (w.avail_ms, w.start_ms, w.channel));
    let (decodable, short): (Vec<Window>, Vec<Window>) = windows
        .iter()
        .partition(|w| (w.end_ms - w.start_ms) * RATE / 1000 >= WHISPER_FLOOR_SAMPLES);
    log("windows_found", json!({
        "audio_secs": audio_secs, "windows": decodable.len(), "below_whisper_floor": short.len(),
        "vad_secs": vad_secs, "checkpoint_secs": args.checkpoint_secs, "max_window_secs": args.max_window_secs,
    }));

    let model = args.model.to_str().context("model path is not valid UTF-8")?;
    silence_native_logs();
    let load_started = Instant::now();
    let ctx = WhisperContext::new_with_params(model, WhisperContextParameters::default())
        .with_context(|| format!("load whisper model {model}"))?;
    let mut state = ctx.create_state().context("create whisper state")?;
    let load_secs = load_started.elapsed().as_secs_f64();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8) as i32;
    log("model_loaded", json!({ "load_secs": load_secs, "threads": threads }));

    let run_full = args.compare_ctx || !args.fit_ctx;
    let run_fit = args.compare_ctx || args.fit_ctx;
    let (mut full_secs, mut fit_secs) = (Vec::new(), Vec::new());
    let (mut shared_words, mut total_words, mut identical) = (0usize, 0usize, 0usize);
    let mut speech_secs = 0.0f64;
    for (seq, w) in decodable.iter().enumerate() {
        let (name, audio) = &channels[w.channel];
        let a = (w.start_ms * RATE / 1000).min(audio.len());
        let b = (w.end_ms * RATE / 1000).min(audio.len());
        let slice = &audio[a..b];
        speech_secs += slice.len() as f64 / RATE as f64;
        let full = if run_full { Some(decode(&mut state, slice, &args.lang, threads, CTX_FULL)?) } else { None };
        let fit = if run_fit { Some(decode(&mut state, slice, &args.lang, threads, fitted_ctx(slice.len()))?) } else { None };
        let mut line = json!({
            "seq": seq, "channel": name, "dur_ms": w.end_ms - w.start_ms, "avail_ms": w.avail_ms,
        });
        if let Some((text, secs)) = &full {
            full_secs.push(*secs);
            line["full_ms"] = json!((secs * 1000.0).round());
            line["full_chars"] = json!(text.chars().count());
        }
        if let Some((text, secs)) = &fit {
            fit_secs.push(*secs);
            line["fit_ms"] = json!((secs * 1000.0).round());
            line["fit_chars"] = json!(text.chars().count());
            line["fit_ctx"] = json!(fitted_ctx(slice.len()));
        }
        if let (Some((full_text, _)), Some((fit_text, _))) = (&full, &fit) {
            let (x, y) = (words(full_text), words(fit_text));
            let common = lcs(&x, &y);
            shared_words += 2 * common;
            total_words += x.len() + y.len();
            identical += usize::from(x == y);
            line["agreement"] = json!(if x.len() + y.len() == 0 { 1.0 } else { 2.0 * common as f64 / (x.len() + y.len()) as f64 });
        }
        if args.print_text {
            line["text"] = json!(fit.as_ref().or(full.as_ref()).map(|(text, _)| text.clone()));
        }
        log("window", line);
    }

    let mut summary = json!({
        "component": "live_probe", "event": "summary", "run_id": run_id,
        "model": args.model.file_stem().and_then(|s| s.to_str()), "lang": args.lang,
        "audio_secs": audio_secs, "speech_secs": speech_secs,
        "windows": decodable.len(), "below_whisper_floor": short.len(),
        "checkpoint_secs": args.checkpoint_secs, "poll_ms": args.poll_ms, "max_window_secs": args.max_window_secs,
        "vad_secs": vad_secs, "model_load_secs": load_secs,
    });
    for (label, secs) in [("full_ctx", &full_secs), ("fit_ctx", &fit_secs)] {
        if secs.is_empty() {
            continue;
        }
        let total: f64 = secs.iter().sum();
        summary[label] = json!({
            "decode_secs": spread(secs),
            "first_decode_secs": secs[0],
            "decode_total_secs": total,
            // Share of the meeting the decoder is busy: above 1.0 it can never keep up.
            "duty": if audio_secs > 0.0 { total / audio_secs } else { 0.0 },
            "lag_secs": spread(&lags(&decodable, secs)),
        });
    }
    if args.compare_ctx && total_words > 0 {
        summary["ctx_agreement"] = json!({
            "word_agreement": shared_words as f64 / total_words as f64,
            "identical_windows": identical,
            "of": decodable.len(),
        });
    }
    println!("{summary}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(start_ms: usize, end_ms: usize, avail_ms: usize) -> Window {
        Window { channel: 0, start_ms, end_ms, avail_ms }
    }

    #[test]
    fn lag_counts_waiting_for_the_checkpoint_and_for_a_busy_decoder() {
        // Speech ends at 4.0 s, readable at 6.5 s, decode takes 1.5 s → done at 8.0 s, 4.0 s behind.
        // The next window is readable at 7.0 s but the decoder is busy until 8.0 s → done at 10.0 s.
        let windows = [window(1_000, 4_000, 6_500), window(5_000, 6_000, 7_000)];
        let got = lags(&windows, &[1.5, 2.0]);
        assert!((got[0] - 4.0).abs() < 1e-9, "{got:?}");
        assert!((got[1] - 4.0).abs() < 1e-9, "{got:?}");
    }

    #[test]
    fn fitted_context_grows_with_the_window_and_never_exceeds_the_full_one() {
        assert_eq!(fitted_ctx(RATE), 128); // 1 s → 82 frames, raised to the floor
        assert_eq!(fitted_ctx(10 * RATE), 536); // 532 rounded up to a multiple of 8
        assert_eq!(fitted_ctx(97 * RATE / 10), 520); // the 9.7 s window that aborted at 517
        assert_eq!(fitted_ctx(40 * RATE), CTX_FULL);
        // Every fitted value is a multiple of 8, or the model's own full context (1500, the default).
        assert!((1..=45).map(|secs| fitted_ctx(secs * RATE)).all(|ctx| ctx == CTX_FULL || ctx % 8 == 0));
    }

    #[test]
    fn word_agreement_ignores_case_and_punctuation() {
        let a = words("Dale, lo vemos el Lunes.");
        let b = words("dale lo vemos el martes");
        assert_eq!(lcs(&a, &b), 4);
        assert_eq!(lcs(&a, &a), a.len());
        assert_eq!(lcs(&a, &[]), 0);
    }

    #[test]
    fn silence_produces_no_window() {
        let silence = vec![0.0f32; 5 * RATE];
        assert!(windows_for(0, &silence, 2 * RATE, 12_000, 500).unwrap().is_empty());
    }
}
