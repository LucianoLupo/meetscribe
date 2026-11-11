//! Whisper ASR wrapper (Phase 2) — one loaded model + ONE reused decode state, transcribe per VAD window.
//!
//! Reuses the whisper-rs sequence proven in `src/bin/rtf_probe.rs` (Phase 1.5).
//! CoreML is still toggled at build time by the crate `coreml` feature.

use anyhow::{Context, Result};
use std::sync::Once;
use std::sync::atomic::{AtomicU64, Ordering};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

/// One decoded word, in seconds relative to the start of the window.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub t_start: f64,
    pub t_end: f64,
}

/// Windows shorter than this are skipped: whisper's own encoder floor is ~100 mel frames
/// (≈ 16 040 samples / 1002 ms), below which `whisper_full` logs "input is too short" and
/// returns zero segments. We skip well below that rather than feed it a window it will drop
/// anyway. (Recovering sub-second utterances would mean padding past the frame floor AND
/// accepting whisper's tendency to hallucinate on trailing silence — a separate feature.)
const MIN_SAMPLES: usize = 16_000 / 10; // 100 ms @ 16 kHz

static NATIVE_LOG_INIT: Once = Once::new();
static NATIVE_WARNINGS: AtomicU64 = AtomicU64::new(0);
static NATIVE_ERRORS: AtomicU64 = AtomicU64::new(0);

// Native message bodies can contain decoded words at any level. Never dereference,
// format or forward them. Atomics cannot unwind across this C callback.
unsafe extern "C" fn native_log_callback(
    level: whisper_rs::whisper_rs_sys::ggml_log_level,
    _text: *const std::ffi::c_char,
    _user_data: *mut std::ffi::c_void,
) {
    match level {
        whisper_rs::whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_WARN => {
            NATIVE_WARNINGS.fetch_add(1, Ordering::Relaxed);
        }
        whisper_rs::whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_ERROR => {
            NATIVE_ERRORS.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

fn install_native_diagnostics() {
    NATIVE_LOG_INIT.call_once(|| {
        // SAFETY: the static callback ignores pointers and only performs atomic operations.
        // Both setters retain it for process lifetime; backend init reuses Whisper's callback.
        unsafe {
            whisper_rs::set_log_callback(Some(native_log_callback), std::ptr::null_mut());
            whisper_rs::whisper_rs_sys::ggml_log_set(Some(native_log_callback), std::ptr::null_mut());
        }
    });
}

// Report outside FFI, where ordinary Rust logging is safe. Preserve severity and
// counts without disclosing native text, including with verbose RUST_LOG.
fn flush_native_diagnostics() {
    let warnings = NATIVE_WARNINGS.swap(0, Ordering::Relaxed);
    let errors = NATIVE_ERRORS.swap(0, Ordering::Relaxed);
    if warnings > 0 { log::warn!("component=asr event=native_warning count={warnings}"); }
    if errors > 0 { log::error!("component=asr event=native_error count={errors}"); }
}

pub struct Asr {
    /// One long-lived decode state, reused across every window. `full()` resets its internal
    /// buffers each call so windows do not bleed into each other; creating a fresh state per
    /// window (the old path) instead rebuilt the whole Metal+BLAS backend every call — an
    /// init/free cycle per window, pure churn.
    state: WhisperState,
    threads: i32,
}

impl Asr {
    pub fn load(model_path: &str) -> Result<Self> {
        install_native_diagnostics();
        let loaded = WhisperContext::new_with_params(model_path, WhisperContextParameters::default());
        flush_native_diagnostics();
        let ctx = loaded.with_context(|| format!("load whisper model {model_path}"))?;
        // Create the decode state once. `WhisperState` holds its own `Arc` to the inner context,
        // so it stays valid after `ctx` is dropped at the end of `load`.
        let created = ctx.create_state();
        flush_native_diagnostics();
        let state = created.context("create whisper state")?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(8) as i32;
        Ok(Self { state, threads })
    }

    /// Transcribe one 16 kHz mono window → (text, mean-token-probability confidence).
    /// Sub-100 ms windows return `("", 0.0)` instead of erroring on too-short input.
    pub fn transcribe(&mut self, audio_16k: &[f32], lang: &str) -> Result<(String, f32)> {
        if audio_16k.len() < MIN_SAMPLES {
            return Ok((String::new(), 0.0));
        }
        self.decode(audio_16k, lang, false)?;
        self.text_and_confidence()
    }

    /// [`Asr::transcribe`] plus per-word timings, for far-end windows that will be split at voice
    /// changes. Text and confidence are built by the same code as `transcribe`; only token
    /// timestamps are added (whisper's plain token timestamps — NOT DTW: whisper.cpp's DTW path
    /// asserts `filter_width < n_audio_tokens` and aborts the whole process when a decode pass
    /// covers ≤ ~150 ms, which took down the daemon on a real meeting). A new word starts at a token with a leading space; special
    /// tokens (`[_…]`, `<|…|>`) are skipped.
    pub fn transcribe_words(&mut self, audio_16k: &[f32], lang: &str) -> Result<(String, f32, Vec<Word>)> {
        if audio_16k.len() < MIN_SAMPLES {
            return Ok((String::new(), 0.0, Vec::new()));
        }
        self.decode(audio_16k, lang, true)?;
        let (text, confidence) = self.text_and_confidence()?;
        let n = self.state.full_n_segments().context("n_segments")?;
        let mut words: Vec<Word> = Vec::new();
        for i in 0..n {
            let nt = self.state.full_n_tokens(i).context("n_tokens")?;
            for tok in 0..nt {
                let piece = self.state.full_get_token_text_lossy(i, tok).unwrap_or_default();
                if piece.starts_with("[_") || piece.starts_with("<|") {
                    continue;
                }
                let d = self.state.full_get_token_data(i, tok).context("token data")?;
                // whisper times are centiseconds.
                let (a, b) = (d.t0 as f64 / 100.0, d.t1 as f64 / 100.0);
                match words.last_mut() {
                    Some(w) if !piece.starts_with(' ') => {
                        w.text.push_str(&piece);
                        w.t_end = b;
                    }
                    _ => words.push(Word { text: piece.trim_start().to_string(), t_start: a, t_end: b }),
                }
            }
        }
        words.retain(|w| !w.text.trim().is_empty());
        Ok((text, confidence, words))
    }

    fn decode(&mut self, audio_16k: &[f32], lang: &str, token_timestamps: bool) -> Result<()> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(lang));
        params.set_n_threads(self.threads);
        params.set_translate(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        if token_timestamps {
            params.set_token_timestamps(true);
        }
        let decoded = self.state.full(params, audio_16k);
        flush_native_diagnostics();
        decoded.context("whisper full()")?;
        Ok(())
    }

    /// Text (trimmed segments, space-joined) + mean token probability of the last decode.
    fn text_and_confidence(&self) -> Result<(String, f32)> {
        let n = self.state.full_n_segments().context("n_segments")?;
        let mut text = String::new();
        let mut prob_sum = 0.0f32;
        let mut prob_n = 0u32;
        for i in 0..n {
            if let Ok(seg) = self.state.full_get_segment_text(i) {
                let t = seg.trim();
                if !t.is_empty() {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(t);
                }
            }
            if let Ok(nt) = self.state.full_n_tokens(i) {
                for tok in 0..nt {
                    if let Ok(p) = self.state.full_get_token_prob(i, tok) {
                        prob_sum += p;
                        prob_n += 1;
                    }
                }
            }
        }
        let confidence = if prob_n > 0 { prob_sum / prob_n as f32 } else { 0.0 };
        Ok((text, confidence))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pipeline, resample};

    #[test]
    fn audit_f7_callback_privacy() {
        const MARKER: &str = "SYNTHETIC_RELEASE_TEXT_MUST_NOT_LOG";
        if std::env::var_os("MEETSCRIBE_TEST_NATIVE_LOG").is_some() {
            crate::diagnostics::init();
            install_native_diagnostics();
            let text = std::ffi::CString::new(MARKER).unwrap();
            for level in 0..=6 {
                // SAFETY: the callback accepts this live C string and ignores its contents.
                unsafe { native_log_callback(level, text.as_ptr(), std::ptr::null_mut()); }
            }
            // SAFETY: the adapter does not dereference either a null or invalid-encoding message.
            unsafe {
                native_log_callback(2, std::ptr::null(), std::ptr::null_mut());
                native_log_callback(3, [0xffu8, 0].as_ptr().cast(), std::ptr::null_mut());
            }
            flush_native_diagnostics();
            println!("{{\"event\":\"native_callback_ack\"}}");
            return;
        }
        for filter in ["info", "debug"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "asr::tests::audit_f7_callback_privacy", "--nocapture"])
                .env("MEETSCRIBE_TEST_NATIVE_LOG", filter).env("RUST_LOG", filter)
                .output().unwrap();
            assert!(output.status.success(), "native callback child failed: {output:?}");
            assert!(String::from_utf8_lossy(&output.stdout).contains("native_callback_ack"));
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!stderr.contains(MARKER), "decoded token leaked at {filter}");
            assert!(stderr.contains("native_warning"), "safe warning missing at {filter}");
            assert!(stderr.contains("native_error"), "safe error missing at {filter}");
        }
        println!("{{\"event\":\"audit_case_complete\",\"case\":\"f7_callback\",\"log_levels\":2}}");
    }

    /// Split on vs off must not change a single transcript byte. Decodes the audit's 30
    /// multi-speaker far-end chunks with `transcribe` and with `transcribe_words` (token
    /// timestamps on) and checks text + confidence are identical and the words rebuild the text.
    /// Needs the model + the private regression set: `cargo test asr -- --ignored`.
    #[test]
    #[ignore]
    fn split_on_leaves_text_and_confidence_byte_identical() {
        let home = std::path::PathBuf::from(std::env::var("HOME").unwrap());
        let eval = home.join(".meetscribe/eval/nemotron-split/ts");
        let sessions = home.join(".meetscribe/sessions");
        let mut far: Vec<Vec<f32>> = Vec::new();
        // Meeting ids come from the private regression set, never from the (public) source.
        let mut meetings: Vec<String> = std::fs::read_dir(&eval)
            .expect("regression set missing (~/.meetscribe/eval/nemotron-split/ts)")
            .filter_map(|e| e.ok()?.file_name().to_str()?.strip_suffix(".chunks.json").map(String::from))
            .collect();
        meetings.sort();
        for m in &meetings {
            let m = m.as_str();
            let load = |ch: &str| {
                let (s, r) = pipeline::read_wav_any_rate(&sessions.join(m).join(format!("{ch}.wav"))).unwrap();
                resample::to_16k_mono(&s, r).unwrap()
            };
            let cut = |a: &[f32], t0: f64, t1: f64| a[(t0 * 16_000.0) as usize..((t1 * 16_000.0) as usize).min(a.len())].to_vec();
            let sys = load("system");
            let chunks: Vec<serde_json::Value> =
                serde_json::from_str(&std::fs::read_to_string(eval.join(format!("{m}.chunks.json"))).unwrap()).unwrap();
            for c in &chunks {
                far.push(cut(&sys, c["t_start"].as_f64().unwrap(), c["t_end"].as_f64().unwrap()));
            }
        }
        assert_eq!(far.len(), 30);
        let model = "models/ggml-large-v3.bin";

        let mut asr = Asr::load(model).unwrap();
        let before_far: Vec<(String, f32)> = far.iter().map(|a| asr.transcribe(a, "es").unwrap()).collect();
        let squash = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        for (i, a) in far.iter().enumerate() {
            let (text, conf, words) = asr.transcribe_words(a, "es").unwrap();
            assert_eq!((&text, conf), (&before_far[i].0, before_far[i].1), "far-end chunk {i}");
            let joined: String = words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
            assert_eq!(squash(&joined), squash(&text), "words of chunk {i} must rebuild its text");
            assert!(words.windows(2).all(|w| w[0].t_start <= w[1].t_start), "chunk {i}: word times not monotonic");
        }
    }
}
