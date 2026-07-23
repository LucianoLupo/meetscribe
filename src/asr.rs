//! Whisper ASR wrapper (Phase 2) — one loaded model + ONE reused decode state, transcribe per VAD window.
//!
//! Reuses the whisper-rs sequence proven in `src/bin/rtf_probe.rs` (Phase 1.5).
//! CoreML is still toggled at build time by the crate `coreml` feature.

use anyhow::{Context, Result};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

/// Windows shorter than this are skipped: whisper's own encoder floor is ~100 mel frames
/// (≈ 16 040 samples / 1002 ms), below which `whisper_full` logs "input is too short" and
/// returns zero segments. We skip well below that rather than feed it a window it will drop
/// anyway. (Recovering sub-second utterances would mean padding past the frame floor AND
/// accepting whisper's tendency to hallucinate on trailing silence — a separate feature.)
const MIN_SAMPLES: usize = 16_000 / 10; // 100 ms @ 16 kHz

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
        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
            .with_context(|| format!("load whisper model {model_path}"))?;
        // Create the decode state once. `WhisperState` holds its own `Arc` to the inner context,
        // so it stays valid after `ctx` is dropped at the end of `load`.
        let state = ctx.create_state().context("create whisper state")?;
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

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(lang));
        params.set_n_threads(self.threads);
        params.set_translate(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        self.state
            .full(params, audio_16k)
            .context("whisper full()")?;

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
