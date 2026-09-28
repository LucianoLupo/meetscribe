//! Whisper ASR wrapper (Phase 2) — one loaded model + ONE reused decode state, transcribe per VAD window.
//!
//! Reuses the whisper-rs sequence proven in `src/bin/rtf_probe.rs` (Phase 1.5).
//! CoreML is still toggled at build time by the crate `coreml` feature.

use anyhow::{Context, Result};
use whisper_rs::{
    DtwMode, DtwModelPreset, DtwParameters, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters, WhisperState,
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

pub struct Asr {
    /// One long-lived decode state, reused across every window. `full()` resets its internal
    /// buffers each call so windows do not bleed into each other; creating a fresh state per
    /// window (the old path) instead rebuilt the whole Metal+BLAS backend every call — an
    /// init/free cycle per window, pure churn.
    state: WhisperState,
    threads: i32,
    /// DTW token timestamps were enabled at load (only when far-end splitting is on).
    dtw: bool,
}

impl Asr {
    /// `dtw` enables DTW token timestamps (large-v3 alignment heads) for [`Asr::transcribe_words`].
    /// It is a context-wide, load-time setting, so it is on ONLY when the far-end diarizer loaded;
    /// otherwise the context is exactly the pre-split default.
    pub fn load(model_path: &str, dtw: bool) -> Result<Self> {
        let mut cp = WhisperContextParameters::default();
        if dtw {
            cp.dtw_parameters(DtwParameters {
                mode: DtwMode::ModelPreset { model_preset: DtwModelPreset::LargeV3 },
                ..Default::default()
            });
        }
        let ctx = WhisperContext::new_with_params(model_path, cp)
            .with_context(|| format!("load whisper model {model_path}"))?;
        // Create the decode state once. `WhisperState` holds its own `Arc` to the inner context,
        // so it stays valid after `ctx` is dropped at the end of `load`.
        let state = ctx.create_state().context("create whisper state")?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(8) as i32;
        Ok(Self { state, threads, dtw })
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
    /// timestamps are added. Word times use DTW when it was enabled at load, else whisper's
    /// plain token timestamps. A new word starts at a token with a leading space; special
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
                // whisper times are centiseconds; DTW gives one point per token (-1 = unavailable).
                let (a, b) = if self.dtw && d.t_dtw >= 0 {
                    (d.t_dtw as f64 / 100.0, d.t_dtw as f64 / 100.0)
                } else {
                    (d.t0 as f64 / 100.0, d.t1 as f64 / 100.0)
                };
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
        self.state.full(params, audio_16k).context("whisper full()")?;
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

    /// Split on vs off must not change a single transcript byte (plan Step 3). Decodes the audit's
    /// 30 multi-speaker far-end chunks plus a sample of mic windows twice — pre-split context with
    /// `transcribe`, DTW context with `transcribe_words` (far-end) / `transcribe` (mic) — and
    /// checks text + confidence are identical and the words concatenate back to the text.
    /// Needs the model + the private regression set: `cargo test asr -- --ignored`.
    #[test]
    #[ignore]
    fn split_on_leaves_text_and_confidence_byte_identical() {
        let home = std::path::PathBuf::from(std::env::var("HOME").unwrap());
        let eval = home.join(".meetscribe/eval/nemotron-split/ts");
        let sessions = home.join(".meetscribe/sessions");
        let mut far: Vec<Vec<f32>> = Vec::new();
        let mut mic: Vec<Vec<f32>> = Vec::new();
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
            let segs: Vec<serde_json::Value> =
                serde_json::from_str(&std::fs::read_to_string(sessions.join(m).join("transcript.json")).unwrap()).unwrap();
            let micw = load("mic");
            for s in segs.iter().filter(|s| s["speaker"] == "you").take(4) {
                mic.push(cut(&micw, s["t_start"].as_f64().unwrap(), s["t_end"].as_f64().unwrap()));
            }
        }
        assert_eq!(far.len(), 30);
        let model = "models/ggml-large-v3.bin";

        let mut plain = Asr::load(model, false).unwrap();
        let before_far: Vec<(String, f32)> = far.iter().map(|a| plain.transcribe(a, "es").unwrap()).collect();
        let before_mic: Vec<(String, f32)> = mic.iter().map(|a| plain.transcribe(a, "es").unwrap()).collect();
        drop(plain);

        let mut dtw = Asr::load(model, true).unwrap();
        let squash = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        for (i, a) in far.iter().enumerate() {
            let (text, conf, words) = dtw.transcribe_words(a, "es").unwrap();
            assert_eq!((&text, conf), (&before_far[i].0, before_far[i].1), "far-end chunk {i}");
            let joined: String = words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
            assert_eq!(squash(&joined), squash(&text), "words of chunk {i} must rebuild its text");
            assert!(words.windows(2).all(|w| w[0].t_start <= w[1].t_start), "chunk {i}: word times not monotonic");
        }
        for (i, a) in mic.iter().enumerate() {
            assert_eq!(dtw.transcribe(a, "es").unwrap(), before_mic[i], "mic window {i}");
        }
    }
}
