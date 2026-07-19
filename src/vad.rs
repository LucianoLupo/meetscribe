//! Voice-activity detection (silero) — gates ASR per channel (Phase 2).
//!
//! Runs silero VAD over a 16 kHz mono buffer and returns coalesced speech windows.
//! Whisper then transcribes only these windows (skipping silence = the "gates ASR"
//! win), and each window's absolute start/end drives the You/Others merge.

use anyhow::{Context, Result};
use silero::{VadConfig, VadSession, VadTransition};

/// Merge speech segments separated by less than this much silence (brief in-turn pauses).
const COALESCE_GAP_MS: usize = 800;
/// Don't coalesce past this length — keeps windows whisper-friendly and preserves
/// turn-level granularity for the cross-channel merge.
const MAX_WINDOW_MS: usize = 30_000;

const SAMPLES_PER_MS: usize = 16; // 16 kHz

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechWindow {
    pub start_ms: usize,
    pub end_ms: usize,
}

impl SpeechWindow {
    /// Sample range into the 16 kHz buffer this window covers.
    pub fn sample_range(&self, len: usize) -> (usize, usize) {
        let a = (self.start_ms * SAMPLES_PER_MS).min(len);
        let b = (self.end_ms * SAMPLES_PER_MS).min(len);
        (a, b.max(a))
    }
}

/// Detect coalesced speech windows in a 16 kHz mono buffer, using silero's default
/// tuning (0.5/0.35 thresholds, 600 ms redemption/pre-pad, 90 ms min speech).
pub fn speech_windows(audio_16k: &[f32]) -> Result<Vec<SpeechWindow>> {
    let total_ms = audio_16k.len() / SAMPLES_PER_MS;
    let mut vad = VadSession::new(VadConfig::default()).context("init silero VAD (static-model)")?;

    let mut raw: Vec<(usize, usize)> = Vec::new();
    for chunk in audio_16k.chunks(16_000) {
        for t in vad.process(chunk).context("vad.process")? {
            if let VadTransition::SpeechEnd { start_timestamp_ms, end_timestamp_ms, .. } = t {
                raw.push((start_timestamp_ms, end_timestamp_ms));
            }
        }
    }
    // Trailing speech still open at end-of-stream → close it at the audio end.
    if vad.is_speaking() {
        let dur = vad.current_speech_duration().as_millis() as usize;
        raw.push((total_ms.saturating_sub(dur), total_ms));
    }

    Ok(coalesce(raw, COALESCE_GAP_MS, MAX_WINDOW_MS))
}

/// Merge adjacent intervals with a gap < `gap_ms`, but never past `max_ms` total length.
fn coalesce(raw: Vec<(usize, usize)>, gap_ms: usize, max_ms: usize) -> Vec<SpeechWindow> {
    let mut out: Vec<SpeechWindow> = Vec::new();
    for (s, e) in raw {
        if let Some(last) = out.last_mut()
            && s.saturating_sub(last.end_ms) < gap_ms
            && e.saturating_sub(last.start_ms) <= max_ms
        {
            last.end_ms = e.max(last.end_ms);
            continue;
        }
        out.push(SpeechWindow { start_ms: s, end_ms: e });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesce_merges_close_splits_far() {
        // gap 500 ms (<800) → merge; gap 3000 ms (>=800) → split.
        let raw = vec![(0, 1000), (1500, 2000), (5000, 6000)];
        let w = coalesce(raw, 800, 30_000);
        assert_eq!(w.len(), 2);
        assert_eq!((w[0].start_ms, w[0].end_ms), (0, 2000));
        assert_eq!((w[1].start_ms, w[1].end_ms), (5000, 6000));
    }

    #[test]
    fn coalesce_respects_max_cap() {
        // gap 500 ms (<800) but merged length 40 s (>30 s) → do NOT merge.
        let raw = vec![(0, 25_000), (25_500, 40_000)];
        let w = coalesce(raw, 800, 30_000);
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn sample_range_clamps_to_len() {
        let w = SpeechWindow { start_ms: 100, end_ms: 200 };
        assert_eq!(w.sample_range(16_000 * 10), (1600, 3200));
        // beyond buffer → clamped
        let w2 = SpeechWindow { start_ms: 1000, end_ms: 2000 };
        assert_eq!(w2.sample_range(8000), (8000, 8000));
    }
}
