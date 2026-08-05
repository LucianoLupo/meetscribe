//! 16 kHz mono resampling of native-rate capture WAVs (Phase 2).
//!
//! Capture writes each channel at the aggregate's native rate (built-in mic 48 kHz,
//! Teams tap ~24 kHz, BT headset 16 kHz). Both silero VAD and whisper require 16 kHz
//! mono f32, so every channel is resampled once here before VAD/ASR.

use anyhow::{Context, Result};
use rubato::{FftFixedIn, Resampler};

pub const TARGET_RATE: u32 = 16_000;

/// Force a sample into the [-1, 1] range silero and whisper require.
///
/// Capture can land a hair outside full scale: the aggregate device's drift-compensating
/// resampler overshoots on transients, so a 64-minute meeting may hold a single sample at
/// e.g. 1.008. silero's `validate_input` rejects the WHOLE buffer on ONE out-of-range
/// sample (`VadError::InvalidData`) — and that check is `#[cfg(debug_assertions)]`, which
/// the daemon binary is built with. Clamping here costs nothing and can't lose speech.
fn normalize(sample: f32) -> f32 {
    if sample.is_finite() { sample.clamp(-1.0, 1.0) } else { 0.0 }
}

/// Resample mono f32 `input` from `src_rate` to 16 kHz. Passthrough copy when already 16 kHz.
///
/// Uses an FFT resampler over fixed-size chunks; the final partial chunk is zero-padded.
/// FFT resamplers add a small constant startup delay (sub-20 ms) — negligible for
/// speaker-labeled batch transcription.
///
/// Output is always in [-1, 1] — see [`normalize`].
pub fn to_16k_mono(input: &[f32], src_rate: u32) -> Result<Vec<f32>> {
    if src_rate == TARGET_RATE {
        return Ok(input.iter().copied().map(normalize).collect());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    const CHUNK: usize = 1024;
    let mut resampler = FftFixedIn::<f32>::new(src_rate as usize, TARGET_RATE as usize, CHUNK, 2, 1)
        .context("build FftFixedIn resampler")?;

    let need = resampler.input_frames_next(); // fixed == CHUNK for FftFixedIn
    let approx_out = input.len() * TARGET_RATE as usize / src_rate as usize + need;
    let mut out: Vec<f32> = Vec::with_capacity(approx_out);

    let mut frame = vec![0.0f32; need];
    let mut pos = 0usize;
    while pos < input.len() {
        let end = (pos + need).min(input.len());
        let n = end - pos;
        // Sanitize on the way IN as well: rubato's FFT path *panics* on a non-finite sample.
        for (dst, &src) in frame[..n].iter_mut().zip(&input[pos..end]) {
            *dst = normalize(src);
        }
        if n < need {
            frame[n..].fill(0.0); // zero-pad the final chunk
        }
        let wave_out = resampler
            .process(&[frame.as_slice()], None)
            .context("resample chunk")?;
        out.extend(wave_out[0].iter().copied().map(normalize));
        pos += need;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_already_16k() {
        let input = vec![0.1f32; 16_000];
        let out = to_16k_mono(&input, 16_000).unwrap();
        assert_eq!(out.len(), 16_000);
        assert_eq!(out, input);
    }

    #[test]
    fn downsample_48k_to_16k_length_ratio() {
        // 1 s of a 440 Hz sine at 48 kHz → ~16 k samples (ratio ~1/3).
        let n = 48_000usize;
        let input: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 48_000.0).sin())
            .collect();
        let out = to_16k_mono(&input, 48_000).unwrap();
        let ratio = out.len() as f64 / input.len() as f64;
        assert!(
            (ratio - 1.0 / 3.0).abs() < 0.02,
            "expected ~1/3 ratio, got {ratio} (out {} samples)",
            out.len()
        );
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(to_16k_mono(&[], 48_000).unwrap().is_empty());
    }

    /// One sample past full scale used to abort a whole meeting at the VAD (silero rejects
    /// the entire buffer on a single out-of-range sample). Both paths must clamp.
    #[test]
    fn output_always_within_unit_range() {
        let mut input = vec![0.1f32; 16_000];
        input[7] = 1.007_948_2; // observed in a real 64-minute capture
        input[9] = -1.5;
        input[11] = f32::NAN;

        for rate in [16_000u32, 48_000] {
            let out = to_16k_mono(&input, rate).unwrap();
            assert!(
                out.iter().all(|s| s.is_finite() && (-1.0..=1.0).contains(s)),
                "{rate} Hz path left a sample outside [-1, 1]"
            );
        }
        // Passthrough clamps rather than mangles: only the offending samples change.
        let out = to_16k_mono(&input, 16_000).unwrap();
        assert_eq!(out[7], 1.0);
        assert_eq!(out[9], -1.0);
        assert_eq!(out[11], 0.0);
        assert_eq!(out[0], 0.1);
    }
}
