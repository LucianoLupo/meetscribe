//! Glue between `spk` (pure embedding math) and `db` (rows): turn a meeting's far-end windows
//! into cluster rows, run auto-match against the enrolled voiceprints, re-embed stored meetings
//! from their WAVs, and pick the clips `speakers play` cuts.
//!
//! Everything here is deterministic given its inputs; the only I/O is reading session WAVs in
//! [`slice_far_end`], which reuses the pipeline's roll walk so a retro-clustered meeting sees the
//! same samples the live path embedded.

use std::path::Path;

use anyhow::{Context, Result};

use crate::{db, pipeline, resample, spk};

/// One far-end window: a stored segment (or a not-yet-stored index), its times, and its
/// embedding when it was long enough to embed.
#[derive(Debug, Clone)]
pub struct Window {
    /// `transcript_segments.id` (retro path) or the index into the segments being inserted
    /// (live path) — whatever `db::SegmentVoice::segment` expects at the call site.
    pub key: i64,
    pub t_start: f64,
    pub t_end: f64,
    pub embedding: Option<Vec<f32>>,
}

/// Clusters + memberships ready to write, plus the decoded centroids for matching.
#[derive(Debug, Default)]
pub struct Assembly {
    pub clusters: Vec<db::NewCluster>,
    pub voices: Vec<db::SegmentVoice>,
    centroids: Vec<Vec<f32>>,
}

impl Assembly {
    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }
}

/// Cluster the embedded windows at `cut`, letter the clusters by speech time, and let short
/// (un-embedded) windows inherit a neighbour's cluster. Pure.
pub fn assemble(windows: &[Window], cut: f32) -> Assembly {
    let embedded: Vec<usize> = (0..windows.len()).filter(|&i| windows[i].embedding.is_some()).collect();
    if embedded.is_empty() {
        return Assembly::default();
    }
    let embs: Vec<&[f32]> = embedded
        .iter()
        .map(|&i| windows[i].embedding.as_deref().expect("filtered"))
        .collect();
    let assignments = spk::cluster(&embs, cut);
    let secs: Vec<f64> = embedded.iter().map(|&i| windows[i].t_end - windows[i].t_start).collect();
    let labels = spk::label_clusters(&assignments, &secs);

    let mut clusters = Vec::with_capacity(labels.len());
    let mut centroids = Vec::with_capacity(labels.len());
    for (c, label) in labels.iter().enumerate() {
        let members: Vec<&[f32]> = assignments
            .iter()
            .zip(&embs)
            .filter(|(a, _)| **a == c)
            .map(|(_, e)| *e)
            .collect();
        let speech_secs: f64 = assignments.iter().zip(&secs).filter(|(a, _)| **a == c).map(|(_, s)| *s).sum();
        let centroid = spk::centroid(&members);
        clusters.push(db::NewCluster {
            cluster: label.clone(),
            speaker_id: None,
            assigned_by: None,
            match_score: None,
            centroid: spk::to_blob(&centroid),
            dim: centroid.len() as i64,
            n_windows: members.len() as i64,
            speech_secs,
        });
        centroids.push(centroid);
    }

    let mut voices: Vec<db::SegmentVoice> = embedded
        .iter()
        .zip(&assignments)
        .map(|(&i, &c)| db::SegmentVoice {
            segment: windows[i].key,
            cluster: c,
            inherited: false,
            embedding: windows[i].embedding.as_deref().map(spk::to_blob),
        })
        .collect();

    let anchors: Vec<(f64, usize)> = embedded.iter().zip(&assignments).map(|(&i, &c)| (windows[i].t_start, c)).collect();
    let short: Vec<usize> = (0..windows.len()).filter(|&i| windows[i].embedding.is_none()).collect();
    let short_starts: Vec<f64> = short.iter().map(|&i| windows[i].t_start).collect();
    for (&i, inherited) in short.iter().zip(spk::inherit_short(&anchors, &short_starts)) {
        if let Some(c) = inherited {
            voices.push(db::SegmentVoice { segment: windows[i].key, cluster: c, inherited: true, embedding: None });
        }
    }

    Assembly { clusters, voices, centroids }
}

pub fn decode_enrolled(rows: &[db::EnrolledRow]) -> Result<Vec<spk::Enrolled>> {
    rows.iter()
        .map(|r| {
            Ok(spk::Enrolled {
                speaker_id: r.speaker_id,
                voiceprints: r.voiceprints.iter().map(|b| spk::from_blob(b)).collect::<Result<_>>()?,
            })
        })
        .collect()
}

/// Run auto-match over every cluster; returns how many were assigned. Only sets `auto`
/// provenance — a manual label is never overwritten here because the assembly has none yet.
pub fn apply_matches(a: &mut Assembly, enrolled: &[spk::Enrolled]) -> usize {
    let mut n = 0;
    for (c, centroid) in a.clusters.iter_mut().zip(&a.centroids) {
        if let Some((sid, score)) = spk::auto_match(centroid, c.n_windows as usize, enrolled) {
            c.speaker_id = Some(sid);
            c.assigned_by = Some("auto".into());
            c.match_score = Some(f64::from(score));
            n += 1;
        }
    }
    n
}

/// Far-end audio at 16 kHz for each `(t_start, t_end)` in meeting time, sliced from the
/// session's `system.wav` rolls exactly as the live pipeline saw them (same roll order, same
/// `segments.txt` gaps, same resampler). A range no roll covers comes back `None`.
pub fn slice_far_end(dir: &Path, ranges: &[(f64, f64)]) -> Result<Vec<Option<Vec<f32>>>> {
    let files = pipeline::discover_channel(dir, "system");
    if files.is_empty() {
        anyhow::bail!("no system.wav in {}", dir.display());
    }
    let gaps = pipeline::read_segment_gaps(dir)?;
    let rate = resample::TARGET_RATE as f64;
    let mut out: Vec<Option<Vec<f32>>> = vec![None; ranges.len()];
    let mut offset = 0.0f64;
    for (i, path) in files.iter().enumerate() {
        if i >= 1 {
            offset += gaps.get(&(i as u32)).copied().unwrap_or(0.0);
        }
        let (samples, sr) = pipeline::read_wav_any_rate(path)?;
        let audio = resample::to_16k_mono(&samples, sr).with_context(|| format!("resample {}", path.display()))?;
        let dur = audio.len() as f64 / rate;
        for (k, &(ts, te)) in ranges.iter().enumerate() {
            if ts < offset || ts >= offset + dur {
                continue;
            }
            // Stored times are `offset + whole_ms / 1000` from the live loop, so the residual is an
            // integer number of milliseconds. Anything else means the pipeline's timing changed or
            // the WAV is not the one that was transcribed — refuse rather than embed the wrong audio.
            for t in [ts, te] {
                let ms = (t - offset) * 1000.0;
                if (ms - ms.round()).abs() > 1e-6 {
                    anyhow::bail!(
                        "segment time {t:.6}s is not on a millisecond boundary of {} (offset {offset:.6}s) — \
                         the recording no longer matches the stored transcript",
                        path.display()
                    );
                }
            }
            let a = ((ts - offset) * rate).round() as usize;
            let b = (((te - offset) * rate).round() as usize).min(audio.len());
            if b > a {
                out[k] = Some(audio[a..b].to_vec());
            }
        }
        offset += dur;
    }
    Ok(out)
}

/// Embed a stored meeting's far-end segments from its WAVs (the retro `speakers cluster` path).
/// Segments shorter than the embedding floor, or outside every roll, come back un-embedded.
pub fn embed_far_end(dir: &Path, segments: &[(i64, f64, f64)], emb: &mut spk::Embedder) -> Result<Vec<Window>> {
    let ranges: Vec<(f64, f64)> = segments.iter().map(|s| (s.1, s.2)).collect();
    let slices = slice_far_end(dir, &ranges)?;
    let floor = spk::MIN_WINDOW_MS * resample::TARGET_RATE as usize / 1000;
    let mut out = Vec::with_capacity(segments.len());
    for (&(key, t_start, t_end), slice) in segments.iter().zip(slices) {
        let embedding = match slice {
            Some(audio) if audio.len() >= floor => Some(emb.embed(&audio)?),
            _ => None,
        };
        out.push(Window { key, t_start, t_end, embedding });
    }
    Ok(out)
}

/// The windows `speakers play` cuts: the `n` embedded windows closest to the centroid, spread
/// out in time — at least `min_gap_secs` apart where the cluster allows it, closer only when it
/// does not (a five-window cluster inside ninety seconds must still play).
pub fn pick_clips<'a>(windows: &'a [db::WindowRow], centroid: &[f32], n: usize, min_gap_secs: f64) -> Vec<&'a db::WindowRow> {
    let mut ranked: Vec<(f32, &db::WindowRow)> = windows
        .iter()
        .filter_map(|w| w.embedding.as_deref().and_then(|b| spk::from_blob(b).ok()).map(|e| (spk::cosine(&e, centroid), w)))
        .collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut picked: Vec<&db::WindowRow> = Vec::new();
    for gap in [min_gap_secs, 0.0] {
        for (_, w) in &ranked {
            if picked.len() >= n {
                break;
            }
            let clashes = picked.iter().any(|p| std::ptr::eq(*p, *w) || (p.t_start - w.t_start).abs() < gap);
            if !clashes {
                picked.push(w);
            }
        }
    }
    picked.sort_by(|a, b| a.t_start.partial_cmp(&b.t_start).unwrap_or(std::cmp::Ordering::Equal));
    picked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(axis: usize, tilt: f32) -> Vec<f32> {
        let mut v = [0.0f32; 3];
        v[axis] = 1.0;
        v[(axis + 1) % 3] = tilt;
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    fn win(key: i64, t: f64, len: f64, e: Option<Vec<f32>>) -> Window {
        Window { key, t_start: t, t_end: t + len, embedding: e }
    }

    #[test]
    fn assemble_letters_by_speech_time_and_inherits_short_windows() {
        // Two voices: axis 0 talks 30 s in two windows, axis 1 talks 5 s once; one 1 s window
        // at t=31 has no embedding and sits next to the axis-0 windows.
        let windows = vec![
            win(10, 0.0, 20.0, Some(unit(0, 0.0))),
            win(11, 25.0, 10.0, Some(unit(0, 0.05))),
            win(12, 100.0, 5.0, Some(unit(1, 0.0))),
            win(13, 36.0, 1.0, None),
            win(14, 500.0, 1.0, None), // nothing within 30 s → unassigned
        ];
        let a = assemble(&windows, spk::CLUSTER_CUT);
        assert_eq!(a.clusters.len(), 2);
        assert_eq!(a.clusters[0].cluster, "A");
        assert_eq!((a.clusters[0].n_windows, a.clusters[0].speech_secs), (2, 30.0));
        assert_eq!(a.clusters[1].cluster, "B");
        assert_eq!(a.clusters[0].dim, 3);
        // 3 embedded voices + 1 inherited; key 14 is absent.
        assert_eq!(a.voices.len(), 4);
        let inherited = a.voices.iter().find(|v| v.segment == 13).unwrap();
        assert!(inherited.inherited && inherited.embedding.is_none() && inherited.cluster == 0);
        assert!(a.voices.iter().all(|v| v.segment != 14));
        assert!(a.voices.iter().filter(|v| !v.inherited).all(|v| v.embedding.is_some()));
    }

    #[test]
    fn assemble_with_nothing_embedded_is_empty() {
        let a = assemble(&[win(1, 0.0, 1.0, None)], spk::CLUSTER_CUT);
        assert!(a.is_empty() && a.voices.is_empty());
    }

    #[test]
    fn matches_are_applied_as_auto_with_a_score() {
        let windows: Vec<Window> = (0..spk::MIN_WINDOWS).map(|i| win(i as i64, i as f64 * 10.0, 5.0, Some(unit(0, 0.0)))).collect();
        let mut a = assemble(&windows, spk::CLUSTER_CUT);
        let enrolled = vec![spk::Enrolled { speaker_id: 42, voiceprints: vec![unit(0, 0.1)] }];
        assert_eq!(apply_matches(&mut a, &enrolled), 1);
        assert_eq!(a.clusters[0].speaker_id, Some(42));
        assert_eq!(a.clusters[0].assigned_by.as_deref(), Some("auto"));
        assert!(a.clusters[0].match_score.unwrap() > 0.9);
        // Too small a cluster is left alone.
        let mut small = assemble(&windows[..2], spk::CLUSTER_CUT);
        assert_eq!(apply_matches(&mut small, &enrolled), 0);
        assert!(small.clusters[0].speaker_id.is_none());
    }

    #[test]
    fn decode_enrolled_rejects_a_corrupt_blob() {
        let ok = db::EnrolledRow { speaker_id: 1, voiceprints: vec![spk::to_blob(&unit(0, 0.0))] };
        assert_eq!(decode_enrolled(std::slice::from_ref(&ok)).unwrap()[0].voiceprints.len(), 1);
        let bad = db::EnrolledRow { speaker_id: 2, voiceprints: vec![vec![1, 2, 3]] };
        assert!(decode_enrolled(&[ok, bad]).is_err());
    }

    fn tempdir() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!("meetscribe-voices-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_wav(path: &Path, samples: &[f32], rate: u32) {
        let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for s in samples {
            w.write_sample(*s).unwrap();
        }
        w.finalize().unwrap();
    }

    /// Two 16 kHz rolls with a recorded 0.5 s gap between them; every sample carries its own
    /// index so a slice proves exactly which samples it holds.
    #[test]
    fn slices_follow_the_pipeline_roll_offsets() {
        let dir = tempdir();
        let roll0: Vec<f32> = (0..16_000).map(|i| i as f32 / 1e6).collect(); // 1.0 s
        let roll1: Vec<f32> = (0..32_000).map(|i| (100_000 + i) as f32 / 1e6).collect(); // 2.0 s
        write_wav(&dir.join("system.wav"), &roll0, 16_000);
        write_wav(&dir.join("system.001.wav"), &roll1, 16_000);
        std::fs::write(dir.join("segments.txt"), "seg 1 mic=x system=y rate=16000 gap_frames=8000\n").unwrap();

        // Meeting time: roll0 covers [0,1), gap 0.5, roll1 covers [1.5, 3.5).
        let got = slice_far_end(&dir, &[(0.25, 0.5), (1.5, 2.0), (2.0, 9.0), (1.2, 1.4), (0.9, 1.3)]).unwrap();
        let s0 = got[0].as_ref().unwrap();
        assert_eq!(s0.len(), 4000);
        assert!((s0[0] - roll0[4000]).abs() < 1e-9, "first roll, sample 4000");
        let s1 = got[1].as_ref().unwrap();
        assert_eq!(s1.len(), 8000);
        assert!((s1[0] - roll1[0]).abs() < 1e-9, "second roll starts at meeting time 1.5 s");
        let s2 = got[2].as_ref().unwrap();
        assert_eq!(s2.len(), 24_000, "clamped to the roll's end");
        assert!(got[3].is_none(), "inside the gap: no roll covers it");
        let s4 = got[4].as_ref().unwrap();
        assert_eq!(s4.len(), 1600, "a range starting in roll0 is clamped to roll0's end");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_time_off_the_millisecond_grid_is_refused() {
        let dir = tempdir();
        write_wav(&dir.join("system.wav"), &vec![0.0f32; 16_000], 16_000);
        assert!(slice_far_end(&dir, &[(0.25, 0.5)]).is_ok());
        let err = slice_far_end(&dir, &[(0.2504, 0.5)]).unwrap_err();
        assert!(err.to_string().contains("millisecond boundary"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn slice_without_a_far_end_track_is_an_error() {
        let dir = tempdir();
        assert!(slice_far_end(&dir, &[(0.0, 1.0)]).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    fn row(id: i64, t: f64, e: &[f32]) -> db::WindowRow {
        db::WindowRow { segment_id: id, t_start: t, t_end: t + 5.0, inherited: false, embedding: Some(spk::to_blob(e)) }
    }

    #[test]
    fn clips_prefer_central_windows_spread_out_in_time() {
        let c = unit(0, 0.0);
        let rows = vec![
            row(1, 0.0, &unit(0, 0.02)),   // very central, t=0
            row(2, 30.0, &unit(0, 0.01)),  // most central, but within 120 s of #1
            row(3, 400.0, &unit(0, 0.3)),  // far in time, less central
            row(4, 800.0, &unit(1, 0.0)),  // far in time, not this voice at all
            db::WindowRow { segment_id: 5, t_start: 900.0, t_end: 901.0, inherited: true, embedding: None },
        ];
        let picked: Vec<i64> = pick_clips(&rows, &c, 3, 120.0).iter().map(|w| w.segment_id).collect();
        assert_eq!(picked, vec![2, 3, 4], "best-first, then spaced, never an inherited window");
        // A tight cluster still yields clips: spacing relaxes rather than starving.
        let tight = vec![row(1, 0.0, &unit(0, 0.0)), row(2, 10.0, &unit(0, 0.0)), row(3, 20.0, &unit(0, 0.0))];
        assert_eq!(pick_clips(&tight, &c, 3, 120.0).len(), 3);
        assert_eq!(pick_clips(&tight, &c, 2, 120.0).len(), 2);
    }
}
