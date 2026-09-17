//! Speaker embeddings + clustering (Batch E) — the far-end identity layer.
//!
//! One VAD window of far-end speech → one 192-d embedding (3D-Speaker CAM++ via ONNX Runtime,
//! Kaldi fbank front-end). Within a meeting, windows are clustered by average-linkage on cosine
//! distance; each cluster's centroid is matched against the enrolled voiceprints of named speakers.
//!
//! This module is **path-included by `src/bin/spk_probe.rs`** as well as declared on the main
//! binary, so it must not use `crate::` paths — pure functions, `Embedder`, and BLOB codecs only.
//!
//! Every threshold below comes from the Batch D listening tests
//! (`plans/2026-09-17-batch-d-speaker-embedding-results.md`, addendum), not from a paper.

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array3;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;

/// Windows shorter than this are not embedded; they inherit a neighbour's cluster instead.
pub const MIN_WINDOW_MS: usize = 1500;
/// Within-meeting agglomerative cut on cosine DISTANCE (1 − cosine): merge while the average
/// cosine between two groups is ≥ 0.55. Six of seven groups were one person each at this cut.
pub const CLUSTER_CUT: f32 = 0.45;
/// Cross-meeting recall: a cluster centroid needs at least this cosine to an enrolled voiceprint.
/// Set from the first day of real labelling (12 people, 8 meetings): every automatic match the
/// owner judged wrong scored 0.55–0.62, every right one 0.76 or above. 0.70 sits in the gap.
pub const MATCH_THRESHOLD: f32 = 0.70;
/// The best speaker must beat the runner-up by this much, or the cluster stays unassigned.
pub const MATCH_MARGIN: f32 = 0.05;
/// Clusters with fewer embedded windows are never auto-named: the one mixed group in the
/// listening test had 4 windows. A wrong name is worse than no name.
pub const MIN_WINDOWS: usize = 5;
/// A short window inherits the cluster of the nearest embedded window within this many seconds.
pub const INHERIT_MAX_SECS: f64 = 30.0;

// ---------------------------------------------------------------- embedder

/// The speaker-embedding model: 16 kHz mono clip in, L2-normalised embedding out.
pub struct Embedder {
    session: Session,
    input_name: String,
    /// Embedding size, known after the first `embed` (0 before).
    pub dim: usize,
}

impl Embedder {
    pub fn load(path: &Path) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(path)
            .with_context(|| format!("load speaker model {}", path.display()))?;
        let input_name = session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .context("speaker model has no inputs")?;
        Ok(Self { session, input_name, dim: 0 })
    }

    /// L2-normalised embedding of a 16 kHz mono clip.
    ///
    /// knf: 25 ms / 10 ms Kaldi fbank, 80 bins, dither 0, then per-utterance mean subtraction —
    /// exactly the model's recipe (`normalize_samples=1`, `feature_normalize_type=global-mean`).
    /// Do NOT "fix" the mel high cutoff to match sherpa-onnx: 3D-Speaker trains with the Kaldi
    /// default, and this path was proven bit-faithful in Batch D.
    pub fn embed(&mut self, audio_16k: &[f32]) -> Result<Vec<f32>> {
        let feats = knf_rs::compute_fbank(audio_16k).map_err(|e| anyhow::anyhow!("fbank: {e}"))?;
        let (t, bins) = feats.dim();
        let x: Array3<f32> = feats
            .into_shape_with_order((1, t, bins))
            .context("reshape fbank to (1,T,80)")?;
        let outputs = self.session.run(ort::inputs![
            self.input_name.as_str() => TensorRef::from_array_view(x.view())?
        ])?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
        let v = normalized(data);
        self.dim = v.len();
        Ok(v)
    }
}

// ---------------------------------------------------------------- vector helpers

/// Cosine similarity of two L2-normalised vectors (a plain dot product).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalized(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|a| a / norm).collect()
}

/// Mean of a set of embeddings, re-normalised. Empty input yields an empty vector.
pub fn centroid<V: AsRef<[f32]>>(vs: &[V]) -> Vec<f32> {
    let Some(first) = vs.first() else { return Vec::new() };
    let dim = first.as_ref().len();
    let mut acc = vec![0.0f32; dim];
    for v in vs {
        for (a, x) in acc.iter_mut().zip(v.as_ref()) {
            *a += x;
        }
    }
    normalized(&acc)
}

/// Embedding → SQLite BLOB (f32 little-endian).
pub fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// SQLite BLOB (f32 little-endian) → embedding. A length that is not a multiple of 4 is corrupt.
pub fn from_blob(b: &[u8]) -> Result<Vec<f32>> {
    if !b.len().is_multiple_of(4) {
        anyhow::bail!("embedding blob has {} bytes, not a multiple of 4", b.len());
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

// ---------------------------------------------------------------- clustering

/// Average-linkage agglomerative clustering on cosine distance (1 − cosine), cut at `cut`.
///
/// Returns one cluster index per input, 0-based, in order of first appearance. Same semantics as
/// scipy's `linkage(method="average", metric="cosine")` + `fcluster(criterion="distance")`, which
/// is what the listening-test clips were cut from. Lance–Williams update, O(n³) worst case — a
/// two-hour meeting has a few hundred windows, so no dependency is worth it.
pub fn cluster<V: AsRef<[f32]>>(embeddings: &[V], cut: f32) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    // Pairwise distance matrix between CURRENT clusters (starts as singletons).
    let mut d = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = 1.0 - cosine(embeddings[i].as_ref(), embeddings[j].as_ref());
            d[i][j] = dist;
            d[j][i] = dist;
        }
    }
    let mut alive: Vec<bool> = vec![true; n];
    let mut size: Vec<usize> = vec![1; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    loop {
        // Closest live pair.
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..n {
            if !alive[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !alive[j] {
                    continue;
                }
                if best.is_none_or(|(_, _, bd)| d[i][j] < bd) {
                    best = Some((i, j, d[i][j]));
                }
            }
        }
        let Some((i, j, dist)) = best else { break };
        if dist >= cut {
            break;
        }
        // Merge j into i: average linkage = size-weighted mean of the two distances.
        for k in 0..n {
            if k == i || k == j || !alive[k] {
                continue;
            }
            let merged = (size[i] as f32 * d[i][k] + size[j] as f32 * d[j][k])
                / (size[i] + size[j]) as f32;
            d[i][k] = merged;
            d[k][i] = merged;
        }
        let moved = std::mem::take(&mut members[j]);
        members[i].extend(moved);
        size[i] += size[j];
        alive[j] = false;
    }

    // A merge always keeps the lower slot, so live slots in index order ARE the clusters in
    // first-appearance order: number them 0, 1, 2, … that way.
    let mut out = vec![0usize; n];
    for (next, m) in members.iter().filter(|m| !m.is_empty()).enumerate() {
        for &idx in m {
            out[idx] = next;
        }
    }
    out
}

/// Excel-style letters: 0 → "A", 25 → "Z", 26 → "AA", …
pub fn letter(mut i: usize) -> String {
    let mut s = Vec::new();
    loop {
        s.push(b'A' + (i % 26) as u8);
        if i < 26 {
            break;
        }
        i = i / 26 - 1;
    }
    s.reverse();
    String::from_utf8(s).expect("ASCII letters")
}

/// Label clusters `A`, `B`, … by descending total speech time. Input: one `(cluster, secs)` per
/// window. Output: label per cluster index (`out[cluster]`). Ties break on cluster index so the
/// result is deterministic.
pub fn label_clusters(assignments: &[usize], secs: &[f64]) -> Vec<String> {
    let k = assignments.iter().map(|c| c + 1).max().unwrap_or(0);
    let mut total = vec![0.0f64; k];
    for (&c, &s) in assignments.iter().zip(secs) {
        total[c] += s;
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| total[b].partial_cmp(&total[a]).unwrap().then(a.cmp(&b)));
    let mut labels = vec![String::new(); k];
    for (rank, c) in order.into_iter().enumerate() {
        labels[c] = letter(rank);
    }
    labels
}

/// A window too short to embed takes the cluster of the nearest-in-time embedded window, if one
/// lies within [`INHERIT_MAX_SECS`]; otherwise it stays unassigned.
///
/// `embedded` = `(t_start, cluster)` of embedded windows; `short` = `t_start` of short ones.
pub fn inherit_short(embedded: &[(f64, usize)], short: &[f64]) -> Vec<Option<usize>> {
    short
        .iter()
        .map(|t| {
            embedded
                .iter()
                .map(|(ts, c)| ((ts - t).abs(), *c))
                .filter(|(gap, _)| *gap <= INHERIT_MAX_SECS)
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .map(|(_, c)| c)
        })
        .collect()
}

/// One enrolled speaker: id + every voiceprint the owner has confirmed for them.
pub struct Enrolled {
    pub speaker_id: i64,
    pub voiceprints: Vec<Vec<f32>>,
}

/// Match a cluster centroid against the enrolled speakers.
///
/// Score per speaker = **max** cosine over that speaker's voiceprints (many prints per speaker is
/// what survives codec drift). Assigns only when the cluster is big enough, the best score clears
/// [`MATCH_THRESHOLD`], and it beats the runner-up by [`MATCH_MARGIN`] (runner-up = 0 when only
/// one speaker is enrolled). Returns `(speaker_id, score)`.
pub fn auto_match(centroid: &[f32], n_windows: usize, enrolled: &[Enrolled]) -> Option<(i64, f32)> {
    if n_windows < MIN_WINDOWS {
        return None;
    }
    let mut scored: Vec<(f32, i64)> = enrolled
        .iter()
        .filter_map(|e| {
            e.voiceprints
                .iter()
                .map(|p| cosine(centroid, p))
                .fold(None, |m: Option<f32>, s| Some(m.map_or(s, |m| m.max(s))))
                .map(|s| (s, e.speaker_id))
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    let (best, id) = *scored.first()?;
    let second = scored.get(1).map_or(0.0, |s| s.0);
    if best >= MATCH_THRESHOLD && best - second >= MATCH_MARGIN {
        Some((id, best))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit vector along axis `axis` in `dim` dimensions, tilted toward the next axis by `tilt`.
    fn unit(dim: usize, axis: usize, tilt: f32) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[axis] = 1.0;
        v[(axis + 1) % dim] = tilt;
        normalized(&v)
    }

    #[test]
    fn cosine_of_normalized_vectors_is_bounded() {
        let a = unit(4, 0, 0.0);
        let b = unit(4, 1, 0.0);
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn centroid_is_the_renormalised_mean() {
        let c = centroid(&[unit(2, 0, 0.0), unit(2, 1, 0.0)]);
        assert!((c[0] - c[1]).abs() < 1e-6);
        assert!((c.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(centroid::<Vec<f32>>(&[]).is_empty());
    }

    #[test]
    fn blob_roundtrip_and_corruption() {
        let v = vec![0.25f32, -1.5, 3.0e-3];
        assert_eq!(from_blob(&to_blob(&v)).unwrap(), v);
        assert!(from_blob(&[1, 2, 3]).is_err());
    }

    #[test]
    fn three_separated_groups_become_three_clusters() {
        // Three orthogonal directions, each with a few slightly tilted members.
        let mut e = Vec::new();
        for axis in 0..3 {
            for t in [0.0, 0.05, 0.1] {
                e.push(unit(3, axis, t));
            }
        }
        let c = cluster(&e, CLUSTER_CUT);
        assert_eq!(c, vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn the_cut_is_on_cosine_distance() {
        // Two vectors at a chosen cosine: join when 1 − cos < cut, stay apart otherwise.
        let a = vec![1.0f32, 0.0];
        let at = |cos: f32| vec![cos, (1.0 - cos * cos).sqrt()];
        assert_eq!(cluster(&[a.clone(), at(0.56)], CLUSTER_CUT), vec![0, 0], "distance 0.44 joins");
        assert_eq!(cluster(&[a, at(0.54)], CLUSTER_CUT), vec![0, 1], "distance 0.46 does not");
    }

    #[test]
    fn empty_and_singleton_inputs() {
        assert!(cluster::<Vec<f32>>(&[], CLUSTER_CUT).is_empty());
        assert_eq!(cluster(&[unit(2, 0, 0.0)], CLUSTER_CUT), vec![0]);
    }

    #[test]
    fn letters_are_excel_style() {
        assert_eq!(letter(0), "A");
        assert_eq!(letter(25), "Z");
        assert_eq!(letter(26), "AA");
        assert_eq!(letter(27), "AB");
        assert_eq!(letter(52), "BA");
    }

    #[test]
    fn labels_follow_descending_speech_time() {
        // cluster 0: 3 s, cluster 1: 10 s, cluster 2: 3 s (tie with 0 → lower index first)
        let assign = [0, 1, 1, 2];
        let secs = [3.0, 4.0, 6.0, 3.0];
        assert_eq!(label_clusters(&assign, &secs), vec!["B", "A", "C"]);
        assert!(label_clusters(&[], &[]).is_empty());
    }

    #[test]
    fn short_windows_inherit_the_nearest_neighbour_within_the_bound() {
        let embedded = [(10.0, 0), (100.0, 1)];
        // 12 → 10 (2 s); 95 → 100 (5 s); 80 → 100 (20 s); 55 is 45 s from both → none.
        let got = inherit_short(&embedded, &[12.0, 95.0, 80.0, 55.0, 200.0]);
        assert_eq!(got, vec![Some(0), Some(1), Some(1), None, None]);
    }

    fn enrolled(id: i64, prints: &[&Vec<f32>]) -> Enrolled {
        Enrolled { speaker_id: id, voiceprints: prints.iter().map(|p| (*p).clone()).collect() }
    }

    #[test]
    fn auto_match_needs_threshold_margin_and_size_independently() {
        let a = unit(3, 0, 0.0);
        let b = unit(3, 1, 0.0);
        let near_a = normalized(&[0.8, 0.6, 0.0]); // cos to a = 0.8, to b = 0.6

        // Happy path: one speaker, clear match.
        assert_eq!(auto_match(&near_a, MIN_WINDOWS, &[enrolled(1, &[&a])]), Some((1, 0.8)));
        // Too few windows flips it.
        assert_eq!(auto_match(&near_a, MIN_WINDOWS - 1, &[enrolled(1, &[&a])]), None);
        // Below threshold flips it (cos 0.5 < 0.55).
        let far = normalized(&[0.5, (1.0f32 - 0.25).sqrt(), 0.0]);
        assert_eq!(auto_match(&far, MIN_WINDOWS, &[enrolled(1, &[&a])]), None);
        // Margin flips it: two speakers within 0.05 of each other.
        let mid = normalized(&[0.72, 0.69, 0.0]);
        assert_eq!(auto_match(&mid, MIN_WINDOWS, &[enrolled(1, &[&a]), enrolled(2, &[&b])]), None);
        // …but a clear winner among two is assigned.
        assert_eq!(
            auto_match(&near_a, MIN_WINDOWS, &[enrolled(1, &[&a]), enrolled(2, &[&b])]).map(|(id, _)| id),
            Some(1)
        );
        // Max over voiceprints: a bad print does not drag a good one down.
        let bad = unit(3, 2, 0.0);
        assert_eq!(auto_match(&near_a, MIN_WINDOWS, &[enrolled(1, &[&bad, &a])]).map(|(id, _)| id), Some(1));
        // No enrolments → nothing.
        assert_eq!(auto_match(&near_a, MIN_WINDOWS, &[]), None);
    }
}
