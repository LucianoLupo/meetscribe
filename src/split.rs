//! Split-then-name (plans/2026-09-26-nemotron-split-naming.md §3–§4): cut far-end Whisper chunks
//! at diarizer voice changes, give each piece its words, and after naming turn the pieces back
//! into stored segments. Pure functions only — the pipeline owns audio, models and the DB.

use std::collections::HashMap;

use anyhow::Result;

use crate::{asr, db, diar, spk, transcript, voices};

/// Same-local-speaker pieces closer than this merge into one (the evaluation's rule).
const MERGE_GAP_SECS: f64 = 1.0;

/// One piece of a far-end chunk, in meeting time.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    /// Index of the parent chunk in the merged segment list.
    pub chunk: usize,
    pub t_start: f64,
    pub t_end: f64,
    /// Diarizer-local speaker of the roll; `None` = the chunk had no turn coverage (kept whole).
    pub local: Option<u32>,
    pub text: String,
}

impl Piece {
    pub fn secs(&self) -> f64 {
        self.t_end - self.t_start
    }
}

/// Round a roll-relative diarizer time to whole milliseconds, then place it in meeting time.
fn on_ms_grid(offset: f64, t: f64) -> f64 {
    offset + (t * 1000.0).round() / 1000.0
}

/// The pieces of one chunk `[c_start, c_end)` (meeting time) given its roll's turns
/// (roll-relative). Pieces = turn ∩ chunk, in turn order; a piece merges into the previous one
/// when both belong to the same local speaker and the gap is under [`MERGE_GAP_SECS`]. No
/// overlapping turn ⇒ the whole chunk is one piece. Overlapping turns of different speakers give
/// overlapping pieces (both are kept).
pub fn pieces_for_chunk(chunk: usize, c_start: f64, c_end: f64, offset: f64, turns: &[diar::Turn]) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::new();
    for t in turns {
        let (a, b) = (on_ms_grid(offset, t.t_start), on_ms_grid(offset, t.t_end));
        let (s, e) = (a.max(c_start), b.min(c_end));
        if e <= s {
            continue;
        }
        match out.last_mut() {
            Some(last) if last.local == Some(t.speaker) && s - last.t_end < MERGE_GAP_SECS => {
                last.t_end = last.t_end.max(e);
            }
            _ => out.push(Piece { chunk, t_start: s, t_end: e, local: Some(t.speaker), text: String::new() }),
        }
    }
    if out.is_empty() {
        out.push(Piece { chunk, t_start: c_start, t_end: c_end, local: None, text: String::new() });
    }
    out
}

/// Give every word of the chunk to exactly one piece and set each piece's text. `words` are
/// relative to the chunk start. A word goes to the piece holding its midpoint; a midpoint in a
/// gap → the nearest piece; a midpoint in two overlapping pieces → the piece covering more of the
/// word, tie → the earlier piece.
pub fn assign_words(pieces: &mut [Piece], c_start: f64, words: &[asr::Word]) {
    if pieces.is_empty() {
        return;
    }
    let mut texts: Vec<Vec<&str>> = vec![Vec::new(); pieces.len()];
    for w in words {
        let (a, b) = (c_start + w.t_start, c_start + w.t_end);
        let mid = (a + b) / 2.0;
        let hits: Vec<usize> = (0..pieces.len()).filter(|&i| pieces[i].t_start <= mid && mid < pieces[i].t_end).collect();
        let k = match hits.as_slice() {
            [] => (0..pieces.len())
                .min_by(|&i, &j| {
                    let d = |p: &Piece| (p.t_start - mid).abs().min((p.t_end - mid).abs());
                    d(&pieces[i]).total_cmp(&d(&pieces[j])).then(i.cmp(&j))
                })
                .expect("non-empty"),
            [one] => *one,
            many => {
                let cover = |p: &Piece| (b.min(p.t_end) - a.max(p.t_start)).max(0.0);
                *many
                    .iter()
                    .max_by(|&&i, &&j| cover(&pieces[i]).total_cmp(&cover(&pieces[j])).then(j.cmp(&i)))
                    .expect("non-empty")
            }
        };
        texts[k].push(w.text.as_str());
    }
    for (p, t) in pieces.iter_mut().zip(texts) {
        p.text = t.join(" ");
    }
}

/// Turn named pieces back into stored segments (§4). Returns the final segment list (both
/// channels, ordered by start) and the assembly to insert, keyed by final segment index.
///
/// - `merged`: today's segments (mic + far-end chunks), `chunk_embeddings[i]` = chunk i's path-A
///   embedding.
/// - `pieces` + `b`: path B after [`voices::rule_c`]; `b.voices[..].segment` = piece index.
///
/// A chunk whose pieces all resolve to one cluster (or none) is stored exactly as today (one
/// segment, today's embedding). Otherwise adjacent pieces with the same cluster are re-joined;
/// a run takes the embedding of its longest embedded piece (nothing is re-embedded), and a run
/// without one is stored inherited. Word-less runs are dropped. Clusters left with no embedded
/// row are dropped; the rest get stats recomputed from what is stored and letters by speech time.
pub fn finalize(
    merged: &[transcript::TranscriptSegment],
    chunk_embeddings: &[Option<Vec<f32>>],
    pieces: &[Piece],
    b: &voices::Assembly,
) -> Result<(Vec<transcript::TranscriptSegment>, voices::Assembly)> {
    let member: HashMap<usize, &db::SegmentVoice> = b.voices.iter().map(|v| (v.segment as usize, v)).collect();
    let mut by_chunk: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, p) in pieces.iter().enumerate() {
        by_chunk.entry(p.chunk).or_default().push(i);
    }

    // (segment, Option<(cluster, inherited, embedding)>)
    type Row = (transcript::TranscriptSegment, Option<(usize, bool, Option<Vec<u8>>)>);
    let mut rows: Vec<Row> = Vec::new();
    for (ci, seg) in merged.iter().enumerate() {
        let Some(idx) = by_chunk.get(&ci) else {
            rows.push((seg.clone(), None));
            continue;
        };
        let mut idx = idx.clone();
        idx.sort_by(|&i, &j| pieces[i].t_start.total_cmp(&pieces[j].t_start).then(i.cmp(&j)));
        let resolved: Vec<Option<usize>> = idx.iter().map(|i| member.get(i).map(|v| v.cluster)).collect();
        if resolved.iter().all(|c| *c == resolved[0]) {
            let voice = resolved[0].map(|c| {
                let emb = chunk_embeddings.get(ci).cloned().flatten();
                (c, emb.is_none(), emb.as_deref().map(spk::to_blob))
            });
            rows.push((seg.clone(), voice));
            continue;
        }
        let mut start = 0;
        while start < idx.len() {
            let mut end = start + 1;
            while end < idx.len() && resolved[end] == resolved[start] {
                end += 1;
            }
            let run = &idx[start..end];
            let text = run.iter().map(|&i| pieces[i].text.as_str()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join(" ");
            if !text.is_empty() {
                let voice = resolved[start].map(|c| {
                    let longest = run
                        .iter()
                        .filter_map(|&i| member.get(&i).filter(|v| !v.inherited).and_then(|v| v.embedding.clone()).map(|e| (pieces[i].secs(), e)))
                        .max_by(|x, y| x.0.total_cmp(&y.0));
                    match longest {
                        Some((_, e)) => (c, false, Some(e)),
                        None => (c, true, None),
                    }
                });
                let t_start = run.iter().map(|&i| pieces[i].t_start).fold(f64::INFINITY, f64::min);
                let t_end = run.iter().map(|&i| pieces[i].t_end).fold(f64::NEG_INFINITY, f64::max);
                rows.push((transcript::TranscriptSegment { speaker: seg.speaker, text, t_start, t_end, confidence: seg.confidence }, voice));
            }
            start = end;
        }
    }
    let rows = transcript::merge_keyed(rows);

    // Keep clusters that still hold an embedded row; re-index them.
    let mut kept: Vec<usize> = rows
        .iter()
        .filter_map(|(_, v)| v.as_ref().filter(|(_, inh, e)| !inh && e.is_some()).map(|(c, _, _)| *c))
        .collect();
    kept.sort_unstable();
    kept.dedup();
    let remap: HashMap<usize, usize> = kept.iter().enumerate().map(|(new, &old)| (old, new)).collect();

    let mut segments = Vec::with_capacity(rows.len());
    let mut out_voices = Vec::new();
    let mut windows: Vec<Vec<db::WindowRow>> = vec![Vec::new(); kept.len()];
    for (i, (seg, voice)) in rows.into_iter().enumerate() {
        if let Some((c, inherited, embedding)) = voice {
            if let Some(&nc) = remap.get(&c) {
                windows[nc].push(db::WindowRow { segment_id: i as i64, t_start: seg.t_start, t_end: seg.t_end, inherited, embedding: embedding.clone() });
                out_voices.push(db::SegmentVoice { segment: i as i64, cluster: nc, inherited, embedding });
            }
        }
        segments.push(seg);
    }

    let mut clusters = Vec::with_capacity(kept.len());
    let mut centroids = Vec::with_capacity(kept.len());
    for (nc, &old) in kept.iter().enumerate() {
        let (centroid, n, secs) = voices::stats_from(&windows[nc])?;
        let src = &b.clusters[old];
        clusters.push(db::NewCluster {
            cluster: String::new(),
            speaker_id: src.speaker_id,
            assigned_by: src.assigned_by.clone(),
            match_score: src.match_score,
            centroid: spk::to_blob(&centroid),
            dim: centroid.len() as i64,
            n_windows: n,
            speech_secs: secs,
        });
        centroids.push(centroid);
    }
    let order: Vec<usize> = (0..clusters.len()).collect();
    let secs: Vec<f64> = clusters.iter().map(|c| c.speech_secs).collect();
    for (c, label) in spk::label_clusters(&order, &secs).into_iter().enumerate() {
        clusters[c].cluster = label;
    }
    Ok((segments, voices::Assembly::from_parts(clusters, out_voices, centroids)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcript::{Speaker, TranscriptSegment};

    fn turn(a: f64, b: f64, s: u32) -> diar::Turn {
        diar::Turn { t_start: a, t_end: b, speaker: s }
    }
    fn word(t: &str, a: f64, b: f64) -> asr::Word {
        asr::Word { text: t.into(), t_start: a, t_end: b }
    }
    fn seg(speaker: Speaker, text: &str, a: f64, b: f64) -> TranscriptSegment {
        TranscriptSegment { speaker, text: text.into(), t_start: a, t_end: b, confidence: 0.9 }
    }

    #[test]
    fn pieces_clip_to_the_chunk_merge_same_speaker_and_keep_uncovered_chunks_whole() {
        // Roll offset 100 s; chunk 102–110 s.
        let turns = [turn(1.0, 4.0, 0), turn(4.5, 5.0, 0), turn(5.0, 8.0, 1), turn(9.0, 20.0, 0)];
        let p = pieces_for_chunk(7, 102.0, 110.0, 100.0, &turns);
        let spans: Vec<(f64, f64, Option<u32>)> = p.iter().map(|p| (p.t_start, p.t_end, p.local)).collect();
        assert_eq!(spans, vec![(102.0, 105.0, Some(0)), (105.0, 108.0, Some(1)), (109.0, 110.0, Some(0))]);
        assert!(p.iter().all(|p| p.chunk == 7));
        let whole = pieces_for_chunk(0, 50.0, 55.0, 0.0, &turns);
        assert_eq!((whole[0].t_start, whole[0].t_end, whole[0].local), (50.0, 55.0, None));
        // Diarizer times land on the millisecond grid.
        let g = pieces_for_chunk(0, 0.0, 10.0, 0.0, &[turn(1.23456, 2.0, 0)]);
        assert_eq!(g[0].t_start, 1.235);
    }

    #[test]
    fn every_word_goes_to_exactly_one_piece() {
        let mut p = pieces_for_chunk(0, 10.0, 20.0, 0.0, &[turn(10.0, 14.0, 0), turn(13.5, 17.0, 1), turn(18.0, 20.0, 0)]);
        let words = [
            word("uno", 0.5, 1.0),   // 10.75 → piece 0
            word("dos", 3.4, 4.2),   // mid 13.8 in pieces 0 and 1; covers 0.6 of 0, 0.7 of 1 → 1
            word("tres", 7.2, 7.4),  // 17.3 in the gap → nearest = piece 1 (0.3 away)
            word("cuatro", 9.0, 9.5),
        ];
        assign_words(&mut p, 10.0, &words);
        assert_eq!(p.iter().map(|p| p.text.as_str()).collect::<Vec<_>>(), ["uno", "dos tres", "cuatro"]);
        let all: Vec<&str> = p.iter().flat_map(|p| p.text.split(' ')).collect();
        assert_eq!(all, ["uno", "dos", "tres", "cuatro"]);
    }

    fn cluster(speaker: Option<i64>, score: Option<f64>) -> db::NewCluster {
        db::NewCluster {
            cluster: "?".into(),
            speaker_id: speaker,
            assigned_by: speaker.map(|_| "auto".into()),
            match_score: score,
            centroid: Vec::new(),
            dim: 0,
            n_windows: 5,
            speech_secs: 10.0,
        }
    }
    fn voice(segment: i64, cluster: usize, emb: Option<&[f32]>) -> db::SegmentVoice {
        db::SegmentVoice { segment, cluster, inherited: emb.is_none(), embedding: emb.map(spk::to_blob) }
    }
    fn piece(chunk: usize, a: f64, b: f64, text: &str) -> Piece {
        Piece { chunk, t_start: a, t_end: b, local: Some(0), text: text.into() }
    }

    #[test]
    fn rule_c_moves_short_pieces_to_the_chunk_speaker_and_is_identity_without_matches() {
        let a = voices::Assembly::from_parts(vec![cluster(Some(7), Some(0.8))], vec![voice(0, 0, Some(&[1.0, 0.0]))], vec![]);
        let mk = |s0: Option<i64>| {
            voices::Assembly::from_parts(
                vec![cluster(s0, Some(0.8)), cluster(None, None)],
                vec![voice(0, 0, Some(&[1.0, 0.0])), voice(1, 1, None), voice(2, 1, None)],
                vec![],
            )
        };
        // p0 long in c0 (speaker 7); p1 short in c1, parent chunk 0 is speaker 7 on path A;
        // p2 short, parent chunk 1 has no path-A voice → untouched.
        let (parent, embedded) = ([0, 0, 1], [true, false, false]);
        let mut b = mk(Some(7));
        voices::rule_c(&a, &mut b, &parent, &embedded);
        let at = |b: &voices::Assembly, p: i64| b.voices.iter().find(|v| v.segment == p).map(|v| (v.cluster, v.inherited));
        assert_eq!(at(&b, 1), Some((0, true)));
        assert_eq!(at(&b, 2), Some((1, true)));
        // Nothing matched on path B → identity.
        let mut b = mk(None);
        voices::rule_c(&a, &mut b, &parent, &embedded);
        assert_eq!(at(&b, 1), Some((1, true)));
    }

    #[test]
    fn rule_c_prefers_the_cluster_holding_a_long_piece_of_the_same_chunk() {
        let a = voices::Assembly::from_parts(vec![cluster(Some(7), Some(0.8))], vec![voice(0, 0, Some(&[1.0]))], vec![]);
        let mut b = voices::Assembly::from_parts(
            vec![cluster(Some(7), Some(0.95)), cluster(Some(7), Some(0.75))],
            vec![voice(0, 1, Some(&[1.0])), voice(1, 0, None)],
            vec![],
        );
        voices::rule_c(&a, &mut b, &[0, 0], &[true, false]);
        assert_eq!(b.voices.iter().find(|v| v.segment == 1).map(|v| v.cluster), Some(1));
    }

    /// Collapse, re-join, word-less drop, cluster drop, renumbering and recomputed stats in one
    /// meeting: every voice row points at the segment that holds its piece, and every stored
    /// cluster row equals `stats_from` over the rows stored for it.
    #[test]
    fn finalize_rejoins_renumbers_and_recomputes_cluster_stats() {
        let merged = vec![
            seg(Speaker::You, "hola", 0.0, 1.0),
            seg(Speaker::Others, "a b c", 1.0, 10.0),
            seg(Speaker::Others, "x", 12.0, 14.0),
            seg(Speaker::Others, "y", 20.0, 21.0),
        ];
        let (e1, e2, ep0) = (vec![0.0f32, 1.0], vec![0.6f32, 0.8], vec![1.0f32, 0.0]);
        let chunk_embs = vec![None, Some(e1), Some(e2.clone()), None];
        let pieces = vec![
            piece(1, 1.0, 5.0, "a b"),  // p0 long, c0
            piece(1, 5.0, 6.0, "c"),    // p1 short, c1
            piece(2, 12.0, 14.0, "x"),  // p2 long, c1 (single piece → chunk collapses)
            piece(1, 9.0, 10.0, ""),    // p3 short, word-less, c0 → dropped
            piece(3, 20.0, 21.0, "y"),  // p4 short, c2 (only inherited rows → cluster dropped)
        ];
        let b = voices::Assembly::from_parts(
            vec![cluster(Some(3), Some(0.9)), cluster(None, None), cluster(Some(5), Some(0.72))],
            vec![voice(0, 0, Some(&ep0)), voice(1, 1, None), voice(2, 1, Some(&[0.6, 0.8])), voice(3, 0, None), voice(4, 2, None)],
            vec![],
        );
        let (segs, asm) = finalize(&merged, &chunk_embs, &pieces, &b).unwrap();
        let view: Vec<(&str, f64, f64)> = segs.iter().map(|s| (s.text.as_str(), s.t_start, s.t_end)).collect();
        assert_eq!(view, [("hola", 0.0, 1.0), ("a b", 1.0, 5.0), ("c", 5.0, 6.0), ("x", 12.0, 14.0), ("y", 20.0, 21.0)]);
        assert_eq!(asm.clusters.len(), 2, "the inherited-only cluster is dropped");
        let row = |seg: i64| asm.voices.iter().find(|v| v.segment == seg).map(|v| (v.cluster, v.inherited));
        assert_eq!(row(0), None);
        assert_eq!(row(4), None, "voice of a dropped cluster goes too");
        let (ca, cb) = (row(1).unwrap().0, row(3).unwrap().0);
        assert_ne!(ca, cb);
        assert_eq!(row(1), Some((ca, false)));
        assert_eq!(row(2), Some((cb, true)));
        assert_eq!(row(3), Some((cb, false)));
        assert_eq!(asm.clusters[ca].speaker_id, Some(3));
        assert_eq!(asm.clusters[ca].cluster, "A", "most speech = A");
        // The collapsed chunk stores today's (chunk) embedding.
        let v3 = asm.voices.iter().find(|v| v.segment == 3).unwrap();
        assert_eq!(v3.embedding.as_deref(), Some(spk::to_blob(&e2).as_slice()));
        // Stored stats == stats_from(stored rows), per cluster.
        for (c, cl) in asm.clusters.iter().enumerate() {
            let rows: Vec<db::WindowRow> = asm
                .voices
                .iter()
                .filter(|v| v.cluster == c)
                .map(|v| db::WindowRow {
                    segment_id: v.segment,
                    t_start: segs[v.segment as usize].t_start,
                    t_end: segs[v.segment as usize].t_end,
                    inherited: v.inherited,
                    embedding: v.embedding.clone(),
                })
                .collect();
            let (centroid, n, secs) = voices::stats_from(&rows).unwrap();
            assert_eq!((cl.n_windows, cl.speech_secs, cl.centroid.clone()), (n, secs, spk::to_blob(&centroid)));
        }
    }

    /// Piece times stay on the millisecond grid `slice_far_end` (`speakers play`) requires.
    #[test]
    fn slice_far_end_accepts_piece_times() {
        let dir = std::env::temp_dir().join(format!("meetscribe-split-slice-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spec = hound::WavSpec { channels: 1, sample_rate: 16_000, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
        let mut w = hound::WavWriter::create(dir.join("system.wav"), spec).unwrap();
        for i in 0..48_000 {
            w.write_sample((i as f32 * 0.01).sin() * 0.1).unwrap();
        }
        w.finalize().unwrap();
        let p = pieces_for_chunk(0, 0.25, 2.5, 0.0, &[turn(0.123456, 1.987654, 0), turn(1.987654, 2.9, 1)]);
        let ranges: Vec<(f64, f64)> = p.iter().map(|p| (p.t_start, p.t_end)).collect();
        let out = voices::slice_far_end(&dir, &ranges).unwrap();
        assert!(out.iter().all(Option::is_some));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// No pieces ⇒ exactly today's segments and no voices (the `None` turns / split-off shape).
    #[test]
    fn finalize_without_pieces_is_today() {
        let merged = vec![seg(Speaker::You, "hola", 0.0, 1.0), seg(Speaker::Others, "qué tal", 1.0, 3.0)];
        let (out, a) = finalize(&merged, &[None, None], &[], &voices::Assembly::default()).unwrap();
        assert_eq!(out, merged);
        assert!(a.is_empty() && a.voices.is_empty());
    }
}
