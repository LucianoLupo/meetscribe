//! Model-free regressions through the production speaker-command dispatch.
use super::*;
use crate::transcript::{Speaker, TranscriptSegment};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "meetscribe-merge-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&path).unwrap();
    path
}

async fn seed(path: &Path, mode: &str) -> (i64, i64, i64) {
    let mut db = db::Db::open(path).await.unwrap();
    let meta = db::MeetingMeta {
        title: "Release readiness sync".into(), source_dir: "/synthetic/release-sync".into(),
        model: "fixture-whisper-large-v3".into(), lang: "en".into(),
        started_at: 1_762_880_000, duration_secs: 24.0, created_at: 1_762_880_030, capture_status: None,
    };
    let clusters: Vec<db::NewCluster> = ["A", "B", "C", "D"].iter().map(|label| db::NewCluster {
        cluster: (*label).into(), speaker_id: None, assigned_by: None, match_score: None,
        centroid: spk::to_blob(&if *label == "D" { [-0.6, 0.8] } else { [0.8, 0.6] }),
        dim: 2, n_windows: 6, speech_secs: 6.0,
    }).collect();
    let mut segments = Vec::new();
    let mut memberships = Vec::new();
    for (cluster, row) in clusters.iter().enumerate() {
        for window in 0..6 {
            let index = cluster * 6 + window;
            segments.push(TranscriptSegment {
                speaker: Speaker::Others, text: format!("Release checklist item {} is ready", index + 1),
                t_start: index as f64, t_end: index as f64 + 1.0, confidence: 0.93,
            });
            memberships.push(db::SegmentVoice {
                segment: index as i64, cluster, inherited: false,
                embedding: Some(if mode == "corrupt" && cluster == 1 && window == 0 {
                    vec![1, 2, 3]
                } else { row.centroid.clone() }),
            });
        }
    }
    let meeting = db.insert_meeting_with_voices(&meta, &segments, &clusters, &memberships).await.unwrap();
    let speaker = db.add_speaker("Morgan", "Sample", 1_762_880_040).await.unwrap();
    let other = db.add_speaker("Avery", "Fixture", 1_762_880_040).await.unwrap();
    for label in ["A", "B", "C", "D"] {
        let row = db.get_cluster(meeting, label).await.unwrap().unwrap();
        let manual = label == "D"
            || label == "A" && matches!(mode, "manual_auto" | "manual_manual" | "conflict")
            || label == "B" && mode != "manual_auto"
            || label == "C" && mode == "conflict";
        let assigned = label != "C" || mode == "conflict";
        if assigned && !(label == "A" && mode == "unnamed") {
            let person = if label == "D" || label == "C" || label == "A" && mode == "different" {
                other
            } else { speaker };
            db.set_cluster_speaker(row.id, Some(person), Some(if manual { "manual" } else { "auto" }), if manual { None } else { Some(0.95) }).await.unwrap();
            if manual { db.add_voiceprint(person, &row, 1_762_880_050).await.unwrap(); }
        }
    }
    db.close().await.unwrap();
    (meeting, speaker, other)
}

async fn snapshot(path: &Path, meeting: i64) -> String {
    let mut db = db::Db::open(path).await.unwrap();
    let clusters = db.list_clusters(meeting).await.unwrap();
    let mut windows = Vec::new();
    for row in &clusters { windows.push((row.id, db.cluster_windows(row.id).await.unwrap())); }
    let enrolled = db.load_enrolled().await.unwrap();
    let people = db.list_speakers().await.unwrap();
    let segments = db.load_segments(meeting).await.unwrap();
    db.close().await.unwrap();
    format!("{clusters:?}\n{windows:?}\n{enrolled:?}\n{people:?}\n{segments:?}")
}

fn merge(path: &Path, meeting: i64, labels: &[&str]) -> Result<()> {
    let mut args = vec!["merge".into(), meeting.to_string()];
    args.extend(labels.iter().map(|label| (*label).to_string()));
    args.extend(["--db".into(), path.to_str().unwrap().into()]);
    run_speakers(&args)
}

#[test]
fn audit_f4_merge_enrollment() {
    let rt = crate::new_runtime().unwrap();
    for iteration in 0..5 {
        for mode in ["auto_manual", "manual_auto", "manual_manual", "multiple"] {
            let dir = scratch(); let path = dir.join("meeting.db");
            let (meeting, speaker, other) = rt.block_on(seed(&path, mode));
            let raw = rt.block_on(async {
                let mut db = db::Db::open(&path).await.unwrap();
                let text: Vec<String> = db.load_segments(meeting).await.unwrap().into_iter().map(|row| row.seg.text).collect();
                db.close().await.unwrap(); text
            });
            let labels: &[&str] = if mode == "multiple" { &["A", "B", "C"] } else { &["A", "B"] };
            merge(&path, meeting, labels).unwrap();
            rt.block_on(async {
                let mut db = db::Db::open(&path).await.unwrap();
                let keep = db.get_cluster(meeting, "A").await.unwrap().unwrap();
                assert_eq!(keep.speaker_id, Some(speaker));
                assert_eq!(keep.assigned_by.as_deref(), Some("manual"), "{mode}, iteration {iteration}");
                assert_eq!(keep.match_score, None); assert!(!keep.skipped);
                let count = if mode == "multiple" { 18 } else { 12 };
                assert_eq!(keep.n_windows, count); assert_eq!(keep.speech_secs, count as f64);
                assert_eq!(db.cluster_windows(keep.id).await.unwrap().len(), count as usize);
                assert!(db.get_cluster(meeting, "B").await.unwrap().is_none());
                let enrolled = db.load_enrolled().await.unwrap();
                assert_eq!(enrolled.len(), 2);
                assert_eq!(enrolled.iter().find(|row| row.speaker_id == speaker).unwrap().voiceprints, vec![keep.centroid.clone()]);
                assert_eq!(enrolled.iter().find(|row| row.speaker_id == other).unwrap().voiceprints, vec![spk::to_blob(&[-0.6, 0.8])]);
                assert_eq!(db.get_speaker(speaker).await.unwrap().unwrap().voiceprints, 1);
                let decoded = voices::decode_enrolled(&enrolled).unwrap();
                assert_eq!(spk::auto_match(&spk::from_blob(&keep.centroid).unwrap(), keep.n_windows as usize, &decoded).map(|(id, _)| id), Some(speaker));
                assert_eq!(db.load_segments(meeting).await.unwrap().into_iter().map(|row| row.seg.text).collect::<Vec<_>>(), raw);
                db.close().await.unwrap();
            });
            std::fs::remove_dir_all(dir).unwrap();
        }
        for (mode, labels) in [
            ("conflict", vec!["A", "B", "C"]), ("unnamed", vec!["A", "B"]),
            ("different", vec!["A", "B"]), ("duplicate", vec!["A", "B", "B"]),
            ("duplicate", vec!["A", "A"]), ("duplicate", vec!["A", "missing"]),
            ("corrupt", vec!["A", "B"]),
        ] {
            let dir = scratch(); let path = dir.join("meeting.db");
            let (meeting, _, _) = rt.block_on(seed(&path, mode));
            let before = rt.block_on(snapshot(&path, meeting));
            let error = merge(&path, meeting, &labels).unwrap_err();
            if mode == "corrupt" { assert!(format!("{error:#}").contains("embedding blob")); }
            assert_eq!(rt.block_on(snapshot(&path, meeting)), before, "rejected {mode} mutated state");
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f4_merge\",\"scenarios\":11,\"iterations_per_scenario\":5}}");
}
