//! Transcript contract (Phase 2) — the storage/export shape for a meeting.
//!
//! Channel-based speaker labeling: mic → You, system tap → Others (no neural diarizer).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Speaker {
    You,
    Others,
}

impl Speaker {
    pub fn label(self) -> &'static str {
        match self {
            Speaker::You => "You",
            Speaker::Others => "Others",
        }
    }

    /// Lowercase token stored in the `speaker` TEXT column (matches the serde rename).
    pub fn as_sql(self) -> &'static str {
        match self {
            Speaker::You => "you",
            Speaker::Others => "others",
        }
    }

    /// Parse a `speaker` column value back into the enum.
    pub fn from_sql(s: &str) -> Option<Speaker> {
        match s {
            "you" => Some(Speaker::You),
            "others" => Some(Speaker::Others),
            _ => None,
        }
    }
}

/// One transcribed span. Timestamps are seconds from meeting start. This is the
/// contract Phase 3 (sqlx storage) and export will persist.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub speaker: Speaker,
    pub text: String,
    pub t_start: f64,
    pub t_end: f64,
    pub confidence: f32,
}

/// Merge segments from both channels into one time-ordered transcript (stable by t_start).
pub fn merge(mut segs: Vec<TranscriptSegment>) -> Vec<TranscriptSegment> {
    segs.sort_by(|a, b| {
        a.t_start
            .partial_cmp(&b.t_start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    segs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(speaker: Speaker, t_start: f64) -> TranscriptSegment {
        TranscriptSegment { speaker, text: format!("@{t_start}"), t_start, t_end: t_start + 1.0, confidence: 1.0 }
    }

    #[test]
    fn merge_orders_both_channels_by_start() {
        let you = vec![seg(Speaker::You, 0.0), seg(Speaker::You, 5.0)];
        let others = vec![seg(Speaker::Others, 2.0), seg(Speaker::Others, 3.0)];
        let merged = merge([you, others].concat());
        let order: Vec<f64> = merged.iter().map(|s| s.t_start).collect();
        assert_eq!(order, vec![0.0, 2.0, 3.0, 5.0]);
        assert_eq!(merged[1].speaker, Speaker::Others);
    }

    #[test]
    fn speaker_sql_roundtrips() {
        for sp in [Speaker::You, Speaker::Others] {
            assert_eq!(Speaker::from_sql(sp.as_sql()), Some(sp));
        }
        assert_eq!(Speaker::from_sql("nobody"), None);
    }

    #[test]
    fn serde_roundtrips() {
        let s = seg(Speaker::Others, 1.5);
        let j = serde_json::to_string(&s).unwrap();
        assert!(j.contains("\"others\""));
        let back: TranscriptSegment = serde_json::from_str(&j).unwrap();
        assert_eq!(back.speaker, Speaker::Others);
    }
}
