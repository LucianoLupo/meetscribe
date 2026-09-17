//! Presentation layer — the ONE place raw stored segments become display text.
//!
//! The database holds raw ASR output and never rewrites it. Speaker names and vocabulary
//! corrections are applied here, at render time, which is what makes both loops retroactive:
//! label a voice once or add one correction and every past meeting improves on re-export, with
//! no whisper re-run (~18 min/meeting).
//!
//! The load-bearing property, unit-tested below: **`render` is the identity function when the
//! identity map and vocabulary are both empty.** That is what guarantees existing exports stay
//! byte-identical for a user who has adopted neither feature.

use std::collections::HashMap;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::db::{SpeakerRow, StoredSegment, VocabRow};
use crate::transcript::{Speaker, TranscriptSegment};

/// A segment ready for display. Serializes byte-identically to `TranscriptSegment` while the
/// identity fields are empty: same field order, and every addition is `Option` + skipped when
/// `None`. That is what keeps `transcript.json` stable for meetings with no speaker labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedSegment {
    pub speaker: Speaker,
    pub text: String,
    pub t_start: f64,
    pub t_end: f64,
    pub confidence: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_cluster: Option<String>,
}

impl RenderedSegment {
    /// What the Markdown line is labeled with: a resolved name when known, else the channel.
    pub fn display_label(&self) -> &str {
        self.speaker_name
            .as_deref()
            .unwrap_or_else(|| self.speaker.label())
    }
}

/// Resolved speaker names: `names` maps a `speakers.id` to "First Last". `your_name` would rename
/// the mic channel; it is not configurable yet (v1.1), so the mic channel always renders "You".
#[derive(Debug, Clone, Default)]
pub struct IdentityMap {
    your_name: Option<String>,
    names: HashMap<i64, String>,
}

impl IdentityMap {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Names for every person in the database. A far-end segment whose cluster is unnamed (or
    /// skipped) has no `speaker_id` and falls back to the channel label.
    pub fn from_db(speakers: &[SpeakerRow]) -> Self {
        Self {
            your_name: None,
            names: speakers.iter().map(|s| (s.id, s.full_name())).collect(),
        }
    }

    fn resolve(&self, speaker: Speaker, speaker_id: Option<i64>) -> Option<String> {
        match speaker {
            Speaker::You => self.your_name.clone(),
            Speaker::Others => speaker_id.and_then(|id| self.names.get(&id).cloned()),
        }
    }
}

/// One compiled correction rule.
struct Rule {
    id: i64,
    re: Regex,
    replacement: String,
    /// Literal rules must NOT expand `$1`/`$name` in the replacement — see [`Vocab::apply`].
    literal: bool,
}

/// Compiled vocabulary corrections, applied in `id` order.
#[derive(Default)]
pub struct Vocab {
    rules: Vec<Rule>,
}

impl Vocab {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Compile rules in `id` order, skipping (and reporting) any that do not compile.
    ///
    /// A malformed pattern must NOT take down rendering: it was typed once, but it would
    /// otherwise break every export and every daemon-written transcript from then on.
    pub fn compile(rows: &[VocabRow]) -> (Self, Vec<String>) {
        let mut rules = Vec::with_capacity(rows.len());
        let mut warnings = Vec::new();
        for row in rows {
            let source = if row.is_regex {
                row.pattern.clone()
            } else {
                word_bounded(&row.pattern)
            };
            // An empty source compiles fine and matches the zero-width position between EVERY
            // character, so one such row would shred every segment of every transcript. Refuse it
            // here as well as at `vocab add`, so a row that predates that guard cannot do damage.
            if source.is_empty() {
                warnings.push(format!(
                    "vocab rule {} has an empty pattern and was skipped (it would match everywhere)",
                    row.id
                ));
                continue;
            }
            match Regex::new(&source) {
                Ok(re) => rules.push(Rule {
                    id: row.id,
                    re,
                    replacement: row.replacement.clone(),
                    literal: !row.is_regex,
                }),
                Err(e) => warnings.push(format!(
                    "vocab rule {} ('{}') does not compile and was skipped: {e}",
                    row.id, row.pattern
                )),
            }
        }
        (Self { rules }, warnings)
    }

    /// Apply every rule in order. Later rules see earlier rules' output — deliberate, so a rule
    /// can build on another, and the reason ordering is pinned to `id` rather than hash order.
    ///
    /// Literal rules substitute through [`regex::NoExpand`]. The `&str` replacer expands `$1` and
    /// `$name` as capture references, which is right for a `--regex` rule but silently DELETES
    /// text for a literal one: the pattern is escaped by `word_bounded` while the replacement is
    /// not, so a replacement like `$USD` would resolve to a group that does not exist and
    /// substitute nothing.
    #[must_use]
    pub fn apply(&self, text: &str) -> String {
        let mut out = text.to_string();
        for rule in &self.rules {
            out = if rule.literal {
                rule.re
                    .replace_all(&out, regex::NoExpand(rule.replacement.as_str()))
                    .into_owned()
            } else {
                rule.re.replace_all(&out, rule.replacement.as_str()).into_owned()
            };
        }
        out
    }

    /// Rule ids that actually change `text` — powers `vocab test`'s preview.
    #[must_use]
    pub fn matching_rules(&self, text: &str) -> Vec<i64> {
        self.rules
            .iter()
            .filter(|r| r.re.is_match(text))
            .map(|r| r.id)
            .collect()
    }
}

/// Wrap a literal pattern in word boundaries, but only on sides that end in a word character —
/// `\b` next to punctuation asserts the opposite of what a reader expects (`C++` would never
/// match). Escaped first so a literal is never accidentally interpreted as a regex.
fn word_bounded(pattern: &str) -> String {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = String::new();
    if pattern.starts_with(is_word) {
        out.push_str(r"\b");
    }
    out.push_str(&regex::escape(pattern));
    if pattern.ends_with(is_word) {
        out.push_str(r"\b");
    }
    out
}

/// Render stored segments for display.
pub fn render(stored: &[StoredSegment], ids: &IdentityMap, vocab: &Vocab) -> Vec<RenderedSegment> {
    stored
        .iter()
        .map(|s| render_one(&s.seg, s.speaker_id, s.voice_cluster.clone(), ids, vocab))
        .collect()
}

/// Render freshly-transcribed segments that have not been persisted (the pipeline's own path).
pub fn render_fresh(
    segs: &[TranscriptSegment],
    ids: &IdentityMap,
    vocab: &Vocab,
) -> Vec<RenderedSegment> {
    segs.iter()
        .map(|s| render_one(s, None, None, ids, vocab))
        .collect()
}

fn render_one(
    seg: &TranscriptSegment,
    speaker_id: Option<i64>,
    voice_cluster: Option<String>,
    ids: &IdentityMap,
    vocab: &Vocab,
) -> RenderedSegment {
    RenderedSegment {
        speaker: seg.speaker,
        text: vocab.apply(&seg.text),
        t_start: seg.t_start,
        t_end: seg.t_end,
        confidence: seg.confidence,
        speaker_name: ids.resolve(seg.speaker, speaker_id),
        speaker_id,
        voice_cluster,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: i64, pattern: &str, replacement: &str, is_regex: bool) -> VocabRow {
        VocabRow {
            id,
            pattern: pattern.into(),
            replacement: replacement.into(),
            is_regex,
            enabled: true,
            created_at: 0,
        }
    }

    fn seg(text: &str) -> TranscriptSegment {
        TranscriptSegment {
            speaker: Speaker::Others,
            text: text.into(),
            t_start: 1.0,
            t_end: 2.0,
            confidence: 0.9,
        }
    }

    #[test]
    fn empty_vocab_and_identity_is_the_identity_function() {
        let stored = vec![
            StoredSegment::unstored(seg("Postgres pays the bills")),
            StoredSegment::unstored(seg("  ragged  spacing  ")),
        ];
        let out = render(&stored, &IdentityMap::empty(), &Vocab::empty());
        for (r, s) in out.iter().zip(&stored) {
            assert_eq!(r.text, s.seg.text, "text must pass through untouched");
            assert_eq!(r.speaker, s.seg.speaker);
            assert!(r.speaker_name.is_none());
            assert!(r.speaker_id.is_none());
            assert!(r.voice_cluster.is_none());
        }
    }

    #[test]
    fn json_is_byte_identical_to_the_raw_contract_when_unlabelled() {
        let raw = seg("hola");
        let rendered = render_fresh(
            std::slice::from_ref(&raw),
            &IdentityMap::empty(),
            &Vocab::empty(),
        );
        assert_eq!(
            serde_json::to_string_pretty(&[raw]).unwrap(),
            serde_json::to_string_pretty(&rendered).unwrap(),
            "identity fields must be skipped, and field order must match, or every stored \
             meeting's transcript.json changes shape"
        );
    }

    #[test]
    fn literal_patterns_match_on_word_boundaries() {
        let (v, w) = Vocab::compile(&[row(1, "Postgre", "Postgres", false)]);
        assert!(w.is_empty());
        assert_eq!(v.apply("Postgre ships"), "Postgres ships");
        // Not a standalone word → untouched. This is what stops a short personal nickname from
        // being rewritten everywhere it appears inside another token.
        assert_eq!(v.apply("Postgresql"), "Postgresql");
    }

    #[test]
    fn literal_patterns_are_escaped_not_interpreted() {
        let (v, _) = Vocab::compile(&[row(1, "a.c", "X", false)]);
        assert_eq!(v.apply("a.c"), "X");
        assert_eq!(v.apply("abc"), "abc", "the dot must be literal, not any-char");
    }

    #[test]
    fn punctuation_edges_do_not_get_word_boundaries() {
        let (v, _) = Vocab::compile(&[row(1, "C++", "Rust", false)]);
        assert_eq!(v.apply("we use C++ here"), "we use Rust here");
    }

    /// `Regex::replace_all` with a `&str` replacer expands `$name`/`$1` as capture references.
    /// For a LITERAL rule that silently deletes the matched text instead of replacing it.
    #[test]
    fn literal_replacements_do_not_expand_dollar_signs() {
        let (v, _) = Vocab::compile(&[row(1, "dolares", "$USD", false)]);
        assert_eq!(v.apply("cuesta cinco dolares hoy"), "cuesta cinco $USD hoy");

        let (v, _) = Vocab::compile(&[row(1, "precio", "US$5", false)]);
        assert_eq!(v.apply("el precio"), "el US$5");

        let (v, _) = Vocab::compile(&[row(1, "x", "${braced}", false)]);
        assert_eq!(v.apply("x"), "${braced}");
    }

    /// …but a --regex rule still gets capture expansion, which is the point of opting in.
    /// Note `${1}` not `$1`: `$1QL` names a group "1QL", which does not exist and expands to
    /// nothing — the same footgun the literal path is protected from.
    #[test]
    fn regex_rules_keep_capture_expansion() {
        let (v, _) = Vocab::compile(&[row(1, r"(\w+)ql", "${1}QL", true)]);
        assert_eq!(v.apply("postgresql"), "postgresQL");
    }

    /// An empty pattern compiles and matches the zero-width position between every character —
    /// one such row would shred every segment of every transcript.
    #[test]
    fn empty_patterns_are_skipped_not_applied() {
        let (v, warnings) = Vocab::compile(&[row(1, "", "BOOM", false), row(2, "ok", "fine", false)]);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("empty pattern"));
        assert_eq!(v.apply("ok text"), "fine text", "only the valid rule may fire");
        assert_eq!(v.matching_rules("anything"), Vec::<i64>::new());
    }

    #[test]
    fn regex_rules_are_opt_in() {
        let (v, _) = Vocab::compile(&[row(1, r"Postgre\w*s", "Postgres", true)]);
        assert_eq!(v.apply("Postgress and Postgres"), "Postgres and Postgres");
    }

    #[test]
    fn rules_apply_in_id_order_and_compose() {
        let (v, _) = Vocab::compile(&[row(1, "a", "b", false), row(2, "b", "c", false)]);
        assert_eq!(v.apply("a"), "c", "rule 2 must see rule 1's output");
    }

    #[test]
    fn a_bad_pattern_is_skipped_not_fatal() {
        let (v, warnings) =
            Vocab::compile(&[row(1, "(unclosed", "x", true), row(2, "ok", "fine", false)]);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("rule 1"));
        assert_eq!(
            v.matching_rules("ok"),
            vec![2],
            "the good rule must survive a bad neighbour"
        );
        assert_eq!(v.apply("ok"), "fine");
    }

    #[test]
    fn corrections_are_applied_to_rendered_text() {
        let (v, _) = Vocab::compile(&[row(1, "NCP", "MCP", false)]);
        let stored = vec![StoredSegment::unstored(seg("the NCP gateway"))];
        let out = render(&stored, &IdentityMap::empty(), &v);
        assert_eq!(out[0].text, "the MCP gateway");
        assert_eq!(
            stored[0].seg.text, "the NCP gateway",
            "the raw segment must not be mutated"
        );
    }

    #[test]
    fn named_clusters_render_the_full_name_and_unnamed_ones_the_channel() {
        let people = vec![SpeakerRow {
            id: 7,
            first_name: "Ada".into(),
            last_name: "Lovelace".into(),
            created_at: 0,
            voiceprints: 1,
        }];
        let ids = IdentityMap::from_db(&people);
        let mut named = StoredSegment::unstored(seg("hola"));
        named.speaker_id = Some(7);
        named.voice_cluster = Some("A".into());
        let mut unnamed = StoredSegment::unstored(seg("chau"));
        unnamed.voice_cluster = Some("B".into());
        let mut you = StoredSegment::unstored(seg("yo"));
        you.seg.speaker = Speaker::You;

        let out = render(&[named, unnamed, you], &ids, &Vocab::empty());
        assert_eq!(out[0].display_label(), "Ada Lovelace");
        assert_eq!(out[0].speaker_name.as_deref(), Some("Ada Lovelace"));
        assert_eq!(out[0].voice_cluster.as_deref(), Some("A"));
        assert_eq!(out[1].display_label(), "Others", "an unnamed cluster keeps the channel label");
        assert!(out[1].speaker_name.is_none());
        assert_eq!(out[1].voice_cluster.as_deref(), Some("B"), "…but the cluster still shows in JSON");
        assert_eq!(out[2].display_label(), "You", "the mic channel is never renamed in v1");
    }

    #[test]
    fn display_label_falls_back_to_the_channel() {
        let out = render_fresh(&[seg("x")], &IdentityMap::empty(), &Vocab::empty());
        assert_eq!(out[0].display_label(), "Others");
    }

    #[test]
    fn matching_rules_reports_only_rules_that_fire() {
        let (v, _) = Vocab::compile(&[row(1, "alpha", "A", false), row(7, "beta", "B", false)]);
        assert_eq!(v.matching_rules("beta only"), vec![7]);
        assert!(v.matching_rules("nothing here").is_empty());
    }
}
