//! Promotion: at hangup the live stream of a recording becomes its stored transcript.
//!
//! The `transcribe` child calls [`wait_and_load`] when it was started with `--promote-live`. A
//! stream that is provably the whole meeting is turned into transcript segments and stored
//! without running whisper again. Every other outcome is a [`Reject`], and the caller runs the
//! batch pass in the same attempt: a stream that cannot be read is a reason to transcribe, never
//! a failed attempt.
//!
//! Nothing here writes: the stream stays on disk as the reader left it.

use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

use crate::live::{self, StreamEvent, StreamLine};
use crate::recovery::{CaptureStatus, State};
use crate::transcript::{self, Speaker, TranscriptSegment};
use crate::vad;

/// A stream larger than this is not one the reader wrote: an hour of speech is under a megabyte.
const MAX_STREAM_BYTES: u64 = 64 * 1024 * 1024;
/// The stream's own `duration_secs` is rounded to 0.1 s, and a segment may end that far past the
/// committed audio for the same reason.
const DURATION_SLACK_SECS: f64 = 0.1;

/// The reasons and their names in the log, declared once: the enum, [`Reject::as_str`] and the
/// list the log allowlist tests against cannot drift apart.
macro_rules! reject_reasons {
    ($($(#[$doc:meta])* $variant:ident => $name:literal,)+) => {
        /// Why a live stream was not stored. The log carries [`Reject::as_str`], so this list is
        /// closed: `diagnostics.rs` drops an event whose reason is not in it.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub(crate) enum Reject { $($(#[$doc])* $variant,)+ }

        impl Reject {
            const ALL: &'static [Reject] = &[$(Reject::$variant,)+];

            pub(crate) fn as_str(self) -> &'static str {
                match self { $(Reject::$variant => $name,)+ }
            }
        }
    };
}

reject_reasons! {
    /// No stream file for this recording: no reader ran.
    NoStream => "no_stream",
    /// Something is at the stream's path that cannot be read as a regular file.
    Unreadable => "unreadable",
    /// The stream stopped changing before the reader wrote its last line.
    ReaderGone => "reader_gone",
    /// The reader was still writing when the wait ended.
    Timeout => "timeout",
    /// A line that does not parse, a wrong version, a gap in `seq`, or an unfinished last line.
    Torn => "torn",
    KeyMismatch => "key_mismatch",
    /// The reader stopped for another reason than the end of the recording.
    NotEnded => "not_ended",
    DurationMismatch => "duration_mismatch",
    CountMismatch => "count_mismatch",
    /// A complete stream with no text. Kept as a rejection on purpose: a live result with nothing
    /// in it is the one case worth a second opinion, and the batch pass on silence is quick.
    Empty => "empty",
    CapturePartial => "capture_partial",
    InvalidSegment => "invalid_segment",
}

impl Reject {
    /// Every reason's name, in declaration order.
    pub(crate) fn names() -> impl Iterator<Item = &'static str> {
        Self::ALL.iter().map(|reject| reject.as_str())
    }

    /// Whether `reason` is one of the reasons above (the log allowlist's test).
    pub(crate) fn is_known(reason: &str) -> bool {
        Self::names().any(|name| name == reason)
    }
}

/// How long the child waits for the reader to finish the stream.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Wait {
    /// The reader writes a heartbeat every 10 s; a stream unchanged for this long has no reader.
    pub stale: Duration,
    /// The longest wait, however alive the reader looks.
    pub cap: Duration,
    pub poll: Duration,
}

impl Default for Wait {
    fn default() -> Self {
        Self { stale: Duration::from_secs(30), cap: Duration::from_secs(300), poll: Duration::from_millis(250) }
    }
}

#[cfg(test)]
std::thread_local! { static TEST_WAIT: std::cell::Cell<Option<Wait>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
pub(crate) fn test_wait(wait: Wait) { TEST_WAIT.set(Some(wait)); }

impl Wait {
    /// The production limits; a test child may shorten them.
    pub(crate) fn current() -> Self {
        #[cfg(test)]
        if let Some(wait) = TEST_WAIT.get() { return wait; }
        Self::default()
    }
}

/// A stream that passed every check, ready to store.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Promotion {
    /// The model that decoded the stream, as the reader recorded it.
    pub model: String,
    /// The language the reader decoded in. `None` for a stream written before readers recorded it.
    pub lang: Option<String>,
    /// The transcript: the stream's segments joined by [`coalesce`], in transcript order.
    pub segments: Vec<TranscriptSegment>,
    /// Segment lines in the stream, before joining.
    pub stream_segments: u64,
    pub short_windows: u64,
    /// The committed audio, exact (the stream's own figure is rounded).
    pub duration_secs: f64,
    pub wait_ms: u64,
}

/// What [`check`] found in a complete stream.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Checked {
    pub model: String,
    pub lang: Option<String>,
    pub segments: Vec<TranscriptSegment>,
    pub short_windows: u64,
    pub duration_secs: f64,
}

/// Wait for the reader of `state`'s recording to finish `<live_dir>/<key>.jsonl`, then check it.
pub(crate) fn wait_and_load(live_dir: &Path, state: &State, wait: &Wait) -> Result<Promotion, Reject> {
    // What the state alone decides is decided before any wait: after a capture that did not end
    // cleanly a reader can keep its stream alive for the whole cap, and the answer is known now.
    if state.capture_status != CaptureStatus::Complete { return Err(Reject::CapturePartial); }
    let path = live_dir.join(format!("{}.jsonl", state.key));
    let started = Instant::now();
    let bytes = loop {
        let (bytes, modified) = read_stream(&path)?;
        if ends_with_meeting_ended(&bytes) { break bytes; }
        // A clock that moved backwards makes the file look new, which only waits longer.
        if SystemTime::now().duration_since(modified).is_ok_and(|age| age >= wait.stale) { return Err(Reject::ReaderGone); }
        if started.elapsed() >= wait.cap { return Err(Reject::Timeout); }
        std::thread::sleep(wait.poll);
    };
    let checked = check(&bytes, state)?;
    let stream_segments = checked.segments.len() as u64;
    Ok(Promotion {
        model: checked.model,
        lang: checked.lang,
        segments: coalesce(checked.segments, &state.roll_offsets()),
        stream_segments,
        short_windows: checked.short_windows,
        duration_secs: checked.duration_secs,
        wait_ms: started.elapsed().as_millis() as u64,
    })
}

/// The stream's bytes and when it last changed. Only a regular file is read: a symlink or a
/// directory at that path is not something the reader made.
fn read_stream(path: &Path) -> Result<(Vec<u8>, SystemTime), Reject> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Err(Reject::NoStream),
        Err(_) => return Err(Reject::Unreadable),
    };
    if !metadata.file_type().is_file() || metadata.len() > MAX_STREAM_BYTES { return Err(Reject::Unreadable); }
    let modified = metadata.modified().map_err(|_| Reject::Unreadable)?;
    let bytes = std::fs::read(path).map_err(|_| Reject::Unreadable)?;
    Ok((bytes, modified))
}

fn ends_with_meeting_ended(bytes: &[u8]) -> bool {
    let Some(body) = bytes.strip_suffix(b"\n") else { return false; };
    let last = body.rsplit(|byte| *byte == b'\n').next().unwrap_or(body);
    serde_json::from_slice::<StreamLine>(last).is_ok_and(|line| matches!(line.event, StreamEvent::MeetingEnded { .. }))
}

/// The completeness gate: `bytes` is the whole stream of the recording `state` describes, written
/// by a reader that reached the end of the recording. Checks run in the order of the reasons.
pub(crate) fn check(bytes: &[u8], state: &State) -> Result<Checked, Reject> {
    let text = std::str::from_utf8(bytes).map_err(|_| Reject::Torn)?;
    if !text.is_empty() && !text.ends_with('\n') { return Err(Reject::Torn); }
    let mut events = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line: StreamLine = serde_json::from_str(line).map_err(|_| Reject::Torn)?;
        if line.v != 1 || line.seq != index as u64 { return Err(Reject::Torn); }
        events.push(line.event);
    }

    let (model, lang) = match events.first() {
        Some(StreamEvent::MeetingStarted { key, model, lang, .. }) if key == &state.key => (model.clone(), lang.clone()),
        _ => return Err(Reject::KeyMismatch),
    };
    let (duration, declared, short_windows) = match events.last() {
        Some(StreamEvent::MeetingEnded { duration_secs, reason, segments, short_windows }) if reason == "ended" =>
            (*duration_secs, *segments, *short_windows),
        _ => return Err(Reject::NotEnded),
    };
    let committed = live::committed_secs(state);
    if (duration - committed).abs() > DURATION_SLACK_SECS { return Err(Reject::DurationMismatch); }

    let mut segments = Vec::new();
    for event in &events[1..events.len() - 1] {
        match event {
            StreamEvent::Segment { speaker, t_start, t_end, text, confidence, .. } => segments.push(
                TranscriptSegment { speaker: *speaker, text: text.clone(), t_start: *t_start, t_end: *t_end, confidence: *confidence as f32 }),
            StreamEvent::Heartbeat { .. } => {}
            // A second start or end inside the stream is two runs written into one file.
            StreamEvent::MeetingStarted { .. } | StreamEvent::MeetingEnded { .. } => return Err(Reject::Torn),
        }
    }
    if segments.len() as u64 != declared { return Err(Reject::CountMismatch); }
    if segments.is_empty() { return Err(Reject::Empty); }
    if state.capture_status != CaptureStatus::Complete { return Err(Reject::CapturePartial); }
    // Against the exact committed audio, not the stream's rounded duration. JSON numbers are
    // finite; the stored confidence is narrower, and a huge value becomes infinite in the cast.
    let valid = |segment: &TranscriptSegment| {
        0.0 <= segment.t_start && segment.t_start <= segment.t_end
            && segment.t_end <= committed + DURATION_SLACK_SECS
            && segment.confidence.is_finite() && !segment.text.trim().is_empty()
    };
    if !segments.iter().all(valid) { return Err(Reject::InvalidSegment); }
    Ok(Checked { model, lang, segments, short_windows, duration_secs: committed })
}

/// Join a channel's neighbouring segments the way the batch pass joins speech before decoding
/// (`vad.rs`: a gap under 800 ms, at most 30 s), then put both channels in transcript order.
///
/// A joined segment takes the first start and the last end as they are stored, with no
/// arithmetic: `speakers cluster` slices the recording at these times and refuses one that is
/// off a millisecond of its roll. For the same reason two segments in different rolls are never
/// joined, whatever the gap: a roll after a device change has its own millisecond grid.
pub(crate) fn coalesce(segments: Vec<TranscriptSegment>, roll_offsets: &[f64]) -> Vec<TranscriptSegment> {
    struct Joined { segment: TranscriptSegment, parts: u32, secs: f64, weighted: f64, plain: f64 }
    let roll = |time: f64| roll_offsets.partition_point(|offset| *offset <= time);
    // Stored times are a roll's offset plus whole milliseconds, so within a roll a difference
    // rounds to the integer the batch rule compares.
    let ms = |from: f64, to: f64| ((to - from) * 1000.0).round();
    let (gap_ms, max_ms) = (vad::COALESCE_GAP_MS as f64, vad::MAX_WINDOW_MS as f64);
    let mut merged = Vec::new();
    for speaker in [Speaker::You, Speaker::Others] {
        let mut channel: Vec<TranscriptSegment> = segments.iter().filter(|segment| segment.speaker == speaker).cloned().collect();
        channel.sort_by(|a, b| a.t_start.total_cmp(&b.t_start));
        let mut joined: Vec<Joined> = Vec::new();
        for next in channel {
            let secs = (next.t_end - next.t_start).max(0.0);
            let confidence = f64::from(next.confidence);
            if let Some(last) = joined.last_mut()
                && roll(last.segment.t_start) == roll(next.t_start)
                && ms(last.segment.t_end, next.t_start) < gap_ms
                && ms(last.segment.t_start, next.t_end) <= max_ms
            {
                if next.t_end > last.segment.t_end { last.segment.t_end = next.t_end; }
                last.segment.text = format!("{} {}", last.segment.text.trim_end(), next.text.trim_start());
                last.parts += 1;
                last.secs += secs;
                last.weighted += confidence * secs;
                last.plain += confidence;
                continue;
            }
            joined.push(Joined { segment: next, parts: 1, secs, weighted: confidence * secs, plain: confidence });
        }
        merged.extend(joined.into_iter().map(|mut run| {
            // A segment that was not joined keeps its confidence bit for bit.
            if run.parts > 1 {
                let confidence = if run.secs > 0.0 { run.weighted / run.secs } else { run.plain / f64::from(run.parts) };
                run.segment.confidence = confidence as f32;
            }
            (run.segment, ())
        }));
    }
    transcript::merge_keyed(merged).into_iter().map(|(segment, ())| segment).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery;
    use serde_json::json;

    const KEY: &str = "a6e79ba0314c4cc2b647f4d25b327011";

    /// A finished recording's state: one 16 kHz roll per entry of `frames`, 0.25 s apart.
    fn state(frames: &[u64]) -> State {
        let rolls: Vec<serde_json::Value> = frames.iter().enumerate().map(|(index, frames)| json!({
            "index": index, "sample_rate": 16_000, "gap_seconds": if index == 0 { 0.0 } else { 0.25 },
            "mic_frames": frames, "system_frames": frames,
        })).collect();
        serde_json::from_value(json!({
            "version": 1, "key": KEY, "started_at": 1_762_880_000i64, "app": "com.google.Chrome",
            "title": "Release readiness sync", "db_path": "/synthetic/archive.db", "export_dir": "/synthetic/recording",
            "lock_dir": "/synthetic/.locks", "archive_id": null, "phase": "ready", "capture_status": "complete",
            "rolls": rolls, "duration_seconds": 3.0, "attempts": 1, "in_flight": true, "next_retry": 0,
            "error_class": null, "meeting_id": null, "publication": null, "unarchived_manual_output": false,
        })).unwrap()
    }

    fn started() -> serde_json::Value {
        json!({"type": "meeting_started", "key": KEY, "app": "Chrome", "started_at": 1_762_880_000i64, "model": "ggml-fixture"})
    }

    fn segment(speaker: &str, t_start: f64, t_end: f64, text: &str) -> serde_json::Value {
        json!({"type": "segment", "speaker": speaker, "t_start": t_start, "t_end": t_end, "text": text,
            "confidence": 0.9, "lag_ms": 3_400, "decode_ms": 900})
    }

    fn ended(duration: f64, segments: u64) -> serde_json::Value {
        json!({"type": "meeting_ended", "duration_secs": duration, "reason": "ended", "segments": segments, "short_windows": 4})
    }

    /// The events as a stream: `v` and `seq` added, one line each.
    fn stream(events: &[serde_json::Value]) -> Vec<u8> {
        events.iter().enumerate().map(|(seq, event)| {
            let mut line = event.clone();
            line["v"] = json!(1);
            line["seq"] = json!(seq);
            format!("{line}\n")
        }).collect::<String>().into_bytes()
    }

    /// A complete stream of the 3 s recording `state(&[48_000])` describes.
    fn complete() -> Vec<serde_json::Value> {
        vec![started(), segment("you", 0.2, 1.1, "The release checklist is ready"),
            json!({"type": "heartbeat", "audio_secs": 2.0, "decoded_secs": 1.1}),
            segment("others", 1.4, 2.6, "We approved the deployment window"), ended(3.0, 2)]
    }

    fn temp_root(tag: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("meetscribe-promote-{tag}-{}", recovery::new_identity()));
        recovery::private_dir(&root).unwrap();
        root
    }

    fn seg(speaker: Speaker, t_start: f64, t_end: f64, text: &str, confidence: f32) -> TranscriptSegment {
        TranscriptSegment { speaker, text: text.into(), t_start, t_end, confidence }
    }

    #[test]
    fn a_complete_stream_passes_and_carries_the_readers_model_and_the_exact_duration() {
        let state = state(&[48_000]);
        let checked = check(&stream(&complete()), &state).unwrap();
        assert_eq!((checked.model.as_str(), checked.short_windows, checked.duration_secs), ("ggml-fixture", 4, 3.0));
        assert_eq!(checked.lang, None, "a stream written before readers recorded their language");
        let mut with_language = complete();
        with_language[0]["lang"] = json!("pt");
        assert_eq!(check(&stream(&with_language), &state).unwrap().lang.as_deref(), Some("pt"));
        assert_eq!(checked.segments, [seg(Speaker::You, 0.2, 1.1, "The release checklist is ready", 0.9),
            seg(Speaker::Others, 1.4, 2.6, "We approved the deployment window", 0.9)]);
    }

    #[test]
    fn every_way_a_stream_is_incomplete_is_named() {
        let state = state(&[48_000]);
        let with = |change: &dyn Fn(&mut Vec<serde_json::Value>)| { let mut events = complete(); change(&mut events); stream(&events) };
        let cases: Vec<(&str, Vec<u8>, Reject)> = vec![
            ("a line that is not JSON", { let mut bytes = stream(&complete()); bytes.splice(0..0, b"not json\n".iter().copied()); bytes }, Reject::Torn),
            ("an unfinished last line", { let mut bytes = stream(&complete()); bytes.pop(); bytes }, Reject::Torn),
            ("bytes that are not text", vec![0xff, 0xfe, b'\n'], Reject::Torn),
            ("an unknown event", with(&|events| events.insert(1, json!({"type": "note", "text": "x"}))), Reject::Torn),
            ("a segment without its end", with(&|events| { events[1].as_object_mut().unwrap().remove("t_end"); }), Reject::Torn),
            ("a gap in seq", { let mut events = complete(); events.remove(2); let mut bytes = stream(&events);
                bytes = String::from_utf8(bytes).unwrap().replace("\"seq\":3", "\"seq\":4").into_bytes(); bytes }, Reject::Torn),
            ("another format version", String::from_utf8(stream(&complete())).unwrap().replace("\"v\":1", "\"v\":2").into_bytes(), Reject::Torn),
            ("a second start inside", with(&|events| events.insert(2, started())), Reject::Torn),
            ("an empty file", Vec::new(), Reject::KeyMismatch),
            ("another recording's key", with(&|events| events[0]["key"] = json!("b".repeat(32))), Reject::KeyMismatch),
            ("no first line", with(&|events| { events.remove(0); }), Reject::KeyMismatch),
            ("no last line", with(&|events| { events.pop(); }), Reject::NotEnded),
            ("a reader that stalled", with(&|events| events[4]["reason"] = json!("stalled")), Reject::NotEnded),
            ("a shorter stream than the recording", with(&|events| events[4]["duration_secs"] = json!(2.8)), Reject::DurationMismatch),
            ("a longer stream than the recording", with(&|events| events[4]["duration_secs"] = json!(3.2)), Reject::DurationMismatch),
            ("fewer segments than declared", with(&|events| events[4]["segments"] = json!(3)), Reject::CountMismatch),
            ("more segments than declared", with(&|events| events[4]["segments"] = json!(1)), Reject::CountMismatch),
            ("no text at all", stream(&[started(), ended(3.0, 0)]), Reject::Empty),
            ("an end before its start", with(&|events| events[1]["t_end"] = json!(0.1)), Reject::InvalidSegment),
            ("a start before the recording", with(&|events| events[1]["t_start"] = json!(-0.5)), Reject::InvalidSegment),
            ("an end past the recording", with(&|events| events[3]["t_end"] = json!(3.2)), Reject::InvalidSegment),
            ("a segment of spaces", with(&|events| events[3]["text"] = json!("  ")), Reject::InvalidSegment),
            ("a confidence too large to store", with(&|events| events[3]["confidence"] = json!(1e300)), Reject::InvalidSegment),
        ];
        for (name, bytes, reason) in cases {
            assert_eq!(check(&bytes, &state), Err(reason), "{name}");
        }
        let mut partial = state.clone();
        partial.capture_status = CaptureStatus::Partial;
        assert_eq!(check(&stream(&complete()), &partial), Err(Reject::CapturePartial), "a capture recovered after a crash");
        // The names are distinct, and nothing else passes for a reason.
        let names: std::collections::HashSet<&str> = Reject::names().collect();
        assert_eq!(names.len(), Reject::names().count());
        assert!(!Reject::is_known("SYNTHETIC_PRIVATE_SENTENCE") && !Reject::is_known(""));
    }

    /// The stream's own duration is rounded to 0.1 s. A segment end is checked against the exact
    /// committed audio: with 1.049 s recorded the stream says 1.0 and an end at 1.14 is inside
    /// the slack; with 1.051 s recorded the stream says 1.1 and an end at 1.17 is past it. A check
    /// against the rounded figure gets both wrong.
    #[test]
    fn a_segment_end_is_checked_against_the_exact_duration_not_the_rounded_one() {
        let short = |frames: u64, rounded: f64, end: f64| check(&stream(&[started(), segment("you", 0.2, end, "The checklist is ready"), ended(rounded, 1)]), &state(&[frames]));
        assert_eq!(short(16_784, 1.0, 1.14).map(|checked| checked.duration_secs), Ok(1.049));
        assert_eq!(short(16_816, 1.1, 1.17), Err(Reject::InvalidSegment));
    }

    #[test]
    fn neighbours_in_a_channel_are_joined_by_the_batch_rule() {
        let offsets = [0.0];
        let joined = coalesce(vec![
            seg(Speaker::You, 1.0, 2.0, "The release checklist ", 0.8),
            seg(Speaker::You, 2.799, 4.799, " is ready for review", 0.5),   // 799 ms later: joined
            seg(Speaker::You, 5.599, 6.0, "Next item", 0.7),                // 800 ms later: a new segment
            seg(Speaker::Others, 1.5, 3.0, "We approved the window", 0.9),  // another channel never joins
        ], &offsets);
        assert_eq!(joined.len(), 3);
        assert_eq!((joined[0].speaker, joined[0].t_start.to_bits(), joined[0].t_end.to_bits()), (Speaker::You, 1.0f64.to_bits(), 4.799f64.to_bits()),
            "the first start and the last end, as stored");
        assert_eq!(joined[0].text, "The release checklist is ready for review");
        assert!((joined[0].confidence - 0.6).abs() < 1e-6, "weighted by duration: (0.8 * 1 s + 0.5 * 2 s) / 3 s, got {}", joined[0].confidence);
        assert_eq!((joined[1].speaker, joined[1].t_start), (Speaker::Others, 1.5), "transcript order is by start");
        assert_eq!((joined[2].text.as_str(), joined[2].confidence.to_bits()), ("Next item", 0.7f32.to_bits()), "a lone segment is copied");
    }

    #[test]
    fn a_joined_segment_never_exceeds_thirty_seconds() {
        // 12 s windows back to back, as a monologue is cut: two fit in 30 s, the third does not.
        let cut: Vec<TranscriptSegment> = (0..5).map(|index| seg(Speaker::Others, 12.0 * f64::from(index), 12.0 * f64::from(index + 1), "word", 0.9)).collect();
        let joined = coalesce(cut, &[0.0]);
        assert_eq!(joined.iter().map(|segment| (segment.t_start, segment.t_end)).collect::<Vec<_>>(), [(0.0, 24.0), (24.0, 48.0), (48.0, 60.0)]);
        assert_eq!(joined[0].text, "word word");
        // Exactly 30 s is still one segment, as in `vad::coalesce`.
        assert_eq!(coalesce(vec![seg(Speaker::You, 0.0, 15.0, "a", 0.9), seg(Speaker::You, 15.0, 30.0, "b", 0.9)], &[0.0]).len(), 1);
        assert_eq!(coalesce(vec![seg(Speaker::You, 0.0, 15.0, "a", 0.9), seg(Speaker::You, 15.0, 30.001, "b", 0.9)], &[0.0]).len(), 2);
    }

    /// The second roll starts where the first ended plus its gap, on no decimal grid. Two
    /// segments 250 ms apart on either side of that boundary would be joined by the gap rule;
    /// they must stay two, or the joined one would start on one roll's grid and end on another's.
    #[test]
    fn segments_on_either_side_of_a_roll_boundary_stay_two() {
        let state = state(&[47_957, 32_000]);
        let offsets = state.roll_offsets();
        assert_eq!(offsets.len(), 2);
        let before = seg(Speaker::Others, offsets[0] + 2.0, offsets[1] - 0.25, "End of the first device", 0.9);
        let after = seg(Speaker::Others, offsets[1], offsets[1] + 1.0, "Start of the second device", 0.8);
        assert!((after.t_start - before.t_end - 0.25).abs() < 1e-9);
        let joined = coalesce(vec![before.clone(), after.clone()], &offsets);
        assert_eq!(joined, [before.clone(), after.clone()]);
        // The same two segments inside one roll are one segment.
        assert_eq!(coalesce(vec![before, after], &offsets[..1]).len(), 1);
    }

    #[test]
    fn the_wait_ends_on_a_finished_stream_a_dead_reader_or_the_cap() {
        let root = temp_root("wait");
        let state = state(&[48_000]);
        let path = root.join(format!("{KEY}.jsonl"));
        let quick = Wait { stale: Duration::from_secs(30), cap: Duration::from_millis(150), poll: Duration::from_millis(10) };
        assert_eq!(wait_and_load(&root, &state, &quick), Err(Reject::NoStream));

        // A directory, and a symlink to a perfectly good stream, are not the reader's file.
        std::fs::create_dir(&path).unwrap();
        assert_eq!(wait_and_load(&root, &state, &quick), Err(Reject::Unreadable));
        std::fs::remove_dir(&path).unwrap();
        let elsewhere = root.join("elsewhere.jsonl");
        std::fs::write(&elsewhere, stream(&complete())).unwrap();
        std::os::unix::fs::symlink(&elsewhere, &path).unwrap();
        assert_eq!(wait_and_load(&root, &state, &quick), Err(Reject::Unreadable));
        std::fs::remove_file(&path).unwrap();

        // The reader is alive (the file is new) but never finishes: the cap ends the wait.
        let unfinished = stream(&complete()[..4]);
        std::fs::write(&path, &unfinished).unwrap();
        assert_eq!(wait_and_load(&root, &state, &quick), Err(Reject::Timeout));
        // After a capture that did not end cleanly nothing is waited for: the state already says no.
        let mut partial = state.clone();
        partial.capture_status = CaptureStatus::Partial;
        let patient = Wait { stale: Duration::from_secs(30), cap: Duration::from_secs(120), poll: Duration::from_millis(10) };
        let asked = Instant::now();
        assert_eq!(wait_and_load(&root, &partial, &patient), Err(Reject::CapturePartial));
        assert!(asked.elapsed() < Duration::from_secs(5), "the child waited for a stream it could never store");
        // The same stream untouched for 31 s has no reader any more.
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(SystemTime::now() - Duration::from_secs(31)).unwrap();
        assert_eq!(wait_and_load(&root, &state, &Wait::default()), Err(Reject::ReaderGone));

        // The reader finishes while the child waits.
        std::fs::write(&path, &unfinished).unwrap();
        let (finish_at, last) = (path.clone(), stream(&complete()));
        let writer = std::thread::spawn(move || { std::thread::sleep(Duration::from_millis(60)); std::fs::write(finish_at, last).unwrap(); });
        let patient = Wait { stale: Duration::from_secs(30), cap: Duration::from_secs(10), poll: Duration::from_millis(10) };
        let promotion = wait_and_load(&root, &state, &patient).unwrap();
        writer.join().unwrap();
        assert_eq!((promotion.stream_segments, promotion.segments.len(), promotion.short_windows), (2, 2, 4));
        assert_eq!((promotion.model.as_str(), promotion.duration_secs), ("ggml-fixture", 3.0));
        assert!(promotion.wait_ms >= 40, "the wait was measured, got {} ms", promotion.wait_ms);
        // A finished stream is judged whatever its age: an old complete stream is still complete.
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(SystemTime::now() - Duration::from_secs(3_600)).unwrap();
        assert!(wait_and_load(&root, &state, &Wait::default()).is_ok());
        std::fs::remove_dir_all(root).unwrap();
    }

    fn args(list: &[&str]) -> Vec<String> { list.iter().map(|arg| (*arg).to_string()).collect() }

    #[test]
    fn the_cli_parses_the_flag_and_hands_it_to_the_pipeline() {
        let keyed = ["/sessions/recording", "--session-key", KEY, "--db", "/archive.db", "--export-dir", "/sessions/recording"];
        let with_flag = args(&[&keyed[..], &["--promote-live", "/streams"]].concat());
        let parsed = crate::parse_transcribe_args(&with_flag).unwrap().unwrap();
        assert_eq!(parsed.promote_live.as_deref(), Some(Path::new("/streams")));
        let without = crate::parse_transcribe_args(&args(&keyed)).unwrap().unwrap();
        assert_eq!(without.promote_live, None);
        assert_eq!(parsed.opts, without.opts, "the flag travels beside the pipeline options, not inside them");
        // No managed session, no stream to find: refused instead of silently transcribing.
        assert!(crate::parse_transcribe_args(&args(&["capture", "--promote-live", "/streams"])).is_err());
        assert!(crate::parse_transcribe_args(&args(&[&keyed[..], &["--promote-live"]].concat())).is_err(), "a flag without its directory");

        let mut handed = None;
        let stopped = crate::run_transcribe_using(&with_flag, |dir, opts, promote_live| {
            handed = Some((dir.to_path_buf(), opts.session_key.clone(), promote_live.map(Path::to_path_buf)));
            anyhow::bail!("the pipeline itself is not under test")
        });
        assert!(stopped.is_err());
        assert_eq!(handed, Some(("/sessions/recording".into(), Some(KEY.to_string()), Some("/streams".into()))));
        let mut handed = None;
        let _ = crate::run_transcribe_using(&args(&keyed), |_, _, promote_live| { handed = Some(promote_live.map(Path::to_path_buf)); anyhow::bail!("stop") });
        assert_eq!(handed, Some(None), "without the flag the pipeline is told nothing");
    }

    /// A live-stored meeting's Markdown names `meetscribe finalize <id>`. An unknown subcommand
    /// starts a recording, so until the command exists it has to be a known one that fails.
    #[test]
    fn finalize_is_refused_instead_of_falling_through_to_a_capture() {
        let refused = crate::run_subcommand(&args(&["meetscribe", "finalize", "57"])).expect("finalize fell through to the capture path");
        assert!(format!("{:#}", refused.unwrap_err()).contains("not part of this build"));
        // The capture path itself is unchanged: no subcommand, or flags only, is still a capture.
        assert!(crate::run_subcommand(&args(&["meetscribe"])).is_none());
        assert!(crate::run_subcommand(&args(&["meetscribe", "--seconds", "300"])).is_none());
    }

    /// The reader's own stream of a recording with no speech: complete in every respect and
    /// empty. The gate sends it to the batch pass (`Reject::Empty`), by decision.
    #[test]
    fn a_complete_stream_of_silence_written_by_the_reader_is_rejected_as_empty() {
        let root = temp_root("silence");
        let dir = root.join("sessions").join("recording");
        let mut session = recovery::Session::begin(&dir, &root.join("archive.db"), &dir, "com.google.Chrome", "Silence fixture", 1_762_880_000).unwrap();
        session.start_roll(0, 16_000, 0.0).unwrap();
        for channel in ["mic", "system"] {
            let mut wav = hound::WavWriter::create(session.dir.join(format!("{channel}.wav")), hound::WavSpec { channels: 1, sample_rate: 16_000,
                bits_per_sample: 32, sample_format: hound::SampleFormat::Float }).unwrap();
            for _ in 0..48_000 { wav.write_sample(0.0f32).unwrap(); }
            wav.finalize().unwrap();
        }
        session.checkpoint(0, 48_000, 48_000).unwrap();
        session.finish_capture(3.0, 0.1).unwrap();
        let opts = live::LiveOpts { model: root.join("unused.bin"), lang: "es".into(), out_dir: root.join("live"),
            base: root.join("base"), max_window_ms: 12_000, print: false };
        let summary = live::run_live(&session.dir, &opts, |_, _| Ok(("decoded".into(), 0.9))).unwrap();
        assert_eq!((summary.reason, summary.segments), ("ended", 0));
        assert_eq!(wait_and_load(&root.join("live"), &session.state, &Wait::default()), Err(Reject::Empty));
        drop(session);
        std::fs::remove_dir_all(root).unwrap();
    }
}
