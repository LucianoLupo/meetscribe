//! The live lane: a provisional transcript a few seconds behind the speaker.
//!
//! `meetscribe live <session-dir>` follows a recording while it grows. It reads only what the
//! capture loop has checkpointed (`session-state.json` plus the flushed WAV prefix), cuts each
//! channel into speech windows as they close, runs whisper on each window and appends the text
//! to a JSON-lines stream outside the session directory. `meetscribe live tail` prints that
//! stream.
//!
//! It is a reader: it takes no session lock, needs no audio permission and writes nothing into
//! the session directory. The transcript stored at hangup is produced by the batch pipeline as
//! before and stays the stored truth; the two can word a sentence differently.
//!
//! The stream holds meeting text, so it is a private file (directory 0700, file 0600). The log
//! carries numbers only (`diagnostics.rs` allowlists the events and fields).

use std::collections::HashMap;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use silero::{VadConfig, VadSession, VadTransition};

use crate::recovery::{Phase, State};
use crate::transcript::Speaker;
use crate::{asr, detect, export, recovery, resample, status};

/// How often the reader looks for a new checkpoint.
const POLL: Duration = Duration::from_millis(500);
/// The VAD is fed at most this much at a time, so a forced cut is considered every second.
const FEED_SAMPLES: usize = resample::TARGET_RATE as usize;
/// At most this much audio per channel is read in one pass, so catching up on a long recording
/// keeps memory bounded and the two channels close to each other in time.
const MAX_READ_SECS: u64 = 30;
/// whisper returns nothing below ~100 mel frames (see `asr.rs`); such windows are counted only.
const WHISPER_FLOOR_SAMPLES: usize = 16_040;
/// A recording checkpoints every 2 s. With no checkpoint for this long and no daemon recording,
/// the writer is gone and the reader stops.
const STALL: Duration = Duration::from_secs(30);
const HEARTBEAT: Duration = Duration::from_secs(10);
const LAG_LOG: Duration = Duration::from_secs(60);
/// While nobody speaks, the VAD keeps this much audio and the rest is dropped every
/// [`TRIM_EVERY_MS`], or its buffer would grow for the whole of a long silence.
const SILENCE_KEEP_MS: u64 = 3_000;
const TRIM_EVERY_MS: u64 = 30_000;
const SAMPLES_PER_MS: usize = resample::TARGET_RATE as usize / 1000;

#[derive(Debug, Clone)]
pub(crate) struct LiveOpts {
    pub model: PathBuf,
    pub lang: String,
    /// Where `<session key>.jsonl` is written.
    pub out_dir: PathBuf,
    /// The daemon's base directory (`status.json` tells whether a call is being recorded).
    pub base: PathBuf,
    /// A speaker who does not pause is cut into windows of about this length.
    pub max_window_ms: usize,
    /// Also print each line to the terminal as the viewer would.
    pub print: bool,
}

/// What one run did, for the caller and the tests. No text.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LiveSummary {
    pub segments: u64,
    pub short_windows: u64,
    pub audio_secs: f64,
    pub reason: &'static str,
    pub stream: PathBuf,
}

/// The live streams' directory under the daemon's base directory.
fn default_out_dir(base: &Path) -> PathBuf {
    base.join("live")
}

/// A closed speech window on one channel, with times on that roll's 16 kHz timeline.
#[derive(Debug, Clone, PartialEq)]
struct Window {
    start_ms: usize,
    end_ms: usize,
    audio: Vec<f32>,
}

/// Turns the VAD's transitions into windows, cutting a long monologue into pieces. It holds no
/// audio: the VAD session does.
#[derive(Debug, Default)]
struct Cutter {
    /// Start of the part of the current speech that has not been emitted yet.
    open_from_ms: Option<usize>,
}

impl Cutter {
    fn on_transition(&mut self, transition: VadTransition) -> Option<Window> {
        match transition {
            VadTransition::SpeechStart { timestamp_ms } => {
                self.open_from_ms = Some(timestamp_ms);
                None
            }
            VadTransition::SpeechEnd { start_timestamp_ms, end_timestamp_ms, samples } => {
                // After a forced cut only the rest of the speech is still owed.
                let from_ms = self.open_from_ms.take().unwrap_or(start_timestamp_ms).max(start_timestamp_ms);
                let skip = ((from_ms - start_timestamp_ms) * SAMPLES_PER_MS).min(samples.len());
                (end_timestamp_ms > from_ms && skip < samples.len())
                    .then(|| Window { start_ms: from_ms, end_ms: end_timestamp_ms, audio: samples[skip..].to_vec() })
            }
        }
    }

    /// The start of a window to force out now, when speech has run for `max_ms` without a pause.
    fn forced_cut(&self, now_ms: usize, max_ms: usize) -> Option<usize> {
        self.open_from_ms.filter(|from_ms| now_ms.saturating_sub(*from_ms) >= max_ms)
    }
}

/// One capture channel being followed: where it is in the recording, and its streaming state.
struct Stream {
    channel: &'static str,
    speaker: Speaker,
    /// Index of the roll being read, and how many of its native-rate frames were read.
    roll: usize,
    consumed: u64,
    resampler: Option<resample::Streamer>,
    vad: VadSession,
    cutter: Cutter,
    trimmed_at_ms: u64,
}

/// A window ready to decode, placed on the meeting's timeline.
struct Pending {
    speaker: Speaker,
    t_start: f64,
    t_end: f64,
    audio: Vec<f32>,
}

impl Stream {
    fn new(channel: &'static str, speaker: Speaker) -> Result<Self> {
        Ok(Self {
            channel, speaker, roll: 0, consumed: 0, resampler: None,
            vad: VadSession::new(VadConfig::default()).context("init silero VAD")?,
            cutter: Cutter::default(), trimmed_at_ms: 0,
        })
    }

    fn finished(&self, state: &State) -> bool {
        state.phase != Phase::Recording && self.roll >= state.rolls.len()
    }

    /// Audio seconds of this channel that have been read, on the meeting's timeline.
    fn audio_secs(&self, state: &State) -> f64 {
        let offsets = state.roll_offsets();
        match state.rolls.get(self.roll) {
            Some(roll) => offsets[self.roll] + self.consumed as f64 / f64::from(roll.sample_rate),
            None => committed_secs(state),
        }
    }

    /// Read what the checkpoint allows (at most [`MAX_READ_SECS`]) and collect the windows that
    /// close. Returns whether anything was read or a roll was closed.
    fn advance(&mut self, dir: &Path, state: &State, max_window_ms: usize, out: &mut Vec<Pending>) -> Result<bool> {
        let Some(roll) = state.rolls.get(self.roll) else { return Ok(false); };
        let offset = state.roll_offsets()[self.roll];
        let committed = if self.channel == "mic" { roll.mic_frames } else { roll.system_frames };
        let to = committed.min(self.consumed + u64::from(roll.sample_rate) * MAX_READ_SECS);
        let mut progressed = false;
        if to > self.consumed {
            let path = dir.join(recovery::roll_name(self.channel, roll.index));
            let native = recovery::read_committed_range(&path, roll.sample_rate, committed, self.consumed, to)?;
            self.consumed = to;
            let audio = match self.resampler.as_mut() {
                Some(resampler) => resampler.push(&native)?,
                None => self.resampler.insert(resample::Streamer::new(roll.sample_rate)?).push(&native)?,
            };
            self.feed(&audio, offset, max_window_ms, out)?;
            progressed = true;
        }
        // A roll is complete once a later roll exists or the recording has ended.
        let closed = self.roll + 1 < state.rolls.len() || state.phase != Phase::Recording;
        if closed && self.consumed == committed {
            if let Some(mut resampler) = self.resampler.take() {
                let tail = resampler.finish()?;
                self.feed(&tail, offset, max_window_ms, out)?;
            }
            if let Some(from_ms) = self.cutter.open_from_ms.take() {
                let audio = self.vad.get_speech(from_ms, None).to_vec();
                let end_ms = from_ms + audio.len() / SAMPLES_PER_MS;
                self.push(Window { start_ms: from_ms, end_ms, audio }, offset, out);
            }
            // The next roll starts its own timeline (and possibly another sample rate).
            self.vad = VadSession::new(VadConfig::default()).context("init silero VAD")?;
            self.trimmed_at_ms = 0;
            self.roll += 1;
            self.consumed = 0;
            progressed = true;
        }
        Ok(progressed)
    }

    fn feed(&mut self, audio: &[f32], offset: f64, max_window_ms: usize, out: &mut Vec<Pending>) -> Result<()> {
        for chunk in audio.chunks(FEED_SAMPLES) {
            for transition in self.vad.process(chunk).context("vad.process")? {
                if let Some(window) = self.cutter.on_transition(transition) {
                    self.push(window, offset, out);
                }
            }
            let now_ms = self.vad.processed_duration().as_millis() as usize;
            if let Some(from_ms) = self.cutter.forced_cut(now_ms, max_window_ms) {
                let audio = self.vad.get_speech(from_ms, Some(now_ms)).to_vec();
                self.cutter.open_from_ms = Some(now_ms);
                self.push(Window { start_ms: from_ms, end_ms: now_ms, audio }, offset, out);
            }
            // Dropping audio during speech would invalidate the start the VAD remembers, so the
            // buffer is trimmed only while nobody speaks, and never up to the present.
            let session_ms = self.vad.session_time().as_millis() as u64;
            if self.cutter.open_from_ms.is_none() && !self.vad.is_speaking()
                && session_ms.saturating_sub(self.trimmed_at_ms) >= TRIM_EVERY_MS {
                let _ = self.vad.take_until(Duration::from_millis(session_ms - SILENCE_KEEP_MS));
                self.trimmed_at_ms = session_ms;
            }
        }
        Ok(())
    }

    fn push(&self, window: Window, offset: f64, out: &mut Vec<Pending>) {
        out.push(Pending {
            speaker: self.speaker,
            t_start: offset + window.start_ms as f64 / 1000.0,
            t_end: offset + window.end_ms as f64 / 1000.0,
            audio: window.audio,
        });
    }
}

/// The append-only stream of one meeting. Every event is one line, written with one `write`, so
/// a reader (or a killed writer) never sees half an event except possibly a last unfinished line.
struct StreamWriter {
    file: File,
    path: PathBuf,
    seq: u64,
    print: bool,
}

impl StreamWriter {
    fn create(out_dir: &Path, key: &str, print: bool) -> Result<Self> {
        recovery::private_dir(out_dir)?;
        let path = out_dir.join(format!("{key}.jsonl"));
        // Opened without truncating: a second reader of the same meeting must meet the first
        // one's lock before it can empty the stream under it. The lock lasts as long as `file`.
        let file = OpenOptions::new().write(true).create(true).truncate(false).mode(0o600).open(&path)
            .with_context(|| format!("create live stream {}", path.display()))?;
        // SAFETY: `file` owns the valid descriptor; flock does not access Rust memory.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::WouldBlock {
                bail!("another live reader already writes this meeting's stream");
            }
            return Err(error).context("lock the live stream");
        }
        file.set_len(0).context("start the live stream")?;
        Ok(Self { file, path, seq: 0, print })
    }

    fn event(&mut self, kind: &str, mut fields: Value) -> Result<()> {
        fields["v"] = json!(1);
        fields["seq"] = json!(self.seq);
        fields["type"] = json!(kind);
        self.file.write_all(format!("{fields}\n").as_bytes()).context("append to the live stream")?;
        self.seq += 1;
        if self.print && let Some(line) = display(&fields) {
            println!("{line}");
        }
        Ok(())
    }
}

fn round(value: f64, decimals: i32) -> f64 {
    let scale = 10f64.powi(decimals);
    (value * scale).round() / scale
}

fn log_event(event: &'static str, key: &str, fields: Value) {
    let mut line = json!({"component": "live", "event": event, "session_id": key, "outcome": "ok"});
    if let (Some(line), Some(fields)) = (line.as_object_mut(), fields.as_object()) {
        line.extend(fields.clone());
    }
    log::info!(target: "meetscribe::event", "{line}");
}

/// The state and when it was written. The checkpoint's write time anchors audio time to the
/// clock: the audio it covers had been captured by then.
fn read_state(dir: &Path) -> Result<(State, SystemTime)> {
    let state = State::read(dir)?;
    let written = std::fs::metadata(dir.join(recovery::STATE_FILE))?.modified()?;
    Ok((state, written))
}

/// Audio seconds the checkpoint covers, on the meeting's timeline: what had been captured when
/// the state file was written.
fn committed_secs(state: &State) -> f64 {
    state.rolls.last().zip(state.roll_offsets().last().copied()).map_or(0.0, |(roll, offset)| {
        offset + roll.mic_frames.max(roll.system_frames) as f64 / f64::from(roll.sample_rate)
    })
}

/// How far behind the speaker the reader is at `now` for a moment `t_end` of the meeting. The
/// audio `state` covers had been captured by `written`, so that moment happened at about
/// `written - (committed_secs - t_end)`. The anchor is the checkpointed audio, not what the
/// reader has consumed: a reader that has fallen behind would otherwise report itself closer
/// to the speaker than it is.
fn lag_ms(state: &State, written: SystemTime, t_end: f64, now: SystemTime) -> u64 {
    written.checked_sub(Duration::from_secs_f64((committed_secs(state) - t_end).max(0.0)))
        .and_then(|spoken_at| now.duration_since(spoken_at).ok())
        .map_or(0, |lag| lag.as_millis() as u64)
}

/// Follow the recording in `dir` until it ends. `decode` turns one 16 kHz window into
/// `(text, confidence)`; the binary passes whisper, tests pass a fake.
pub(crate) fn run_live(
    dir: &Path,
    opts: &LiveOpts,
    mut decode: impl FnMut(&[f32], &str) -> Result<(String, f32)>,
) -> Result<LiveSummary> {
    let (mut state, mut written) = read_state(dir).context("the session has no readable state yet")?;
    let key = state.key.clone();
    let mut stream = StreamWriter::create(&opts.out_dir, &key, opts.print)?;
    stream.event("meeting_started", json!({
        "key": key, "app": detect::app_label(&state.app), "started_at": state.started_at,
        "model": crate::pipeline::model_name(&opts.model),
    }))?;
    log_event("live_started", &key, json!({}));

    let mut channels = [Stream::new("mic", Speaker::You)?, Stream::new("system", Speaker::Others)?];
    let (mut segments, mut short_windows, mut decoded_secs) = (0u64, 0u64, 0.0f64);
    let (mut last_heartbeat, mut last_lag_log) = (Instant::now(), Instant::now());
    let mut last_lag_ms = 0u64;
    let reason = loop {
        // A state file is replaced atomically, so a failed read is a transient race at worst.
        if let Ok((fresh, fresh_written)) = read_state(dir) {
            if fresh.key != key { bail!("the session directory now belongs to another recording"); }
            (state, written) = (fresh, fresh_written);
        }
        let mut pending = Vec::new();
        let mut progressed = false;
        for channel in &mut channels {
            progressed |= channel.advance(dir, &state, opts.max_window_ms, &mut pending)?;
        }
        let audio_secs = channels.iter().map(|channel| channel.audio_secs(&state)).fold(0.0, f64::max);
        pending.sort_by(|a, b| a.t_end.total_cmp(&b.t_end));
        for window in pending {
            if window.audio.len() < WHISPER_FLOOR_SAMPLES {
                short_windows += 1;
                continue;
            }
            let decode_started = Instant::now();
            let (text, confidence) = decode(&window.audio, &opts.lang)?;
            let decode_ms = decode_started.elapsed().as_millis() as u64;
            decoded_secs = decoded_secs.max(window.t_end);
            if text.is_empty() { continue; }
            last_lag_ms = lag_ms(&state, written, window.t_end, SystemTime::now());
            stream.event("segment", segment_fields(&window, &text, confidence, last_lag_ms, decode_ms))?;
            segments += 1;
        }
        if last_heartbeat.elapsed() >= HEARTBEAT {
            stream.event("heartbeat", json!({"audio_secs": round(audio_secs, 1), "decoded_secs": round(decoded_secs, 1)}))?;
            last_heartbeat = Instant::now();
        }
        if last_lag_log.elapsed() >= LAG_LOG {
            log_event("live_lag", &key, json!({
                "audio_secs": round(audio_secs, 1), "decoded_secs": round(decoded_secs, 1),
                "segments": segments, "short_windows": short_windows, "lag_ms": last_lag_ms,
            }));
            last_lag_log = Instant::now();
        }
        if channels.iter().all(|channel| channel.finished(&state)) { break "ended"; }
        if progressed { continue; }
        let quiet_for = SystemTime::now().duration_since(written).unwrap_or_default();
        if state.phase == Phase::Recording && quiet_for >= STALL && !status::call_is_live(&opts.base) {
            break "stalled";
        }
        std::thread::sleep(POLL);
    };

    let audio_secs = channels.iter().map(|channel| channel.audio_secs(&state)).fold(0.0, f64::max);
    stream.event("meeting_ended", json!({
        "duration_secs": round(audio_secs, 1), "reason": reason, "segments": segments, "short_windows": short_windows,
    }))?;
    log_event("live_stopped", &key, json!({
        "audio_secs": round(audio_secs, 1), "decoded_secs": round(decoded_secs, 1),
        "segments": segments, "short_windows": short_windows, "lag_ms": last_lag_ms,
        "outcome": if reason == "ended" { "ok" } else { "attention" },
    }));
    Ok(LiveSummary { segments, short_windows, audio_secs, reason, stream: stream.path })
}

/// The fields of one `segment` line. The times are written as computed (the roll's offset plus
/// whole milliseconds), not rounded: `speakers cluster` slices the recording at stored times and
/// refuses one that is off a millisecond boundary of its roll.
fn segment_fields(window: &Pending, text: &str, confidence: f32, lag_ms: u64, decode_ms: u64) -> Value {
    json!({
        "speaker": window.speaker, "t_start": window.t_start, "t_end": window.t_end,
        "text": text, "confidence": round(f64::from(confidence), 3), "lag_ms": lag_ms, "decode_ms": decode_ms,
    })
}

/// One line of the stream as the viewer shows it; `None` for events with nothing to show.
fn display(event: &Value) -> Option<String> {
    match event["type"].as_str()? {
        "meeting_started" => Some(format!(
            "── {} · {} UTC ──", event["app"].as_str().unwrap_or("meeting"),
            export::fmt_utc(event["started_at"].as_i64().unwrap_or(0)))),
        "segment" => {
            let speaker = if event["speaker"] == "you" { Speaker::You } else { Speaker::Others };
            Some(format!("[{}] {}: {}", export::fmt_timestamp(event["t_start"].as_f64().unwrap_or(0.0)),
                speaker.label(), event["text"].as_str().unwrap_or("").trim()))
        }
        "meeting_ended" => Some(format!(
            "── ended · {} ──", export::fmt_duration(event["duration_secs"].as_f64().unwrap_or(0.0)))),
        _ => None,
    }
}

/// The most recently written stream in `out_dir`.
fn newest_stream(out_dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(out_dir).ok()?;
    entries.filter_map(|entry| {
        let path = entry.ok()?.path();
        let modified = path.metadata().ok()?.modified().ok()?;
        (path.extension()? == "jsonl").then_some((modified, path))
    }).max().map(|(_, path)| path)
}

/// Print the complete lines of `path` from byte `offset` on; returns the new offset. A last line
/// without its newline is still being written and is left for the next call.
fn print_from(path: &Path, offset: u64, out: &mut impl Write) -> Result<u64> {
    let mut reader = BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    // A stream that shrank was started again for the same meeting: show it from its beginning.
    let mut offset = if reader.get_ref().metadata()?.len() < offset { 0 } else { offset };
    reader.seek(SeekFrom::Start(offset))?;
    let mut line = String::new();
    loop {
        line.clear();
        let read = reader.read_line(&mut line)?;
        if read == 0 || !line.ends_with('\n') { return Ok(offset); }
        offset += read as u64;
        if let Some(shown) = serde_json::from_str::<Value>(&line).ok().as_ref().and_then(display) {
            writeln!(out, "{shown}")?;
            out.flush()?;
        }
    }
}

/// `meetscribe live tail [--follow] [--key <k>] [--dir <live dir>]`.
fn run_tail(argv: &[String]) -> Result<()> {
    let (mut follow, mut key, mut out_dir) = (false, None, None);
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--follow" | "-f" => follow = true,
            "--key" => key = Some(it.next().context("--key needs a value")?.clone()),
            "--dir" => out_dir = Some(PathBuf::from(it.next().context("--dir needs a value")?)),
            other => bail!("unknown argument {other} (usage: meetscribe live tail [--follow] [--key <k>] [--dir <dir>])"),
        }
    }
    let out_dir = match out_dir {
        Some(dir) => dir,
        None => default_out_dir(&crate::base_dir().context("cannot resolve $HOME")?),
    };
    let pick = || key.as_ref().map(|key| out_dir.join(format!("{key}.jsonl"))).filter(|path| path.exists())
        .or_else(|| if key.is_none() { newest_stream(&out_dir) } else { None });
    let mut stdout = std::io::stdout().lock();
    let mut offsets = HashMap::new();
    loop {
        // With --follow the viewer moves to the next meeting's stream when one appears.
        match pick() {
            Some(path) => show(&path, &mut offsets, &mut stdout)?,
            None if !follow => bail!("no live stream in {}", out_dir.display()),
            None => {}
        }
        if !follow { return Ok(()); }
        std::thread::sleep(Duration::from_millis(300));
    }
}

/// Print what `path` has gained since it was last shown. Each stream keeps its own position, so
/// two meetings written at the same time are each shown once however often the newest one changes.
fn show(path: &Path, offsets: &mut HashMap<PathBuf, u64>, out: &mut impl Write) -> Result<()> {
    let offset = print_from(path, offsets.get(path).copied().unwrap_or(0), out)?;
    offsets.insert(path.to_path_buf(), offset);
    Ok(())
}

/// `meetscribe live replay <recording dir> <new session dir> [--speed <n>] [--seconds <n>]`:
/// a test tool. It regrows a finished recording into a new managed session the way the capture
/// loop writes one (append to both WAVs, flush both, then save the checkpoint every 2 s of
/// audio), so `meetscribe live` can be tried without a call. The copy holds real audio: put it
/// in a private place and remove it afterwards.
fn run_replay(argv: &[String]) -> Result<()> {
    let (mut paths, mut speed, mut seconds) = (Vec::new(), 1.0f64, None);
    let mut it = argv.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--speed" => speed = it.next().context("--speed needs a value")?.parse().context("--speed")?,
            "--seconds" => seconds = Some(it.next().context("--seconds needs a value")?.parse::<f64>().context("--seconds")?),
            other => paths.push(PathBuf::from(other)),
        }
    }
    let [source, destination] = paths.as_slice() else {
        bail!("usage: meetscribe live replay <recording dir> <new session dir> [--speed <n>] [--seconds <n>]");
    };
    anyhow::ensure!(speed > 0.0, "--speed must be positive");
    let (mic, rate) = crate::pipeline::read_wav_any_rate(&source.join("mic.wav"))?;
    let (system, system_rate) = crate::pipeline::read_wav_any_rate(&source.join("system.wav"))?;
    anyhow::ensure!(rate == system_rate, "mic and system differ in sample rate; not one capture roll");
    let frames = mic.len().min(system.len()).min(seconds.map_or(usize::MAX, |secs| (secs * f64::from(rate)) as usize));
    let mut session = recovery::Session::begin(destination, &destination.join("replay.db"), destination,
        "org.meetscribe.replay", "Replay", crate::now_epoch())?;
    session.start_roll(0, rate, 0.0)?;
    let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
    let mut mic_out = hound::WavWriter::create(session.dir.join("mic.wav"), spec)?;
    let mut system_out = hound::WavWriter::create(session.dir.join("system.wav"), spec)?;
    let step = rate as usize * 2;
    let started = Instant::now();
    let mut position = 0usize;
    while position < frames {
        let end = (position + step).min(frames);
        for index in position..end {
            mic_out.write_sample(mic[index])?;
            system_out.write_sample(system[index])?;
        }
        mic_out.flush()?;
        system_out.flush()?;
        session.checkpoint(0, end, end)?;
        position = end;
        // Keep to the recording's own clock, scaled: checkpoint k is due at k * 2 s / speed.
        let due = Duration::from_secs_f64(position as f64 / f64::from(rate) / speed);
        std::thread::sleep(due.saturating_sub(started.elapsed()));
    }
    mic_out.finalize()?;
    system_out.finalize()?;
    session.finish_capture(frames as f64 / f64::from(rate), 0.0)?;
    println!("{}", json!({"event": "replay_finished", "audio_secs": round(frames as f64 / f64::from(rate), 1), "speed": speed}));
    Ok(())
}

/// The session being recorded right now under `sessions`, if any: its state says `Recording`
/// and its last checkpoint is recent. `followed` holds the keys already transcribed, so a
/// recording whose writer died is not picked up again.
fn recording_now(sessions: &Path, followed: &std::collections::HashSet<String>) -> Option<PathBuf> {
    let entries = std::fs::read_dir(sessions).ok()?;
    entries.filter_map(|entry| {
        let dir = entry.ok()?.path();
        let (state, written) = read_state(&dir).ok()?;
        let fresh = SystemTime::now().duration_since(written).is_ok_and(|age| age < STALL);
        (state.phase == Phase::Recording && fresh && !followed.contains(&state.key)).then_some((state.started_at, dir))
    }).max().map(|(_, dir)| dir)
}

/// `meetscribe live watch`: wait for a recording, print its transcript while it runs, then wait
/// for the next one. The daemon is not involved: this only reads what it checkpoints.
fn run_watch(opts: &LiveOpts) -> Result<()> {
    let model = opts.model.to_str().context("model path is not valid UTF-8")?;
    let mut asr = asr::Asr::load(model)?;
    let sessions = opts.base.join("sessions");
    let mut followed = std::collections::HashSet::new();
    eprintln!("waiting for a recording under {} (Ctrl-C to stop)", sessions.display());
    loop {
        let Some(dir) = recording_now(&sessions, &followed) else {
            std::thread::sleep(Duration::from_secs(1));
            continue;
        };
        let key = State::read(&dir).ok().map(|state| state.key);
        if let Some(key) = &key { followed.insert(key.clone()); }
        // One failed meeting must not end the watch: the next call still gets its transcript.
        if let Err(error) = run_live(&dir, opts, |audio, lang| asr.transcribe(audio, lang)) {
            let mut line = json!({"component": "live", "event": "live_failed", "outcome": "error",
                "error_class": crate::diagnostics::error_class(&error)});
            if let Some(key) = &key { line["session_id"] = json!(key); }
            log::warn!(target: "meetscribe::event", "{line}");
            eprintln!("this recording could not be followed to its end; waiting for the next one");
        }
    }
}

/// The options of `live <session-dir>` and `live watch`, and the session directory when one was
/// given. `None` after `--help`.
fn parse_args(argv: &[String], watch: bool, base: PathBuf) -> Result<Option<(LiveOpts, Option<PathBuf>)>> {
    let mut base = base;
    let (mut model, mut lang, mut out_dir) = (None, None, None);
    let (mut max_window_ms, mut print, mut dir) = (12_000, watch, None);
    let mut it = argv.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().cloned().with_context(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--model" | "-m" => model = Some(PathBuf::from(value("--model")?)),
            "--lang" | "-l" => lang = Some(value("--lang")?),
            "--out" => out_dir = Some(PathBuf::from(value("--out")?)),
            "--base" => base = PathBuf::from(value("--base")?),
            "--max-window-secs" => {
                let secs: f64 = value("--max-window-secs")?.parse().context("--max-window-secs")?;
                anyhow::ensure!(secs >= 2.0, "--max-window-secs must be at least 2");
                max_window_ms = (secs * 1000.0) as usize;
            }
            "--print" => print = true,
            "-h" | "--help" => {
                eprintln!("usage: meetscribe live watch [options]            wait for calls and print each transcript\n       \
                    meetscribe live <session-dir> [options]    follow one recording\n       \
                    meetscribe live tail [--follow] [--key <k>] [--dir <dir>]\n       \
                    meetscribe live replay <recording dir> <new session dir> [--speed <n>] [--seconds <n>]\n\
                    options: [--model <ggml>] [--lang <code>] [--out <dir>] [--base <dir>] [--max-window-secs <n>] [--print]");
                return Ok(None);
            }
            other if !watch && dir.is_none() && !other.starts_with('-') => dir = Some(PathBuf::from(other)),
            other => bail!("unknown argument {other} (see `meetscribe live --help`)"),
        }
    }
    // What no flag set follows the base directory, and the language follows the daemon's config,
    // so the live text is decoded like the transcript the daemon will store. `Config::load` only
    // reads: following a recording must not create a config file.
    let opts = LiveOpts {
        model: model.unwrap_or_else(|| base.join("models/ggml-large-v3.bin")),
        lang: lang.unwrap_or_else(|| crate::config::Config::load(&base).daemon.lang),
        out_dir: out_dir.unwrap_or_else(|| default_out_dir(&base)),
        base, max_window_ms, print,
    };
    Ok(Some((opts, dir)))
}

/// `meetscribe live <session-dir> [--model <ggml>] [--lang <code>] [--out <dir>] [--base <dir>]
/// [--max-window-secs <n>] [--print]`, plus the `watch`, `tail` and `replay` subcommands.
pub(crate) fn run(argv: &[String]) -> Result<()> {
    match argv.first().map(String::as_str) {
        Some("tail") => return run_tail(&argv[1..]),
        Some("replay") => return run_replay(&argv[1..]),
        _ => {}
    }
    let watch = argv.first().is_some_and(|arg| arg == "watch");
    let base = crate::base_dir().context("cannot resolve $HOME")?;
    let Some((opts, dir)) = parse_args(&argv[usize::from(watch)..], watch, base)? else { return Ok(()); };
    if watch { return run_watch(&opts); }
    let dir = dir.context("usage: meetscribe live <session-dir> (see --help)")?;
    let model = opts.model.to_str().context("model path is not valid UTF-8")?;
    let mut asr = asr::Asr::load(model)?;
    let summary = run_live(&dir, &opts, |audio, lang| asr.transcribe(audio, lang))?;
    // Numbers and the stream's path only: the text is in the stream.
    println!("{}", json!({
        "event": "live_summary", "segments": summary.segments, "short_windows": summary.short_windows,
        "audio_secs": round(summary.audio_secs, 1), "reason": summary.reason, "stream": summary.stream,
    }));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("meetscribe-live-{tag}-{}", recovery::new_identity()));
        recovery::private_dir(&root).unwrap();
        root
    }

    fn opts(root: &Path) -> LiveOpts {
        LiveOpts { model: root.join("unused.bin"), lang: "es".into(), out_dir: root.join("live"),
            base: root.join("base"), max_window_ms: 12_000, print: false }
    }

    fn events(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path).unwrap().lines().map(|line| serde_json::from_str(line).unwrap()).collect()
    }

    fn wav(path: &Path, rate: u32) -> hound::WavWriter<std::io::BufWriter<File>> {
        hound::WavWriter::create(path, hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float }).unwrap()
    }

    #[test]
    fn a_pause_closes_a_window_and_a_monologue_is_cut_into_pieces() {
        let mut cutter = Cutter::default();
        assert!(cutter.on_transition(VadTransition::SpeechStart { timestamp_ms: 1_000 }).is_none());
        assert_eq!(cutter.forced_cut(12_999, 12_000), None);
        assert_eq!(cutter.forced_cut(13_000, 12_000), Some(1_000));
        // The caller emits 1.0–13.0 s itself and moves the start; the pause then owes only the rest.
        cutter.open_from_ms = Some(13_000);
        let samples: Vec<f32> = (0..15_000 * SAMPLES_PER_MS).map(|index| index as f32).collect();
        let rest = cutter.on_transition(VadTransition::SpeechEnd {
            start_timestamp_ms: 1_000, end_timestamp_ms: 16_000, samples: samples.clone() }).unwrap();
        assert_eq!((rest.start_ms, rest.end_ms, rest.audio.len()), (13_000, 16_000, 3_000 * SAMPLES_PER_MS));
        assert_eq!(rest.audio[0], (12_000 * SAMPLES_PER_MS) as f32, "the rest starts where the forced cut ended");
        assert_eq!(cutter.forced_cut(99_000, 12_000), None, "nothing is open after the pause");
        // Speech that ends without a forced cut is emitted whole.
        cutter.on_transition(VadTransition::SpeechStart { timestamp_ms: 20_000 });
        let whole = cutter.on_transition(VadTransition::SpeechEnd {
            start_timestamp_ms: 20_000, end_timestamp_ms: 22_000, samples: samples[..2_000 * SAMPLES_PER_MS].to_vec() }).unwrap();
        assert_eq!((whole.start_ms, whole.end_ms, whole.audio.len()), (20_000, 22_000, 2_000 * SAMPLES_PER_MS));
    }

    #[test]
    fn a_ranged_read_returns_exactly_the_frames_inside_the_committed_prefix() {
        let root = temp_root("range");
        let path = root.join("mic.wav");
        let mut writer = wav(&path, 16_000);
        for index in 0..1_000 { writer.write_sample(index as f32).unwrap(); }
        writer.flush().unwrap();
        assert_eq!(recovery::read_committed_range(&path, 16_000, 1_000, 250, 253).unwrap(), vec![250.0, 251.0, 252.0]);
        assert!(recovery::read_committed_range(&path, 16_000, 1_000, 0, 0).unwrap().is_empty());
        // More frames are appended but not flushed or checkpointed: still only the prefix is trusted.
        for index in 1_000..1_500 { writer.write_sample(index as f32).unwrap(); }
        assert!(recovery::read_committed_range(&path, 16_000, 1_000, 990, 1_001).is_err(), "past the checkpoint");
        assert!(recovery::read_committed_range(&path, 16_000, 2_000_000, 0, 10).is_err(), "checkpoint beyond the file");
        assert!(recovery::read_committed_range(&path, 48_000, 1_000, 0, 10).is_err(), "wrong sample rate");
        drop(writer);
        std::fs::remove_dir_all(root).unwrap();
    }

    /// A recording that grows while the reader follows it: two rolls at different rates, silence
    /// only (no model in a unit test). Proves the reader consumes every checkpointed frame, ends
    /// when the recording ends, writes a valid private stream outside the session directory and
    /// puts no meeting text in it beyond what the decoder returned.
    #[test]
    fn a_growing_recording_is_followed_to_its_end_and_the_session_dir_is_untouched() {
        let root = temp_root("follow");
        let dir = root.join("sessions").join("recording");
        let mut session = recovery::Session::begin(&dir, &root.join("archive.db"), &dir,
            "com.google.Chrome", "Live fixture", 1_762_880_000).unwrap();
        let dir = session.dir.clone();
        session.start_roll(0, 16_000, 0.0).unwrap();
        let (mut mic, mut system) = (wav(&dir.join("mic.wav"), 16_000), wav(&dir.join("system.wav"), 16_000));
        let reader_dir = dir.clone();
        let reader_opts = opts(&root);
        let reader = std::thread::spawn(move || run_live(&reader_dir, &reader_opts, |_, _| Ok(("decoded".into(), 0.9))));
        let mut written = 0usize;
        for _ in 0..3 {
            for _ in 0..16_000 { mic.write_sample(0.0f32).unwrap(); system.write_sample(0.0f32).unwrap(); }
            written += 16_000;
            mic.flush().unwrap(); system.flush().unwrap();
            session.checkpoint(0, written, written).unwrap();
            std::thread::sleep(Duration::from_millis(120));
        }
        // A route change rolls to 48 kHz after a 0.25 s gap; the mic ends 100 frames short.
        session.start_roll(1, 48_000, 0.25).unwrap();
        let (mut mic_1, mut system_1) = (wav(&dir.join("mic.001.wav"), 48_000), wav(&dir.join("system.001.wav"), 48_000));
        for index in 0..96_000 { if index < 95_900 { mic_1.write_sample(0.0f32).unwrap(); } system_1.write_sample(0.0f32).unwrap(); }
        mic_1.flush().unwrap(); system_1.flush().unwrap();
        session.checkpoint(1, 95_900, 96_000).unwrap();
        session.finish_capture(5.25, 0.1).unwrap();
        let summary = reader.join().unwrap().unwrap();

        assert_eq!((summary.reason, summary.segments, summary.short_windows), ("ended", 0, 0));
        assert!((summary.audio_secs - 5.25).abs() < 1e-6, "3 s + 0.25 s gap + 2 s, got {}", summary.audio_secs);
        assert_eq!(summary.stream, root.join("live").join(format!("{}.jsonl", session.state.key)));
        let lines = events(&summary.stream);
        assert_eq!(lines.first().unwrap()["type"], "meeting_started");
        assert_eq!((lines[0]["app"].as_str(), lines[0]["key"].as_str()), (Some("Chrome"), Some(session.state.key.as_str())));
        assert_eq!(lines.last().unwrap()["type"], "meeting_ended");
        assert_eq!(lines.last().unwrap()["reason"], "ended");
        // What a later reader of the stream needs to judge it complete: the model that decoded
        // it, and how many segment lines it should hold.
        assert_eq!(lines[0]["model"], "unused");
        let ended = lines.last().unwrap();
        assert_eq!((ended["segments"].as_u64(), ended["short_windows"].as_u64()), (Some(0), Some(0)));
        assert_eq!(lines.iter().filter(|line| line["type"] == "segment").count() as u64, summary.segments);
        assert!(lines.iter().enumerate().all(|(index, line)| line["seq"] == index as u64 && line["v"] == 1));
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&summary.stream).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(root.join("live")).unwrap().permissions().mode() & 0o777, 0o700);
        let mut names: Vec<String> = std::fs::read_dir(&dir).unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        assert_eq!(names, ["mic.001.wav", "mic.wav", "session-state.json", "system.001.wav", "system.wav"],
            "the reader must leave the session directory as retention expects it");
        drop((mic, system, mic_1, system_1));
        std::fs::remove_dir_all(root).unwrap();
    }

    /// A reader that is behind reads at most 30 s per pass. Its lag must still be counted from
    /// the audio the checkpoint covers, or it reports itself closer to the speaker than it is.
    #[test]
    fn lag_is_counted_from_the_checkpoint_not_from_what_the_reader_has_consumed() {
        let root = temp_root("lag");
        let dir = root.join("recording");
        let mut session = recovery::Session::begin(&dir, &root.join("archive.db"), &dir,
            "com.google.Chrome", "Lag fixture", 1_762_880_000).unwrap();
        session.start_roll(0, 16_000, 0.0).unwrap();
        let dir = session.dir.clone();
        let (mut mic, mut system) = (wav(&dir.join("mic.wav"), 16_000), wav(&dir.join("system.wav"), 16_000));
        for _ in 0..60 * 16_000 { mic.write_sample(0.0f32).unwrap(); system.write_sample(0.0f32).unwrap(); }
        mic.flush().unwrap(); system.flush().unwrap();
        session.checkpoint(0, 60 * 16_000, 60 * 16_000).unwrap();
        let (state, written) = read_state(&dir).unwrap();
        let mut stream = Stream::new("mic", Speaker::You).unwrap();
        assert!(stream.advance(&dir, &state, 12_000, &mut Vec::new()).unwrap());
        assert!((stream.audio_secs(&state) - 30.0).abs() < 1e-6, "one pass reads 30 s of the 60 s");
        assert!((committed_secs(&state) - 60.0).abs() < 1e-6);
        // A sentence that ended at 25 s was 35 s old when the checkpoint was written; 2 s later, 37 s.
        // Counted from the 30 s the reader has consumed it would look 7 s old.
        let now = written + Duration::from_secs(2);
        assert_eq!(lag_ms(&state, written, 25.0, now), 37_000);
        assert_eq!(lag_ms(&state, written, 60.0, now), 2_000, "the newest audio is as old as the checkpoint");
        assert_eq!(lag_ms(&state, written, 61.0, now), 2_000, "a moment past the checkpoint is not in the future");
        drop((mic, system));
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Two readers on one meeting (a daemon-started one and a hand-started `live watch`) must
    /// not both write its stream: the second is refused before it can empty the first one's file.
    #[test]
    fn a_second_writer_is_refused_and_the_first_stream_is_intact() {
        let root = temp_root("writer-lock");
        let out = root.join("live");
        let key = "a6e79ba0314c4cc2b647f4d25b327011";
        let mut first = StreamWriter::create(&out, key, false).unwrap();
        first.event("meeting_started", json!({"key": key})).unwrap();
        let before = std::fs::read(&first.path).unwrap();
        assert!(StreamWriter::create(&out, key, false).is_err(), "a second writer was let in");
        assert_eq!(std::fs::read(&first.path).unwrap(), before, "the refused writer emptied the stream");
        first.event("heartbeat", json!({})).unwrap();
        assert_eq!(events(&first.path).iter().map(|line| line["seq"].as_u64().unwrap()).collect::<Vec<_>>(), [0, 1]);
        // Once the first writer is gone, a new reader starts the stream again from nothing.
        let path = first.path.clone();
        drop(first);
        let again = StreamWriter::create(&out, key, false).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
        drop(again);
        std::fs::remove_dir_all(root).unwrap();
    }

    /// `speakers cluster` slices the recording at stored times and refuses a time that is off a
    /// millisecond boundary of its roll. A roll after a device change starts at an offset on no
    /// decimal grid, so a time rounded for the stream would no longer match the audio.
    #[test]
    fn segment_times_survive_the_stream_on_a_millisecond_of_their_roll() {
        // The second roll of the fixture above: 3 s of roll 0 and a 0.25 s gap would put it on a
        // grid, so start it after 95,900 frames at 48 kHz instead.
        let offset = 0.25 + 95_900.0 / 48_000.0;
        let window = Pending { speaker: Speaker::Others, t_start: offset + 1_232.0 / 1000.0,
            t_end: offset + 4_567.0 / 1000.0, audio: Vec::new() };
        let line = segment_fields(&window, "decoded", 0.91, 3_400, 1_500).to_string();
        let read: Value = serde_json::from_str(&line).unwrap();
        for (field, whole_ms) in [("t_start", 1_232.0), ("t_end", 4_567.0)] {
            let ms = (read[field].as_f64().unwrap() - offset) * 1000.0;
            assert!((ms - whole_ms).abs() <= 1e-6, "{field} is {ms} ms into its roll, not {whole_ms}");
        }
        assert_eq!((read["speaker"].as_str(), read["confidence"].as_f64()), (Some("others"), Some(0.91)));
    }

    #[test]
    fn a_recording_whose_writer_died_ends_as_stalled_instead_of_waiting_forever() {
        let root = temp_root("stalled");
        let dir = root.join("recording");
        let mut session = recovery::Session::begin(&dir, &root.join("archive.db"), &dir,
            "com.google.Chrome", "Stalled fixture", 1_762_880_000).unwrap();
        session.start_roll(0, 16_000, 0.0).unwrap();
        let dir = session.dir.clone();
        let mut mic = wav(&dir.join("mic.wav"), 16_000);
        let mut system = wav(&dir.join("system.wav"), 16_000);
        for _ in 0..8_000 { mic.write_sample(0.0f32).unwrap(); system.write_sample(0.0f32).unwrap(); }
        mic.flush().unwrap(); system.flush().unwrap();
        session.checkpoint(0, 8_000, 8_000).unwrap();
        // The last checkpoint is 31 s old and no daemon says it is recording.
        let state_file = File::options().write(true).open(dir.join(recovery::STATE_FILE)).unwrap();
        state_file.set_modified(SystemTime::now() - Duration::from_secs(31)).unwrap();
        let summary = run_live(&dir, &opts(&root), |_, _| Ok((String::new(), 0.0))).unwrap();
        assert_eq!(summary.reason, "stalled");
        assert!((summary.audio_secs - 0.5).abs() < 1e-6);
        assert_eq!(events(&summary.stream).last().unwrap()["reason"], "stalled");
        drop((mic, system));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn watch_picks_the_recording_in_progress_and_nothing_else() {
        let root = temp_root("watch");
        let sessions = root.join("sessions");
        let begin = |name: &str, started_at: i64| {
            let dir = sessions.join(name);
            let mut session = recovery::Session::begin(&dir, &root.join("archive.db"), &dir,
                "com.google.Chrome", "Watch fixture", started_at).unwrap();
            session.start_roll(0, 16_000, 0.0).unwrap();
            session
        };
        let mut finished = begin("finished", 300);
        finished.finish_capture(60.0, 0.1).unwrap();
        let abandoned = begin("abandoned", 200);
        File::options().write(true).open(abandoned.dir.join(recovery::STATE_FILE)).unwrap()
            .set_modified(SystemTime::now() - Duration::from_secs(120)).unwrap();
        let mut followed = std::collections::HashSet::new();
        // Compared by directory name: the temp dir is reached through a symlink on macOS.
        let picked = |followed: &std::collections::HashSet<String>| recording_now(&sessions, followed)
            .map(|dir| dir.file_name().unwrap().to_string_lossy().into_owned());
        assert_eq!(picked(&followed), None, "a finished and a dead recording are not calls in progress");
        let older = begin("older", 100);
        let newer = begin("newer", 150);
        assert_eq!(picked(&followed).as_deref(), Some("newer"));
        followed.insert(newer.state.key.clone());
        assert_eq!(picked(&followed).as_deref(), Some("older"), "a recording already followed is skipped");
        drop((finished, abandoned, older));
        std::fs::remove_dir_all(root).unwrap();
    }

    fn args(list: &[&str]) -> Vec<String> { list.iter().map(|arg| (*arg).to_string()).collect() }

    /// `live watch` follows the daemon's recordings, so it decodes in the daemon's language and
    /// keeps everything under the daemon's base directory unless a flag says otherwise.
    #[test]
    fn defaults_follow_the_base_directory_and_the_daemons_language() {
        let root = temp_root("defaults");
        let (home, other) = (root.join("home"), root.join("other"));
        recovery::private_dir(&home).unwrap();
        std::fs::write(home.join("config.toml"), "version = 1\n[daemon]\nlang = \"en\"\n").unwrap();
        let (opts, dir) = parse_args(&[], true, home.clone()).unwrap().unwrap();
        assert_eq!((opts.lang.as_str(), dir, opts.print), ("en", None, true));
        assert_eq!((opts.out_dir, opts.model), (home.join("live"), home.join("models/ggml-large-v3.bin")));
        // Another base directory moves the stream, the model and the config with it.
        let moved = args(&["session", "--base", other.to_str().unwrap()]);
        let (opts, dir) = parse_args(&moved, false, home.clone()).unwrap().unwrap();
        assert_eq!((opts.lang.as_str(), dir), ("es", Some(PathBuf::from("session"))));
        assert_eq!((opts.base.clone(), opts.out_dir, opts.model), (other.clone(), other.join("live"), other.join("models/ggml-large-v3.bin")));
        assert!(!other.exists(), "a reader must not create a config file or its directory");
        // Each flag still overrides its own default, whatever the order.
        let explicit = args(&["--out", "/streams", "--lang", "pt", "--base", other.to_str().unwrap(), "--model", "/m.bin"]);
        let (opts, _) = parse_args(&explicit, true, home).unwrap().unwrap();
        assert_eq!((opts.lang.as_str(), opts.out_dir, opts.model), ("pt", PathBuf::from("/streams"), PathBuf::from("/m.bin")));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_viewer_shows_each_line_once_when_two_streams_take_turns() {
        let root = temp_root("turns");
        let (first, second) = (root.join("first.jsonl"), root.join("second.jsonl"));
        let segment = |seq: u64, text: &str| format!("{}\n", json!({"v":1,"seq":seq,"type":"segment","speaker":"you","t_start":1.0,"t_end":2.0,"text":text}));
        let append = |path: &Path, line: String| OpenOptions::new().create(true).append(true).open(path).unwrap().write_all(line.as_bytes()).unwrap();
        let (mut offsets, mut shown) = (HashMap::new(), Vec::new());
        append(&first, segment(0, "uno"));
        show(&first, &mut offsets, &mut shown).unwrap();
        append(&second, segment(0, "dos"));
        show(&second, &mut offsets, &mut shown).unwrap();
        append(&first, segment(1, "tres"));
        show(&first, &mut offsets, &mut shown).unwrap();
        show(&second, &mut offsets, &mut shown).unwrap();
        let shown = String::from_utf8(shown).unwrap();
        let texts: Vec<&str> = shown.lines().map(|line| line.rsplit(": ").next().unwrap()).collect();
        assert_eq!(texts, ["uno", "dos", "tres"]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_viewer_shows_segments_and_waits_for_an_unfinished_line() {
        let root = temp_root("tail");
        let path = root.join("stream.jsonl");
        let lines = [
            json!({"v":1,"seq":0,"type":"meeting_started","key":"k","app":"Chrome","started_at":1_794_443_305}),
            json!({"v":1,"seq":1,"type":"segment","speaker":"others","t_start":41.2,"t_end":47.9,"text":" Hola, ¿se escucha? ","confidence":0.91,"lag_ms":3400}),
            json!({"v":1,"seq":2,"type":"heartbeat","audio_secs":52.0,"decoded_secs":48.6}),
            json!({"v":1,"seq":3,"type":"segment","speaker":"you","t_start":3725.0,"t_end":3727.0,"text":"Sí, perfecto.","confidence":0.9,"lag_ms":2900}),
        ];
        let body: String = lines.iter().map(|line| format!("{line}\n")).collect();
        let unfinished = r#"{"v":1,"seq":4,"type":"segment","speaker":"you","t_start":3730.0,"text":"cut off"#;
        std::fs::write(&path, format!("{body}{unfinished}")).unwrap();
        let mut shown = Vec::new();
        let offset = print_from(&path, 0, &mut shown).unwrap();
        assert_eq!(offset, body.len() as u64);
        let shown = String::from_utf8(shown).unwrap();
        let expected_start = format!("── Chrome · {} UTC ──", export::fmt_utc(1_794_443_305));
        assert_eq!(shown.lines().collect::<Vec<_>>(), [expected_start.as_str(),
            &format!("[{}] Others: Hola, ¿se escucha?", export::fmt_timestamp(41.2)),
            &format!("[{}] You: Sí, perfecto.", export::fmt_timestamp(3725.0))]);
        // The rest arrives: only the new line is printed.
        let ending = json!({"v":1,"seq":5,"type":"meeting_ended","duration_secs":3731.0,"reason":"ended"});
        std::fs::write(&path, format!("{body}{ending}\n")).unwrap();
        let mut more = Vec::new();
        print_from(&path, offset, &mut more).unwrap();
        assert_eq!(String::from_utf8(more).unwrap().trim_end(), format!("── ended · {} ──", export::fmt_duration(3731.0)));
        std::fs::remove_dir_all(root).unwrap();
    }
}
