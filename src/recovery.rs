//! Private session state and lifetime locks shared by capture and recovery.
//! Checkpoints describe a paired committed prefix, never an unflushed tail.
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

pub const STATE_FILE: &str = "session-state.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase { Recording, Ready, Archived, Complete, SkippedShort, NeedsAttention }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureStatus { Unknown, Complete, Partial }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ProofScope { Archive, Retention }

impl CaptureStatus {
    pub fn as_str(self) -> &'static str {
        match self { Self::Unknown => "unknown", Self::Complete => "complete", Self::Partial => "partial" }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Roll {
    pub index: u32,
    pub sample_rate: u32,
    pub gap_seconds: f64,
    pub mic_frames: u64,
    pub system_frames: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub version: u32,
    pub key: String,
    pub started_at: i64,
    pub app: String,
    pub title: String,
    pub db_path: PathBuf,
    pub export_dir: PathBuf,
    pub lock_dir: PathBuf,
    pub archive_id: Option<String>,
    pub phase: Phase,
    pub capture_status: CaptureStatus,
    pub rolls: Vec<Roll>,
    pub duration_seconds: f64,
    pub attempts: u8,
    pub in_flight: bool,
    pub next_retry: i64,
    pub error_class: Option<String>,
    pub meeting_id: Option<i64>,
    pub publication: Option<PathBuf>,
    pub unarchived_manual_output: bool,
}

impl State {
    pub fn read(dir: &Path) -> Result<Self> {
        let path = dir.join(STATE_FILE);
        if fs::metadata(&path)?.len() > 1_048_576 { bail!("session metadata exceeds its size limit"); }
        let state: Self = serde_json::from_reader(File::open(path)?)?;
        state.validate()?;
        Ok(state)
    }

    pub fn due(&self, now: i64) -> bool {
        matches!(self.phase, Phase::Recording | Phase::Ready | Phase::Archived)
            && (self.attempts < 4 || self.in_flight) && self.next_retry <= now
    }

    fn validate(&self) -> Result<()> {
        if self.version != 1 || self.key.len() != 32 || !self.key.bytes().all(|b| b.is_ascii_hexdigit()) {
            bail!("unknown or invalid session identity/version");
        }
        if !self.db_path.is_absolute() || !self.export_dir.is_absolute() || !self.lock_dir.is_absolute() {
            bail!("session destinations and lock directory must be absolute");
        }
        if !self.duration_seconds.is_finite() || self.duration_seconds < 0.0 || self.attempts > 4
            || self.in_flight && self.attempts == 0 {
            bail!("invalid session duration or retry state");
        }
        for (index, roll) in self.rolls.iter().enumerate() {
            if roll.index as usize != index || roll.sample_rate == 0
                || !roll.gap_seconds.is_finite() || roll.gap_seconds < 0.0 {
                bail!("invalid capture roll boundary");
            }
        }
        Ok(())
    }
}

pub fn new_identity() -> String {
    let mut bytes = [0u8; 16];
    // SAFETY: the writable stack buffer is valid for exactly the supplied length.
    unsafe { libc::arc4random_buf(bytes.as_mut_ptr().cast(), bytes.len()); }
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn absolute(path: &Path) -> Result<PathBuf> {
    let path = if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir()?.join(path) };
    Ok(fs::canonicalize(&path).unwrap_or(path))
}

pub fn private_dir(path: &Path) -> Result<()> {
    fs::DirBuilder::new().recursive(true).mode(0o700).create(path)
        .with_context(|| format!("create private directory {}", path.display()))
}

pub fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().context("state target has no parent")?;
    let temporary = parent.join(format!(".state-{}.tmp", new_identity()));
    let result = (|| {
        let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(&temporary)?;
        serde_json::to_writer(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() { let _ = fs::remove_file(&temporary); }
    result
}

pub struct SessionLock { _file: File }

impl Drop for SessionLock {
    fn drop(&mut self) {
        // Release logical ownership now, even if an unrelated concurrent fork briefly inherited
        // this open file description before exec. The lock is never cloned into another owner.
        // SAFETY: self owns the live descriptor, and flock does not access Rust memory.
        if unsafe { libc::flock(self._file.as_raw_fd(), libc::LOCK_UN) } != 0 {
            log::warn!("component=recovery event=session_unlock_failed");
        }
    }
}

impl SessionLock {
    fn acquire(lock_dir: &Path, key: &str) -> Result<Option<Self>> {
        private_dir(lock_dir)?;
        let file = OpenOptions::new().write(true).create(true).truncate(false).mode(0o600)
            .open(lock_dir.join(format!("{key}.lock")))?;
        // SAFETY: file owns the valid descriptor; flock does not access Rust memory.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::WouldBlock { return Ok(None); }
            return Err(error.into());
        }
        Ok(Some(Self { _file: file }))
    }
}

pub struct Session {
    pub dir: PathBuf,
    pub state: State,
    _lock: SessionLock,
}

impl Session {
    pub fn begin(dir: &Path, db: &Path, export: &Path, app: &str, title: &str, started_at: i64) -> Result<Self> {
        let parent = dir.parent().context("session directory has no parent")?;
        private_dir(parent)?;
        fs::DirBuilder::new().mode(0o700).create(dir).context("create exclusive session directory")?;
        let dir = fs::canonicalize(dir)?;
        let absolute = |path: &Path| -> Result<PathBuf> {
            Ok(if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir()?.join(path) })
        };
        let state = State {
            version: 1, key: new_identity(), started_at, app: app.into(), title: title.into(),
            db_path: absolute(db)?, export_dir: absolute(export)?, lock_dir: fs::canonicalize(parent)?.join(".locks"),
            archive_id: None, phase: Phase::Recording, capture_status: CaptureStatus::Unknown,
            rolls: Vec::new(), duration_seconds: 0.0, attempts: 0, in_flight: false,
            next_retry: 0, error_class: None, meeting_id: None, publication: None,
            unarchived_manual_output: false,
        };
        let lock = SessionLock::acquire(&state.lock_dir, &state.key)?.context("new capture session is busy")?;
        let session = Self { dir, state, _lock: lock };
        session.save()?;
        session.event("capture_started");
        Ok(session)
    }

    pub fn open(dir: &Path) -> Result<Option<Self>> {
        let state = State::read(dir)?;
        let Some(lock) = SessionLock::acquire(&state.lock_dir, &state.key)? else { return Ok(None); };
        let current = State::read(dir)?;
        if current.key != state.key || current.lock_dir != state.lock_dir {
            bail!("session identity changed while acquiring its lock");
        }
        Ok(Some(Self { dir: fs::canonicalize(dir)?, state: current, _lock: lock }))
    }

    pub fn reserve(&mut self, now: i64, explicit_resume: bool) -> Result<()> {
        if explicit_resume {
            self.state.attempts = 0;
            self.state.next_retry = 0;
            self.state.in_flight = false;
            self.state.phase = if self.state.meeting_id.is_some() { Phase::Archived } else { Phase::Ready };
        } else if self.state.in_flight {
            self.failed("interrupted_attempt", now)?;
            bail!("abandoned attempt charged; retry deadline preserved");
        }
        if !self.state.due(now) { bail!("session is not eligible for an automatic attempt"); }
        self.state.attempts += 1;
        self.state.in_flight = true;
        self.state.error_class = None;
        self.save()?;
        self.event("attempt_reserved");
        Ok(())
    }

    pub async fn archive(&mut self) -> Result<crate::db::Db> {
        if let Some(identity) = &self.state.archive_id {
            match crate::db::Db::open_existing(&self.state.db_path, identity).await {
                Ok(db) => Ok(db),
                Err(error) => {
                    self.needs_attention("archive_identity")?;
                    Err(error.context("bound archive unavailable; no database was created"))
                }
            }
        } else {
            let mut db = crate::db::Db::open(&self.state.db_path).await?;
            let identity = db.archive_id().await?;
            self.state.archive_id = Some(identity);
            self.save()?; // Binding precedes every possible keyed commit.
            self.event("archive_bound");
            Ok(db)
        }
    }

    pub fn committed(&mut self, meeting_id: i64) -> Result<()> {
        self.state.meeting_id = Some(meeting_id);
        self.state.phase = Phase::Archived;
        self.save()?;
        self.event("archive_committed");
        Ok(())
    }

    /// Names alone do not prove ownership of nested/unknown material. Only exact capture
    /// files and, for archived sessions, the independently verified root export pair may go.
    fn capture_files_owned(&self, exports: bool) -> Result<bool> {
        if self.state.rolls.is_empty() { return Ok(false); }
        let mut known = std::collections::HashSet::from([STATE_FILE.to_owned()]);
        if exports { known.extend(["transcript.md".into(), "transcript.json".into()]); }
        for roll in &self.state.rolls {
            for (channel, frames) in [("mic", roll.mic_frames), ("system", roll.system_frames)] {
                let name = roll_name(channel, roll.index); let path = self.dir.join(&name);
                let offset = committed_data_offset(&path, roll.sample_rate, frames)?;
                let reader = hound::WavReader::open(&path)?; let spec = reader.spec();
                if spec.channels != 1 || spec.sample_rate != roll.sample_rate || spec.bits_per_sample != 32
                    || spec.sample_format != hound::SampleFormat::Float || u64::from(reader.len()) != frames
                    || frames.checked_mul(4).and_then(|bytes| offset.checked_add(bytes)) != Some(fs::metadata(&path)?.len()) {
                    return Ok(false);
                }
                known.insert(name);
            }
        }
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() || !entry.file_name().to_str().is_some_and(|name| known.contains(name)) { return Ok(false); }
        }
        Ok(true)
    }

    pub fn prunable_short(&self) -> Result<bool> {
        if self.state.phase != Phase::SkippedShort || self.state.capture_status != CaptureStatus::Complete
            || self.state.unarchived_manual_output || self.state.archive_id.is_some()
            || self.state.meeting_id.is_some() || self.state.publication.is_some() { return Ok(false); }
        self.capture_files_owned(false)
    }

    /// Retention proof is read-only and never initializes a missing database.
    /// The caller holds this session's stable lock through any subsequent deletion.
    pub fn proof(&self, scope: ProofScope) -> Result<bool> {
        if self.state.phase != Phase::Complete
            || scope == ProofScope::Retention && self.state.unarchived_manual_output { return Ok(false); }
        let (Some(archive), Some(id), Some(publication)) = (&self.state.archive_id, self.state.meeting_id, &self.state.publication) else { return Ok(false); };
        if scope == ProofScope::Retention && !self.capture_files_owned(true)? { return Ok(false); }
        let runtime = crate::new_runtime()?;
        let stored = runtime.block_on(async {
            let mut db = crate::db::Db::open_existing(&self.state.db_path, archive).await?;
            if db.ingest_id(&self.state.key).await? != Some(id) { return anyhow::Ok(None); }
            let Some(row) = db.get_meeting(id).await? else { return Ok(None); };
            let segments = db.load_segments(id).await?;
            if segments.len() as i64 != row.segment_count { return Ok(None); }
            let vocabulary = db.load_enabled_vocab().await?;
            let people = db.list_speakers().await?;
            db.close().await?;
            Ok(Some((row, segments, vocabulary, people)))
        })?;
        let Some((row, segments, vocabulary, people)) = stored else { return Ok(false); };
        let (vocabulary, _) = crate::render::Vocab::compile(&vocabulary);
        let rendered = crate::render::render(&segments, &crate::render::IdentityMap::from_db(&people), &vocabulary);
        let expected_md = crate::export::to_markdown(&row, &rendered);
        let expected_json = crate::export::to_json(&rendered)?;
        let matches = |dir: &Path| -> bool {
            fs::read(dir.join("transcript.md")).is_ok_and(|bytes| bytes == expected_md.as_bytes())
                && fs::read(dir.join("transcript.json")).is_ok_and(|bytes| bytes == expected_json.as_bytes())
        };
        if !matches(publication) { return Ok(false); }
        if scope == ProofScope::Retention && publication != &self.state.export_dir
            && (self.state.export_dir.join("transcript.md").exists() || self.state.export_dir.join("transcript.json").exists())
            && !matches(&self.state.export_dir) { return Ok(false); }
        Ok(true)
    }

    pub fn complete(&mut self, publication: &Path) -> Result<()> {
        let markdown = publication.join("transcript.md");
        let json_path = publication.join("transcript.json");
        if fs::metadata(markdown)?.len() == 0
            || !serde_json::from_reader::<_, serde_json::Value>(File::open(json_path)?)?.is_array() {
            bail!("selected publication is not a valid export pair");
        }
        if self.state.meeting_id.is_none() { bail!("completion requires a durable meeting"); }
        self.state.publication = Some(publication.to_path_buf());
        self.state.phase = Phase::Complete;
        self.state.in_flight = false;
        self.state.next_retry = 0;
        self.state.error_class = None;
        self.save()?;
        self.event("session_complete");
        Ok(())
    }

    pub fn protect_outputs(&mut self) -> Result<()> {
        self.state.unarchived_manual_output = true;
        self.save()
    }

    pub fn needs_attention(&mut self, class: &'static str) -> Result<()> {
        self.state.phase = Phase::NeedsAttention;
        self.state.in_flight = false;
        self.state.error_class = Some(class.into());
        self.save()?;
        self.event("attention_required");
        Ok(())
    }

    pub fn failed(&mut self, class: &'static str, now: i64) -> Result<()> {
        self.state.in_flight = false;
        if self.state.phase != Phase::NeedsAttention || self.state.error_class.is_none() {
            self.state.error_class = Some(class.into());
        }
        if self.state.phase != Phase::NeedsAttention {
            if self.state.attempts >= 4 {
                self.state.phase = Phase::NeedsAttention;
            } else {
                self.state.phase = if self.state.meeting_id.is_some() { Phase::Archived } else { Phase::Ready };
                let delay = [60, 300, 1800][self.state.attempts.saturating_sub(1).min(2) as usize];
                self.state.next_retry = now.saturating_add(delay);
            }
        }
        self.save()?;
        self.event("attempt_failed");
        Ok(())
    }

    fn save(&self) -> Result<()> {
        self.state.validate()?;
        atomic_json(&self.dir.join(STATE_FILE), &self.state)
    }

    fn event(&self, event: &'static str) {
        crate::diagnostics::event("recovery", event, &self.state.key, &format!("{:?}", self.state.phase), self.state.attempts,
            crate::now_epoch().saturating_sub(self.state.started_at).saturating_mul(1000), self.state.error_class.as_deref());
    }

    pub fn start_roll(&mut self, index: u32, sample_rate: u32, gap_seconds: f64) -> Result<()> {
        if self.state.phase != Phase::Recording || index as usize != self.state.rolls.len() {
            bail!("invalid capture roll transition");
        }
        self.state.rolls.push(Roll { index, sample_rate, gap_seconds, mic_frames: 0, system_frames: 0 });
        self.save()?;
        self.event("roll_started");
        Ok(())
    }

    pub fn checkpoint(&mut self, index: u32, mic_frames: usize, system_frames: usize) -> Result<()> {
        let roll = self.state.rolls.last_mut().context("checkpoint before capture roll")?;
        if roll.index != index || (mic_frames as u64) < roll.mic_frames || (system_frames as u64) < roll.system_frames {
            bail!("checkpoint moved backwards or crossed a roll");
        }
        roll.mic_frames = mic_frames as u64;
        roll.system_frames = system_frames as u64;
        self.save()?;
        self.event("checkpoint_committed");
        Ok(())
    }

    pub fn finish_capture(&mut self, seconds: f64, minimum: f64) -> Result<()> {
        self.state.duration_seconds = seconds;
        self.state.capture_status = CaptureStatus::Complete;
        self.state.phase = if seconds < minimum { Phase::SkippedShort } else { Phase::Ready };
        self.save()?;
        self.event("capture_finalized");
        Ok(())
    }

    /// Validate every committed roll. Repair only into fresh copies, preserving original tails.
    pub fn inputs(&mut self) -> Result<PathBuf> {
        if !self.state.rolls.iter().any(|roll| roll.mic_frames > 0 || roll.system_frames > 0) {
            bail!("recording has no trusted nonzero checkpoint; files preserved");
        }
        let mut copy_required = false;
        let mut sources = Vec::new();
        for roll in &self.state.rolls {
            for (channel, frames) in [("mic", roll.mic_frames), ("system", roll.system_frames)] {
                let name = roll_name(channel, roll.index);
                let path = self.dir.join(&name);
                let offset = if frames == 0 && !path.exists() { None } else {
                    Some(committed_data_offset(&path, roll.sample_rate, frames)?)
                };
                let exact = hound::WavReader::open(&path).is_ok_and(|reader| {
                    let spec = reader.spec();
                    spec.channels == 1 && spec.sample_rate == roll.sample_rate
                        && spec.bits_per_sample == 32 && spec.sample_format == hound::SampleFormat::Float
                        && u64::from(reader.len()) == frames
                });
                copy_required |= !exact;
                sources.push((path, name, roll.sample_rate, frames, offset));
            }
        }
        if self.state.capture_status == CaptureStatus::Unknown {
            self.state.capture_status = CaptureStatus::Partial;
        }
        if self.state.phase == Phase::Recording { self.state.phase = Phase::Ready; }
        self.save()?;
        if !copy_required { return Ok(self.dir.clone()); }
        let root = self.dir.join(".inputs");
        private_dir(&root)?;
        let destination = root.join(new_identity());
        fs::DirBuilder::new().mode(0o700).create(&destination)?;
        for (path, name, sample_rate, frames, offset) in sources {
            let spec = hound::WavSpec { channels: 1, sample_rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
            let file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(destination.join(name))?;
            let mut writer = hound::WavWriter::new(std::io::BufWriter::new(file), spec)?;
            if let Some(offset) = offset {
                use std::io::{Read, Seek};
                let mut source = std::io::BufReader::new(File::open(path)?);
                source.seek(std::io::SeekFrom::Start(offset))?;
                for _ in 0..frames {
                    let mut bytes = [0u8; 4];
                    source.read_exact(&mut bytes)?;
                    writer.write_sample(f32::from_le_bytes(bytes))?;
                }
            }
            writer.finalize()?;
        }
        self.event("committed_inputs_recovered");
        Ok(destination)
    }

    pub fn capture_failed(&mut self) -> Result<()> {
        self.state.phase = Phase::NeedsAttention;
        self.state.error_class = Some("capture_io".into());
        self.save()?;
        self.event("capture_failed");
        Ok(())
    }
}

pub(crate) fn roll_name(channel: &str, index: u32) -> String {
    if index == 0 { format!("{channel}.wav") } else { format!("{channel}.{index:03}.wav") }
}

// RIFF chunk boundaries, not a guessed universal WAV header offset. Only our mono
// float format with a validated metadata checkpoint is eligible for this reader.
fn committed_data_offset(path: &Path, rate: u32, frames: u64) -> Result<u64> {
    use std::io::{Read, Seek, SeekFrom};
    if !fs::symlink_metadata(path)?.file_type().is_file() { bail!("capture channel is not a regular file"); }
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    let mut riff = [0u8; 12];
    file.read_exact(&mut riff)?;
    if &riff[..4] != b"RIFF" || &riff[8..] != b"WAVE" { bail!("invalid captured RIFF format"); }
    let mut format_valid = false;
    loop {
        let mut chunk = [0u8; 8];
        file.read_exact(&mut chunk).context("missing captured WAV data chunk")?;
        let size = u32::from_le_bytes(chunk[4..8].try_into()?) as u64;
        let start = file.stream_position()?;
        if &chunk[..4] == b"data" {
            let bytes = frames.checked_mul(4).context("checkpoint size overflow")?;
            if !format_valid || start.checked_add(bytes).is_none_or(|end| end > length) {
                bail!("captured WAV format/prefix does not match checkpoint");
            }
            return Ok(start);
        }
        let end = start.checked_add(size).and_then(|end| end.checked_add(size % 2)).context("WAV chunk overflow")?;
        if end > length { bail!("captured WAV chunk exceeds file size"); }
        if &chunk[..4] == b"fmt " {
            if !(16..=4096).contains(&size) { bail!("invalid captured format chunk"); }
            let mut format = vec![0u8; size as usize];
            file.read_exact(&mut format)?;
            let tag = u16::from_le_bytes(format[0..2].try_into()?);
            let float_format = tag == 3 || (tag == 0xfffe && format.len() >= 40
                && format[24..40] == [3, 0, 0, 0, 0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113]);
            format_valid = float_format && u16::from_le_bytes(format[2..4].try_into()?) == 1
                && u32::from_le_bytes(format[4..8].try_into()?) == rate
                && u16::from_le_bytes(format[12..14].try_into()?) == 4
                && u16::from_le_bytes(format[14..16].try_into()?) == 32;
        }
        file.seek(SeekFrom::Start(end))?;
    }
}
