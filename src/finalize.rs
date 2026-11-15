//! `meetscribe finalize <id>… | --all-live` — replace a live-stored meeting's transcript with the
//! batch pass's, under the same meeting id.
//!
//! A meeting stored from its live stream (`transcribe --promote-live`) has the words and no
//! voices. This command runs the same pass the daemon runs after a call (whisper, the diarizer,
//! the speaker embeddings), replaces the meeting's rows in one transaction and then replaces the
//! published pair in place. It is run by hand, in the foreground, and takes about as long as the
//! pass does.
//!
//! Two rules shape everything here:
//!
//! - **Before its transaction begins, it writes nothing under the session directory.** It holds
//!   the session lock, leaves the session `Complete` and reserves no attempt, checks the audio
//!   without saving the state, and gives the diarizer a directory under `<base>/scratch`. A run
//!   that is refused, fails or is killed during the pass has changed no session file and no row,
//!   so the daemon has nothing to recover. (`--overwrite` is the one exception, and it is asked
//!   for: it first records the files it is about to replace.)
//! - **A published file is replaced only if this tool wrote it.** Promotion records the sha256 of
//!   the pair it publishes in the session state ([`Provisional`]). A file whose hash is on record
//!   is replaced; a file someone edited is left alone until `--overwrite`.
//!
//! After the commit the database is right and the files are not yet. A run that stops there is
//! finished by `meetscribe finalize <id>` again: it finds a `batch` row with a record still in the
//! state and only publishes, without reading any audio.

use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::recovery::{Phase, Provisional, Session};
use crate::{db, export, pipeline, recovery, render, status, voices};

/// How long the replacement waits for another process's database lock, at `BEGIN` and at `COMMIT`.
/// SQLite does the waiting; there is no retry loop here.
const BUSY_TIMEOUT: Duration = Duration::from_secs(60);

const USAGE: &str = "usage: meetscribe finalize <id>… | --all-live [--db <path>] [--base <dir>] [--session <dir>]\n\
    \x20                          [--drop-names] [--overwrite] [--lang <code>] [--model <ggml.bin>]\n\
    \x20                          [--speaker-model <onnx>] [--diarizer-model <gguf>] [--diarizer-bin <exe>] [--no-split]\n\
    \n\
    Replaces a live-stored meeting's transcript with the full pass over its recording (whisper,\n\
    voices, names), under the same meeting id. Takes about as long as that pass; run it with no\n\
    call in progress. Nothing is changed before the pass has finished.\n\
    \n\
    \x20 --all-live      every live-stored meeting, oldest first\n\
    \x20 --session <dir> the meeting's session directory, when its stored path is relative\n\
    \x20 --drop-names    replace also when a voice of the meeting was named by hand (the names and\n\
    \x20                 their voiceprints are deleted)\n\
    \x20 --overwrite     replace the published transcript files as they are now, also when they were\n\
    \x20                 edited or no record says meetscribe wrote them\n\
    \x20 --base <dir>    where config.toml, the models and status.json are (default ~/.meetscribe)";

#[derive(Debug, Default)]
struct Args {
    ids: Vec<i64>,
    all_live: bool,
    db_path: PathBuf,
    base: PathBuf,
    session: Option<PathBuf>,
    drop_names: bool,
    overwrite: bool,
    lang: Option<String>,
    model: Option<PathBuf>,
    speaker_model: Option<PathBuf>,
    diarizer_model: Option<PathBuf>,
    diarizer_bin: Option<PathBuf>,
    no_split: bool,
    diarizer_rttm: Option<PathBuf>,
}

/// A usage error: the problem and the usage are said on stderr, and the error itself stays short.
/// The log files an error under a class chosen from its words, so the usage text (which names
/// models) must not ride in it.
fn usage(problem: &str) -> anyhow::Error {
    eprintln!("finalize: {problem}\n\n{USAGE}");
    anyhow::anyhow!("finalize: invalid arguments")
}

fn parse(argv: &[String]) -> Result<Option<Args>> {
    let mut args = Args { db_path: crate::default_db_path(), ..Args::default() };
    let mut base: Option<PathBuf> = None;
    let mut it = argv.iter();
    let value = |it: &mut std::slice::Iter<'_, String>, flag: &str| -> Result<String> {
        it.next().cloned().ok_or_else(|| usage(&format!("{flag} needs a value")))
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--all-live" => args.all_live = true,
            "--drop-names" => args.drop_names = true,
            "--overwrite" => args.overwrite = true,
            "--no-split" => args.no_split = true,
            "--db" => args.db_path = value(&mut it, "--db")?.into(),
            "--base" => base = Some(value(&mut it, "--base")?.into()),
            "--session" => args.session = Some(value(&mut it, "--session")?.into()),
            "--model" | "-m" => args.model = Some(value(&mut it, "--model")?.into()),
            "--speaker-model" => args.speaker_model = Some(value(&mut it, "--speaker-model")?.into()),
            "--diarizer-model" => args.diarizer_model = Some(value(&mut it, "--diarizer-model")?.into()),
            "--diarizer-bin" => args.diarizer_bin = Some(value(&mut it, "--diarizer-bin")?.into()),
            "--diarizer-rttm" => args.diarizer_rttm = Some(value(&mut it, "--diarizer-rttm")?.into()),
            "--lang" | "-l" => args.lang = Some(value(&mut it, "--lang")?),
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return Ok(None);
            }
            id if !id.starts_with('-') => args.ids.push(id.parse().map_err(|_| usage(&format!("'{id}' is not a meeting id")))?),
            other => return Err(usage(&format!("unknown flag '{other}'"))),
        }
    }
    if args.all_live != args.ids.is_empty() {
        return Err(usage("pass one or more meeting ids, or --all-live"));
    }
    if args.session.is_some() && args.ids.len() != 1 {
        return Err(usage("--session goes with exactly one meeting id"));
    }
    args.base = match base {
        Some(base) => base,
        None => crate::base_dir().context("finalize requires an absolute HOME, or --base")?,
    };
    Ok(Some(args))
}

/// The options of the pass: the ones the daemon's own transcription of this session would run
/// with (`[daemon] lang` and `[speakers] split` from `<base>/config.toml`, the installed models
/// under `<base>`), with this command's flags on top. A test pins them to `daemon::job_for`.
fn pass_options(args: &Args, state: &recovery::State) -> pipeline::PipelineOpts {
    let config = crate::config::Config::load(&args.base);
    let installed = |given: &Option<PathBuf>, relative: &str| given.clone().unwrap_or_else(|| args.base.join(relative));
    pipeline::PipelineOpts {
        model: installed(&args.model, "models/ggml-large-v3.bin"),
        lang: args.lang.clone().unwrap_or(config.daemon.lang),
        title: Some(state.title.clone()),
        db_path: state.db_path.clone(),
        export_dir: state.export_dir.clone(),
        no_store: false,
        session_key: Some(state.key.clone()),
        resume: false,
        speaker_model: Some(installed(&args.speaker_model, crate::SPEAKER_MODEL_REL)),
        split: config.speakers.split && !args.no_split,
        diarizer_bin: installed(&args.diarizer_bin, crate::DIARIZER_BIN_REL),
        diarizer_model: installed(&args.diarizer_model, crate::DIARIZER_MODEL_REL),
        diarizer_rttm: args.diarizer_rttm.clone(),
    }
}

/// What one meeting's run came to. Every variant but the error leaves the meeting consistent.
enum Outcome {
    /// The batch pass ran and replaced the live transcript.
    Finalized { segments: i64, clusters: usize, minutes: f64, dir: PathBuf },
    /// The row was already the batch transcript and only the files were still to publish.
    Published { dir: PathBuf },
    /// Nothing to do: the stored transcript is the batch one and its files are published.
    AlreadyFull { stale_files: Option<PathBuf> },
    /// The recording was pruned; the live transcript is the final one.
    RecordingGone,
}

/// `meetscribe finalize …`: one line per meeting on stdout (never the transcript), the reason of
/// each failure on stderr, and an error at the end if any meeting failed.
pub(crate) fn run(argv: &[String]) -> Result<()> {
    let Some(args) = parse(argv)? else { return Ok(()) };
    ensure!(args.db_path.exists(), "db {} does not exist", args.db_path.display());
    let ids = if args.all_live { live_meetings(&args.db_path)? } else { args.ids.clone() };
    if ids.is_empty() {
        println!("no live-stored meeting in {}", args.db_path.display());
        return Ok(());
    }
    let mut failed = 0usize;
    for id in &ids {
        let mut key = None;
        match finalize_meeting(&args, *id, &mut key) {
            Ok(Outcome::Finalized { segments, clusters, minutes, dir }) =>
                println!("meeting {id}: finalized — {segments} segments, {clusters} voice(s), {minutes:.1} min — {}", dir.display()),
            Ok(Outcome::Published { dir }) => println!("meeting {id}: already replaced; its files are now published — {}", dir.display()),
            Ok(Outcome::AlreadyFull { stale_files: None }) => println!("meeting {id}: already the full transcript"),
            Ok(Outcome::AlreadyFull { stale_files: Some(dir) }) => println!(
                "meeting {id}: already the full transcript; the files in {} are not its current render (`meetscribe rerender {id} --write` rewrites them)", dir.display()),
            Ok(Outcome::RecordingGone) if args.all_live => println!("meeting {id}: skipped — the recording is gone; the live transcript is final"),
            Ok(Outcome::RecordingGone) => {
                failed += 1;
                eprintln!("meeting {id}: the recording is gone; the live transcript is final");
            }
            Err(error) => {
                failed += 1;
                // The log carries a class only; the reason is said here, where the command was typed.
                eprintln!("meeting {id}: {error:#}");
                event("finalize_failed", key.as_deref(), serde_json::json!({"outcome": "error", "error_class": crate::diagnostics::error_class(&error)}));
            }
        }
    }
    ensure!(failed == 0, "finalize: {failed} of {} meeting(s) not finalized", ids.len());
    Ok(())
}

fn event(name: &'static str, session: Option<&str>, mut fields: serde_json::Value) {
    fields["component"] = serde_json::json!("pipeline");
    fields["event"] = serde_json::json!(name);
    if let Some(session) = session { fields["session_id"] = serde_json::json!(session); }
    log::info!(target: "meetscribe::event", "{fields}");
}

fn live_meetings(db_path: &Path) -> Result<Vec<i64>> {
    crate::new_runtime()?.block_on(async {
        let mut database = db::Db::open_read_only(db_path).await?;
        let ids = database.live_meeting_ids().await?;
        database.close().await?;
        Ok(ids)
    })
}

/// Where a meeting's session is, judged from its row alone. `source_dir` is historical data: a
/// path as it was spelled on the command line that stored the meeting.
pub(crate) enum Located {
    Session(PathBuf),
    /// An absolute directory that no longer exists: retention pruned the recording.
    Gone,
    /// A relative path: only `--session` can say where it is.
    Relative,
    /// A directory with no session state in it (a meeting transcribed from plain files).
    Unmanaged,
}

pub(crate) fn locate_session(row: &db::MeetingRow, given: Option<&Path>) -> Located {
    if let Some(dir) = given { return Located::Session(dir.to_path_buf()); }
    let dir = PathBuf::from(&row.source_dir);
    if !dir.is_absolute() { return Located::Relative; }
    if std::fs::symlink_metadata(&dir).is_err() { return Located::Gone; }
    if std::fs::symlink_metadata(dir.join(recovery::STATE_FILE)).is_ok() { Located::Session(dir) } else { Located::Unmanaged }
}

/// What the database says about a meeting and the session that claims it, read without writing.
struct Facts {
    archive: String,
    ingest: Option<i64>,
    row: Option<db::MeetingRow>,
    manual_names: i64,
}

/// One read-only look at the database: it is never created, never migrated, and a journal left by
/// a killed writer makes this fail instead of being replayed by a command that may still refuse.
fn facts(db_path: &Path, id: i64, key: Option<&str>) -> Result<Facts> {
    crate::new_runtime()?.block_on(async {
        let mut database = db::Db::open_read_only(db_path).await?;
        let facts = Facts {
            archive: database.archive_id().await?,
            ingest: match key { Some(key) => database.ingest_id(key).await?, None => None },
            row: database.get_meeting(id).await?,
            manual_names: database.manual_names(id).await?,
        };
        database.close().await?;
        anyhow::Ok(facts)
    }).with_context(|| format!("read {} without writing to it (if a command was killed while writing, one `meetscribe list --db {}` repairs the file)",
        db_path.display(), db_path.display()))
}

/// Take a session's lock without waiting. A held lock is another meetscribe command at work on
/// that session; the words avoid "busy", which the log would count as the daemon's own retry.
pub(crate) fn lock_session(dir: &Path, id: i64) -> Result<Session> {
    Session::open(dir)
        .with_context(|| format!("read the session state in {}", dir.display()))?
        .with_context(|| format!("the session of meeting {id} is in use by another meetscribe command; nothing was changed"))
}

/// Refuse a session that is not this meeting's: its state must name the meeting, `db_path` must be
/// the database it is bound to (`archive` is that database's identity), and the database must map
/// the session's key to the meeting (`ingest`).
pub(crate) fn ensure_meeting_session(session: &Session, id: i64, db_path: &Path, archive: &str, ingest: Option<i64>) -> Result<()> {
    let same_database = recovery::absolute(db_path)? == recovery::absolute(&session.state.db_path)?;
    ensure!(session.state.meeting_id == Some(id) && ingest == Some(id) && same_database && session.state.archive_id.as_deref() == Some(archive),
        "{} does not hold the session of meeting {id} in {}; nothing was changed (--session names the right directory)",
        session.dir.display(), db_path.display());
    Ok(())
}

/// The locked session behind a meeting, checked to be that meeting's; `Err` says why the row
/// leads to no session.
fn open_meeting_session(db_path: &Path, id: i64, row: &db::MeetingRow, given: Option<&Path>) -> Result<Result<Session, Located>> {
    let dir = match locate_session(row, given) {
        Located::Session(dir) => dir,
        other => return Ok(Err(other)),
    };
    let session = lock_session(&dir, id)?;
    let seen = facts(db_path, id, Some(&session.state.key))?;
    ensure_meeting_session(&session, id, db_path, &seen.archive, seen.ingest)?;
    Ok(Ok(session))
}

/// The meeting behind a session whose `finalize` committed and has not published yet: the state
/// still carries the record, and the database already says `batch`. `transcribe --resume` and the
/// daemon's own visit ask this before they reserve anything, because recovery would publish that
/// meeting a second time under `.recovered/` and leave the provisional files where they are.
///
/// Read-only and silent: no migration, no state save, and any failure to read (an archive that is
/// missing, foreign or older) is "no": the caller then does exactly what it did before.
pub(crate) fn unfinished(state: &recovery::State) -> Option<i64> {
    state.provisional()?;
    let archive = state.archive_id.as_deref()?;
    crate::new_runtime().ok()?.block_on(async {
        let mut database = db::Db::open_read_only(&state.db_path).await.ok()?;
        let found = async {
            if database.archive_id().await.ok()? != archive { return None; }
            let id = database.ingest_id(&state.key).await.ok()??;
            (database.recorded_origin(id).await.ok()? == Some(db::TranscriptOrigin::Batch)).then_some(id)
        }.await;
        let _ = database.close().await;
        found
    })
}

fn finalize_meeting(args: &Args, id: i64, key: &mut Option<String>) -> Result<Outcome> {
    // Section 1 of the design: find the session and check everything that can refuse, with no
    // write to any row and to no file under the session directory.
    let row = facts(&args.db_path, id, None)?.row.with_context(|| format!("no meeting with id {id} in {}", args.db_path.display()))?;
    let mut session = match open_meeting_session(&args.db_path, id, &row, args.session.as_deref())? {
        Ok(session) => session,
        // A row can go from `live` to `batch` and never back, so a `batch` read here stays true.
        Err(_) if row.origin == db::TranscriptOrigin::Batch => return Ok(Outcome::AlreadyFull { stale_files: None }),
        Err(Located::Gone) => return Ok(Outcome::RecordingGone),
        Err(Located::Relative) => bail!("meeting {id} was stored from a relative path ('{}'); name its session directory with --session", row.source_dir),
        Err(_) => bail!("'{}' holds no session of meeting {id}; name its session directory with --session", row.source_dir),
    };
    *key = Some(session.state.key.clone());
    ensure!(session.state.phase == Phase::Complete,
        "the session of meeting {id} is not complete; finish it with `meetscribe transcribe {} --resume` first", session.dir.display());
    // Under the lock now: this read of the origin is the one that decides.
    let seen = facts(&args.db_path, id, None)?;
    let row = seen.row.with_context(|| format!("meeting {id} vanished from {}", args.db_path.display()))?;
    // Only a record for the directory the state says is published counts as one.
    let record = session.state.provisional().filter(|record| session.state.publication.as_deref() == Some(record.dir.as_path()));

    if row.origin == db::TranscriptOrigin::Batch {
        // Already replaced. With a record, an earlier run stopped after its commit: publish.
        return match record {
            Some(record) => {
                let record = if args.overwrite { adopt(&mut session, id)? } else { record };
                publish(&mut session, id, record).map(|(_, _, dir)| Outcome::Published { dir })
            }
            None => Ok(Outcome::AlreadyFull { stale_files: stale_publication(&session, id)? }),
        };
    }

    ensure!(!status::call_is_live(&args.base), "a call is being recorded; run `meetscribe finalize {id}` when it has ended");
    let scratch = safe_scratch(&args.base, &session)?;
    ensure!(args.drop_names || seen.manual_names == 0,
        "meeting {id} has {} manual name(s) on its voices; nothing was changed (--drop-names deletes them and their voiceprints)", seen.manual_names);
    if !args.overwrite { own_files(&session, record.as_ref(), id)?; }
    ensure!(session.inputs_intact()?,
        "the recording of meeting {id} is not exactly what its session recorded as committed; finalize transcribes only an intact recording");
    let opts = pass_options(args, &session.state);

    // The one write before the transaction, and only when asked for by flag.
    let record = match record {
        _ if args.overwrite => adopt(&mut session, id)?,
        Some(record) => record,
        None => bail!("meeting {id} has no record of its published files"), // `own_files` refused this above
    };

    // Section 2: the pass, with the lock held, no attempt reserved and no write in the session.
    event("finalize_started", key.as_deref(), serde_json::json!({"outcome": "ok"}));
    yield_to_calls(&args.base);
    let mut decoded = {
        let work = WorkDir::create(&scratch, &session.state.key)?;
        pipeline::decode(&session.dir, &session.dir, work.path(), &opts, Some(&session.state))?
    };

    // Section 4: the pass took a while, so the files are checked again, then one transaction.
    own_files(&session, Some(&record), id)?;
    replace(&session, id, &mut decoded, args.drop_names)?;
    #[cfg(test)]
    pipeline::cutpoint("finalize_committed");

    // Section 5: the database is right; now the files.
    let (segments, clusters, dir) = publish(&mut session, id, record)?;
    Ok(Outcome::Finalized { segments, clusters, minutes: decoded.meeting_secs / 60.0, dir })
}

/// Drop to background priority while a call is recorded, as the daemon's own child does. Once per
/// process: `--all-live` runs many passes.
fn yield_to_calls(base: &Path) {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| status::spawn_call_yield(base.to_path_buf()));
}

/// `<base>/scratch`, safe to keep the diarizer's temporary audio in: a real directory (created
/// here when absent), not a symlink, and neither inside the session nor inside the publication
/// directory, so that nothing written or removed there can touch a session file.
fn safe_scratch(base: &Path, session: &Session) -> Result<PathBuf> {
    use std::os::unix::fs::DirBuilderExt;
    let base = std::fs::canonicalize(base).with_context(|| format!("--base {} is not a directory", base.display()))?;
    let scratch = base.join("scratch");
    let publication = session.state.publication.as_deref().map(|dir| std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf()));
    ensure!(!scratch.starts_with(&session.dir) && !publication.is_some_and(|dir| scratch.starts_with(dir)),
        "{} is inside the session of the meeting; pass a --base outside it", scratch.display());
    match std::fs::symlink_metadata(&scratch) {
        Ok(found) => ensure!(found.file_type().is_dir(),
            "{} is not a real directory (a symlink or a file is in its place); finalize keeps temporary audio there and does not follow it", scratch.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound =>
            std::fs::DirBuilder::new().mode(0o700).create(&scratch).with_context(|| format!("create {}", scratch.display()))?,
        Err(error) => return Err(error).with_context(|| format!("inspect {}", scratch.display())),
    }
    Ok(scratch)
}

/// One run's directory under the scratch place, removed when the run ends however it ends. A
/// killed run leaves it; the next run of the same meeting removes what it finds.
struct WorkDir(PathBuf);

impl WorkDir {
    fn create(scratch: &Path, key: &str) -> Result<Self> {
        use std::os::unix::fs::DirBuilderExt;
        let prefix = format!("finalize-{key}-");
        // The session lock is held, so no run of this meeting is alive: these are leftovers.
        // `DirEntry::file_type` does not follow a symlink, so only real directories are removed.
        for entry in std::fs::read_dir(scratch).with_context(|| format!("read {}", scratch.display()))? {
            let entry = entry?;
            if entry.file_name().to_str().is_some_and(|name| name.starts_with(&prefix)) && entry.file_type()?.is_dir() {
                std::fs::remove_dir_all(entry.path()).with_context(|| format!("remove {}", entry.path().display()))?;
            }
        }
        let path = scratch.join(format!("{prefix}{}", recovery::new_identity()));
        std::fs::DirBuilder::new().mode(0o700).create(&path).with_context(|| format!("create {}", path.display()))?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path { &self.0 }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn read_if_present(path: &Path) -> Result<Option<Vec<u8>>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("read {}", path.display())),
    }
}

/// Refuse unless both published files are the tool's own: each is read once and its hash looked
/// up in the record. An absent file is not a changed one; it is simply written later.
fn own_files(session: &Session, record: Option<&Provisional>, id: i64) -> Result<()> {
    let record = record.with_context(|| format!(
        "meeting {id} has no record of the transcript files meetscribe published for it (an earlier build stored it, or saved its \
         state since); nothing was changed. `meetscribe finalize {id} --overwrite` replaces the files as they are"))?;
    for (name, known) in [("transcript.md", &record.markdown), ("transcript.json", &record.json)] {
        let path = record.dir.join(name);
        if let Some(bytes) = read_if_present(&path)? {
            ensure!(known.contains(&export::sha256_hex(&bytes)),
                "{} was changed since meetscribe wrote it; nothing was replaced. `meetscribe finalize {id} --overwrite` replaces it as it is", path.display());
        }
    }
    if record.dir != session.state.export_dir && (session.state.export_dir.join("transcript.md").exists() || session.state.export_dir.join("transcript.json").exists()) {
        eprintln!("meeting {id}: the transcript files in {} are an older copy and are left as they are; the published ones are in {} \
            (`meetscribe rerender {id} --write` rewrites the older copy)", session.state.export_dir.display(), record.dir.display());
    }
    Ok(())
}

/// `--overwrite`: take the published files as they are now for the tool's own. The record is
/// saved before anything else happens, so a run that dies after its commit can still be finished.
fn adopt(session: &mut Session, id: i64) -> Result<Provisional> {
    let dir = session.state.publication.clone().with_context(|| format!("meeting {id} has no published files to replace"))?;
    let hash = |name: &str| -> Result<Option<String>> { Ok(read_if_present(&dir.join(name))?.map(|bytes| export::sha256_hex(&bytes))) };
    let record = Provisional::new(&dir, hash("transcript.md")?, hash("transcript.json")?);
    session.save_provisional(&record)?;
    Ok(record)
}

/// The transaction: name the voices, then replace the meeting's rows. After any error the
/// connection is closed, never reused, and whether the replacement happened is read on a new one,
/// because a `COMMIT` that reports an error may still have committed.
fn replace(session: &Session, id: i64, decoded: &mut pipeline::Decoded, drop_names: bool) -> Result<()> {
    let archive = session.state.archive_id.as_deref().context("the session is bound to no archive")?;
    let attempt: Result<()> = crate::new_runtime()?.block_on(async {
        let mut database = db::Db::open_existing_patient(&session.state.db_path, archive, BUSY_TIMEOUT).await?;
        let stored = store(&mut database, id, decoded, drop_names).await;
        let closed = database.close().await;
        stored.and(closed)
    });
    #[cfg(test)]
    let attempt = attempt.and_then(|()| if pipeline::fault("replace_reports_an_error") { Err(anyhow::anyhow!("synthetic failure reported after the commit")) } else { Ok(()) });
    let Err(error) = attempt else { return Ok(()) };
    match facts(&session.state.db_path, id, None) {
        Ok(Facts { row: Some(row), .. }) if row.origin == db::TranscriptOrigin::Batch => {
            log::warn!("component=pipeline event=finalize_commit_reported_an_error; the replacement is in the database");
            Ok(())
        }
        _ => Err(error),
    }
}

async fn store(database: &mut db::Db, id: i64, decoded: &mut pipeline::Decoded, drop_names: bool) -> Result<()> {
    let enrolled = voices::decode_enrolled(&database.load_enrolled().await?)?;
    let (_, split) = pipeline::name_voices(decoded, &enrolled)?;
    let (segments, assembly) = decoded.to_store(&split);
    database.replace_meeting_transcript(id, &decoded.meta, segments, &assembly.clusters, &assembly.voices, drop_names).await
}

/// The stored meeting as it publishes now: its row, its rendered pair, and how many voices it has.
fn stored_render(session: &Session, id: i64) -> Result<(db::MeetingRow, export::Pair, usize)> {
    let (row, segments, vocabulary, people, clusters) = crate::new_runtime()?.block_on(async {
        let mut database = db::Db::open_read_only(&session.state.db_path).await?;
        let row = database.get_meeting(id).await?.with_context(|| format!("meeting {id} vanished"))?;
        let segments = database.load_segments(id).await?;
        let vocabulary = database.load_enabled_vocab().await?;
        let people = database.list_speakers().await?;
        let clusters = database.list_clusters(id).await?.len();
        database.close().await?;
        anyhow::Ok((row, segments, vocabulary, people, clusters))
    })?;
    let (vocabulary, _) = render::Vocab::compile(&vocabulary);
    let rendered = render::render(&segments, &render::IdentityMap::from_db(&people), &vocabulary);
    Ok((row.clone(), export::Pair::render(&row, &rendered)?, clusters))
}

/// For a meeting that is already final: its publication directory when the files there are not
/// the render of the stored rows (someone edited them, or a rule changed since).
fn stale_publication(session: &Session, id: i64) -> Result<Option<PathBuf>> {
    let Some(dir) = session.state.publication.clone() else { return Ok(None) };
    let (_, pair, _) = stored_render(session, id)?;
    let current = read_if_present(&dir.join("transcript.md"))?.is_some_and(|bytes| bytes == pair.markdown.as_bytes())
        && read_if_present(&dir.join("transcript.json"))?.is_some_and(|bytes| bytes == pair.json.as_bytes());
    Ok((!current).then_some(dir))
}

/// After the commit: replace the published pair in place and finish the session state. Runs
/// again, from the top, after any stop: every step is safe to repeat.
///
/// The hashes of the new render are saved BEFORE either file is replaced. A run killed between
/// the two files leaves a file this tool wrote; without its hash on record the next run could
/// not tell it from a file someone edited (a vocabulary rule added in between changes the render).
fn publish(session: &mut Session, id: i64, mut record: Provisional) -> Result<(i64, usize, PathBuf)> {
    let (row, pair, clusters) = stored_render(session, id)?;
    let (markdown, json) = pair.hashes();
    ensure!(record.append(&markdown, &json),
        "meeting {id} was published {} times without finishing; `meetscribe finalize {id} --overwrite` starts again from the files as they are",
        recovery::PROVISIONAL_HASHES);
    session.save_provisional(&record)?;
    event("finalize_committed", Some(&session.state.key), serde_json::json!({"outcome": "ok", "segments": row.segment_count, "clusters": clusters}));
    #[cfg(test)]
    pipeline::cutpoint("finalize_hashes_saved");

    place(&record.dir.join("transcript.md"), &pair.markdown, &record.markdown, id)?;
    #[cfg(test)]
    pipeline::cutpoint("finalize_markdown_replaced");
    place(&record.dir.join("transcript.json"), &pair.json, &record.json, id)?;
    #[cfg(test)]
    pipeline::cutpoint("finalize_json_replaced");

    session.finalized(row.duration_secs, &record.dir)?;
    event("finalize_published", Some(&session.state.key), serde_json::json!({"outcome": "ok"}));
    Ok((row.segment_count, clusters, record.dir))
}

/// One published file against the render it should hold: already that render, absent, the tool's
/// own (replaced), or someone's (an error, and the file is left exactly as it is).
fn place(path: &Path, fresh: &str, known: &[String], id: i64) -> Result<()> {
    match read_if_present(path)? {
        Some(bytes) if bytes == fresh.as_bytes() => Ok(()),
        Some(bytes) if known.contains(&export::sha256_hex(&bytes)) => export::replace_file(path, fresh),
        Some(_) => bail!("{} was changed since meetscribe wrote it and was left as it is; the database already holds the full transcript. \
            `meetscribe finalize {id} --overwrite` replaces the file", path.display()),
        None => export::create_file(path, fresh),
    }
}
