//! Plaintext SQLite storage (Phase 3) — one meeting + its ordered segments per row set.
//!
//! Single `SqliteConnection` (this is a one-shot batch CLI, one writer, one transaction);
//! no pool. All async work runs inside a single `block_on` from the sync pipeline. File
//! perms are `0600` (encryption-at-rest is v1.1). The `speaker` column stores the lowercase
//! token from `Speaker::as_sql` and is `CHECK`-constrained to it.
//!
//! # Schema evolution
//!
//! `init_schema` is **FROZEN as the v0 baseline** and must never change again; every later
//! change is a rung on the `PRAGMA user_version` ladder in `apply_migrations`. Adding columns
//! to `init_schema` instead would kill every FRESH install with a duplicate-column error while
//! existing installs kept working — a failure invisible to any test that migrates a copy of a
//! real database.
//!
//! Ladder rules, learned the hard way:
//! - **`CREATE TABLE` before any `ALTER TABLE … REFERENCES` it.** With `foreign_keys(true)`, an
//!   `ADD COLUMN … REFERENCES missing_table(id)` *succeeds* and `SELECT`s keep working, but every
//!   subsequent `INSERT` fails at prepare with `no such table` — the daemon silently loses the
//!   ability to store meetings while `list` and `export` look healthy.
//! - **Every rung runs in one `BEGIN IMMEDIATE`** that re-reads `user_version` inside the
//!   transaction and writes the bump before `COMMIT`. `PRAGMA user_version` is transactional, so
//!   a half-applied migration cannot leave the version bumped (which would brick every later open
//!   with duplicate-column).
//! - **Additive only** — never a NOT NULL column, never a rename. That is what keeps an older
//!   binary able to read a newer database, since all reads select explicit columns.

use anyhow::{Context, Result, anyhow};
use sqlx::sqlite::{SqliteConnectOptions, SqliteConnection, SqliteRow};
use sqlx::{Connection, Row};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use crate::transcript::{Speaker, TranscriptSegment};

/// Metadata for one meeting (segment_count is derived from the segments at insert time).
#[derive(Debug, Clone)]
pub struct MeetingMeta {
    pub title: String,
    pub source_dir: String,
    pub model: String,
    pub lang: String,
    pub started_at: i64, // unix epoch seconds — real meeting time (earliest capture WAV mtime)
    pub duration_secs: f64,
    pub created_at: i64, // unix epoch seconds — ingest time
}

/// A persisted meeting row (what `list`/`export`/`get_meeting` read back).
#[derive(Debug, Clone)]
pub struct MeetingRow {
    pub id: i64,
    pub title: String,
    pub source_dir: String,
    pub model: String,
    pub lang: String,
    pub started_at: i64,
    pub duration_secs: f64,
    pub segment_count: i64,
    pub created_at: i64,
}

const MEETING_COLS: &str =
    "id, title, source_dir, model, lang, started_at, duration_secs, segment_count, created_at";

/// Schema version this binary knows how to produce. Bump with each new ladder rung.
const SCHEMA_VERSION: i32 = 1;

/// A stored segment: the raw ASR contract plus the row identity and the (nullable) speaker
/// identity. `speaker_id`/`voice_cluster` are always `None` until the speaker-ID work populates
/// them; they are carried now so widening this does not churn every read and export signature.
#[derive(Debug, Clone)]
pub struct StoredSegment {
    pub id: i64,
    pub speaker_id: Option<i64>,
    pub voice_cluster: Option<String>,
    pub seg: TranscriptSegment,
}

impl StoredSegment {
    /// Wrap a segment that has not been persisted (so it has no row id). Production code renders
    /// fresh transcripts through `render::render_fresh`; this exists for tests.
    #[cfg(test)]
    pub fn unstored(seg: TranscriptSegment) -> Self {
        Self { id: 0, speaker_id: None, voice_cluster: None, seg }
    }
}

/// One vocabulary-correction rule. Application order is `id` ascending, permanently — which is
/// why the CLI disables rather than deletes.
#[derive(Debug, Clone)]
pub struct VocabRow {
    pub id: i64,
    pub pattern: String,
    pub replacement: String,
    pub is_regex: bool,
    pub enabled: bool,
    pub created_at: i64,
}

pub struct Db {
    conn: SqliteConnection,
}

impl Db {
    /// Open (creating if missing) the DB at `path`, initialize the schema, and set `0600`.
    /// The parent directory is created and set to `0700` only when we create it (never
    /// re-permissioned if it already existed — so `--db ./x.db` won't chmod the cwd).
    pub async fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            let existed = parent.exists();
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create db dir {}", parent.display()))?;
            if !existed {
                let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
            }
        }
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .foreign_keys(true);
        let mut conn = SqliteConnection::connect_with(&opts)
            .await
            .with_context(|| format!("open sqlite db {}", path.display()))?;
        prepare_schema(&mut conn).await?;
        // 0600 on the db file. The transient rollback `-journal` inherits these perms.
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod 0600 {}", path.display()))?;
        Ok(Self { conn })
    }

    /// In-memory DB for tests. Routes through the IDENTICAL prepare path as `open` so tests
    /// exercise the real ladder rather than a hand-built schema.
    #[cfg(test)]
    pub async fn open_in_memory() -> Result<Self> {
        let opts = SqliteConnectOptions::new()
            .filename(":memory:")
            .create_if_missing(true)
            .foreign_keys(true);
        let mut conn = SqliteConnection::connect_with(&opts).await?;
        prepare_schema(&mut conn).await?;
        Ok(Self { conn })
    }

    /// The schema version currently on disk.
    #[cfg(test)]
    pub async fn schema_version(&mut self) -> Result<i32> {
        read_user_version(&mut self.conn).await
    }

    /// `PRAGMA table_info` as `(name, type, notnull, dflt, pk)` tuples — used to assert a fresh
    /// database and a migrated legacy one are structurally identical.
    #[cfg(test)]
    pub async fn table_info(&mut self, table: &str) -> Result<Vec<String>> {
        let rows = sqlx::query(&format!("PRAGMA table_info({table})"))
            .fetch_all(&mut self.conn)
            .await
            .with_context(|| format!("table_info({table})"))?;
        rows.iter()
            .map(|r| {
                Ok(format!(
                    "{}|{}|{}|{}",
                    r.try_get::<String, _>("name")?,
                    r.try_get::<String, _>("type")?,
                    r.try_get::<i64, _>("notnull")?,
                    r.try_get::<i64, _>("pk")?,
                ))
            })
            .collect()
    }

    /// Insert a meeting and its segments in one transaction; returns the meeting id.
    /// Zero segments is valid (a silent recording where VAD found no speech).
    pub async fn insert_meeting(
        &mut self,
        meta: &MeetingMeta,
        segs: &[TranscriptSegment],
    ) -> Result<i64> {
        let mut tx = self.conn.begin().await.context("begin insert tx")?;
        let res = sqlx::query(
            "INSERT INTO meetings \
             (title, source_dir, model, lang, started_at, duration_secs, segment_count, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(meta.title.as_str())
        .bind(meta.source_dir.as_str())
        .bind(meta.model.as_str())
        .bind(meta.lang.as_str())
        .bind(meta.started_at)
        .bind(meta.duration_secs)
        .bind(segs.len() as i64)
        .bind(meta.created_at)
        .execute(&mut *tx)
        .await
        .context("insert meeting")?;
        let meeting_id = res.last_insert_rowid();

        for (i, s) in segs.iter().enumerate() {
            sqlx::query(
                "INSERT INTO transcript_segments \
                 (meeting_id, seq, speaker, text, t_start, t_end, confidence) \
                 VALUES (?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(meeting_id)
            .bind(i as i64)
            .bind(s.speaker.as_sql())
            .bind(s.text.as_str())
            .bind(s.t_start)
            .bind(s.t_end)
            .bind(s.confidence)
            .execute(&mut *tx)
            .await
            .with_context(|| format!("insert segment {i}"))?;
        }

        tx.commit().await.context("commit insert tx")?;
        Ok(meeting_id)
    }

    /// All meetings, newest first.
    pub async fn list_meetings(&mut self) -> Result<Vec<MeetingRow>> {
        let sql = format!("SELECT {MEETING_COLS} FROM meetings ORDER BY started_at DESC, id DESC");
        let rows = sqlx::query(&sql)
            .fetch_all(&mut self.conn)
            .await
            .context("list meetings")?;
        rows.iter().map(meeting_from_row).collect()
    }

    pub async fn get_meeting(&mut self, id: i64) -> Result<Option<MeetingRow>> {
        let sql = format!("SELECT {MEETING_COLS} FROM meetings WHERE id = ?");
        let row = sqlx::query(&sql)
            .bind(id)
            .fetch_optional(&mut self.conn)
            .await
            .context("get meeting")?;
        match row {
            Some(r) => Ok(Some(meeting_from_row(&r)?)),
            None => Ok(None),
        }
    }

    /// Segments for a meeting, in stored order (`seq`).
    ///
    /// `text` is RAW ASR output and is never rewritten in place — presentation is a pure
    /// function of (raw, identity, vocab) applied by `render`.
    pub async fn load_segments(&mut self, meeting_id: i64) -> Result<Vec<StoredSegment>> {
        let rows = sqlx::query(
            "SELECT id, speaker, text, t_start, t_end, confidence \
             FROM transcript_segments WHERE meeting_id = ? ORDER BY seq",
        )
        .bind(meeting_id)
        .fetch_all(&mut self.conn)
        .await
        .context("load segments")?;

        let mut out = Vec::with_capacity(rows.len());
        for r in &rows {
            let sp: String = r.try_get("speaker")?;
            let speaker =
                Speaker::from_sql(&sp).ok_or_else(|| anyhow!("unknown speaker token '{sp}'"))?;
            out.push(StoredSegment {
                id: r.try_get("id")?,
                // Populated by the speaker-ID work (ladder rung 2); always None today.
                speaker_id: None,
                voice_cluster: None,
                seg: TranscriptSegment {
                    speaker,
                    text: r.try_get("text")?,
                    t_start: r.try_get("t_start")?,
                    t_end: r.try_get("t_end")?,
                    confidence: r.try_get("confidence")?,
                },
            });
        }
        Ok(out)
    }

    /// Add a correction rule; returns its id (which fixes its place in the application order).
    pub async fn add_vocab(
        &mut self,
        pattern: &str,
        replacement: &str,
        is_regex: bool,
        created_at: i64,
    ) -> Result<i64> {
        let res = sqlx::query(
            "INSERT INTO vocab_corrections (pattern, replacement, is_regex, enabled, created_at) \
             VALUES (?, ?, ?, 1, ?)",
        )
        .bind(pattern)
        .bind(replacement)
        .bind(i64::from(is_regex))
        .bind(created_at)
        .execute(&mut self.conn)
        .await
        .context("insert vocab correction")?;
        Ok(res.last_insert_rowid())
    }

    /// All correction rules, enabled or not, in application order.
    pub async fn list_vocab(&mut self) -> Result<Vec<VocabRow>> {
        self.query_vocab(false).await
    }

    /// Only the enabled rules, in application order — what `render` consumes.
    pub async fn load_enabled_vocab(&mut self) -> Result<Vec<VocabRow>> {
        self.query_vocab(true).await
    }

    async fn query_vocab(&mut self, enabled_only: bool) -> Result<Vec<VocabRow>> {
        let sql = if enabled_only {
            "SELECT id, pattern, replacement, is_regex, enabled, created_at \
             FROM vocab_corrections WHERE enabled = 1 ORDER BY id"
        } else {
            "SELECT id, pattern, replacement, is_regex, enabled, created_at \
             FROM vocab_corrections ORDER BY id"
        };
        let rows = sqlx::query(sql)
            .fetch_all(&mut self.conn)
            .await
            .context("list vocab corrections")?;
        rows.iter()
            .map(|r| {
                Ok(VocabRow {
                    id: r.try_get("id")?,
                    pattern: r.try_get("pattern")?,
                    replacement: r.try_get("replacement")?,
                    is_regex: r.try_get::<i64, _>("is_regex")? != 0,
                    enabled: r.try_get::<i64, _>("enabled")? != 0,
                    created_at: r.try_get("created_at")?,
                })
            })
            .collect()
    }

    /// Enable/disable a rule. `false` means no rule with that id. Rules are never deleted, so
    /// ids — and therefore the documented application order — stay stable forever.
    pub async fn set_vocab_enabled(&mut self, id: i64, enabled: bool) -> Result<bool> {
        let res = sqlx::query("UPDATE vocab_corrections SET enabled = ? WHERE id = ?")
            .bind(i64::from(enabled))
            .bind(id)
            .execute(&mut self.conn)
            .await
            .context("update vocab correction")?;
        Ok(res.rows_affected() > 0)
    }

    pub async fn close(self) -> Result<()> {
        self.conn.close().await.context("close sqlite connection")?;
        Ok(())
    }
}

fn meeting_from_row(r: &SqliteRow) -> Result<MeetingRow> {
    Ok(MeetingRow {
        id: r.try_get("id")?,
        title: r.try_get("title")?,
        source_dir: r.try_get("source_dir")?,
        model: r.try_get("model")?,
        lang: r.try_get("lang")?,
        started_at: r.try_get("started_at")?,
        duration_secs: r.try_get("duration_secs")?,
        segment_count: r.try_get("segment_count")?,
        created_at: r.try_get("created_at")?,
    })
}

/// Bring a connection's schema to `SCHEMA_VERSION`: the frozen v0 baseline, then the ladder.
async fn prepare_schema(conn: &mut SqliteConnection) -> Result<()> {
    init_schema(conn).await?;
    apply_migrations(conn).await
}

async fn read_user_version(conn: &mut SqliteConnection) -> Result<i32> {
    sqlx::query_scalar::<_, i32>("PRAGMA user_version")
        .fetch_one(&mut *conn)
        .await
        .context("read PRAGMA user_version")
}

/// Walk the ladder to `SCHEMA_VERSION`, one transactional rung at a time.
///
/// A database from a NEWER binary is left alone with a warning rather than failing: the ladder is
/// additive-only and every read selects explicit columns, so extra columns are harmless.
async fn apply_migrations(conn: &mut SqliteConnection) -> Result<()> {
    loop {
        let v = read_user_version(conn).await?;
        if v > SCHEMA_VERSION {
            log::warn!(
                "database schema v{v} is newer than this binary knows (v{SCHEMA_VERSION}) — \
                 continuing read-only-compatible; upgrade meetscribe if something looks missing"
            );
            return Ok(());
        }
        if v == SCHEMA_VERSION {
            return Ok(());
        }
        match v {
            0 => migrate_v0_to_v1(conn).await?,
            other => anyhow::bail!("no migration registered from schema v{other}"),
        }
    }
}

/// v0 → v1: add `vocab_corrections`.
///
/// Deliberately contains NO speaker tables — that rung ships with the speaker-ID work, so a
/// database never carries dead tables for a feature that might not pass its calibration gate.
async fn migrate_v0_to_v1(conn: &mut SqliteConnection) -> Result<()> {
    run_migration(conn, 0, 1, |c| {
        Box::pin(async move {
            sqlx::query(
                "CREATE TABLE IF NOT EXISTS vocab_corrections (\
                    id          INTEGER PRIMARY KEY AUTOINCREMENT, \
                    pattern     TEXT    NOT NULL, \
                    replacement TEXT    NOT NULL, \
                    is_regex    INTEGER NOT NULL DEFAULT 0 CHECK (is_regex IN (0,1)), \
                    enabled     INTEGER NOT NULL DEFAULT 1 CHECK (enabled  IN (0,1)), \
                    created_at  INTEGER NOT NULL)",
            )
            .execute(&mut *c)
            .await
            .context("create vocab_corrections table")?;
            Ok(())
        })
    })
    .await
}

/// Run one ladder rung inside `BEGIN IMMEDIATE`, re-checking the version inside the transaction
/// and bumping it before COMMIT.
///
/// `BEGIN IMMEDIATE` (not sqlx's deferred `begin()`) takes the write lock up front, so the
/// re-read below cannot race another process that migrates between our check and our first write.
/// Bumping `user_version` inside the same transaction is what makes a partial migration
/// impossible: without it, an applied DDL plus a lost version write bricks every later open.
async fn run_migration<F>(conn: &mut SqliteConnection, from: i32, to: i32, body: F) -> Result<()>
where
    F: for<'c> FnOnce(
        &'c mut SqliteConnection,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<()>> + Send + 'c>,
    >,
{
    sqlx::query("BEGIN IMMEDIATE")
        .execute(&mut *conn)
        .await
        .with_context(|| format!("begin migration v{from}→v{to}"))?;

    let res = async {
        // Another process may have won the race while we waited for the write lock.
        let current = read_user_version(conn).await?;
        if current != from {
            return Ok(false);
        }
        body(conn).await?;
        sqlx::query(&format!("PRAGMA user_version = {to}"))
            .execute(&mut *conn)
            .await
            .with_context(|| format!("bump user_version to {to}"))?;
        Ok(true)
    }
    .await;

    match res {
        Ok(applied) => {
            sqlx::query("COMMIT")
                .execute(&mut *conn)
                .await
                .with_context(|| format!("commit migration v{from}→v{to}"))?;
            if applied {
                log::info!("database schema migrated v{from} → v{to}");
            }
            Ok(())
        }
        Err(e) => {
            let _ = sqlx::query("ROLLBACK").execute(&mut *conn).await;
            Err(e)
        }
    }
}

/// The FROZEN v0 baseline. Never add to this — see the module docs; new schema goes on the ladder.
async fn init_schema(conn: &mut SqliteConnection) -> Result<()> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS meetings (\
            id            INTEGER PRIMARY KEY AUTOINCREMENT, \
            title         TEXT    NOT NULL, \
            source_dir    TEXT    NOT NULL, \
            model         TEXT    NOT NULL, \
            lang          TEXT    NOT NULL, \
            started_at    INTEGER NOT NULL, \
            duration_secs REAL    NOT NULL, \
            segment_count INTEGER NOT NULL, \
            created_at    INTEGER NOT NULL)",
    )
    .execute(&mut *conn)
    .await
    .context("create meetings table")?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS transcript_segments (\
            id          INTEGER PRIMARY KEY AUTOINCREMENT, \
            meeting_id  INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE, \
            seq         INTEGER NOT NULL, \
            speaker     TEXT    NOT NULL, \
            text        TEXT    NOT NULL, \
            t_start     REAL    NOT NULL, \
            t_end       REAL    NOT NULL, \
            confidence  REAL    NOT NULL, \
            CHECK (speaker IN ('you','others')))",
    )
    .execute(&mut *conn)
    .await
    .context("create transcript_segments table")?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_segments_meeting \
         ON transcript_segments(meeting_id, seq)",
    )
    .execute(&mut *conn)
    .await
    .context("create segments index")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(sp: Speaker, t: f64, text: &str) -> TranscriptSegment {
        TranscriptSegment { speaker: sp, text: text.into(), t_start: t, t_end: t + 1.0, confidence: 0.9 }
    }

    /// A per-test scratch directory (mirrors the pattern in maintenance/status/config tests).
    /// Suffixed per call so tests in the same process never share a database file.
    fn tempdir() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "meetscribe-db-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn meta() -> MeetingMeta {
        MeetingMeta {
            title: "team sync".into(),
            source_dir: "capture".into(),
            model: "ggml-large-v3.bin".into(),
            lang: "es".into(),
            started_at: 1_752_900_000,
            duration_secs: 312.5,
            created_at: 1_752_900_400,
        }
    }

    #[tokio::test]
    async fn insert_and_load_roundtrip() {
        let mut db = Db::open_in_memory().await.unwrap();
        let segs = vec![seg(Speaker::You, 0.0, "hola"), seg(Speaker::Others, 2.0, "qué tal")];
        let id = db.insert_meeting(&meta(), &segs).await.unwrap();

        let back = db.load_segments(id).await.unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].seg.speaker, Speaker::You);
        assert_eq!(back[0].seg.text, "hola");
        assert_eq!(back[1].seg.speaker, Speaker::Others);
        assert!((back[1].seg.t_start - 2.0).abs() < 1e-9);
        assert!((back[0].seg.confidence - 0.9).abs() < 1e-6);
        assert!(back[0].id > 0, "stored segments carry their row id");
        assert!(back[0].speaker_id.is_none(), "identity is not populated yet");

        let m = db.get_meeting(id).await.unwrap().unwrap();
        assert_eq!(m.segment_count, 2);
        assert_eq!(m.title, "team sync");
        assert_eq!(m.lang, "es");
        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn empty_transcript_is_valid() {
        let mut db = Db::open_in_memory().await.unwrap();
        let id = db.insert_meeting(&meta(), &[]).await.unwrap();
        assert!(db.load_segments(id).await.unwrap().is_empty());
        let m = db.get_meeting(id).await.unwrap().unwrap();
        assert_eq!(m.segment_count, 0);
        let all = db.list_meetings().await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, id);
    }

    /// Open a file-backed DB at `path` the way the real `open` does.
    async fn open_file(path: &Path) -> Result<Db> {
        Db::open(path).await
    }

    /// Build a v0 database: the frozen baseline schema, WITH ROWS, and no `user_version`.
    /// This is what every existing installation looks like.
    async fn seed_v0(path: &Path) {
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .foreign_keys(true);
        let mut conn = SqliteConnection::connect_with(&opts).await.unwrap();
        init_schema(&mut conn).await.unwrap();
        sqlx::query("INSERT INTO meetings (title, source_dir, model, lang, started_at, duration_secs, segment_count, created_at) VALUES ('old','/tmp/x','m','es',1,1.0,1,1)")
            .execute(&mut conn)
            .await
            .unwrap();
        sqlx::query("INSERT INTO transcript_segments (meeting_id, seq, speaker, text, t_start, t_end, confidence) VALUES (1,0,'you','legacy',0.0,1.0,0.5)")
            .execute(&mut conn)
            .await
            .unwrap();
        assert_eq!(read_user_version(&mut conn).await.unwrap(), 0);
        conn.close().await.unwrap();
    }

    #[tokio::test]
    async fn ladder_upgrades_a_populated_v0_db_and_is_idempotent() {
        let tmp = tempdir();
        let path = tmp.join("m.db");
        seed_v0(&path).await;

        let mut db = open_file(&path).await.unwrap();
        assert_eq!(db.schema_version().await.unwrap(), SCHEMA_VERSION);
        // Pre-existing rows survive the migration.
        assert_eq!(db.list_meetings().await.unwrap().len(), 1);
        assert_eq!(db.load_segments(1).await.unwrap()[0].seg.text, "legacy");
        assert!(db.list_vocab().await.unwrap().is_empty());
        db.close().await.unwrap();

        // Re-running the ladder is a no-op, not an error.
        let mut db = open_file(&path).await.unwrap();
        assert_eq!(db.schema_version().await.unwrap(), SCHEMA_VERSION);
        db.close().await.unwrap();
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// The check that a read-only assertion cannot make: a migrated database must still ACCEPT
    /// WRITES. An `ALTER TABLE … REFERENCES <missing table>` leaves SELECTs working while every
    /// INSERT fails at prepare — so "the ladder ran" is not evidence the tool still works.
    #[tokio::test]
    async fn migrated_db_still_accepts_inserts() {
        let tmp = tempdir();
        let path = tmp.join("m.db");
        seed_v0(&path).await;

        let mut db = open_file(&path).await.unwrap();
        let segs = vec![seg(Speaker::You, 0.0, "after migration")];
        let id = db
            .insert_meeting(&meta(), &segs)
            .await
            .expect("a migrated database must still accept new meetings");
        assert_eq!(db.load_segments(id).await.unwrap().len(), 1);
        db.close().await.unwrap();
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// A fresh database and a migrated legacy one must be structurally identical, or the two
    /// populations drift and only one of them gets tested.
    #[tokio::test]
    async fn fresh_and_migrated_schemas_match() {
        let tmp = tempdir();
        let legacy = tmp.join("legacy.db");
        let fresh = tmp.join("fresh.db");
        seed_v0(&legacy).await;

        let mut a = open_file(&legacy).await.unwrap();
        let mut b = open_file(&fresh).await.unwrap();
        for table in ["meetings", "transcript_segments", "vocab_corrections"] {
            assert_eq!(
                a.table_info(table).await.unwrap(),
                b.table_info(table).await.unwrap(),
                "schema drift in {table} between a migrated v0 db and a fresh one"
            );
        }
        assert_eq!(a.schema_version().await.unwrap(), b.schema_version().await.unwrap());
        a.close().await.unwrap();
        b.close().await.unwrap();
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[tokio::test]
    async fn vocab_crud_orders_by_id_and_disables_without_deleting() {
        let mut db = Db::open_in_memory().await.unwrap();
        let a = db.add_vocab("Postgre", "Postgres", false, 10).await.unwrap();
        let b = db.add_vocab(r"Postgre\w+s", "Postgres", true, 11).await.unwrap();
        assert!(b > a);

        let all = db.list_vocab().await.unwrap();
        assert_eq!(all.iter().map(|r| r.id).collect::<Vec<_>>(), vec![a, b]);
        assert!(!all[0].is_regex && all[1].is_regex);
        assert!(all.iter().all(|r| r.enabled));

        assert!(db.set_vocab_enabled(a, false).await.unwrap());
        assert!(!db.set_vocab_enabled(9999, false).await.unwrap());

        let enabled = db.load_enabled_vocab().await.unwrap();
        assert_eq!(enabled.len(), 1);
        assert_eq!(enabled[0].id, b);
        // Disabled, not deleted — the id (and therefore the application order) is preserved.
        assert_eq!(db.list_vocab().await.unwrap().len(), 2);
        db.close().await.unwrap();
    }

    /// Guard the invariant the whole render split rests on: stored transcript text is written
    /// once and never updated. If this fails, someone added an UPDATE and retroactive
    /// re-rendering silently stops being possible.
    #[test]
    fn no_code_path_updates_stored_transcript_text() {
        // Assembled at compile time so this assertion cannot match its own source text.
        // db.rs is the only module that issues SQL, so scanning it covers the crate.
        let needle = concat!("UPDATE ", "transcript_segments");
        let src = include_str!("db.rs");
        assert!(
            !src.contains(needle),
            "stored transcript text is raw ASR output and must never be rewritten in place — \
             presentation belongs in render.rs, which is what keeps re-rendering retroactive"
        );
    }
}
