//! Plaintext SQLite storage (Phase 3) — one meeting + its ordered segments per row set.
//!
//! Single `SqliteConnection` (this is a one-shot batch CLI, one writer, one transaction);
//! no pool. All async work runs inside a single `block_on` from the sync pipeline. File
//! perms are `0600` (encryption-at-rest is v1.1). The `speaker` column stores the lowercase
//! token from `Speaker::as_sql` and is `CHECK`-constrained to it.

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
        init_schema(&mut conn).await?;
        // 0600 on the db file. The transient rollback `-journal` inherits these perms.
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod 0600 {}", path.display()))?;
        Ok(Self { conn })
    }

    #[cfg(test)]
    pub async fn open_in_memory() -> Result<Self> {
        let opts = SqliteConnectOptions::new()
            .filename(":memory:")
            .create_if_missing(true)
            .foreign_keys(true);
        let mut conn = SqliteConnection::connect_with(&opts).await?;
        init_schema(&mut conn).await?;
        Ok(Self { conn })
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
    pub async fn load_segments(&mut self, meeting_id: i64) -> Result<Vec<TranscriptSegment>> {
        let rows = sqlx::query(
            "SELECT speaker, text, t_start, t_end, confidence \
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
            out.push(TranscriptSegment {
                speaker,
                text: r.try_get("text")?,
                t_start: r.try_get("t_start")?,
                t_end: r.try_get("t_end")?,
                confidence: r.try_get("confidence")?,
            });
        }
        Ok(out)
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
        assert_eq!(back[0].speaker, Speaker::You);
        assert_eq!(back[0].text, "hola");
        assert_eq!(back[1].speaker, Speaker::Others);
        assert!((back[1].t_start - 2.0).abs() < 1e-9);
        assert!((back[0].confidence - 0.9).abs() < 1e-6);

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
}
