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
const SCHEMA_VERSION: i32 = 2;

/// A stored segment: the raw ASR contract plus the row identity and the (nullable) speaker
/// identity, resolved through `segment_voices` → `voice_clusters`. Both are `None` for the mic
/// channel and for any far-end segment whose meeting has not been clustered.
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

/// A named person. Display form is "First Last"; the two fields are stored separately.
#[derive(Debug, Clone)]
pub struct SpeakerRow {
    pub id: i64,
    pub first_name: String,
    pub last_name: String,
    pub created_at: i64,
    /// How many confirmed voiceprints this person has (0 = named but nothing enrolled).
    pub voiceprints: i64,
}

impl SpeakerRow {
    pub fn full_name(&self) -> String {
        format!("{} {}", self.first_name, self.last_name)
    }
}

/// One far-end voice cluster inside one meeting.
#[derive(Debug, Clone)]
pub struct ClusterRow {
    pub id: i64,
    pub meeting_id: i64,
    /// `A`, `B`, … — by descending speech time at cluster time; merge/split leave gaps.
    pub cluster: String,
    pub speaker_id: Option<i64>,
    /// `"auto"` (matched against enrolled voiceprints) or `"manual"` (the owner named it).
    pub assigned_by: Option<String>,
    pub match_score: Option<f64>,
    /// The owner marked this voice as unknown-by-choice; it is not pending.
    pub skipped: bool,
    /// f32 little-endian, L2-normalised (decode with `spk::from_blob`).
    pub centroid: Vec<u8>,
    pub dim: i64,
    /// Embedded windows only (inherited short windows do not count).
    pub n_windows: i64,
    pub speech_secs: f64,
}

/// A cluster to write (no id yet). Written through `insert_meeting_with_voices`,
/// `replace_meeting_clusters` or `replace_cluster`.
#[derive(Debug, Clone)]
pub struct NewCluster {
    pub cluster: String,
    pub speaker_id: Option<i64>,
    pub assigned_by: Option<String>,
    pub match_score: Option<f64>,
    pub centroid: Vec<u8>,
    pub dim: i64,
    pub n_windows: i64,
    pub speech_secs: f64,
}

/// Membership of one segment in one cluster. `cluster` indexes the `NewCluster` slice written in
/// the same call; `segment` is either an index into the segments being inserted
/// (`insert_meeting_with_voices`) or an existing `transcript_segments.id` (the replace calls).
#[derive(Debug, Clone)]
pub struct SegmentVoice {
    pub segment: i64,
    pub cluster: usize,
    pub inherited: bool,
    /// f32 little-endian; `None` for inherited (short) windows.
    pub embedding: Option<Vec<u8>>,
}

/// One far-end window as stored: its segment, its times, and its embedding (if it was embedded).
#[derive(Debug, Clone)]
pub struct WindowRow {
    pub segment_id: i64,
    pub t_start: f64,
    pub t_end: f64,
    pub inherited: bool,
    pub embedding: Option<Vec<u8>>,
}

/// An enrolled person: every confirmed voiceprint, as raw blobs.
#[derive(Debug, Clone)]
pub struct EnrolledRow {
    pub speaker_id: i64,
    pub voiceprints: Vec<Vec<u8>>,
}

/// A pending cluster with the meeting it belongs to (for `speakers list --pending`).
#[derive(Debug, Clone)]
pub struct PendingCluster {
    pub cluster: ClusterRow,
    pub meeting_title: String,
    pub started_at: i64,
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

    /// `PRAGMA table_info` as `name|type|notnull|pk` rows — used to assert a fresh database and a
    /// migrated legacy one are structurally identical. Column DEFAULTS are deliberately not
    /// compared: `dflt_value` comes back as a nullable string and adds a decoding branch for a
    /// property the ladder's additive-only rule already constrains.
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

    /// Insert a meeting and its segments with no voice clusters. Production goes through
    /// `insert_meeting_with_voices` (an empty assembly is the no-identity case); this shorthand
    /// keeps the storage tests readable.
    #[cfg(test)]
    pub async fn insert_meeting(
        &mut self,
        meta: &MeetingMeta,
        segs: &[TranscriptSegment],
    ) -> Result<i64> {
        self.insert_meeting_with_voices(meta, segs, &[], &[]).await
    }

    /// Insert a meeting, its segments, and its far-end voice clusters in ONE transaction.
    /// `voices[i].segment` indexes `segs`; `voices[i].cluster` indexes `clusters`.
    pub async fn insert_meeting_with_voices(
        &mut self,
        meta: &MeetingMeta,
        segs: &[TranscriptSegment],
        clusters: &[NewCluster],
        voices: &[SegmentVoice],
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

        let mut seg_ids = Vec::with_capacity(segs.len());
        for (i, s) in segs.iter().enumerate() {
            let r = sqlx::query(
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
            seg_ids.push(r.last_insert_rowid());
        }

        let by_id: Vec<SegmentVoice> = voices
            .iter()
            .map(|v| {
                let idx = usize::try_from(v.segment).ok().filter(|i| *i < seg_ids.len());
                idx.map(|i| SegmentVoice { segment: seg_ids[i], ..v.clone() })
                    .ok_or_else(|| anyhow!("voice refers to segment index {} of {}", v.segment, seg_ids.len()))
            })
            .collect::<Result<_>>()?;
        write_clusters(&mut tx, meeting_id, clusters, &by_id, meta.created_at).await?;

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
            "SELECT ts.id, ts.speaker, ts.text, ts.t_start, ts.t_end, ts.confidence, \
                    vc.speaker_id AS speaker_id, vc.cluster AS voice_cluster \
             FROM transcript_segments ts \
             LEFT JOIN segment_voices sv ON sv.segment_id = ts.id \
             LEFT JOIN voice_clusters vc ON vc.id = sv.cluster_id \
             WHERE ts.meeting_id = ? ORDER BY ts.seq",
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
                speaker_id: r.try_get("speaker_id")?,
                voice_cluster: r.try_get("voice_cluster")?,
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


    // ------------------------------------------------------------------ speakers (Batch E)

    /// Every named person, with voiceprint counts, ordered by last then first name.
    pub async fn list_speakers(&mut self) -> Result<Vec<SpeakerRow>> {
        let rows = sqlx::query(
            "SELECT s.id, s.first_name, s.last_name, s.created_at, \
                    (SELECT count(*) FROM voiceprints v WHERE v.speaker_id = s.id) AS voiceprints \
             FROM speakers s ORDER BY s.last_name COLLATE NOCASE, s.first_name COLLATE NOCASE, s.id",
        )
        .fetch_all(&mut self.conn)
        .await
        .context("list speakers")?;
        rows.iter().map(speaker_from_row).collect()
    }

    pub async fn get_speaker(&mut self, id: i64) -> Result<Option<SpeakerRow>> {
        let row = sqlx::query(
            "SELECT s.id, s.first_name, s.last_name, s.created_at, \
                    (SELECT count(*) FROM voiceprints v WHERE v.speaker_id = s.id) AS voiceprints \
             FROM speakers s WHERE s.id = ?",
        )
        .bind(id)
        .fetch_optional(&mut self.conn)
        .await
        .context("get speaker")?;
        row.as_ref().map(speaker_from_row).transpose()
    }

    /// Case-insensitive exact match on first + last name, so labelling the same person twice
    /// reuses one row.
    pub async fn find_speaker(&mut self, first: &str, last: &str) -> Result<Option<i64>> {
        sqlx::query_scalar::<_, i64>(
            "SELECT id FROM speakers WHERE lower(first_name) = lower(?) AND lower(last_name) = lower(?) \
             ORDER BY id LIMIT 1",
        )
        .bind(first)
        .bind(last)
        .fetch_optional(&mut self.conn)
        .await
        .context("find speaker")
    }

    pub async fn add_speaker(&mut self, first: &str, last: &str, created_at: i64) -> Result<i64> {
        let res = sqlx::query("INSERT INTO speakers (first_name, last_name, created_at) VALUES (?, ?, ?)")
            .bind(first)
            .bind(last)
            .bind(created_at)
            .execute(&mut self.conn)
            .await
            .context("insert speaker")?;
        Ok(res.last_insert_rowid())
    }

    /// `false` = no speaker with that id.
    pub async fn rename_speaker(&mut self, id: i64, first: &str, last: &str) -> Result<bool> {
        let res = sqlx::query("UPDATE speakers SET first_name = ?, last_name = ? WHERE id = ?")
            .bind(first)
            .bind(last)
            .bind(id)
            .execute(&mut self.conn)
            .await
            .context("rename speaker")?;
        Ok(res.rows_affected() > 0)
    }

    /// Every enrolled person with all their voiceprints — what auto-match runs against.
    pub async fn load_enrolled(&mut self) -> Result<Vec<EnrolledRow>> {
        let rows = sqlx::query("SELECT speaker_id, embedding FROM voiceprints ORDER BY speaker_id, id")
            .fetch_all(&mut self.conn)
            .await
            .context("load voiceprints")?;
        let mut out: Vec<EnrolledRow> = Vec::new();
        for r in &rows {
            let sid: i64 = r.try_get("speaker_id")?;
            let emb: Vec<u8> = r.try_get("embedding")?;
            match out.last_mut() {
                Some(e) if e.speaker_id == sid => e.voiceprints.push(emb),
                _ => out.push(EnrolledRow { speaker_id: sid, voiceprints: vec![emb] }),
            }
        }
        Ok(out)
    }

    // ------------------------------------------------------------------ clusters (Batch E)

    /// A meeting's clusters, most speech first.
    pub async fn list_clusters(&mut self, meeting_id: i64) -> Result<Vec<ClusterRow>> {
        let sql = format!("SELECT {CLUSTER_COLS} FROM voice_clusters WHERE meeting_id = ? \
                           ORDER BY speech_secs DESC, cluster");
        let rows = sqlx::query(&sql)
            .bind(meeting_id)
            .fetch_all(&mut self.conn)
            .await
            .context("list clusters")?;
        rows.iter().map(cluster_from_row).collect()
    }

    pub async fn get_cluster(&mut self, meeting_id: i64, label: &str) -> Result<Option<ClusterRow>> {
        let sql = format!("SELECT {CLUSTER_COLS} FROM voice_clusters WHERE meeting_id = ? AND cluster = ?");
        let row = sqlx::query(&sql)
            .bind(meeting_id)
            .bind(label)
            .fetch_optional(&mut self.conn)
            .await
            .context("get cluster")?;
        row.as_ref().map(cluster_from_row).transpose()
    }

    /// Unnamed, unskipped clusters with at least `min_windows` embedded windows, across all
    /// meetings, newest meeting first — what the owner still has to listen to.
    pub async fn pending_clusters(&mut self, min_windows: i64) -> Result<Vec<PendingCluster>> {
        let cols = CLUSTER_COLS
            .split(',')
            .map(|c| format!("vc.{}", c.trim()))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT {cols}, m.title AS meeting_title, m.started_at AS meeting_started_at \
             FROM voice_clusters vc JOIN meetings m ON m.id = vc.meeting_id \
             WHERE vc.speaker_id IS NULL AND vc.skipped = 0 AND vc.n_windows >= ? \
             ORDER BY m.started_at DESC, vc.meeting_id DESC, vc.speech_secs DESC"
        );
        let rows = sqlx::query(&sql)
            .bind(min_windows)
            .fetch_all(&mut self.conn)
            .await
            .context("pending clusters")?;
        rows.iter()
            .map(|r| {
                Ok(PendingCluster {
                    cluster: cluster_from_row(r)?,
                    meeting_title: r.try_get("meeting_title")?,
                    started_at: r.try_get("meeting_started_at")?,
                })
            })
            .collect()
    }

    /// Set (or clear, with `None`) the person behind a cluster. Clearing also clears the
    /// provenance and score; setting clears `skipped`.
    pub async fn set_cluster_speaker(
        &mut self,
        cluster_id: i64,
        speaker_id: Option<i64>,
        assigned_by: Option<&str>,
        match_score: Option<f64>,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE voice_clusters SET speaker_id = ?, assigned_by = ?, match_score = ?, \
             skipped = CASE WHEN ? IS NULL THEN skipped ELSE 0 END WHERE id = ?",
        )
        .bind(speaker_id)
        .bind(assigned_by)
        .bind(match_score)
        .bind(speaker_id)
        .bind(cluster_id)
        .execute(&mut self.conn)
        .await
        .context("set cluster speaker")?;
        Ok(())
    }

    pub async fn set_cluster_skipped(&mut self, cluster_id: i64, skipped: bool) -> Result<()> {
        sqlx::query("UPDATE voice_clusters SET skipped = ? WHERE id = ?")
            .bind(i64::from(skipped))
            .bind(cluster_id)
            .execute(&mut self.conn)
            .await
            .context("set cluster skipped")?;
        Ok(())
    }

    /// Recompute a cluster's centroid and counts (after `merge`).
    pub async fn update_cluster_stats(
        &mut self,
        cluster_id: i64,
        centroid: &[u8],
        dim: i64,
        n_windows: i64,
        speech_secs: f64,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE voice_clusters SET centroid = ?, dim = ?, n_windows = ?, speech_secs = ? WHERE id = ?",
        )
        .bind(centroid)
        .bind(dim)
        .bind(n_windows)
        .bind(speech_secs)
        .bind(cluster_id)
        .execute(&mut self.conn)
        .await
        .context("update cluster stats")?;
        Ok(())
    }

    /// Move every window of `from` into `to` (the `merge` primitive).
    pub async fn move_segments(&mut self, from_cluster: i64, to_cluster: i64) -> Result<u64> {
        let res = sqlx::query("UPDATE segment_voices SET cluster_id = ? WHERE cluster_id = ?")
            .bind(to_cluster)
            .bind(from_cluster)
            .execute(&mut self.conn)
            .await
            .context("move segments between clusters")?;
        Ok(res.rows_affected())
    }

    /// Delete a cluster; its windows and its voiceprints go with it (cascade).
    pub async fn delete_cluster(&mut self, cluster_id: i64) -> Result<()> {
        sqlx::query("DELETE FROM voice_clusters WHERE id = ?")
            .bind(cluster_id)
            .execute(&mut self.conn)
            .await
            .context("delete cluster")?;
        Ok(())
    }

    /// Replace ALL of a meeting's clusters (the retro `speakers cluster` path). One short
    /// transaction; `voices[i].segment` are existing `transcript_segments.id`s.
    pub async fn replace_meeting_clusters(
        &mut self,
        meeting_id: i64,
        clusters: &[NewCluster],
        voices: &[SegmentVoice],
        created_at: i64,
    ) -> Result<()> {
        let mut tx = self.conn.begin().await.context("begin replace clusters tx")?;
        sqlx::query("DELETE FROM voice_clusters WHERE meeting_id = ?")
            .bind(meeting_id)
            .execute(&mut *tx)
            .await
            .context("delete meeting clusters")?;
        write_clusters(&mut tx, meeting_id, clusters, voices, created_at).await?;
        tx.commit().await.context("commit replace clusters tx")?;
        Ok(())
    }

    /// Replace ONE cluster with several (the `split` primitive), in one transaction.
    pub async fn replace_cluster(
        &mut self,
        cluster_id: i64,
        meeting_id: i64,
        clusters: &[NewCluster],
        voices: &[SegmentVoice],
        created_at: i64,
    ) -> Result<()> {
        let mut tx = self.conn.begin().await.context("begin split tx")?;
        sqlx::query("DELETE FROM voice_clusters WHERE id = ?")
            .bind(cluster_id)
            .execute(&mut *tx)
            .await
            .context("delete split cluster")?;
        write_clusters(&mut tx, meeting_id, clusters, voices, created_at).await?;
        tx.commit().await.context("commit split tx")?;
        Ok(())
    }

    /// The windows of one cluster, in time order.
    pub async fn cluster_windows(&mut self, cluster_id: i64) -> Result<Vec<WindowRow>> {
        let rows = sqlx::query(
            "SELECT sv.segment_id, sv.inherited, sv.embedding, ts.t_start, ts.t_end \
             FROM segment_voices sv JOIN transcript_segments ts ON ts.id = sv.segment_id \
             WHERE sv.cluster_id = ? ORDER BY ts.t_start",
        )
        .bind(cluster_id)
        .fetch_all(&mut self.conn)
        .await
        .context("cluster windows")?;
        rows.iter().map(window_from_row).collect()
    }

    /// Every clustered window of a meeting, in time order (for `--recluster`).
    pub async fn meeting_windows(&mut self, meeting_id: i64) -> Result<Vec<WindowRow>> {
        let rows = sqlx::query(
            "SELECT sv.segment_id, sv.inherited, sv.embedding, ts.t_start, ts.t_end \
             FROM segment_voices sv JOIN transcript_segments ts ON ts.id = sv.segment_id \
             WHERE ts.meeting_id = ? ORDER BY ts.t_start",
        )
        .bind(meeting_id)
        .fetch_all(&mut self.conn)
        .await
        .context("meeting windows")?;
        rows.iter().map(window_from_row).collect()
    }

    /// The far-end segments of a meeting as `(segment_id, t_start, t_end)`, in time order — the
    /// slices the retro path embeds.
    pub async fn far_end_segments(&mut self, meeting_id: i64) -> Result<Vec<(i64, f64, f64)>> {
        let rows = sqlx::query(
            "SELECT id, t_start, t_end FROM transcript_segments \
             WHERE meeting_id = ? AND speaker = ? ORDER BY t_start, seq",
        )
        .bind(meeting_id)
        .bind(Speaker::Others.as_sql())
        .fetch_all(&mut self.conn)
        .await
        .context("far-end segments")?;
        rows.iter()
            .map(|r| Ok((r.try_get("id")?, r.try_get("t_start")?, r.try_get("t_end")?)))
            .collect()
    }

    // ------------------------------------------------------------------ voiceprints (Batch E)

    /// Enrol a cluster's centroid as one voiceprint of `speaker_id`. A voiceprint IS a confirmed
    /// cluster's centroid — that is why it takes the row rather than loose fields.
    pub async fn add_voiceprint(&mut self, speaker_id: i64, cluster: &ClusterRow, created_at: i64) -> Result<i64> {
        let res = sqlx::query(
            "INSERT INTO voiceprints (speaker_id, meeting_id, cluster_id, embedding, dim, sample_secs, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(speaker_id)
        .bind(cluster.meeting_id)
        .bind(cluster.id)
        .bind(cluster.centroid.as_slice())
        .bind(cluster.dim)
        .bind(cluster.speech_secs)
        .bind(created_at)
        .execute(&mut self.conn)
        .await
        .context("insert voiceprint")?;
        Ok(res.last_insert_rowid())
    }

    /// Remove whatever this cluster contributed to the enrolment set (`unlabel`, `skip`, re-`label`).
    pub async fn delete_voiceprints_for_cluster(&mut self, cluster_id: i64) -> Result<u64> {
        let res = sqlx::query("DELETE FROM voiceprints WHERE cluster_id = ?")
            .bind(cluster_id)
            .execute(&mut self.conn)
            .await
            .context("delete cluster voiceprints")?;
        Ok(res.rows_affected())
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

const CLUSTER_COLS: &str = "id, meeting_id, cluster, speaker_id, assigned_by, match_score, skipped, \
                            centroid, dim, n_windows, speech_secs";

fn speaker_from_row(r: &SqliteRow) -> Result<SpeakerRow> {
    Ok(SpeakerRow {
        id: r.try_get("id")?,
        first_name: r.try_get("first_name")?,
        last_name: r.try_get("last_name")?,
        created_at: r.try_get("created_at")?,
        voiceprints: r.try_get("voiceprints")?,
    })
}

fn cluster_from_row(r: &SqliteRow) -> Result<ClusterRow> {
    Ok(ClusterRow {
        id: r.try_get("id")?,
        meeting_id: r.try_get("meeting_id")?,
        cluster: r.try_get("cluster")?,
        speaker_id: r.try_get("speaker_id")?,
        assigned_by: r.try_get("assigned_by")?,
        match_score: r.try_get("match_score")?,
        skipped: r.try_get::<i64, _>("skipped")? != 0,
        centroid: r.try_get("centroid")?,
        dim: r.try_get("dim")?,
        n_windows: r.try_get("n_windows")?,
        speech_secs: r.try_get("speech_secs")?,
    })
}

fn window_from_row(r: &SqliteRow) -> Result<WindowRow> {
    Ok(WindowRow {
        segment_id: r.try_get("segment_id")?,
        t_start: r.try_get("t_start")?,
        t_end: r.try_get("t_end")?,
        inherited: r.try_get::<i64, _>("inherited")? != 0,
        embedding: r.try_get("embedding")?,
    })
}

/// Write clusters + memberships inside a caller-owned transaction. `voices[i].segment` are
/// `transcript_segments.id`s; `voices[i].cluster` indexes `clusters`.
async fn write_clusters(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    meeting_id: i64,
    clusters: &[NewCluster],
    voices: &[SegmentVoice],
    created_at: i64,
) -> Result<()> {
    let mut ids = Vec::with_capacity(clusters.len());
    for c in clusters {
        let r = sqlx::query(
            "INSERT INTO voice_clusters \
             (meeting_id, cluster, speaker_id, assigned_by, match_score, skipped, centroid, dim, \
              n_windows, speech_secs, created_at) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)",
        )
        .bind(meeting_id)
        .bind(c.cluster.as_str())
        .bind(c.speaker_id)
        .bind(c.assigned_by.as_deref())
        .bind(c.match_score)
        .bind(c.centroid.as_slice())
        .bind(c.dim)
        .bind(c.n_windows)
        .bind(c.speech_secs)
        .bind(created_at)
        .execute(&mut **tx)
        .await
        .with_context(|| format!("insert cluster {}", c.cluster))?;
        ids.push(r.last_insert_rowid());
    }
    for v in voices {
        let cid = *ids
            .get(v.cluster)
            .ok_or_else(|| anyhow!("voice refers to cluster index {} of {}", v.cluster, ids.len()))?;
        sqlx::query(
            "INSERT INTO segment_voices (segment_id, cluster_id, inherited, embedding) VALUES (?, ?, ?, ?)",
        )
        .bind(v.segment)
        .bind(cid)
        .bind(i64::from(v.inherited))
        .bind(v.embedding.as_deref())
        .execute(&mut **tx)
        .await
        .with_context(|| format!("insert segment voice for segment {}", v.segment))?;
    }
    Ok(())
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
            0 => run_migration(conn, 0, 1, MIGRATION_V1).await?,
            1 => run_migration(conn, 1, 2, MIGRATION_V2).await?,
            other => anyhow::bail!("no migration registered from schema v{other}"),
        }
    }
}

/// v0 → v1: add `vocab_corrections`.
///
/// Deliberately contains NO speaker tables — that rung ships with the speaker-ID work, so a
/// database never carries dead tables for a feature that might not pass its calibration gate.
/// The statements for v0 → v1, applied IN ORDER.
///
/// Order is load-bearing for any future rung: a `CREATE TABLE` must precede any `ALTER TABLE …
/// REFERENCES` that names it, or the ALTER succeeds and every later INSERT fails at prepare
/// (see the module docs). Expressing a rung as an ordered statement list rather than a closure
/// keeps that requirement visible as the literal order of this array.
const MIGRATION_V1: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS vocab_corrections (\
        id          INTEGER PRIMARY KEY AUTOINCREMENT, \
        pattern     TEXT    NOT NULL, \
        replacement TEXT    NOT NULL, \
        is_regex    INTEGER NOT NULL DEFAULT 0 CHECK (is_regex IN (0,1)), \
        enabled     INTEGER NOT NULL DEFAULT 1 CHECK (enabled  IN (0,1)), \
        created_at  INTEGER NOT NULL)",
];

/// v1 → v2: speaker identity (Batch E). Four new tables, NO change to `transcript_segments`.
///
/// Identity hangs off `transcript_segments.id` through `segment_voices`, so a transcript row is
/// never rewritten when a voice is clustered, named, merged or split. Every `REFERENCES` names a
/// table created earlier in this list (or in v0), and there is no ALTER at all.
///
/// `voiceprints.cluster_id` cascades: a voiceprint exists only while the cluster that produced it
/// exists, so `merge`/`split`/`--recluster` can never leave a stale print feeding auto-match.
const MIGRATION_V2: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS speakers (\
        id         INTEGER PRIMARY KEY AUTOINCREMENT, \
        first_name TEXT    NOT NULL, \
        last_name  TEXT    NOT NULL, \
        created_at INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS voice_clusters (\
        id          INTEGER PRIMARY KEY AUTOINCREMENT, \
        meeting_id  INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE, \
        cluster     TEXT    NOT NULL, \
        speaker_id  INTEGER NULL REFERENCES speakers(id) ON DELETE SET NULL, \
        assigned_by TEXT    NULL CHECK (assigned_by IN ('auto','manual')), \
        match_score REAL    NULL, \
        skipped     INTEGER NOT NULL DEFAULT 0 CHECK (skipped IN (0,1)), \
        centroid    BLOB    NOT NULL, \
        dim         INTEGER NOT NULL, \
        n_windows   INTEGER NOT NULL, \
        speech_secs REAL    NOT NULL, \
        created_at  INTEGER NOT NULL, \
        UNIQUE (meeting_id, cluster))",
    "CREATE TABLE IF NOT EXISTS segment_voices (\
        segment_id INTEGER PRIMARY KEY REFERENCES transcript_segments(id) ON DELETE CASCADE, \
        cluster_id INTEGER NOT NULL REFERENCES voice_clusters(id) ON DELETE CASCADE, \
        inherited  INTEGER NOT NULL DEFAULT 0 CHECK (inherited IN (0,1)), \
        embedding  BLOB    NULL)",
    "CREATE TABLE IF NOT EXISTS voiceprints (\
        id          INTEGER PRIMARY KEY AUTOINCREMENT, \
        speaker_id  INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE, \
        meeting_id  INTEGER NULL REFERENCES meetings(id) ON DELETE SET NULL, \
        cluster_id  INTEGER NOT NULL REFERENCES voice_clusters(id) ON DELETE CASCADE, \
        embedding   BLOB    NOT NULL, \
        dim         INTEGER NOT NULL, \
        sample_secs REAL    NOT NULL, \
        created_at  INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_segment_voices_cluster ON segment_voices(cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_voice_clusters_meeting ON voice_clusters(meeting_id)",
];

/// Run one ladder rung inside `BEGIN IMMEDIATE`, re-checking the version inside the transaction
/// and bumping it before COMMIT.
///
/// `BEGIN IMMEDIATE` (not sqlx's deferred `begin()`) takes the write lock up front, so the
/// re-read below cannot race another process that migrates between our check and our first write.
/// Bumping `user_version` inside the same transaction is what makes a partial migration
/// impossible: without it, an applied DDL plus a lost version write bricks every later open.
async fn run_migration(
    conn: &mut SqliteConnection,
    from: i32,
    to: i32,
    stmts: &[&str],
) -> Result<()> {
    sqlx::query("BEGIN IMMEDIATE")
        .execute(&mut *conn)
        .await
        .with_context(|| format!("begin migration v{from}→v{to}"))?;

    let res = async {
        // Another process may have won the race while we waited for the write lock.
        if read_user_version(conn).await? != from {
            return Ok(false);
        }
        for (i, sql) in stmts.iter().enumerate() {
            sqlx::query(sql)
                .execute(&mut *conn)
                .await
                .with_context(|| format!("migration v{from}→v{to} statement {i}"))?;
        }
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

    /// Every table the ladder produces. A rung that adds a table must add it here, or the
    /// fresh-vs-migrated check silently stops covering it.
    const ALL_TABLES: [&str; 7] = [
        "meetings",
        "transcript_segments",
        "vocab_corrections",
        "speakers",
        "voice_clusters",
        "segment_voices",
        "voiceprints",
    ];

    /// A 4-byte-per-value blob for a small fake embedding.
    fn blob(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn new_cluster(label: &str, n: i64, secs: f64) -> NewCluster {
        NewCluster {
            cluster: label.into(),
            speaker_id: None,
            assigned_by: None,
            match_score: None,
            centroid: blob(&[1.0, 0.0]),
            dim: 2,
            n_windows: n,
            speech_secs: secs,
        }
    }

    fn voice(segment: i64, cluster: usize, inherited: bool) -> SegmentVoice {
        SegmentVoice {
            segment,
            cluster,
            inherited,
            embedding: if inherited { None } else { Some(blob(&[1.0, 0.0])) },
        }
    }

    /// Insert a meeting with two far-end clusters and name one of them — touches every v2
    /// table, which is the only way to prove their foreign keys resolve.
    async fn insert_with_voices_and_label(db: &mut Db) -> (i64, i64, i64) {
        let segs = vec![
            seg(Speaker::You, 0.0, "hola"),
            seg(Speaker::Others, 1.0, "buenas"),
            seg(Speaker::Others, 2.0, "qué tal"),
            seg(Speaker::Others, 3.0, "sí"), // short window: inherits
        ];
        let clusters = vec![new_cluster("A", 2, 2.0), new_cluster("B", 1, 1.0)];
        let voices = vec![voice(1, 0, false), voice(2, 1, false), voice(3, 0, true)];
        let mid = db
            .insert_meeting_with_voices(&meta(), &segs, &clusters, &voices)
            .await
            .expect("insert with voices");
        let a = db.get_cluster(mid, "A").await.unwrap().expect("cluster A");
        let sid = db.add_speaker("Ada", "Lovelace", 5).await.unwrap();
        db.set_cluster_speaker(a.id, Some(sid), Some("manual"), None).await.unwrap();
        db.add_voiceprint(sid, &a, 6).await.unwrap();
        (mid, a.id, sid)
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
        // …and every v2 table must accept writes too. SQLite lets `REFERENCES <missing>` through at
        // CREATE time and only fails at DML, so a read-only assertion would prove nothing here.
        let (mid, _, sid) = insert_with_voices_and_label(&mut db).await;
        assert_eq!(db.list_clusters(mid).await.unwrap().len(), 2);
        assert_eq!(db.get_speaker(sid).await.unwrap().unwrap().voiceprints, 1);
        db.close().await.unwrap();
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Build a v1 database (what every installation running the vocab release looks like).
    async fn seed_v1(path: &Path) {
        seed_v0(path).await;
        let opts = SqliteConnectOptions::new().filename(path).foreign_keys(true);
        let mut conn = SqliteConnection::connect_with(&opts).await.unwrap();
        run_migration(&mut conn, 0, 1, MIGRATION_V1).await.unwrap();
        sqlx::query("INSERT INTO vocab_corrections (pattern, replacement, is_regex, enabled, created_at) VALUES ('a','b',0,1,1)")
            .execute(&mut conn)
            .await
            .unwrap();
        assert_eq!(read_user_version(&mut conn).await.unwrap(), 1);
        conn.close().await.unwrap();
    }

    #[tokio::test]
    async fn ladder_upgrades_a_populated_v1_db_to_v2_and_is_idempotent() {
        let tmp = tempdir();
        let path = tmp.join("m.db");
        seed_v1(&path).await;

        let mut db = open_file(&path).await.unwrap();
        assert_eq!(db.schema_version().await.unwrap(), 2);
        assert_eq!(db.list_meetings().await.unwrap().len(), 1);
        assert_eq!(db.list_vocab().await.unwrap().len(), 1, "v1 rows survive the v2 rung");
        // Unclustered legacy segments read back with no identity at all.
        let legacy = db.load_segments(1).await.unwrap();
        assert!(legacy[0].speaker_id.is_none() && legacy[0].voice_cluster.is_none());
        assert!(db.list_speakers().await.unwrap().is_empty());
        assert!(db.pending_clusters(1).await.unwrap().is_empty());
        db.close().await.unwrap();

        let mut db = open_file(&path).await.unwrap();
        assert_eq!(db.schema_version().await.unwrap(), 2);
        db.close().await.unwrap();
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[tokio::test]
    async fn voices_roundtrip_through_load_segments() {
        let mut db = Db::open_in_memory().await.unwrap();
        let (mid, a_id, sid) = insert_with_voices_and_label(&mut db).await;

        let segs = db.load_segments(mid).await.unwrap();
        assert_eq!(segs.len(), 4);
        assert!(segs[0].voice_cluster.is_none(), "the mic channel is never clustered");
        assert_eq!(segs[1].voice_cluster.as_deref(), Some("A"));
        assert_eq!(segs[1].speaker_id, Some(sid), "labelled cluster resolves to the person");
        assert_eq!(segs[2].voice_cluster.as_deref(), Some("B"));
        assert!(segs[2].speaker_id.is_none(), "unlabelled cluster has no person");
        assert_eq!(segs[3].voice_cluster.as_deref(), Some("A"), "short window inherited A");

        let windows = db.cluster_windows(a_id).await.unwrap();
        assert_eq!(windows.len(), 2);
        assert!(!windows[0].inherited && windows[0].embedding.is_some());
        assert!(windows[1].inherited && windows[1].embedding.is_none());
        assert_eq!(db.meeting_windows(mid).await.unwrap().len(), 3);
        assert_eq!(db.far_end_segments(mid).await.unwrap().len(), 3);

        // A meeting inserted with no voices reads back exactly as before.
        let plain = db.insert_meeting(&meta(), &[seg(Speaker::Others, 0.0, "x")]).await.unwrap();
        let s = db.load_segments(plain).await.unwrap();
        assert!(s[0].speaker_id.is_none() && s[0].voice_cluster.is_none());
        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn enrolment_follows_labels_and_cluster_deletion() {
        let mut db = Db::open_in_memory().await.unwrap();
        let (mid, a_id, sid) = insert_with_voices_and_label(&mut db).await;

        let enrolled = db.load_enrolled().await.unwrap();
        assert_eq!(enrolled.len(), 1);
        assert_eq!(enrolled[0].speaker_id, sid);
        assert_eq!(enrolled[0].voiceprints.len(), 1);

        // Only unnamed, unskipped, big-enough clusters are pending.
        let pending = db.pending_clusters(1).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].cluster.cluster, "B");
        assert_eq!(pending[0].meeting_title, "team sync");
        assert!(db.pending_clusters(2).await.unwrap().is_empty(), "B has only 1 window");
        let b = db.get_cluster(mid, "B").await.unwrap().unwrap();
        db.set_cluster_skipped(b.id, true).await.unwrap();
        assert!(db.pending_clusters(1).await.unwrap().is_empty(), "skipped is not pending");
        // Naming a skipped cluster un-skips it.
        db.set_cluster_speaker(b.id, Some(sid), Some("auto"), Some(0.7)).await.unwrap();
        let b = db.get_cluster(mid, "B").await.unwrap().unwrap();
        assert!(!b.skipped && b.assigned_by.as_deref() == Some("auto") && b.match_score == Some(0.7));
        // Clearing the person clears provenance too but leaves skipped alone.
        db.set_cluster_speaker(b.id, None, None, None).await.unwrap();
        let b = db.get_cluster(mid, "B").await.unwrap().unwrap();
        assert!(b.speaker_id.is_none() && b.assigned_by.is_none() && b.match_score.is_none());

        // unlabel: the voiceprint this cluster contributed goes away, nothing else.
        assert_eq!(db.delete_voiceprints_for_cluster(a_id).await.unwrap(), 1);
        assert!(db.load_enrolled().await.unwrap().is_empty());
        let a = db.get_cluster(mid, "A").await.unwrap().unwrap();
        db.add_voiceprint(sid, &a, 7).await.unwrap();

        // Deleting the cluster cascades to its windows AND its voiceprints (no stale prints).
        db.delete_cluster(a_id).await.unwrap();
        assert!(db.cluster_windows(a_id).await.unwrap().is_empty());
        assert!(db.load_enrolled().await.unwrap().is_empty());
        assert_eq!(db.get_speaker(sid).await.unwrap().unwrap().voiceprints, 0);
        assert!(db.load_segments(mid).await.unwrap()[1].voice_cluster.is_none());
        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn merge_split_and_replace_primitives() {
        let mut db = Db::open_in_memory().await.unwrap();
        let (mid, a_id, _) = insert_with_voices_and_label(&mut db).await;
        let b = db.get_cluster(mid, "B").await.unwrap().unwrap();

        // merge B into A
        assert_eq!(db.move_segments(b.id, a_id).await.unwrap(), 1);
        db.delete_cluster(b.id).await.unwrap();
        db.update_cluster_stats(a_id, &blob(&[0.0, 1.0]), 2, 3, 3.0).await.unwrap();
        assert_eq!(db.cluster_windows(a_id).await.unwrap().len(), 3);
        let a = db.get_cluster(mid, "A").await.unwrap().unwrap();
        assert_eq!((a.n_windows, a.speech_secs), (3, 3.0));
        assert_eq!(a.centroid, blob(&[0.0, 1.0]));

        // split A into C and D (existing segment ids are 2, 3, 4)
        let ids: Vec<i64> = db.cluster_windows(a_id).await.unwrap().iter().map(|w| w.segment_id).collect();
        db.replace_cluster(
            a_id,
            mid,
            &[new_cluster("C", 1, 1.0), new_cluster("D", 1, 1.0)],
            &[voice(ids[0], 0, false), voice(ids[1], 1, false), voice(ids[2], 1, true)],
            9,
        )
        .await
        .unwrap();
        let labels: Vec<String> = db.list_clusters(mid).await.unwrap().into_iter().map(|c| c.cluster).collect();
        assert_eq!(labels, vec!["C", "D"]);
        assert_eq!(db.load_segments(mid).await.unwrap()[3].voice_cluster.as_deref(), Some("D"));

        // replace everything (the retro path)
        db.replace_meeting_clusters(mid, &[new_cluster("A", 3, 3.0)], &[voice(ids[0], 0, false)], 10)
            .await
            .unwrap();
        let clusters = db.list_clusters(mid).await.unwrap();
        assert_eq!(clusters.len(), 1);
        assert_eq!(db.meeting_windows(mid).await.unwrap().len(), 1);

        // a bad cluster index rolls the whole write back
        let err = db
            .replace_meeting_clusters(mid, &[new_cluster("Z", 1, 1.0)], &[voice(ids[0], 5, false)], 11)
            .await;
        assert!(err.is_err());
        assert_eq!(db.list_clusters(mid).await.unwrap()[0].cluster, "A", "rolled back");
        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn speakers_are_found_case_insensitively_and_renamed() {
        let mut db = Db::open_in_memory().await.unwrap();
        let id = db.add_speaker("Grace", "Hopper", 1).await.unwrap();
        assert_eq!(db.find_speaker("grace", "HOPPER").await.unwrap(), Some(id));
        assert_eq!(db.find_speaker("Grace", "Hoppe").await.unwrap(), None);
        assert!(db.rename_speaker(id, "Grace", "Brewster Hopper").await.unwrap());
        assert!(!db.rename_speaker(999, "x", "y").await.unwrap());
        let rows = db.list_speakers().await.unwrap();
        assert_eq!(rows[0].full_name(), "Grace Brewster Hopper");
        assert_eq!(rows[0].voiceprints, 0);
        db.close().await.unwrap();
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
        for table in ALL_TABLES {
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
