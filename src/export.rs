//! Per-meeting export (Phase 3) — human-readable Markdown + machine JSON.
//!
//! JSON is `serde_json::to_string_pretty(&[RenderedSegment])`, which is byte-identical to the
//! old `&[TranscriptSegment]` output while no speaker labels exist (the identity fields are
//! `Option` + skipped when `None`). Markdown adds a metadata header + `**[mm:ss] Speaker:** text`
//! lines. Dates are UTC-labeled (no local offset — that's a `time` soundness footgun and the
//! transcript body uses relative times).
//!
//! Export never decides what the text says: it receives already-rendered segments from
//! [`crate::render`], the single place corrections and names are applied.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use time::OffsetDateTime;
use time::macros::format_description;

use crate::db::MeetingRow;
use crate::render::RenderedSegment;

/// `mm:ss`, or `h:mm:ss` past an hour. Floored to whole seconds.
pub fn fmt_timestamp(secs: f64) -> String {
    let total = secs.max(0.0) as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// Human duration: `45s`, `5m 12s`, `1h 1m 1s`. Floored to whole seconds.
pub fn fmt_duration(secs: f64) -> String {
    let total = secs.max(0.0) as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}h {m}m {s}s")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

/// `YYYY-MM-DD HH:MM UTC` from a unix epoch (UTC — sound, no local-offset dependency).
pub fn fmt_utc(epoch: i64) -> String {
    let fmt = format_description!("[year]-[month]-[day] [hour]:[minute]");
    OffsetDateTime::from_unix_timestamp(epoch)
        .ok()
        .and_then(|dt| dt.format(&fmt).ok())
        .map(|s| format!("{s} UTC"))
        .unwrap_or_else(|| format!("epoch {epoch}"))
}

/// `YYYYMMDD-HHMMSS` (UTC) from a unix epoch — a filesystem-safe session directory name.
pub fn stamp_compact(epoch: i64) -> String {
    let fmt = format_description!("[year][month][day]-[hour][minute][second]");
    OffsetDateTime::from_unix_timestamp(epoch)
        .ok()
        .and_then(|dt| dt.format(&fmt).ok())
        .unwrap_or_else(|| format!("{epoch}"))
}

/// Pretty JSON — identical to the pipeline's transcript.json shape.
pub fn to_json(segs: &[RenderedSegment]) -> Result<String> {
    serde_json::to_string_pretty(segs).context("serialize transcript json")
}

/// Speaker-labeled, timestamped Markdown for one meeting.
pub fn to_markdown(m: &MeetingRow, segs: &[RenderedSegment]) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {}\n\n", m.title));
    out.push_str(&format!("- **Date:** {}\n", fmt_utc(m.started_at)));
    out.push_str(&format!("- **Duration:** {}\n", fmt_duration(m.duration_secs)));
    out.push_str(&format!("- **Model:** {} · **Language:** {}\n", m.model, m.lang));
    out.push_str(&format!("- **Capture:** {}\n", m.capture_status));
    out.push_str(&format!("- **Segments:** {}\n\n", m.segment_count));
    out.push_str("---\n\n");
    if segs.is_empty() {
        out.push_str("_(no speech detected)_\n");
    } else {
        for s in segs {
            out.push_str(&format!(
                "**[{}] {}:** {}\n\n",
                fmt_timestamp(s.t_start),
                s.display_label(),
                s.text.trim()
            ));
        }
    }
    out.push_str(&format!(
        "\n---\n_Source: {} · transcribed {}_\n",
        m.source_dir,
        fmt_utc(m.created_at)
    ));
    out
}

/// Write `<dir>/<basename>.md` and `<dir>/<basename>.json`; returns their paths.
pub fn write_exports(
    dir: &Path,
    basename: &str,
    meeting: &MeetingRow,
    segs: &[RenderedSegment],
) -> Result<(PathBuf, PathBuf)> {
    std::fs::create_dir_all(dir).with_context(|| format!("create export dir {}", dir.display()))?;
    let md_path = dir.join(format!("{basename}.md"));
    let json_path = dir.join(format!("{basename}.json"));
    atomic_export(&md_path, &to_markdown(meeting, segs), PublishMode::Replace)?;
    #[cfg(test)] crate::pipeline::cutpoint("manual_markdown_published");
    atomic_export(&json_path, &to_json(segs)?, PublishMode::Replace)?;
    Ok((md_path, json_path))
}

#[derive(Clone, Copy)]
enum PublishMode { Replace, CreateNew }

fn atomic_export(path: &Path, contents: &str, mode: PublishMode) -> Result<()> {
    use std::io::Write;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;
    let parent = path.parent().context("export target has no parent")?;
    let temporary = parent.join(format!(".export-{}.tmp", crate::recovery::new_identity()));
    let result = (|| -> Result<()> {
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&temporary)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
        match mode {
            PublishMode::Replace => std::fs::rename(&temporary, path)?,
            PublishMode::CreateNew => {
                let from = std::ffi::CString::new(temporary.as_os_str().as_bytes())?;
                let to = std::ffi::CString::new(path.as_os_str().as_bytes())?;
                // SAFETY: the owned C strings outlive the call; EXCL atomically refuses an existing target.
                if unsafe { libc::renamex_np(from.as_ptr(), to.as_ptr(), libc::RENAME_EXCL) } != 0 {
                    return Err(std::io::Error::last_os_error().into());
                }
            }
        }
        Ok(())
    })();
    if result.is_err() { let _ = std::fs::remove_file(&temporary); }
    result
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Publication { Initial, Recovered }

pub fn write_managed_exports(dir: &Path, key: &str, mode: Publication, meeting: &MeetingRow, segs: &[RenderedSegment]) -> Result<(PathBuf, PathBuf)> {
    write_managed_using(dir, key, mode, meeting, segs, || {})
}

fn write_managed_using(dir: &Path, key: &str, mode: Publication, meeting: &MeetingRow, segs: &[RenderedSegment], after_first: impl FnOnce()) -> Result<(PathBuf, PathBuf)> {
    anyhow::ensure!(key.len() == 32 && key.bytes().all(|byte| byte.is_ascii_hexdigit()), "invalid publication identity");
    crate::recovery::private_dir(dir)?;
    let markdown = to_markdown(meeting, segs);
    let json = to_json(segs)?;
    let md_path = dir.join("transcript.md");
    let json_path = dir.join("transcript.json");
    let collision = |error: &anyhow::Error| error.downcast_ref::<std::io::Error>()
        .is_some_and(|error| error.kind() == std::io::ErrorKind::AlreadyExists);
    if mode == Publication::Initial && !md_path.exists() && !json_path.exists() {
        match atomic_export(&md_path, &markdown, PublishMode::CreateNew) {
            Ok(()) => {
                #[cfg(test)]
                crate::pipeline::cutpoint("markdown_published");
                after_first();
                match atomic_export(&json_path, &json, PublishMode::CreateNew) {
                    Ok(()) => return Ok((md_path, json_path)),
                    Err(error) if collision(&error) => {},
                    Err(error) => return Err(error),
                }
            }
            Err(error) if collision(&error) => {},
            Err(error) => return Err(error),
        }
    }
    let root = dir.join(".recovered").join(key);
    crate::recovery::private_dir(&root)?;
    let destination = root.join(crate::recovery::new_identity());
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new().mode(0o700).create(&destination)?;
    let md_path = destination.join("transcript.md");
    let json_path = destination.join("transcript.json");
    atomic_export(&md_path, &markdown, PublishMode::CreateNew)?;
    #[cfg(test)]
    crate::pipeline::cutpoint("markdown_published");
    atomic_export(&json_path, &json, PublishMode::CreateNew)?;
    Ok((md_path, json_path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{IdentityMap, Vocab, render_fresh};
    use crate::transcript::{Speaker, TranscriptSegment};

    fn row() -> MeetingRow {
        MeetingRow {
            id: 1,
            title: "team sync".into(),
            source_dir: "capture".into(),
            model: "ggml-large-v3".into(),
            lang: "es".into(),
            started_at: 1_752_900_000,
            duration_secs: 312.5,
            segment_count: 2,
            created_at: 1_752_900_400,
            capture_status: "unknown".into(),
        }
    }

    fn seg(sp: Speaker, t: f64, text: &str) -> RenderedSegment {
        let raw = TranscriptSegment {
            speaker: sp,
            text: text.into(),
            t_start: t,
            t_end: t + 1.0,
            confidence: 0.9,
        };
        render_fresh(&[raw], &IdentityMap::empty(), &Vocab::empty())
            .pop()
            .expect("one in, one out")
    }

    #[test]
    fn timestamp_formats() {
        assert_eq!(fmt_timestamp(0.0), "00:00");
        assert_eq!(fmt_timestamp(65.0), "01:05");
        assert_eq!(fmt_timestamp(3661.0), "1:01:01");
    }

    #[test]
    fn duration_formats() {
        assert_eq!(fmt_duration(45.0), "45s");
        assert_eq!(fmt_duration(312.5), "5m 12s");
        assert_eq!(fmt_duration(3661.0), "1h 1m 1s");
    }

    #[test]
    fn markdown_has_labels_and_timestamps() {
        let segs = vec![seg(Speaker::You, 0.0, "hola"), seg(Speaker::Others, 65.0, "chau")];
        let md = to_markdown(&row(), &segs);
        assert!(md.contains("# team sync"));
        assert!(md.contains("**[00:00] You:** hola"));
        assert!(md.contains("**[01:05] Others:** chau"));
        assert!(md.contains("UTC"));
    }

    #[test]
    fn markdown_empty_transcript_renders_marker() {
        let mut m = row();
        m.segment_count = 0;
        let md = to_markdown(&m, &[]);
        assert!(md.contains("_(no speech detected)_"));
    }
}
