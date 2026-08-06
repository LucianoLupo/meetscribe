//! meetscribe — local-first, background macOS meeting transcriber.
//!
//! CLI entry + subcommand dispatch over the proven layers:
//!   - default (no subcommand) = a capture session (mic "You" + system-tap "Others", one clock);
//!   - `transcribe <dir>`      = batch resample → VAD → whisper → merge → store + export;
//!   - `list` / `export <id>`  = read back stored meetings;
//!   - `vocab …`               = correction rules applied at RENDER time (never to stored text);
//!   - `rerender [--all]`      = re-render stored meetings into their session dirs, preview by
//!     default — how a new correction reaches past transcripts;
//!   - `detect [--watch]`      = which allowlisted app (if any) holds the mic (Phase-4 detector).
//!
//! Capture usage:
//!   meetscribe [--out <dir>] [--seconds <n>] [--rebuild-after <n>]
//!   (no --seconds ⇒ records until you press Enter)

mod capture;
mod resample;
mod vad;
mod asr;
mod transcript;
mod db;
mod render;
mod export;
mod detect;
mod session;
mod pipeline;
mod daemon;
mod launchd;
mod config;
mod maintenance;
mod status;
mod tray;

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

struct Args {
    out_dir: PathBuf,
    seconds: Option<u64>,
    /// Debug hook: force a rebuild this many seconds in, to exercise the request→poll→rebuild
    /// path without a real route change.
    rebuild_after: Option<u64>,
}

fn parse_args() -> Args {
    let mut out_dir = PathBuf::from("capture");
    let mut seconds = None;
    let mut rebuild_after = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" | "-o" => {
                if let Some(v) = it.next() {
                    out_dir = PathBuf::from(v);
                }
            }
            "--seconds" | "-s" => {
                seconds = it.next().and_then(|v| v.parse().ok());
            }
            "--rebuild-after" => {
                rebuild_after = it.next().and_then(|v| v.parse().ok());
            }
            "-h" | "--help" => {
                eprintln!("usage: meetscribe [--out <dir>] [--seconds <n>] [--rebuild-after <n>]");
                std::process::exit(0);
            }
            _ => {}
        }
    }
    Args {
        out_dir,
        seconds,
        rebuild_after,
    }
}

/// `meetscribe transcribe <dir> [...]` — thin CLI wrapper over `pipeline::transcribe_and_store`:
/// parse args, run the pipeline (which stores AND exports), then print the transcript + RTF.
fn run_transcribe(argv: &[String]) -> Result<()> {
    let mut dir: Option<PathBuf> = None;
    let mut model = String::from("models/ggml-large-v3.bin");
    let mut lang = String::from("es");
    let mut title: Option<String> = None;
    let mut db_path: Option<PathBuf> = None;
    let mut export_dir: Option<PathBuf> = None;
    let mut no_store = false;
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" | "-m" => {
                if let Some(v) = it.next() {
                    model = v.clone();
                }
            }
            "--lang" | "-l" => {
                if let Some(v) = it.next() {
                    lang = v.clone();
                }
            }
            "--title" | "-t" => {
                if let Some(v) = it.next() {
                    title = Some(v.clone());
                }
            }
            "--db" => {
                if let Some(v) = it.next() {
                    db_path = Some(PathBuf::from(v));
                }
            }
            "--export-dir" => {
                if let Some(v) = it.next() {
                    export_dir = Some(PathBuf::from(v));
                }
            }
            "--no-store" => no_store = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: meetscribe transcribe <dir> [--title <t>] [--model <ggml.bin>] \
                     [--lang <code>] [--db <path>] [--export-dir <dir>] [--no-store]"
                );
                return Ok(());
            }
            s if !s.starts_with('-') && dir.is_none() => dir = Some(PathBuf::from(s)),
            _ => {}
        }
    }
    let dir = dir.context("transcribe: missing <dir> (e.g. `meetscribe transcribe capture`)")?;
    let opts = pipeline::PipelineOpts {
        model: PathBuf::from(model),
        lang,
        title,
        db_path: db_path.unwrap_or_else(default_db_path),
        export_dir: export_dir.unwrap_or_else(|| dir.clone()),
        no_store,
    };
    let out = pipeline::transcribe_and_store(&dir, &opts)?;

    println!("\n===== TRANSCRIPT ({}) =====", dir.display());
    for s in &out.segments {
        println!(
            "[{:7.2}-{:7.2}] {:<7} {}",
            s.t_start,
            s.t_end,
            s.speaker.label(),
            s.text
        );
    }
    println!(
        "\nsegments: {} | meeting: {:.1}s | wall: {:.1}s | RTF: {:.3}x (both channels through whisper)",
        out.segments.len(),
        out.meeting_secs,
        out.wall_secs,
        out.rtf
    );
    println!("wrote {}", out.md_path.display());
    println!("wrote {}", out.json_path.display());
    Ok(())
}

/// `~/.meetscribe` — the app's data/config directory (`None` if `$HOME` is unset). The single
/// owner of the base-dir join; the daemon/launchd build their own validated-absolute variant.
pub(crate) fn base_dir() -> Option<PathBuf> {
    home_dir().map(|h| h.join(".meetscribe"))
}

/// `~/.meetscribe/meetscribe.db` (falls back to a repo-local path if `$HOME` is unset).
pub(crate) fn default_db_path() -> PathBuf {
    base_dir()
        .map(|b| b.join("meetscribe.db"))
        .unwrap_or_else(|| PathBuf::from("meetscribe.db"))
}

/// `~/.meetscribe/exports/` (the default target for `export <id>`).
pub(crate) fn default_export_dir() -> PathBuf {
    base_dir()
        .map(|b| b.join("exports"))
        .unwrap_or_else(|| PathBuf::from("exports"))
}

pub(crate) fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

pub(crate) fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub(crate) fn new_runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")
}

/// `meetscribe list [--db <path>]` — the stored meetings, newest first.
fn run_list(argv: &[String]) -> Result<()> {
    let mut db_path = default_db_path();
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--db" => {
                if let Some(v) = it.next() {
                    db_path = PathBuf::from(v);
                }
            }
            "-h" | "--help" => {
                eprintln!("usage: meetscribe list [--db <path>]");
                return Ok(());
            }
            _ => {}
        }
    }
    if !db_path.exists() {
        println!("no meetings yet — db {} does not exist", db_path.display());
        return Ok(());
    }

    let rt = new_runtime()?;
    let meetings = rt.block_on(async {
        let mut database = db::Db::open(&db_path).await?;
        let m = database.list_meetings().await?;
        database.close().await?;
        anyhow::Ok(m)
    })?;

    if meetings.is_empty() {
        println!("no meetings stored in {}", db_path.display());
        return Ok(());
    }
    println!(
        "{:>3}  {:<20}  {:>9}  {:>5}  TITLE",
        "ID", "DATE", "DURATION", "SEGS"
    );
    for m in &meetings {
        println!(
            "{:>3}  {:<20}  {:>9}  {:>5}  {}",
            m.id,
            export::fmt_utc(m.started_at),
            export::fmt_duration(m.duration_secs),
            m.segment_count,
            m.title
        );
    }
    Ok(())
}

/// Load a meeting's row plus its segments rendered with the current vocabulary (and, later,
/// speaker names). `raw` skips rendering entirely — the escape hatch for diffing what changed.
fn load_rendered(
    db_path: &Path,
    id: i64,
    raw: bool,
) -> Result<(db::MeetingRow, Vec<render::RenderedSegment>)> {
    let rt = new_runtime()?;
    let (row, segs, vocab_rows) = rt.block_on(async {
        let mut database = db::Db::open(db_path).await?;
        let row = database
            .get_meeting(id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("no meeting with id {id} in {}", db_path.display()))?;
        let segs = database.load_segments(id).await?;
        let vocab = if raw { Vec::new() } else { database.load_enabled_vocab().await? };
        database.close().await?;
        anyhow::Ok((row, segs, vocab))
    })?;

    let (vocab, warnings) = render::Vocab::compile(&vocab_rows);
    for w in &warnings {
        log::warn!("{w}");
    }
    let rendered = render::render(&segs, &render::IdentityMap::empty(), &vocab);
    Ok((row, rendered))
}

/// `meetscribe export <id> [--db <path>] [--export-dir <dir>] [--raw]` — re-export from the DB.
/// This exercises the real DB read path (round-trip proof).
fn run_export(argv: &[String]) -> Result<()> {
    let mut id: Option<i64> = None;
    let mut db_path = default_db_path();
    let mut export_dir: Option<PathBuf> = None;
    let mut raw = false;
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--db" => {
                if let Some(v) = it.next() {
                    db_path = PathBuf::from(v);
                }
            }
            "--export-dir" => {
                if let Some(v) = it.next() {
                    export_dir = Some(PathBuf::from(v));
                }
            }
            "--raw" => raw = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: meetscribe export <id> [--db <path>] [--export-dir <dir>] [--raw]"
                );
                return Ok(());
            }
            s if !s.starts_with('-') && id.is_none() => {
                id = Some(
                    s.parse::<i64>()
                        .with_context(|| format!("export: invalid id '{s}'"))?,
                );
            }
            _ => {}
        }
    }
    let id = id.context("export: missing <id> (e.g. `meetscribe export 1`)")?;
    let export_dir = export_dir.unwrap_or_else(default_export_dir);
    if !db_path.exists() {
        anyhow::bail!("db {} does not exist — nothing to export", db_path.display());
    }

    let (row, rendered) = load_rendered(&db_path, id, raw)?;
    let (md_path, json_path) =
        export::write_exports(&export_dir, &format!("meeting-{id}"), &row, &rendered)?;
    println!("wrote {}", md_path.display());
    println!("wrote {}", json_path.display());
    Ok(())
}

/// `meetscribe rerender [--all | <id>…] [--db <path>] [--write]` — re-render stored meetings
/// into their ORIGINAL session directories, where the transcripts people actually read live
/// (`~/.meetscribe/exports/` is only ever written by an explicit `export <id>`).
///
/// This is what makes the corrections loop retroactive: one rule, then every past meeting
/// improves — with no whisper re-run. Preview by default; `--write` commits. Overwriting dozens
/// of real session files deserves at least the caution `vocab test` gives a single rule.
fn run_rerender(argv: &[String]) -> Result<()> {
    let mut db_path = default_db_path();
    let mut ids: Vec<i64> = Vec::new();
    let mut all = false;
    let mut write = false;
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--db" => {
                if let Some(v) = it.next() {
                    db_path = PathBuf::from(v);
                }
            }
            "--all" => all = true,
            "--write" => write = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: meetscribe rerender [--all | <id>…] [--db <path>] [--write]\n\
                     \n\
                     Re-renders stored meetings into their session directories using the current\n\
                     vocabulary. Previews by default; pass --write to overwrite the files."
                );
                return Ok(());
            }
            s if !s.starts_with('-') => {
                ids.push(
                    s.parse::<i64>()
                        .with_context(|| format!("rerender: invalid id '{s}'"))?,
                );
            }
            other => anyhow::bail!("rerender: unknown flag '{other}'"),
        }
    }
    if !all && ids.is_empty() {
        anyhow::bail!("rerender: pass --all or one or more meeting ids");
    }
    if !db_path.exists() {
        anyhow::bail!("db {} does not exist", db_path.display());
    }

    if all {
        let rt = new_runtime()?;
        let meetings = rt.block_on(async {
            let mut database = db::Db::open(&db_path).await?;
            let m = database.list_meetings().await?;
            database.close().await?;
            anyhow::Ok(m)
        })?;
        ids = meetings.iter().map(|m| m.id).collect();
    }

    let (mut changed, mut unchanged, mut skipped) = (0usize, 0usize, 0usize);
    for id in ids {
        let (row, rendered) = load_rendered(&db_path, id, false)?;

        // source_dir is untrusted historical data: early rows hold a relative path long since
        // overwritten. Skip with a log rather than failing the whole run.
        let dir = PathBuf::from(&row.source_dir);
        if !dir.is_absolute() || !dir.is_dir() {
            log::warn!(
                "meeting {id}: source dir '{}' is not a usable absolute directory — skipped",
                row.source_dir
            );
            skipped += 1;
            continue;
        }

        let md = export::to_markdown(&row, &rendered);
        let json = export::to_json(&rendered)?;
        let md_path = dir.join("transcript.md");
        let json_path = dir.join("transcript.json");

        let md_differs = std::fs::read_to_string(&md_path).map(|c| c != md).unwrap_or(true);
        let json_differs = std::fs::read_to_string(&json_path).map(|c| c != json).unwrap_or(true);
        if !md_differs && !json_differs {
            unchanged += 1;
            continue;
        }
        changed += 1;

        if write {
            std::fs::write(&md_path, &md).with_context(|| format!("write {}", md_path.display()))?;
            std::fs::write(&json_path, &json)
                .with_context(|| format!("write {}", json_path.display()))?;
            println!("meeting {id}: rewrote {}", dir.display());
        } else {
            println!("meeting {id}: would rewrite {}", dir.display());
            if let Some((before, after)) = first_difference(&md_path, &md) {
                println!("    - {before}");
                println!("    + {after}");
            }
        }
    }

    let hint = match (write, changed) {
        (true, _) => " (written)",
        (false, 0) => "",
        (false, _) => " — re-run with --write to apply",
    };
    println!("\n{changed} to change · {unchanged} already current · {skipped} skipped{hint}");
    Ok(())
}

/// First differing non-empty line between a file on disk and the freshly rendered text.
fn first_difference(path: &Path, fresh: &str) -> Option<(String, String)> {
    let old = std::fs::read_to_string(path).ok()?;
    old.lines()
        .zip(fresh.lines())
        .find(|(a, b)| a != b && !a.trim().is_empty())
        .map(|(a, b)| (a.to_string(), b.to_string()))
}

const VOCAB_USAGE: &str = "usage: meetscribe vocab <add|list|enable|disable|test> [...]\n\
    \n\
    \x20 add <pattern> <replacement> [--regex]   add a correction rule\n\
    \x20 list                                    all rules, in application order\n\
    \x20 enable <id> | disable <id>              toggle a rule (rules are never deleted)\n\
    \x20 test [<id> | --all]                     preview affected segments before committing\n\
    \n\
    common: [--db <path>]";

/// `meetscribe vocab …` — manage the correction rules applied at render time.
///
/// Rules are disabled, never deleted, so ids never shift and the documented "applied in id
/// order" guarantee holds permanently.
fn run_vocab(argv: &[String]) -> Result<()> {
    let mut db_path = default_db_path();
    let mut is_regex = false;
    let mut all = false;
    let mut positional: Vec<&str> = Vec::new();
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--db" => {
                if let Some(v) = it.next() {
                    db_path = PathBuf::from(v);
                }
            }
            "--regex" => is_regex = true,
            "--all" => all = true,
            "-h" | "--help" => {
                eprintln!("{VOCAB_USAGE}");
                return Ok(());
            }
            s if !s.starts_with('-') => positional.push(s),
            other => anyhow::bail!("vocab: unknown flag '{other}'\n\n{VOCAB_USAGE}"),
        }
    }
    let sub = *positional.first().context(VOCAB_USAGE)?;

    let rt = new_runtime()?;
    match sub {
        "add" => {
            let pattern = positional
                .get(1)
                .context("vocab add: missing <pattern>\n\nexample: vocab add \"Cloud Code\" \"Claude Code\"")?;
            let replacement = positional.get(2).context("vocab add: missing <replacement>")?;
            // Compile before storing: a rule that cannot compile is a typo, and catching it here
            // beats discovering it as a warning on every future export.
            let probe = db::VocabRow {
                id: 0,
                pattern: (*pattern).to_string(),
                replacement: (*replacement).to_string(),
                is_regex,
                enabled: true,
                created_at: 0,
            };
            let (_, warnings) = render::Vocab::compile(std::slice::from_ref(&probe));
            if let Some(w) = warnings.first() {
                anyhow::bail!("{w}");
            }
            let id = rt.block_on(async {
                let mut database = db::Db::open(&db_path).await?;
                let id = database
                    .add_vocab(pattern, replacement, is_regex, now_epoch())
                    .await?;
                database.close().await?;
                anyhow::Ok(id)
            })?;
            println!(
                "added rule {id}: '{pattern}' → '{replacement}'{}",
                if is_regex { " (regex)" } else { "" }
            );
            println!("preview it with:  meetscribe vocab test {id}");
        }
        "list" => {
            let rows = rt.block_on(async {
                let mut database = db::Db::open(&db_path).await?;
                let r = database.list_vocab().await?;
                database.close().await?;
                anyhow::Ok(r)
            })?;
            if rows.is_empty() {
                println!("no vocabulary rules yet — add one with `meetscribe vocab add`");
                return Ok(());
            }
            println!("{:>4}  {:<5}  {:<20}  {:<24}  REPLACEMENT", "ID", "STATE", "ADDED", "PATTERN");
            for r in &rows {
                println!(
                    "{:>4}  {:<5}  {:<20}  {:<24}  {}{}",
                    r.id,
                    if r.enabled { "on" } else { "off" },
                    export::fmt_utc(r.created_at),
                    r.pattern,
                    r.replacement,
                    if r.is_regex { "   (regex)" } else { "" }
                );
            }
        }
        "enable" | "disable" => {
            let want = sub == "enable";
            let id: i64 = positional
                .get(1)
                .context("vocab enable/disable: missing <id>")?
                .parse()
                .context("vocab: <id> must be a number")?;
            let found = rt.block_on(async {
                let mut database = db::Db::open(&db_path).await?;
                let found = database.set_vocab_enabled(id, want).await?;
                database.close().await?;
                anyhow::Ok(found)
            })?;
            if found {
                println!("rule {id} {}", if want { "enabled" } else { "disabled" });
            } else {
                anyhow::bail!("no vocabulary rule with id {id}");
            }
        }
        "test" => {
            let only: Option<i64> = match positional.get(1) {
                Some(s) => Some(s.parse().context("vocab test: <id> must be a number")?),
                None if all => None,
                None => anyhow::bail!("vocab test: pass an <id> or --all"),
            };
            run_vocab_test(&db_path, only)?;
        }
        other => anyhow::bail!("vocab: unknown subcommand '{other}'\n\n{VOCAB_USAGE}"),
    }
    Ok(())
}

/// Preview which stored segments a rule (or all rules) would change. Read-only: nothing on disk
/// moves until `rerender --write`.
fn run_vocab_test(db_path: &Path, only: Option<i64>) -> Result<()> {
    if !db_path.exists() {
        anyhow::bail!("db {} does not exist", db_path.display());
    }
    // One DB open for the whole preview — segments come back raw (never rendered), so what is
    // compared is the stored text, not the output of some other rule.
    let rt = new_runtime()?;
    let (meetings, rows) = rt.block_on(async {
        let mut database = db::Db::open(db_path).await?;
        let meta = database.list_meetings().await?;
        let mut meetings = Vec::with_capacity(meta.len());
        for m in meta {
            let segs = database.load_segments(m.id).await?;
            meetings.push((m.id, segs));
        }
        let mut rows = database.list_vocab().await?;
        if let Some(id) = only {
            rows.retain(|r| r.id == id);
        } else {
            rows.retain(|r| r.enabled);
        }
        database.close().await?;
        anyhow::Ok((meetings, rows))
    })?;

    if rows.is_empty() {
        match only {
            Some(id) => anyhow::bail!("no vocabulary rule with id {id}"),
            None => {
                println!("no enabled vocabulary rules to test");
                return Ok(());
            }
        }
    }
    let (vocab, warnings) = render::Vocab::compile(&rows);
    for w in &warnings {
        log::warn!("{w}");
    }

    let mut hits = 0usize;
    let mut shown = 0usize;
    let mut touched_meetings = 0usize;
    const MAX_SHOWN: usize = 20;
    for (meeting_id, segs) in &meetings {
        let before = hits;
        for s in segs {
            let after = vocab.apply(&s.seg.text);
            if after == s.seg.text {
                continue;
            }
            hits += 1;
            if shown < MAX_SHOWN {
                shown += 1;
                println!(
                    "meeting {meeting_id} segment {} [{}] rules {:?}",
                    s.id,
                    export::fmt_timestamp(s.seg.t_start),
                    vocab.matching_rules(&s.seg.text)
                );
                println!("    - {}", s.seg.text.trim());
                println!("    + {}", after.trim());
            }
        }
        if hits > before {
            touched_meetings += 1;
        }
    }
    if hits > shown {
        println!("\n… and {} more affected segment(s)", hits - shown);
    }
    if hits == 0 {
        println!("no stored segment would change");
        return Ok(());
    }
    println!(
        "\n{hits} segment(s) across {touched_meetings} meeting(s) would change. \
         Apply with: meetscribe rerender --all --write"
    );
    Ok(())
}

/// `meetscribe detect [--watch]` — print which allowlisted app (if any) currently holds the mic,
/// plus the live audio-active process table. The mechanical verify tool for the detector: run it
/// while a real meeting app holds the mic (Active) vs while only music plays (Idle).
fn run_detect(argv: &[String]) -> Result<()> {
    let watch = argv.iter().any(|a| a == "--watch" || a == "-w");
    // Mirror the daemon: honor ~/.meetscribe/config.toml so `detect` shows what the daemon sees.
    // Uses the PURE `Config::load` (no disk writes) — a diagnostic must not materialize config.
    let base = base_dir();
    let allowlist: Vec<String> = base
        .as_deref()
        .map(|b| config::Config::load(b).effective_allowlist())
        .unwrap_or_else(|| {
            detect::DEFAULT_ALLOWLIST
                .iter()
                .map(|s| (*s).to_string())
                .collect()
        });
    let al: Vec<&str> = allowlist.iter().map(String::as_str).collect();
    loop {
        let snap = detect::snapshot()?;
        let active = detect::active_app_in(&snap, &al);
        match &active {
            Some(app) => println!("MEETING ACTIVE — {app} holds the mic  ({} processes)", snap.len()),
            None => println!("idle — no allowlisted app holds the mic  ({} processes)", snap.len()),
        }
        for p in &snap {
            let Some(bid) = p.bundle_id.as_deref() else {
                continue;
            };
            if !p.input && !p.output {
                continue;
            }
            let allowed = detect::is_allowlisted(bid, &al);
            println!(
                "  {:<42} input={:<5} output={:<5} allowlisted={}",
                bid, p.input, p.output, allowed
            );
        }
        if !watch {
            break;
        }
        println!("---");
        std::thread::sleep(Duration::from_millis(1500));
    }
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Subcommand dispatch. Anything unrecognized falls through to the capture flow
    // (back-compatible: `meetscribe --seconds 300` still records).
    let argv: Vec<String> = std::env::args().collect();
    match argv.get(1).map(String::as_str) {
        Some("transcribe") => return run_transcribe(&argv[2..]),
        Some("list") => return run_list(&argv[2..]),
        Some("export") => return run_export(&argv[2..]),
        Some("rerender") => return run_rerender(&argv[2..]),
        Some("vocab") => return run_vocab(&argv[2..]),
        Some("detect") => return run_detect(&argv[2..]),
        Some("daemon") => return daemon::run_daemon(&argv[2..]),
        Some("tray") => return tray::run_tray(&argv[2..]),
        Some("install") => return launchd::run_install(&argv[2..]),
        Some("uninstall") => return launchd::run_uninstall(&argv[2..]),
        _ => {}
    }

    let args = parse_args();

    log::info!("meetscribe capture — starting single-aggregate mic+tap capture");
    log::info!(
        "(first run triggers TWO prompts: Microphone and \"record system audio\" — approve both)"
    );

    // Stop control: a --seconds deadline, else press Enter. Only wire the stdin thread when there's
    // no deadline — a non-interactive stdin hits EOF instantly and would otherwise stop capture on
    // the first loop iteration. Both are folded into the `stop` predicate the session polls.
    let stop_flag = Arc::new(AtomicBool::new(false));
    if args.seconds.is_none() {
        let sf = stop_flag.clone();
        std::thread::spawn(move || {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
            sf.store(true, Ordering::Release);
        });
        log::info!("recording… press Enter to stop.");
    } else if let Some(s) = args.seconds {
        log::info!("recording for {s}s…");
    }
    let deadline = args.seconds.map(|s| Instant::now() + Duration::from_secs(s));
    let sf = stop_flag.clone();
    let stop = move || sf.load(Ordering::Acquire) || deadline.is_some_and(|d| Instant::now() >= d);

    let summary = session::run_capture(&args.out_dir, stop, args.rebuild_after)?;
    log::info!(
        "capture finished: {:.1}s, {} segment(s), {} rebuild(s), num_buffers={}, {} sample(s) dropped",
        summary.duration_secs,
        summary.segments,
        summary.rebuilds,
        summary.num_buffers,
        summary.dropped
    );
    Ok(())
}
