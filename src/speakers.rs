//! `meetscribe speakers …` — name the far-end voices (Batch E).
//!
//! The owner's loop: `list <meeting>` → `play <meeting> <cluster>` → `label <meeting> <cluster>
//! "<first>" "<last>"`. A labelled cluster enrols one voiceprint; every later meeting where that
//! voice recurs is recognised at transcription time. `skip` marks a voice unknown-by-choice so it
//! stops appearing in `list --pending`. `cluster` is the retro path for meetings stored before
//! speaker identity existed; `match` re-runs recognition after new labels.
//!
//! Nothing here writes to `transcript_segments`: identity lives in its own tables, and rendering
//! picks it up through `db::load_segments` (so `export`/`rerender` show the names).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::{db, export, spk, voices};

const USAGE: &str = "usage: meetscribe speakers <list|play|label|unlabel|skip|merge|split|cluster|match|rename> [...]\n\
    \n\
    \x20 list [--pending]                          people (with voiceprint counts) | voices still to name\n\
    \x20 list <meeting-id>                         that meeting's far-end voice clusters\n\
    \x20 play <meeting-id> <cluster> [--clips 3] [--secs 6]\n\
    \x20                                           hear a voice before naming it (afplay)\n\
    \x20 label <meeting-id> <cluster> \"<first>\" \"<last>\"\n\
    \x20                                           name a voice (creates or reuses the person; enrols it)\n\
    \x20 unlabel <meeting-id> <cluster>            forget the name (and this cluster's voiceprint)\n\
    \x20 skip <meeting-id> <cluster>               leave this voice unknown on purpose\n\
    \x20 merge <meeting-id> <A> <B> [<C>…]         B… are the same voice as A\n\
    \x20 split <meeting-id> <cluster> [--cut 0.30] a cluster that mixes two voices\n\
    \x20 cluster <meeting-id> | --all [--recluster] [--speaker-model <onnx>]\n\
    \x20                                           find the voices in meetings stored before speaker identity\n\
    \x20 match <meeting-id> | --all                re-run recognition against the current voiceprints\n\
    \x20 rename <speaker-id> \"<first>\" \"<last>\"   fix a typo\n\
    \n\
    common: [--db <path>]   (an unlabelled voice renders as \"Others\"; relabel = unlabel + label)";

/// Minimum spacing between `play` clips when the cluster allows it.
const PLAY_MIN_GAP_SECS: f64 = 120.0;

pub(crate) fn run_speakers(argv: &[String]) -> Result<()> {
    let mut db_path = crate::default_db_path();
    let mut speaker_model = PathBuf::from(crate::SPEAKER_MODEL_REL);
    let mut all = false;
    let mut recluster = false;
    let mut pending = false;
    let mut clips = 3usize;
    let mut secs = 6.0f64;
    let mut cut = 0.30f32;
    let mut positional: Vec<&str> = Vec::new();
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--db" => db_path = PathBuf::from(it.next().context("--db needs a path")?),
            "--speaker-model" => speaker_model = PathBuf::from(it.next().context("--speaker-model needs a path")?),
            "--clips" => clips = it.next().context("--clips needs a number")?.parse().context("--clips must be a number")?,
            "--secs" => secs = it.next().context("--secs needs a number")?.parse().context("--secs must be a number")?,
            "--cut" => cut = it.next().context("--cut needs a number")?.parse().context("--cut must be a number")?,
            "--all" => all = true,
            "--recluster" => recluster = true,
            "--pending" => pending = true,
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return Ok(());
            }
            s if !s.starts_with('-') => positional.push(s),
            other => bail!("speakers: unknown flag '{other}'\n\n{USAGE}"),
        }
    }
    let sub = *positional.first().context(USAGE)?;
    // Never conjure a database out of a typo'd path: every subcommand here reads first.
    if !db_path.exists() {
        bail!("db {} does not exist", db_path.display());
    }
    let args = &positional[1..];
    let rt = crate::new_runtime()?;

    match sub {
        "list" => match args.first() {
            Some(id) => list_meeting(&rt, &db_path, parse_id(id, "meeting-id")?),
            None if pending => list_pending(&rt, &db_path),
            None => list_people(&rt, &db_path),
        },
        "play" => {
            let (mid, label) = meeting_and_cluster(args)?;
            play(&rt, &db_path, mid, label, clips.max(1), secs.max(1.0))
        }
        "label" => {
            let (mid, label) = meeting_and_cluster(args)?;
            let (first, last) = names(&args[2..], "label")?;
            label_cluster(&rt, &db_path, mid, label, first, last)
        }
        "unlabel" => {
            let (mid, label) = meeting_and_cluster(args)?;
            unlabel_cluster(&rt, &db_path, mid, label, false)
        }
        "skip" => {
            let (mid, label) = meeting_and_cluster(args)?;
            unlabel_cluster(&rt, &db_path, mid, label, true)
        }
        "merge" => {
            let (mid, label) = meeting_and_cluster(args)?;
            if args.len() < 3 {
                bail!("speakers merge: pass the cluster to keep, then one or more clusters to fold into it");
            }
            merge_clusters(&rt, &db_path, mid, label, &args[2..])
        }
        "split" => {
            let (mid, label) = meeting_and_cluster(args)?;
            split_cluster(&rt, &db_path, mid, label, cut)
        }
        "cluster" => {
            let ids = meeting_ids(args, all, "cluster")?;
            cluster_meetings(&rt, &db_path, ids, recluster, &speaker_model)
        }
        "match" => {
            let ids = meeting_ids(args, all, "match")?;
            match_meetings(&rt, &db_path, ids)
        }
        "rename" => {
            let sid = parse_id(args.first().context("speakers rename: missing <speaker-id>")?, "speaker-id")?;
            let (first, last) = names(&args[1..], "rename")?;
            rename(&rt, &db_path, sid, first, last)
        }
        other => bail!("speakers: unknown subcommand '{other}'\n\n{USAGE}"),
    }
}

// ---------------------------------------------------------------- argument helpers

fn parse_id(s: &str, what: &str) -> Result<i64> {
    s.parse::<i64>().with_context(|| format!("speakers: <{what}> must be a number, got '{s}'"))
}

fn meeting_and_cluster<'a>(args: &[&'a str]) -> Result<(i64, &'a str)> {
    let mid = parse_id(args.first().context("speakers: missing <meeting-id>")?, "meeting-id")?;
    let label = *args.get(1).context("speakers: missing <cluster> (a letter from `speakers list <meeting-id>`)")?;
    Ok((mid, label))
}

/// Exactly two positionals: a compound first or last name must be shell-quoted, otherwise the
/// split is a guess and a guess in a name is worse than an error.
fn names<'a>(args: &[&'a str], sub: &str) -> Result<(&'a str, &'a str)> {
    match args {
        [first, last] if !first.trim().is_empty() && !last.trim().is_empty() => Ok((first.trim(), last.trim())),
        _ => bail!(
            "speakers {sub}: pass exactly \"<first>\" \"<last>\" (quote compound names, e.g. \"María José\" \"García Lorca\")"
        ),
    }
}

/// `<meeting-id>` or `--all` (an empty list means every meeting).
fn meeting_ids(args: &[&str], all: bool, sub: &str) -> Result<Vec<i64>> {
    if all {
        return Ok(Vec::new());
    }
    let ids: Vec<i64> = args.iter().map(|s| parse_id(s, "meeting-id")).collect::<Result<_>>()?;
    if ids.is_empty() {
        bail!("speakers {sub}: pass one or more <meeting-id>s or --all");
    }
    Ok(ids)
}

// ---------------------------------------------------------------- db helpers

async fn cluster_or_bail(db: &mut db::Db, mid: i64, label: &str) -> Result<db::ClusterRow> {
    db.get_meeting(mid).await?.with_context(|| format!("no meeting with id {mid}"))?;
    db.get_cluster(mid, label)
        .await?
        .with_context(|| format!("meeting {mid} has no cluster '{label}' — see `meetscribe speakers list {mid}`"))
}

fn describe(c: &db::ClusterRow, name: Option<&str>) -> String {
    let who = match (name, c.skipped) {
        (Some(n), _) => n.to_string(),
        (None, true) => "(unknown, skipped)".to_string(),
        (None, false) => "?".to_string(),
    };
    let via = match (c.assigned_by.as_deref(), c.match_score) {
        (Some("auto"), Some(s)) => format!("auto {s:.2}"),
        (Some("manual"), _) => "manual".to_string(),
        _ => String::new(),
    };
    format!(
        "{:<4} {:>4} win  {:>6}  {:<28} {}",
        c.cluster,
        c.n_windows,
        export::fmt_duration(c.speech_secs),
        who,
        via
    )
}

// ---------------------------------------------------------------- list

fn list_people(rt: &tokio::runtime::Runtime, db_path: &Path) -> Result<()> {
    let (people, pending) = rt.block_on(async {
        let mut db = db::Db::open(db_path).await?;
        let p = db.list_speakers().await?;
        let n = db.pending_clusters(spk::MIN_WINDOWS as i64).await?.len();
        db.close().await?;
        anyhow::Ok((p, n))
    })?;
    if people.is_empty() {
        println!("no named speakers yet — `meetscribe speakers list --pending` shows voices waiting for a name");
    } else {
        println!("{:>4}  {:<30}  {:<20}  VOICEPRINTS", "ID", "NAME", "ADDED");
        for p in &people {
            println!("{:>4}  {:<30}  {:<20}  {}", p.id, p.full_name(), export::fmt_utc(p.created_at), p.voiceprints);
        }
    }
    if pending > 0 {
        println!("\n{pending} voice(s) still unnamed — `meetscribe speakers list --pending`");
    }
    Ok(())
}

fn list_pending(rt: &tokio::runtime::Runtime, db_path: &Path) -> Result<()> {
    let rows = rt.block_on(async {
        let mut db = db::Db::open(db_path).await?;
        let r = db.pending_clusters(spk::MIN_WINDOWS as i64).await?;
        db.close().await?;
        anyhow::Ok(r)
    })?;
    if rows.is_empty() {
        println!("nothing pending — every far-end voice with ≥ {} windows is named or skipped", spk::MIN_WINDOWS);
        return Ok(());
    }
    println!("{:>4}  {:<20}  {:<4} {:>4}  {:>6}  TITLE", "MTG", "DATE", "CL", "WIN", "SPEECH");
    for r in &rows {
        let c = &r.cluster;
        println!(
            "{:>4}  {:<20}  {:<4} {:>4}  {:>6}  {}",
            c.meeting_id,
            export::fmt_utc(r.started_at),
            c.cluster,
            c.n_windows,
            export::fmt_duration(c.speech_secs),
            r.meeting_title
        );
    }
    println!("\nhear one:  meetscribe speakers play <mtg> <cl>     name it:  meetscribe speakers label <mtg> <cl> \"<first>\" \"<last>\"");
    Ok(())
}

fn list_meeting(rt: &tokio::runtime::Runtime, db_path: &Path, mid: i64) -> Result<()> {
    let (row, clusters, people) = rt.block_on(async {
        let mut db = db::Db::open(db_path).await?;
        let row = db.get_meeting(mid).await?.with_context(|| format!("no meeting with id {mid}"))?;
        let c = db.list_clusters(mid).await?;
        let p = db.list_speakers().await?;
        db.close().await?;
        anyhow::Ok((row, c, p))
    })?;
    println!("meeting {mid} — {} — {}", export::fmt_utc(row.started_at), row.title);
    if clusters.is_empty() {
        println!("no far-end voice clusters stored — run `meetscribe speakers cluster {mid}`");
        return Ok(());
    }
    println!("{:<4} {:>8}  {:>6}  {:<28} VIA", "CL", "WINDOWS", "SPEECH", "NAME");
    for c in &clusters {
        let name = c.speaker_id.and_then(|id| people.iter().find(|p| p.id == id)).map(|p| p.full_name());
        println!("{}", describe(c, name.as_deref()));
    }
    Ok(())
}

// ---------------------------------------------------------------- play

fn play(rt: &tokio::runtime::Runtime, db_path: &Path, mid: i64, label: &str, clips: usize, secs: f64) -> Result<()> {
    let (row, cluster, windows) = rt.block_on(async {
        let mut db = db::Db::open(db_path).await?;
        let row = db.get_meeting(mid).await?.with_context(|| format!("no meeting with id {mid}"))?;
        let c = cluster_or_bail(&mut db, mid, label).await?;
        let w = db.cluster_windows(c.id).await?;
        db.close().await?;
        anyhow::Ok((row, c, w))
    })?;
    let dir = PathBuf::from(&row.source_dir);
    if !dir.is_absolute() || !dir.is_dir() {
        bail!(
            "meeting {mid}: source dir '{}' is not a usable directory — the recording is gone, so this voice \
             cannot be played (it can still be labelled)",
            row.source_dir
        );
    }
    let centroid = spk::from_blob(&cluster.centroid)?;
    let picked = voices::pick_clips(&windows, &centroid, clips, PLAY_MIN_GAP_SECS);
    if picked.is_empty() {
        bail!("cluster {label} has no embedded windows to play");
    }
    let ranges: Vec<(f64, f64)> = picked.iter().map(|w| (w.t_start, w.t_end.min(w.t_start + secs))).collect();
    let slices = voices::slice_far_end(&dir, &ranges)?;

    let out_dir = std::env::temp_dir().join("meetscribe-play").join(format!("{mid}-{label}"));
    std::fs::create_dir_all(&out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let mut files = Vec::new();
    for ((w, slice), k) in picked.iter().zip(slices).zip(1..) {
        let Some(audio) = slice else { continue };
        let mins = (w.t_start / 60.0).floor() as u64;
        let s = (w.t_start % 60.0).floor() as u64;
        let path = out_dir.join(format!("clip-{k}-{mins:02}m{s:02}s.wav"));
        write_wav_16k(&path, &audio)?;
        files.push((path, w.t_start));
    }
    println!(
        "meeting {mid} cluster {label}: {} windows, {} of speech — playing {} clip(s) of ≤ {secs:.0}s",
        cluster.n_windows,
        export::fmt_duration(cluster.speech_secs),
        files.len()
    );
    for (path, t) in &files {
        println!("  ▶ {}  ({})", export::fmt_timestamp(*t), path.display());
        let status = Command::new("afplay").arg(path).status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => log::warn!("afplay exited with {s} for {}", path.display()),
            Err(e) => bail!("could not run afplay ({e}); the clips are at {}", out_dir.display()),
        }
    }
    println!("name it:  meetscribe speakers label {mid} {label} \"<first>\" \"<last>\"   ·   or:  speakers skip {mid} {label}");
    Ok(())
}

fn write_wav_16k(path: &Path, audio: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: crate::resample::TARGET_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).with_context(|| format!("create {}", path.display()))?;
    for s in audio {
        w.write_sample(*s)?;
    }
    w.finalize()?;
    Ok(())
}

// ---------------------------------------------------------------- label / unlabel / skip / rename

fn label_cluster(rt: &tokio::runtime::Runtime, db_path: &Path, mid: i64, label: &str, first: &str, last: &str) -> Result<()> {
    let (person, existed, prints) = rt.block_on(async {
        let mut db = db::Db::open(db_path).await?;
        let c = cluster_or_bail(&mut db, mid, label).await?;
        let now = crate::now_epoch();
        let (sid, existed) = match db.find_speaker(first, last).await? {
            Some(id) => (id, true),
            None => (db.add_speaker(first, last, now).await?, false),
        };
        // Re-labelling replaces this cluster's contribution rather than stacking a second print.
        db.delete_voiceprints_for_cluster(c.id).await?;
        db.set_cluster_speaker(c.id, Some(sid), Some("manual"), None).await?;
        db.add_voiceprint(sid, &c, now).await?;
        let person = db.get_speaker(sid).await?.context("speaker vanished")?;
        let prints = person.voiceprints;
        db.close().await?;
        anyhow::Ok((person, existed, prints))
    })?;
    println!(
        "meeting {mid} cluster {label} → {} (speaker #{}, {}; {prints} voiceprint(s) enrolled)",
        person.full_name(),
        person.id,
        if existed { "existing person" } else { "new person" }
    );
    println!("later meetings with this voice are recognised automatically; past ones: `meetscribe speakers match --all`, then `rerender`");
    Ok(())
}

fn unlabel_cluster(rt: &tokio::runtime::Runtime, db_path: &Path, mid: i64, label: &str, skip: bool) -> Result<()> {
    let removed = rt.block_on(async {
        let mut db = db::Db::open(db_path).await?;
        let c = cluster_or_bail(&mut db, mid, label).await?;
        db.set_cluster_speaker(c.id, None, None, None).await?;
        db.set_cluster_skipped(c.id, skip).await?;
        let removed = db.delete_voiceprints_for_cluster(c.id).await?;
        db.close().await?;
        anyhow::Ok(removed)
    })?;
    if skip {
        println!("meeting {mid} cluster {label}: left unknown on purpose (no longer pending){}", if removed > 0 { "; its voiceprint was removed" } else { "" });
    } else {
        println!("meeting {mid} cluster {label}: name cleared{}", if removed > 0 { "; its voiceprint was removed" } else { "" });
    }
    Ok(())
}

fn rename(rt: &tokio::runtime::Runtime, db_path: &Path, sid: i64, first: &str, last: &str) -> Result<()> {
    let found = rt.block_on(async {
        let mut db = db::Db::open(db_path).await?;
        let f = db.rename_speaker(sid, first, last).await?;
        db.close().await?;
        anyhow::Ok(f)
    })?;
    if !found {
        bail!("no speaker with id {sid} — see `meetscribe speakers list`");
    }
    println!("speaker #{sid} is now {first} {last} (every past export picks it up on `rerender`)");
    Ok(())
}

// ---------------------------------------------------------------- merge / split

/// Centroid + counts recomputed from a cluster's stored embeddings.
fn stats_from(windows: &[db::WindowRow]) -> Result<(Vec<f32>, i64, f64)> {
    let mut embs = Vec::new();
    let mut secs = 0.0;
    for w in windows.iter().filter(|w| !w.inherited) {
        if let Some(b) = &w.embedding {
            embs.push(spk::from_blob(b)?);
            secs += w.t_end - w.t_start;
        }
    }
    Ok((spk::centroid(&embs), embs.len() as i64, secs))
}

fn merge_clusters(rt: &tokio::runtime::Runtime, db_path: &Path, mid: i64, keep: &str, others: &[&str]) -> Result<()> {
    let (moved, n, secs) = rt.block_on(async {
        let mut db = db::Db::open(db_path).await?;
        let a = cluster_or_bail(&mut db, mid, keep).await?;
        let mut victims = Vec::new();
        for label in others {
            if *label == keep {
                bail!("speakers merge: '{keep}' cannot be merged into itself");
            }
            let b = cluster_or_bail(&mut db, mid, label).await?;
            if b.assigned_by.as_deref() == Some("manual") && b.speaker_id != a.speaker_id {
                bail!(
                    "cluster {label} carries a different manual name than {keep} — `speakers unlabel {mid} {label}` first"
                );
            }
            victims.push(b);
        }
        let mut moved = 0;
        for b in &victims {
            moved += db.move_segments(b.id, a.id).await?;
            db.delete_cluster(b.id).await?;
        }
        let windows = db.cluster_windows(a.id).await?;
        let (centroid, n, secs) = stats_from(&windows)?;
        db.update_cluster_stats(a.id, &spk::to_blob(&centroid), centroid.len() as i64, n, secs).await?;
        // A manual label's voiceprint IS the centroid; refresh it to the merged one.
        if let (Some("manual"), Some(sid)) = (a.assigned_by.as_deref(), a.speaker_id) {
            db.delete_voiceprints_for_cluster(a.id).await?;
            let fresh = db.get_cluster(mid, keep).await?.context("cluster vanished")?;
            db.add_voiceprint(sid, &fresh, crate::now_epoch()).await?;
        }
        db.close().await?;
        anyhow::Ok((moved, n, secs))
    })?;
    println!(
        "meeting {mid}: {} → {keep} ({moved} window(s) moved; {keep} now {n} windows, {})",
        others.join(", "),
        export::fmt_duration(secs)
    );
    Ok(())
}

fn split_cluster(rt: &tokio::runtime::Runtime, db_path: &Path, mid: i64, label: &str, cut: f32) -> Result<()> {
    let (labels, matched) = rt.block_on(async {
        let mut db = db::Db::open(db_path).await?;
        let c = cluster_or_bail(&mut db, mid, label).await?;
        let windows: Vec<voices::Window> = db
            .cluster_windows(c.id)
            .await?
            .into_iter()
            .map(|w| {
                Ok(voices::Window {
                    key: w.segment_id,
                    t_start: w.t_start,
                    t_end: w.t_end,
                    embedding: w.embedding.as_deref().map(spk::from_blob).transpose()?,
                })
            })
            .collect::<Result<_>>()?;
        let mut assembly = voices::assemble(&windows, cut);
        if assembly.clusters.len() < 2 {
            bail!("cluster {label} does not split at cut {cut:.2} — try a smaller --cut");
        }
        // Fresh letters that do not collide with the meeting's other clusters.
        let taken: Vec<String> = db.list_clusters(mid).await?.into_iter().map(|c| c.cluster).collect();
        let mut next = 0;
        for nc in &mut assembly.clusters {
            while taken.contains(&spk::letter(next)) {
                next += 1;
            }
            nc.cluster = spk::letter(next);
            next += 1;
        }
        let enrolled = voices::decode_enrolled(&db.load_enrolled().await?)?;
        let matched = voices::apply_matches(&mut assembly, &enrolled);
        let labels: Vec<String> = assembly.clusters.iter().map(|c| c.cluster.clone()).collect();
        db.replace_cluster(c.id, mid, &assembly.clusters, &assembly.voices, crate::now_epoch()).await?;
        db.close().await?;
        anyhow::Ok((labels, matched))
    })?;
    println!(
        "meeting {mid}: {label} split into {} ({matched} recognised); {label}'s name and voiceprint were dropped — listen and label the parts",
        labels.join(", ")
    );
    Ok(())
}

// ---------------------------------------------------------------- cluster / match

fn cluster_meetings(rt: &tokio::runtime::Runtime, db_path: &Path, ids: Vec<i64>, recluster: bool, speaker_model: &Path) -> Result<()> {
    // The model loads lazily: `--recluster` alone never needs audio or the model.
    let mut embedder: Option<spk::Embedder> = None;
    let summary = rt.block_on(async {
        let mut db = db::Db::open(db_path).await?;
        let wanted: Vec<i64> = if ids.is_empty() {
            db.list_meetings().await?.iter().map(|m| m.id).collect()
        } else {
            ids
        };
        let enrolled = voices::decode_enrolled(&db.load_enrolled().await?)?;
        let (mut done, mut skipped) = (0usize, Vec::<String>::new());
        for mid in wanted {
            let Some(row) = db.get_meeting(mid).await? else {
                skipped.push(format!("{mid}: no such meeting"));
                continue;
            };
            let existing = db.list_clusters(mid).await?;
            let windows: Vec<voices::Window> = if !existing.is_empty() {
                if !recluster {
                    skipped.push(format!("{mid}: already clustered (pass --recluster to redo; drops its names)"));
                    continue;
                }
                let stored = db.meeting_windows(mid).await?;
                stored
                    .into_iter()
                    .map(|w| {
                        Ok(voices::Window {
                            key: w.segment_id,
                            t_start: w.t_start,
                            t_end: w.t_end,
                            embedding: w.embedding.as_deref().map(spk::from_blob).transpose()?,
                        })
                    })
                    .collect::<Result<_>>()?
            } else {
                let dir = PathBuf::from(&row.source_dir);
                if !dir.is_absolute() || !dir.is_dir() {
                    skipped.push(format!("{mid}: source dir '{}' is not a usable directory", row.source_dir));
                    continue;
                }
                let segs = db.far_end_segments(mid).await?;
                if segs.is_empty() {
                    skipped.push(format!("{mid}: no far-end segments"));
                    continue;
                }
                if embedder.is_none() {
                    embedder = Some(
                        spk::Embedder::load(speaker_model)
                            .with_context(|| format!("load speaker model {} (run `bash models/provision.sh`)", speaker_model.display()))?,
                    );
                }
                // Audio work happens with NO transaction open; the write below is one short one.
                match voices::embed_far_end(&dir, &segs, embedder.as_mut().expect("loaded")) {
                    Ok(w) => w,
                    Err(e) => {
                        skipped.push(format!("{mid}: {e:#}"));
                        continue;
                    }
                }
            };
            let mut assembly = voices::assemble(&windows, spk::CLUSTER_CUT);
            if assembly.is_empty() {
                skipped.push(format!("{mid}: no far-end window long enough to embed"));
                continue;
            }
            let matched = voices::apply_matches(&mut assembly, &enrolled);
            db.replace_meeting_clusters(mid, &assembly.clusters, &assembly.voices, crate::now_epoch()).await?;
            println!("meeting {mid}: {} cluster(s), {matched} recognised — {}", assembly.clusters.len(), row.title);
            done += 1;
        }
        db.close().await?;
        anyhow::Ok((done, skipped))
    })?;
    let (done, skipped) = summary;
    for s in &skipped {
        log::warn!("skipped {s}");
    }
    println!("\nclustered {done} · skipped {}", skipped.len());
    if done > 0 {
        println!("next: `meetscribe speakers list --pending`, then `play` + `label`; `rerender` to update the exports");
    }
    Ok(())
}

fn match_meetings(rt: &tokio::runtime::Runtime, db_path: &Path, ids: Vec<i64>) -> Result<()> {
    let (assigned, cleared, kept) = rt.block_on(async {
        let mut db = db::Db::open(db_path).await?;
        let wanted: Vec<i64> = if ids.is_empty() {
            db.list_meetings().await?.iter().map(|m| m.id).collect()
        } else {
            ids
        };
        let enrolled = voices::decode_enrolled(&db.load_enrolled().await?)?;
        let (mut assigned, mut cleared, mut kept) = (0usize, 0usize, 0usize);
        for mid in wanted {
            for c in db.list_clusters(mid).await? {
                // Manual names and deliberate unknowns are the owner's; only auto/unnamed move.
                if c.assigned_by.as_deref() == Some("manual") || c.skipped {
                    kept += 1;
                    continue;
                }
                let centroid = spk::from_blob(&c.centroid)?;
                match spk::auto_match(&centroid, c.n_windows as usize, &enrolled) {
                    Some((sid, score)) => {
                        if c.speaker_id != Some(sid) || c.match_score != Some(f64::from(score)) {
                            db.set_cluster_speaker(c.id, Some(sid), Some("auto"), Some(f64::from(score))).await?;
                        }
                        assigned += 1;
                    }
                    None => {
                        if c.speaker_id.is_some() {
                            db.set_cluster_speaker(c.id, None, None, None).await?;
                            cleared += 1;
                        }
                    }
                }
            }
        }
        db.close().await?;
        anyhow::Ok((assigned, cleared, kept))
    })?;
    println!("recognised {assigned} · cleared {cleared} (no longer match) · left alone {kept} (manual or skipped)");
    if assigned > 0 {
        println!("update the exports with `meetscribe rerender --all` (preview) then `--write`");
    }
    Ok(())
}
