//! Prepare install artifacts before stopping an agent. Never unlink a live model to replace it.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

fn event(name: &'static str) {
    log::info!(target: "meetscribe::event", "{}", serde_json::json!({"component":"provision", "event":name, "outcome":"ok"}));
}
fn same(a: &Path, b: &Path) -> bool {
    fs::metadata(a).ok().zip(fs::metadata(b).ok()).is_some_and(|(a,b)| a.dev() == b.dev() && a.ino() == b.ino())
}
fn destination(path: &Path) -> Result<PathBuf> {
    let parent = path.parent().context("artifact destination has no parent")?;
    crate::recovery::private_dir(parent)?;
    Ok(fs::canonicalize(parent)?.join(path.file_name().context("artifact destination has no name")?))
}
fn sync_dir(path: &Path) -> Result<()> { File::open(path)?.sync_all()?; Ok(()) }

pub struct ModelStage { destination: PathBuf, staged: Option<PathBuf> }
impl ModelStage {
    pub fn prepare(source: &Path, target: &Path, copy: bool) -> Result<Self> {
        Self::prepare_with(source, target, copy, &mut |from, to| fs::copy(from, to))
    }
    pub fn prepare_with(source: &Path, target: &Path, copy: bool,
        copier: &mut impl FnMut(&Path, &Path) -> std::io::Result<u64>) -> Result<Self> {
        let source_parent = source.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let source_entry = fs::canonicalize(source_parent)?.join(source.file_name().context("model source has no filename")?);
        let source = fs::canonicalize(source).context("model source unavailable")?;
        let metadata = fs::metadata(&source)?;
        ensure!(metadata.is_file() && metadata.len() > 0, "model source must be a nonempty regular file");
        File::open(&source).context("model source unreadable")?;
        let target = destination(target)?;
        // --copy may detach an installed link when its independent external target was supplied.
        // Supplying the installed entry, an alias to it, or a hardlink remains an explicit no-op.
        let detach_link = copy && source_entry == source
            && fs::symlink_metadata(&target).is_ok_and(|entry| entry.file_type().is_symlink())
            && fs::canonicalize(&target).is_ok_and(|resolved| resolved == source);
        if same(&source, &target) && !detach_link { event("source_unchanged"); return Ok(Self { destination: target, staged: None }); }
        ensure!(!target.is_dir(), "model destination is a directory");
        let path = target.with_file_name(format!(".model-stage-{}", crate::recovery::new_identity()));
        let stage = Self { destination: target, staged: Some(path.clone()) };
        if copy {
            ensure!(copier(&source, &path)? == metadata.len(), "model source changed during staging");
            File::open(&path)?.sync_all()?;
        } else { std::os::unix::fs::symlink(&source, &path)?; }
        ensure!(fs::metadata(&path)?.len() == metadata.len(), "staged model is incomplete");
        sync_dir(path.parent().unwrap())?;
        event("model_stage_ready");
        Ok(stage)
    }
    pub fn publish(self) -> Result<()> { self.publish_with(&mut |from, to| fs::rename(from, to)) }
    pub fn publish_with(mut self, rename: &mut impl FnMut(&Path, &Path) -> std::io::Result<()>) -> Result<()> {
        if let Some(path) = &self.staged {
            rename(path, &self.destination).context("publish staged model; original destination preserved on failure")?;
            self.staged = None;
            sync_dir(self.destination.parent().unwrap())?;
            event("model_published");
        }
        Ok(())
    }
}
impl Drop for ModelStage {
    fn drop(&mut self) { if let Some(path) = &self.staged { let _ = fs::remove_file(path); } }
}

type Tree = BTreeMap<String, u64>;
fn tree(path: &Path, executable: &str) -> Result<Tree> {
    let mut files = Tree::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = fs::metadata(entry.path())?;
        ensure!(metadata.is_file() && metadata.len() > 0, "runtime tree contains an unsupported or empty entry");
        let name = entry.file_name().into_string().map_err(|_| anyhow::anyhow!("runtime filename is not UTF-8"))?;
        files.insert(name, metadata.len());
    }
    let metadata = fs::metadata(path.join(executable)).context("runtime executable absent")?;
    ensure!(metadata.is_file() && metadata.permissions().mode() & 0o111 != 0, "runtime executable is not executable");
    Ok(files)
}
fn matches(path: &Path, expected: &Tree, executable: &str) -> bool {
    tree(path, executable).is_ok_and(|actual| &actual == expected)
}

#[derive(Serialize, Deserialize)]
struct Swap {
    version: u8, destination: PathBuf, staged: PathBuf, backup: PathBuf,
    executable: String, old: Option<Tree>, new: Tree,
    #[serde(default)] published: bool,
}
fn owned_path(path: &Path, parent: &Path, prefix: &str) -> bool {
    path.parent() == Some(parent) && path.file_name().and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix(prefix)).is_some_and(|nonce| nonce.len() == 32 && nonce.bytes().all(|byte| byte.is_ascii_hexdigit()))
}
fn journal_path(target: &Path) -> PathBuf { target.with_file_name(".runtime-swap.json") }

pub struct RuntimeStage { destination: PathBuf, staged: Option<PathBuf>, executable: String, retained: bool }
impl RuntimeStage {
    /// Reconcile only named, validated sibling trees. Unknown artifacts are never deleted.
    pub fn reconcile(target: &Path, rename: &mut impl FnMut(&Path, &Path) -> std::io::Result<()>) -> Result<()> {
        let journal = journal_path(target);
        match fs::symlink_metadata(&journal) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error).context("runtime journal availability unknown"),
            Ok(metadata) => ensure!(metadata.is_file(), "runtime journal requires manual attention"),
        }
        let normalized = fs::canonicalize(target.parent().context("runtime target has no parent")?)?
            .join(target.file_name().context("runtime target has no filename")?);
        let target = normalized.as_path();
        let journal = journal_path(target);
        let state: Swap = serde_json::from_reader(File::open(&journal)?)?;
        let parent = target.parent().context("runtime target has no parent")?;
        ensure!(state.version == 1 && state.destination == target && state.executable == "nemo-speech-diar"
            && owned_path(&state.staged, parent, ".runtime-stage-") && owned_path(&state.backup, parent, ".runtime-backup-"),
            "runtime swap journal requires manual attention");
        // After a completed publication, extra stale files can safely travel into the next backup.
        // Missing/partial required files still fail closed; a pending swap must match exactly.
        let published = state.published && tree(target, &state.executable).is_ok_and(|actual|
            state.new.iter().all(|(name, length)| actual.get(name) == Some(length)));
        if !published && !matches(target, &state.new, &state.executable) && !state.old.as_ref().is_some_and(|old| matches(target, old, &state.executable)) {
            if state.old.as_ref().is_some_and(|old| matches(&state.backup, old, &state.executable)) {
                if fs::symlink_metadata(target).is_ok() {
                    rename(target, &parent.join(format!(".runtime-uncertain-{}", crate::recovery::new_identity())))?;
                }
                rename(&state.backup, target).context("runtime restore failed; retained trees require attention")?;
                event("runtime_restored");
            } else if state.old.is_none() && matches(&state.staged, &state.new, &state.executable) && !target.exists() {
                rename(&state.staged, target)?;
            } else { bail!("runtime swap has no verified complete destination or backup; preserved for attention"); }
        }
        sync_dir(parent)?;
        rename(&journal, &parent.join(format!(".runtime-history-{}.json", crate::recovery::new_identity())))?;
        sync_dir(parent)?;
        Ok(())
    }

    pub fn prepare(source: &Path, target: &Path) -> Result<Self> {
        Self::prepare_with(source, target, &mut |from, to| fs::copy(from, to))
    }
    pub fn prepare_with(source: &Path, target: &Path,
        copier: &mut impl FnMut(&Path, &Path) -> std::io::Result<u64>) -> Result<Self> {
        let target = destination(target)?;
        Self::reconcile(&target, &mut |from, to| fs::rename(from, to))?;
        let source = fs::canonicalize(source)?;
        let executable = "nemo-speech-diar".to_string();
        let expected = tree(&source, &executable)?;
        if same(&source, &target) { event("source_unchanged"); return Ok(Self { destination: target, staged: None, executable, retained: false }); }
        let path = target.with_file_name(format!(".runtime-stage-{}", crate::recovery::new_identity()));
        crate::recovery::private_dir(&path)?;
        let stage = Self { destination: target, staged: Some(path.clone()), executable, retained: false };
        for (name, size) in &expected {
            let source = source.join(name); let output = path.join(name);
            ensure!(copier(&source, &output)? == *size, "runtime source changed during staging");
            File::open(output)?.sync_all()?;
        }
        ensure!(matches(&path, &expected, &stage.executable), "staged runtime is incomplete");
        sync_dir(&path)?; sync_dir(path.parent().unwrap())?;
        event("runtime_stage_ready");
        Ok(stage)
    }

    pub fn publish(self) -> Result<()> { self.publish_with(&mut |from, to| fs::rename(from, to)) }
    pub fn publish_with(mut self, rename: &mut impl FnMut(&Path, &Path) -> std::io::Result<()>) -> Result<()> {
        let Some(staged) = &self.staged else { return Ok(()); };
        let parent = self.destination.parent().unwrap();
        let old = if self.destination.exists() { Some(tree(&self.destination, &self.executable)?) } else { None };
        let mut state = Swap { version: 1, destination: self.destination.clone(), staged: staged.clone(),
            backup: parent.join(format!(".runtime-backup-{}", crate::recovery::new_identity())),
            executable: self.executable.clone(), old, new: tree(staged, &self.executable)?, published: false };
        crate::recovery::atomic_json(&journal_path(&self.destination), &state)?;
        sync_dir(parent)?;
        self.retained = true;
        if state.old.is_some() { rename(&self.destination, &state.backup)?; sync_dir(parent)?; event("runtime_backup_retained"); }
        if let Err(error) = rename(staged, &self.destination) {
            if state.old.is_some() {
                rename(&state.backup, &self.destination).context("runtime publication and restore failed; all recovery trees retained")?;
                sync_dir(parent)?;
                event("runtime_restored");
            }
            return Err(error).context("runtime publication failed; recovery journal and trees retained");
        }
        ensure!(matches(&self.destination, &state.new, &self.executable), "published runtime incomplete; recovery journal retained");
        sync_dir(parent)?;
        state.published = true;
        crate::recovery::atomic_json(&journal_path(&self.destination), &state)?;
        sync_dir(parent)?;
        event("runtime_published");
        Ok(())
    }
}
impl Drop for RuntimeStage {
    fn drop(&mut self) {
        if !self.retained && let Some(path) = &self.staged { let _ = fs::remove_dir_all(path); }
    }
}
