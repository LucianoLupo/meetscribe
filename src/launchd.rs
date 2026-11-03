//! launchd LaunchAgent install/uninstall for the background daemon.
//!
//! `install` makes the daemon run at login from a STABLE, re-signed copy (`~/.meetscribe/bin/`)
//! so dev rebuilds of `target/debug/meetscribe` never disturb the running daemon or its TCC grant.
//! It is idempotent (boots out any running instance first) and refuses a coreml-compiled binary
//! (which would trigger the one-time ~23-min CoreML ANE compile on a live meeting). The plist uses
//! ABSOLUTE paths and pins `HOME` — under a LaunchAgent `cwd=/` and `~` is not expanded.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

/// FROZEN, permanently. The bundle-id binds the TCC grant (microphone + system audio) and
/// is also the launchd label, the plist filename, and the `codesign --identifier`.
///
/// This may not change. After publication, changing it silently zeroes the mic and
/// system-audio approvals of every existing installation — the daemon simply starts failing
/// with `StartError::SystemAudioTccMissing`, whose remedy string is itself keyed to this id.
/// A change would require a legacy-label migration (boot out + unlink the old plist before
/// the install copy, in `run_install`, `run_uninstall`, and the tray's stop path) plus a
/// documented re-approval step in the release notes.
pub(crate) const LABEL: &str = "com.lucianolupo.meetscribe";

/// Env var holding the codesigning identity — tier 2 of [`resolve_identity`].
const SIGN_IDENTITY_ENV: &str = "MEETSCRIBE_SIGN_IDENTITY";
/// True if THIS binary was compiled with the coreml feature — refused for the unattended daemon.
const BUILT_WITH_COREML: bool = cfg!(feature = "coreml");

/// System tools by ABSOLUTE path. A bare name resolves through `PATH`, where user-writable dirs
/// (`/opt/homebrew/bin`) precede `/usr/bin`, so any process running as the user could shadow
/// them and fake a successful sign or load.
pub(crate) const LAUNCHCTL: &str = "/bin/launchctl";
const CODESIGN: &str = "/usr/bin/codesign";
const SECURITY: &str = "/usr/bin/security";

/// `Umask` for the agent: 0o077, so the daemon's files and launchd's log files are owner-only.
const AGENT_UMASK: u32 = 0o077;

fn uid() -> u32 {
    // SAFETY: getuid is always safe.
    unsafe { libc::getuid() }
}

fn plist_path(home: &Path) -> PathBuf {
    home.join("Library/LaunchAgents")
        .join(format!("{LABEL}.plist"))
}

/// Extract the SHA-1 hashes from `security find-identity -v -p codesigning` output.
///
/// Each identity is one line of the form:
/// `  1) 155971FE…304C "Apple Development: someone@example.com (TEAMID)"`
/// The trailing `N valid identities found` summary carries no hash and is skipped.
#[must_use]
fn parse_identities(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| {
            // Take the token after the `N)` index marker and keep it only if it looks like
            // a SHA-1: exactly 40 hex digits.
            let hash = line.split_whitespace().nth(1)?;
            let is_sha1 = hash.len() == 40 && hash.bytes().all(|b| b.is_ascii_hexdigit());
            is_sha1.then(|| hash.to_owned())
        })
        .collect()
}

/// Pick the codesigning identity: `--identity` flag > `$MEETSCRIBE_SIGN_IDENTITY` > the sole
/// auto-detected identity. Ambiguity and absence are both hard errors with a copy-pasteable fix.
///
/// The explicit tiers are never validated against `found` — a user passing a hash we did not
/// detect is taken at their word, and `codesign` reports the real error if it is wrong.
fn resolve_identity(flag: Option<&str>, env: Option<&str>, found: &[String]) -> Result<String> {
    if let Some(id) = flag.or(env) {
        return Ok(id.to_owned());
    }
    match found {
        [only] => Ok(only.clone()),
        [] => bail!(
            "no codesigning identity found.\n\
             meetscribe re-signs its daemon binary so the TCC grant survives a rebuild, which \
             needs an Apple Development certificate.\n\
             Create one in Xcode (Settings → Accounts → Manage Certificates → +), then re-run.\n\
             Already have one elsewhere? Pass it explicitly:\n    \
             meetscribe install --identity <sha1>\n    \
             {SIGN_IDENTITY_ENV}=<sha1> meetscribe install"
        ),
        many => bail!(
            "{} codesigning identities found — refusing to guess which one to sign with.\n\
             Re-run naming one:\n{}\n\
             Or set {SIGN_IDENTITY_ENV}=<sha1>. List them with:\n    \
             security find-identity -v -p codesigning",
            many.len(),
            many.iter()
                .map(|h| format!("    meetscribe install --identity {h}"))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    }
}

/// Ask the keychain for the available codesigning identities.
fn find_identities() -> Result<Vec<String>> {
    let out = Command::new(SECURITY)
        .args(["find-identity", "-v", "-p", "codesigning"])
        .output()
        .context(
            "run `security find-identity` — is the Xcode command-line toolchain installed? \
             (xcode-select --install)",
        )?;
    Ok(parse_identities(&String::from_utf8_lossy(&out.stdout)))
}

/// The `Identifier=` value from `codesign -d --verbose=2` output (which codesign prints on
/// stderr), or `None` for an unsigned binary.
#[must_use]
fn signed_identifier(display: &str) -> Option<&str> {
    display.lines().find_map(|line| line.strip_prefix("Identifier="))
}

/// Remove-then-sign at `bin` with the resolved identity + the frozen identifier, then VERIFY
/// the result: a strict signature check plus the identifier the TCC grant is keyed to.
///
/// The explicit `--identifier` is mandatory: rustc's default id embeds a per-build hash, so
/// omitting it changes the signed identifier on every rebuild and breaks TCC persistence.
/// `--remove-signature` first (rather than `--force`) matches the documented manual recipe.
fn codesign(bin: &Path, identity: &str) -> Result<()> {
    let path = bin.to_str().context("binary path not UTF-8")?;
    // `--remove-signature` may fail on an unsigned file — ignore its status.
    let _ = Command::new(CODESIGN)
        .args(["--remove-signature", path])
        .status();
    let signed = Command::new(CODESIGN)
        .args([
            "--sign",
            identity,
            "--identifier",
            LABEL,
            "--timestamp=none",
            path,
        ])
        .status()
        .context("run codesign --sign")?;
    if !signed.success() {
        bail!(
            "codesign --sign failed for {} using identity {identity}",
            bin.display()
        );
    }
    let verified = Command::new(CODESIGN)
        .args(["--verify", "--strict", path])
        .status()
        .context("run codesign --verify")?;
    anyhow::ensure!(verified.success(), "codesign --verify --strict failed for {}", bin.display());
    let display = Command::new(CODESIGN)
        .args(["-d", "--verbose=2", path])
        .output()
        .context("run codesign -d")?;
    let display = String::from_utf8_lossy(&display.stderr);
    let id = signed_identifier(&display);
    anyhow::ensure!(
        id == Some(LABEL),
        "signed identifier is {id:?}, expected {LABEL} — the TCC grant would not carry over"
    );
    Ok(())
}

/// XML-escape a value interpolated into the plist, so any path yields a well-formed plist.
#[must_use]
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

/// Write `contents` to `path` atomically: a synced sibling temp file, then `rename` over the
/// target, so an interrupted install or a power loss never leaves a truncated plist behind.
fn write_atomic(path: &Path, contents: &str) -> Result<()> {
    use std::io::Write;
    let name = path.file_name().context("atomic write target has no file name")?;
    let tmp = path.with_file_name(format!(".{}.tmp", name.to_string_lossy()));
    let written = std::fs::File::create(&tmp)
        .and_then(|mut f| f.write_all(contents.as_bytes()).and_then(|()| f.sync_all()))
        .with_context(|| format!("write {}", tmp.display()))
        .and_then(|()| {
            std::fs::rename(&tmp, path)
                .with_context(|| format!("rename {} → {}", tmp.display(), path.display()))
        });
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

/// `ProcessType` is `Standard`, not `Background`: launchd spawns a Background job with a policy the
/// process cannot lift (`spawn type = background`), and whisper ran 2.4× slower under it (measured
/// 2026-10-02: RTF 0.62 vs 0.25). Capture is unaffected either way. Whisper still yields to calls —
/// the transcription child drops ITSELF to background while one is recorded (`status::spawn_call_yield`).
fn plist_contents(dest_bin: &Path, home: &Path, out_log: &Path, err_log: &Path) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{bin}</string>
        <string>daemon</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ProcessType</key>
    <string>Standard</string>
    <key>Umask</key>
    <integer>{umask}</integer>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>{home}</string>
    </dict>
    <key>StandardOutPath</key>
    <string>{out}</string>
    <key>StandardErrorPath</key>
    <string>{err}</string>
</dict>
</plist>
"#,
        LABEL = LABEL,
        umask = AGENT_UMASK,
        bin = xml_escape(&dest_bin.display().to_string()),
        home = xml_escape(&home.display().to_string()),
        out = xml_escape(&out_log.display().to_string()),
        err = xml_escape(&err_log.display().to_string()),
    )
}

/// Provision the model at the daemon's absolute path — a symlink to the repo model by default
/// (avoids duplicating 2.9 GB), or a full copy with `--copy` (repo-independent install).
fn provision_model(src: &Path, dest: &Path, copy: bool) -> Result<()> {
    // Replace any stale symlink/file at dest.
    if std::fs::symlink_metadata(dest).is_ok() {
        std::fs::remove_file(dest).with_context(|| format!("remove stale {}", dest.display()))?;
    }
    if copy {
        std::fs::copy(src, dest)
            .with_context(|| format!("copy model {} → {}", src.display(), dest.display()))?;
        log::info!("model copied → {}", dest.display());
    } else {
        std::os::unix::fs::symlink(src, dest)
            .with_context(|| format!("symlink model {} → {}", src.display(), dest.display()))?;
        log::info!("model symlinked {} → {}", dest.display(), src.display());
    }
    Ok(())
}

/// Install step 3c — the far-end diarizer. The model is provisioned like the others (symlink, or
/// copy with `--copy`); the runtime's directory is always COPIED, never symlinked, because the
/// executable finds its dylibs through `@executable_path`. Missing sources warn and install
/// nothing: the daemon then transcribes without splitting (fail-open).
fn install_diarizer(model_src: &Path, bin_src: &Path, base: &Path, copy_model: bool) -> Result<()> {
    let model_dest = base.join(crate::DIARIZER_MODEL_REL);
    let bin_dest = base.join(crate::DIARIZER_BIN_REL);
    let (Ok(model_abs), Ok(bin_abs)) = (std::fs::canonicalize(model_src), std::fs::canonicalize(bin_src)) else {
        log::warn!(
            "diarizer {} / {} not found — daemon will transcribe but NOT split far-end voices until \
             both are installed (run `bash models/provision.sh` and `bash models/build-diarizer.sh`, \
             then re-run install)",
            model_src.display(),
            bin_src.display()
        );
        return Ok(());
    };
    let model_dir = model_dest.parent().expect("diarizer model path has a parent");
    std::fs::create_dir_all(model_dir).with_context(|| format!("create {}", model_dir.display()))?;
    provision_model(&model_abs, &model_dest, copy_model)?;

    let src_dir = bin_abs.parent().context("diarizer binary has no parent dir")?;
    let dest_dir = bin_dest.parent().expect("diarizer bin path has a parent");
    // Replace the whole directory so no stale dylib from an older build survives.
    if std::fs::symlink_metadata(dest_dir).is_ok() {
        std::fs::remove_dir_all(dest_dir).with_context(|| format!("remove stale {}", dest_dir.display()))?;
    }
    std::fs::create_dir_all(dest_dir).with_context(|| format!("create {}", dest_dir.display()))?;
    for entry in std::fs::read_dir(src_dir).with_context(|| format!("read {}", src_dir.display()))? {
        let path = entry?.path();
        if path.is_file() {
            let to = dest_dir.join(path.file_name().expect("file has a name"));
            std::fs::copy(&path, &to).with_context(|| format!("copy {} → {}", path.display(), to.display()))?;
        }
    }
    anyhow::ensure!(bin_dest.is_file(), "diarizer binary missing after copy: {}", bin_dest.display());
    log::info!("diarizer installed → {} (+ model {})", dest_dir.display(), model_dest.display());
    Ok(())
}

/// `meetscribe install [--model <ggml.bin>] [--copy]` — install + load the login LaunchAgent.
pub(crate) fn run_install(argv: &[String]) -> Result<()> {
    let mut model_src = PathBuf::from("models/ggml-large-v3.bin");
    let mut speaker_src = PathBuf::from(crate::SPEAKER_MODEL_REL);
    let mut diarizer_model_src = PathBuf::from(crate::DIARIZER_MODEL_REL);
    let mut diarizer_bin_src = PathBuf::from(crate::DIARIZER_BIN_REL);
    let mut copy_model = false;
    let mut identity: Option<String> = None;
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" | "-m" => {
                if let Some(v) = it.next() {
                    model_src = PathBuf::from(v);
                }
            }
            "--speaker-model" => {
                if let Some(v) = it.next() {
                    speaker_src = PathBuf::from(v);
                }
            }
            "--diarizer-model" => {
                if let Some(v) = it.next() {
                    diarizer_model_src = PathBuf::from(v);
                }
            }
            "--diarizer-bin" => {
                if let Some(v) = it.next() {
                    diarizer_bin_src = PathBuf::from(v);
                }
            }
            "--identity" | "-i" => {
                if let Some(v) = it.next() {
                    identity = Some(v.clone());
                }
            }
            "--copy" => copy_model = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: meetscribe install [--model <ggml.bin>] [--speaker-model <onnx>] \
                     [--diarizer-model <gguf>] [--diarizer-bin <nemo-speech-diar>] [--copy] \
                     [--identity <sha1>]\n\n\
                     --diarizer-bin  the diarizer executable; its whole directory (dylibs) is copied.\n\
                     --identity  codesigning identity to re-sign the daemon binary with.\n\
                     {SIGN_IDENTITY_ENV} is consulted next; otherwise the sole identity from\n\
                     `security find-identity -v -p codesigning` is used, and anything else errors."
                );
                return Ok(());
            }
            _ => {}
        }
    }

    if BUILT_WITH_COREML {
        bail!(
            "refusing to install a coreml-compiled binary as the unattended daemon — the one-time \
             ~23-min CoreML ANE compile would fire on a live meeting. Rebuild metal-only \
             (`cargo build`) and re-run install."
        );
    }

    let home = crate::home_dir().context("install: cannot resolve $HOME")?;
    anyhow::ensure!(home.is_absolute(), "install: home {} not absolute", home.display());
    let base = home.join(".meetscribe");
    let bin_dir = base.join("bin");
    let dest_bin = bin_dir.join("meetscribe");
    let models_dir = base.join("models");
    let speaker_dest = base.join(crate::SPEAKER_MODEL_REL);
    let logs_dir = base.join("logs");
    let out_log = logs_dir.join("meetscribe.out.log");
    let err_log = logs_dir.join("meetscribe.err.log");
    let speaker_dir = speaker_dest.parent().expect("speaker model path has a parent").to_path_buf();
    for d in [&bin_dir, &models_dir, &speaker_dir, &logs_dir, &base.join("sessions")] {
        std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))?;
    }

    // 0. Resolve the signing identity BEFORE anything else, so an identity problem fails fast
    //    with a copy-pasteable fix. (Step 1 signs a staged copy, so even a failed sign there
    //    leaves the working install untouched.)
    let env_identity = std::env::var(SIGN_IDENTITY_ENV).ok();
    let identity = resolve_identity(
        identity.as_deref(),
        env_identity.as_deref(),
        &find_identities()?,
    )?;
    log::info!("signing identity: {identity}");

    // 1. Copy THIS binary (the one being run) to a temp file beside the stable path, sign and
    //    VERIFY it there. A failed sign leaves the working install (binary + running agent)
    //    untouched.
    let src = std::env::current_exe().context("resolve current exe")?;
    let staged = dest_bin.with_file_name(".meetscribe.new");
    std::fs::copy(&src, &staged)
        .with_context(|| format!("copy {} → {}", src.display(), staged.display()))?;
    if let Err(e) = codesign(&staged, &identity) {
        let _ = std::fs::remove_file(&staged);
        return Err(e);
    }

    // 1b. Logs are owner-only: the dir, and any log files launchd created before `Umask` was
    //     set. Done before the bootout so a failure here also leaves the agent running.
    use std::os::unix::fs::PermissionsExt;
    let restrict = |path: &Path, mode: u32| {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
            .with_context(|| format!("chmod {mode:o} {}", path.display()))
    };
    let owner_only = restrict(&logs_dir, 0o700).and_then(|()| {
        [&out_log, &err_log]
            .into_iter()
            .filter(|log| log.exists())
            .try_for_each(|log| restrict(log, 0o600))
    });
    if let Err(e) = owner_only {
        let _ = std::fs::remove_file(&staged);
        return Err(e);
    }

    // 2. Only now stop any running instance, then swap the verified binary in atomically
    //    (`rename` within one directory keeps the signature and never exposes a partial file).
    let _ = Command::new(LAUNCHCTL)
        .args(["bootout", &format!("gui/{}/{LABEL}", uid())])
        .status();
    if let Err(e) = std::fs::rename(&staged, &dest_bin) {
        let _ = std::fs::remove_file(&staged);
        bail!(
            "move {} → {} failed ({e}). The agent was stopped; the previous binary is unchanged — \
             restart it with: launchctl bootstrap gui/{} {}",
            staged.display(),
            dest_bin.display(),
            uid(),
            plist_path(&home).display()
        );
    }
    log::info!("installed binary → {} (re-signed + verified)", dest_bin.display());

    // 3. Provision the model at the daemon's absolute path.
    match std::fs::canonicalize(&model_src) {
        Ok(abs) => provision_model(&abs, &models_dir.join("ggml-large-v3.bin"), copy_model)?,
        Err(_) => log::warn!(
            "model {} not found — daemon will CAPTURE but not transcribe until a model is placed \
             at {} (re-run install with --model <path>)",
            model_src.display(),
            models_dir.join("ggml-large-v3.bin").display()
        ),
    }

    // 3b. Same for the speaker-embedding model — warn, never fail: the daemon transcribes without
    //     speaker identity until it is provisioned.
    match std::fs::canonicalize(&speaker_src) {
        Ok(abs) => provision_model(&abs, &speaker_dest, copy_model)?,
        Err(_) => log::warn!(
            "speaker model {} not found — daemon will transcribe but NOT identify far-end voices \
             until it is placed at {} (run `bash models/provision.sh`, then re-run install)",
            speaker_src.display(),
            speaker_dest.display()
        ),
    }

    // 3c. The far-end diarizer (split-then-name) — warn, never fail: without it the daemon
    //     transcribes exactly as before, just without splitting far-end chunks.
    if let Err(e) = install_diarizer(&diarizer_model_src, &diarizer_bin_src, &base, copy_model) {
        log::warn!("diarizer not installed ({e:#}) — daemon will transcribe without splitting");
    }

    // 4. Write the plist (absolute paths + pinned HOME).
    let plist = plist_contents(&dest_bin, &home, &out_log, &err_log);
    let pp = plist_path(&home);
    std::fs::create_dir_all(pp.parent().expect("plist has parent"))
        .with_context(|| format!("create {}", pp.parent().unwrap().display()))?;
    write_atomic(&pp, &plist)?;
    log::info!("wrote LaunchAgent → {}", pp.display());

    // 5. Load it (RunAtLoad starts the daemon now + at every login).
    let pp_str = pp.to_str().context("plist path not UTF-8")?;
    let status = Command::new(LAUNCHCTL)
        .args(["bootstrap", &format!("gui/{}", uid()), pp_str])
        .status()
        .context("run launchctl bootstrap")?;
    if !status.success() {
        bail!(
            "launchctl bootstrap failed (status {:?}). The plist is written at {} — you can load it \
             manually with: launchctl bootstrap gui/{} {}",
            status.code(),
            pp.display(),
            uid(),
            pp.display()
        );
    }

    println!("✅ installed + loaded {LABEL} — meetscribe now auto-records meetings at login.");
    println!("   logs:      {}", err_log.display());
    println!("   status:    launchctl print gui/{}/{LABEL}", uid());
    println!("   uninstall: meetscribe uninstall");
    Ok(())
}

/// `meetscribe uninstall` — unload + remove the LaunchAgent. Keeps all data in `~/.meetscribe`.
pub(crate) fn run_uninstall(argv: &[String]) -> Result<()> {
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: meetscribe uninstall  (unloads the LaunchAgent; keeps ~/.meetscribe data)");
        return Ok(());
    }
    let home = crate::home_dir().context("uninstall: cannot resolve $HOME")?;
    let pp = plist_path(&home);
    let _ = Command::new(LAUNCHCTL)
        .args(["bootout", &format!("gui/{}/{LABEL}", uid())])
        .status();
    if pp.exists() {
        std::fs::remove_file(&pp).with_context(|| format!("remove {}", pp.display()))?;
    }
    println!("✅ uninstalled {LABEL} (data kept in {})", home.join(".meetscribe").display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim shape of `security find-identity -v -p codesigning` on a machine with one cert.
    const ONE: &str = "  1) 1111222233334444555566667777888899990000 \"Apple Development: someone@example.com (TEAMID1234)\"\n     1 valid identities found\n";

    const TWO: &str = "  1) 1111222233334444555566667777888899990000 \"Apple Development: someone@example.com (TEAMID1234)\"\n  2) AAAA1111BBBB2222CCCC3333DDDD4444EEEE5555 \"Apple Distribution: Someone (TEAMID1234)\"\n     2 valid identities found\n";

    /// Step 3c on a temp base dir: model provisioned, the runtime dir COPIED (a real directory,
    /// dylibs alongside, executable bit kept), re-install replaces stale files, and missing sources
    /// install nothing without failing. Never touches launchd or the real ~/.meetscribe.
    #[test]
    fn install_diarizer_copies_the_runtime_dir_and_warns_on_missing_sources() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = std::env::temp_dir().join(format!("meetscribe-diar-install-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        let base = tmp.join("base");
        std::fs::create_dir_all(src.join("bin")).unwrap();
        let model = src.join("model.gguf");
        let exe = src.join("bin/nemo-speech-diar");
        std::fs::write(&model, b"gguf").unwrap();
        std::fs::write(&exe, b"#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::write(src.join("bin/libggml.0.dylib"), b"dylib").unwrap();

        install_diarizer(&model, &exe, &base, false).unwrap();
        let bin_dest = base.join(crate::DIARIZER_BIN_REL);
        let dir_dest = bin_dest.parent().unwrap();
        assert!(std::fs::symlink_metadata(dir_dest).unwrap().is_dir(), "bin dir must be a real dir");
        assert!(dir_dest.join("libggml.0.dylib").is_file());
        assert_eq!(std::fs::metadata(&bin_dest).unwrap().permissions().mode() & 0o111, 0o111);
        assert!(base.join(crate::DIARIZER_MODEL_REL).exists());

        // Re-install: a dylib that is no longer in the source must not survive.
        std::fs::write(dir_dest.join("stale.dylib"), b"old").unwrap();
        install_diarizer(&model, &exe, &base, true).unwrap();
        assert!(!dir_dest.join("stale.dylib").exists());
        assert!(!std::fs::symlink_metadata(base.join(crate::DIARIZER_MODEL_REL)).unwrap().file_type().is_symlink());

        // Missing sources: Ok, nothing new installed.
        let empty = tmp.join("empty");
        install_diarizer(&src.join("nope.gguf"), &src.join("nope"), &empty, false).unwrap();
        assert!(!empty.exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn parses_a_single_identity() {
        assert_eq!(
            parse_identities(ONE),
            ["1111222233334444555566667777888899990000"]
        );
    }

    #[test]
    fn parses_multiple_identities_in_order() {
        assert_eq!(
            parse_identities(TWO),
            [
                "1111222233334444555566667777888899990000",
                "AAAA1111BBBB2222CCCC3333DDDD4444EEEE5555"
            ]
        );
    }

    #[test]
    fn parses_empty_when_no_identities() {
        assert!(parse_identities("     0 valid identities found\n").is_empty());
        assert!(parse_identities("").is_empty());
    }

    /// The summary line's leading token is a count, not a hash — it must not be mistaken for one.
    #[test]
    fn ignores_the_summary_line() {
        assert!(parse_identities("     3 valid identities found\n").is_empty());
    }

    #[test]
    fn auto_detects_the_sole_identity() {
        let found = parse_identities(ONE);
        assert_eq!(
            resolve_identity(None, None, &found).unwrap(),
            "1111222233334444555566667777888899990000"
        );
    }

    #[test]
    fn flag_beats_env_and_auto_detect() {
        let found = parse_identities(ONE);
        assert_eq!(
            resolve_identity(Some("FLAG"), Some("ENV"), &found).unwrap(),
            "FLAG"
        );
    }

    #[test]
    fn env_beats_auto_detect() {
        let found = parse_identities(ONE);
        assert_eq!(resolve_identity(None, Some("ENV"), &found).unwrap(), "ENV");
    }

    /// An explicit choice is honored even when the keychain offers nothing — `codesign`
    /// surfaces the real error if the hash is bogus.
    #[test]
    fn explicit_identity_works_with_no_detected_identities() {
        assert_eq!(resolve_identity(Some("FLAG"), None, &[]).unwrap(), "FLAG");
    }

    #[test]
    fn errors_with_actionable_help_when_none_found() {
        let err = resolve_identity(None, None, &[]).unwrap_err().to_string();
        assert!(err.contains("no codesigning identity found"), "{err}");
        assert!(err.contains("--identity"), "{err}");
        assert!(err.contains(SIGN_IDENTITY_ENV), "{err}");
    }

    /// A Background agent throttles whisper 2.4× and launchd's policy cannot be lifted from inside.
    #[test]
    fn the_agent_runs_at_standard_priority_so_transcription_is_not_throttled() {
        let p = plist_contents(
            Path::new("/Users/someone/.meetscribe/bin/meetscribe"),
            Path::new("/Users/someone"),
            Path::new("/Users/someone/.meetscribe/logs/meetscribe.out.log"),
            Path::new("/Users/someone/.meetscribe/logs/meetscribe.err.log"),
        );
        assert!(p.contains("<key>ProcessType</key>\n    <string>Standard</string>"), "{p}");
        assert!(!p.contains("<string>Background</string>"));
    }

    /// A legitimate home path containing `&` (e.g. "Tom & Jerry") must still yield a plist that
    /// macOS accepts — every interpolated path is XML-escaped. Checked with the system `plutil`.
    #[test]
    fn plist_stays_valid_xml_when_paths_contain_special_characters() {
        let home = Path::new("/Users/Tom & Jerry <dev>");
        let p = plist_contents(
            &home.join(".meetscribe/bin/meetscribe"),
            home,
            &home.join(".meetscribe/logs/meetscribe.out.log"),
            &home.join(".meetscribe/logs/meetscribe.err.log"),
        );
        let file = std::env::temp_dir().join(format!("meetscribe-plist-test-{}.plist", std::process::id()));
        std::fs::write(&file, &p).expect("temp dir is writable");
        let lint = Command::new("/usr/bin/plutil").arg("-lint").arg(&file).output().expect("plutil exists on macOS");
        let _ = std::fs::remove_file(&file);
        assert!(
            lint.status.success(),
            "implementation bug: plist is not valid XML: {}\n{p}",
            String::from_utf8_lossy(&lint.stdout)
        );
        assert!(p.contains("/Users/Tom &amp; Jerry &lt;dev&gt;"), "{p}");
    }

    /// System tools are invoked by absolute path: a bare name resolves through `PATH`, where a
    /// user-writable dir (e.g. `/opt/homebrew/bin`) precedes `/usr/bin`. Scans the call sites
    /// themselves, so reverting any one of them back to a bare name fails this test.
    #[test]
    fn system_tools_are_invoked_by_absolute_path() {
        for tool in [LAUNCHCTL, CODESIGN, SECURITY] {
            assert!(Path::new(tool).is_absolute(), "implementation bug: {tool} is not absolute");
            assert!(Path::new(tool).exists(), "{tool} missing on this macOS");
        }
        let sources = [
            ("launchd.rs", include_str!("launchd.rs")),
            ("tray.rs", include_str!("tray.rs")),
            ("speakers.rs", include_str!("speakers.rs")),
        ];
        for (file, src) in sources {
            for tool in ["launchctl", "codesign", "security", "afplay", "open"] {
                let bare = format!("Command::new(\"{tool}\")");
                assert!(!src.contains(&bare), "implementation bug: {file} runs `{tool}` through PATH");
            }
        }
    }

    /// Verbatim stderr shape of `codesign -d --verbose=2 <bin>`.
    const DISPLAY: &str = "Executable=/Users/someone/.meetscribe/bin/meetscribe\n\
        Identifier=com.lucianolupo.meetscribe\nFormat=Mach-O thin (arm64)\n\
        CodeDirectory v=20400 size=66878 flags=0x0(none) hashes=2084+2 location=embedded\n";

    #[test]
    fn reads_the_signed_identifier() {
        assert_eq!(signed_identifier(DISPLAY), Some("com.lucianolupo.meetscribe"));
        assert_eq!(signed_identifier("code object is not signed at all\n"), None);
    }

    /// Real keychain + real codesign, on a temp copy of this test binary: sign, strict-verify,
    /// identifier check. Never touches launchd or ~/.meetscribe. Needs a codesigning identity, so
    /// it is opt-in: `cargo test --bin meetscribe -- --ignored signs_and_verifies`.
    #[test]
    #[ignore = "needs a real codesigning identity in the keychain"]
    fn signs_and_verifies_a_staged_copy_with_the_real_identity() {
        let identity = resolve_identity(None, None, &find_identities().expect("security runs"))
            .expect("exactly one codesigning identity");
        let staged = std::env::temp_dir().join(format!("meetscribe-sign-test-{}", std::process::id()));
        std::fs::copy(std::env::current_exe().expect("test exe"), &staged).expect("copy test exe");
        let result = codesign(&staged, &identity);
        let _ = std::fs::remove_file(&staged);
        result.expect("sign + verify + identifier check all pass");
    }

    /// Files the daemon (and launchd, for the log paths) creates are owner-only.
    #[test]
    fn the_agent_creates_files_owner_only() {
        let p = plist_contents(
            Path::new("/Users/someone/.meetscribe/bin/meetscribe"),
            Path::new("/Users/someone"),
            Path::new("/Users/someone/.meetscribe/logs/meetscribe.out.log"),
            Path::new("/Users/someone/.meetscribe/logs/meetscribe.err.log"),
        );
        assert!(p.contains("<key>Umask</key>\n    <integer>63</integer>"), "{p}");
    }

    /// The plist is replaced by rename, not rewritten in place: the target gets a NEW inode (a
    /// plain `fs::write` would keep the old one), has the new contents, and no temp file remains.
    #[test]
    fn write_atomic_swaps_in_a_new_file_and_leaves_no_temp() {
        use std::os::unix::fs::MetadataExt;
        let dir = std::env::temp_dir().join(format!("meetscribe-atomic-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir is writable");
        let file = dir.join("agent.plist");
        std::fs::write(&file, "old").expect("seed file");
        let before = std::fs::metadata(&file).expect("seeded").ino();
        write_atomic(&file, "new").expect("atomic write succeeds");
        assert_ne!(std::fs::metadata(&file).expect("exists").ino(), before, "rewritten in place, not swapped");
        assert_eq!(std::fs::read_to_string(&file).expect("file exists"), "new");
        let leftovers: Vec<_> = std::fs::read_dir(&dir).expect("dir readable").collect();
        assert_eq!(leftovers.len(), 1, "temp file left behind: {leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Ambiguity must never be resolved by guessing — it lists every candidate instead.
    #[test]
    fn errors_and_lists_candidates_when_ambiguous() {
        let found = parse_identities(TWO);
        let err = resolve_identity(None, None, &found).unwrap_err().to_string();
        assert!(err.contains("2 codesigning identities found"), "{err}");
        for hash in &found {
            assert!(err.contains(hash.as_str()), "missing {hash} in: {err}");
        }
    }
}
