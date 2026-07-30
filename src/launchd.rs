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
    let out = Command::new("security")
        .args(["find-identity", "-v", "-p", "codesigning"])
        .output()
        .context(
            "run `security find-identity` — is the Xcode command-line toolchain installed? \
             (xcode-select --install)",
        )?;
    Ok(parse_identities(&String::from_utf8_lossy(&out.stdout)))
}

/// Remove-then-sign at `bin` with the resolved identity + the frozen identifier.
///
/// The explicit `--identifier` is mandatory: rustc's default id embeds a per-build hash, so
/// omitting it changes the signed identifier on every rebuild and breaks TCC persistence.
/// `--remove-signature` first (rather than `--force`) matches the documented manual recipe.
fn codesign(bin: &Path, identity: &str) -> Result<()> {
    let path = bin.to_str().context("binary path not UTF-8")?;
    // `--remove-signature` may fail on an unsigned file — ignore its status.
    let _ = Command::new("codesign")
        .args(["--remove-signature", path])
        .status();
    let signed = Command::new("codesign")
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
    Ok(())
}

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
    <string>Background</string>
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
        bin = dest_bin.display(),
        home = home.display(),
        out = out_log.display(),
        err = err_log.display(),
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

/// `meetscribe install [--model <ggml.bin>] [--copy]` — install + load the login LaunchAgent.
pub(crate) fn run_install(argv: &[String]) -> Result<()> {
    let mut model_src = PathBuf::from("models/ggml-large-v3.bin");
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
            "--identity" | "-i" => {
                if let Some(v) = it.next() {
                    identity = Some(v.clone());
                }
            }
            "--copy" => copy_model = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: meetscribe install [--model <ggml.bin>] [--copy] [--identity <sha1>]\n\n\
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
    let logs_dir = base.join("logs");
    let out_log = logs_dir.join("meetscribe.out.log");
    let err_log = logs_dir.join("meetscribe.err.log");
    for d in [&bin_dir, &models_dir, &logs_dir, &base.join("sessions")] {
        std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))?;
    }

    // 0. Resolve the signing identity BEFORE anything destructive. Step 1 stops the running
    //    daemon and step 2 overwrites its binary, so failing here — rather than at the sign —
    //    leaves a working install untouched.
    let env_identity = std::env::var(SIGN_IDENTITY_ENV).ok();
    let identity = resolve_identity(
        identity.as_deref(),
        env_identity.as_deref(),
        &find_identities()?,
    )?;
    log::info!("signing identity: {identity}");

    // 1. Idempotent: stop any running instance so the copy below never leaves it on a stale inode.
    let _ = Command::new("launchctl")
        .args(["bootout", &format!("gui/{}/{LABEL}", uid())])
        .status();

    // 2. Copy THIS binary (the one being run) to the stable path, then re-sign there.
    let src = std::env::current_exe().context("resolve current exe")?;
    std::fs::copy(&src, &dest_bin)
        .with_context(|| format!("copy {} → {}", src.display(), dest_bin.display()))?;
    codesign(&dest_bin, &identity)?;
    log::info!("installed binary → {} (re-signed)", dest_bin.display());

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

    // 4. Write the plist (absolute paths + pinned HOME).
    let plist = plist_contents(&dest_bin, &home, &out_log, &err_log);
    let pp = plist_path(&home);
    std::fs::create_dir_all(pp.parent().expect("plist has parent"))
        .with_context(|| format!("create {}", pp.parent().unwrap().display()))?;
    std::fs::write(&pp, plist).with_context(|| format!("write {}", pp.display()))?;
    log::info!("wrote LaunchAgent → {}", pp.display());

    // 5. Load it (RunAtLoad starts the daemon now + at every login).
    let pp_str = pp.to_str().context("plist path not UTF-8")?;
    let status = Command::new("launchctl")
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
    let _ = Command::new("launchctl")
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
    const ONE: &str = "  1) 155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C \"Apple Development: someone@example.com (L634X3YJBF)\"\n     1 valid identities found\n";

    const TWO: &str = "  1) 155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C \"Apple Development: someone@example.com (L634X3YJBF)\"\n  2) AAAA1111BBBB2222CCCC3333DDDD4444EEEE5555 \"Apple Distribution: Someone (L634X3YJBF)\"\n     2 valid identities found\n";

    #[test]
    fn parses_a_single_identity() {
        assert_eq!(
            parse_identities(ONE),
            ["155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C"]
        );
    }

    #[test]
    fn parses_multiple_identities_in_order() {
        assert_eq!(
            parse_identities(TWO),
            [
                "155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C",
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
            "155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C"
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
