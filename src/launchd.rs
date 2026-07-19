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

pub(crate) const LABEL: &str = "com.lucianolupo.meetscribe";
/// Frozen signing identity (Apple Development, Team L634X3YJBF) — the TCC grant binds to it.
const SIGN_IDENTITY: &str = "155971FEAE6B0B537B4BC9C1F216AB3D0EAE304C";
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

/// Remove-then-sign at `bin` with the frozen identity + identifier (matches the manual recipe).
fn codesign(bin: &Path) -> Result<()> {
    let path = bin.to_str().context("binary path not UTF-8")?;
    // `--remove-signature` may fail on an unsigned file — ignore its status.
    let _ = Command::new("codesign")
        .args(["--remove-signature", path])
        .status();
    let signed = Command::new("codesign")
        .args([
            "--sign",
            SIGN_IDENTITY,
            "--identifier",
            LABEL,
            "--timestamp=none",
            path,
        ])
        .status()
        .context("run codesign --sign")?;
    if !signed.success() {
        bail!("codesign --sign failed for {}", bin.display());
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
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" | "-m" => {
                if let Some(v) = it.next() {
                    model_src = PathBuf::from(v);
                }
            }
            "--copy" => copy_model = true,
            "-h" | "--help" => {
                eprintln!("usage: meetscribe install [--model <ggml.bin>] [--copy]");
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

    // 1. Idempotent: stop any running instance so the copy below never leaves it on a stale inode.
    let _ = Command::new("launchctl")
        .args(["bootout", &format!("gui/{}/{LABEL}", uid())])
        .status();

    // 2. Copy THIS binary (the one being run) to the stable path, then re-sign there.
    let src = std::env::current_exe().context("resolve current exe")?;
    std::fs::copy(&src, &dest_bin)
        .with_context(|| format!("copy {} → {}", src.display(), dest_bin.display()))?;
    codesign(&dest_bin)?;
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
