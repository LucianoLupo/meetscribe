//! Exercise the real CLI stderr boundary with synthetic private paths only.
use std::os::unix::fs::DirBuilderExt;
use std::process::Command;

#[test]
fn cli_errors_do_not_disclose_private_paths() {
    const MARKER: &str = "SYNTHETIC_PRIVATE_ERROR_PATH";
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let root = std::env::temp_dir().join(format!("meetscribe-cli-log-{}-{nonce}", std::process::id()));
    std::fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
    for filter in ["info", "debug"] {
        let output = Command::new(env!("CARGO_BIN_EXE_meetscribe"))
            .args(["export", "1", "--db"])
            .arg(root.join(format!("{MARKER}.db")))
            .env("HOME", &root).env("RUST_LOG", filter)
            .output().unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains(MARKER), "private error path reached CLI diagnostics at {filter}");
        let events: Vec<serde_json::Value> = stderr.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert!(events.iter().any(|event| event["event"] == "command_failed"));
        assert!(events.iter().all(|event| event["run_id"].is_string()));
    }
    std::fs::remove_dir_all(root).unwrap();
}
