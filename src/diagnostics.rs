//! Content-free JSON-lines diagnostics. Arbitrary text is never a safe diagnostic field.
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};
use serde_json::{Map, Value, json};

fn approved_class(class: &str) -> bool {
    matches!(class, "capture_io" | "interrupted_attempt" | "child_launch" | "archive_identity" | "checkpoint_invalid" |
        "processing" | "storage" | "session_busy" | "permission" | "model" | "not_found" |
        "invalid_args" | "input_format" | "speaker_conflict" | "operation_failed" | "invalid_state" | "invalid_fields")
}

fn classify(message: &str) -> &'static str {
    let lower = message.to_ascii_lowercase();
    if lower.contains("archive identity") || lower.contains("bound archive") { "archive_identity" }
    else if lower.contains("expected mono 32-bit float wav") { "input_format" }
    else if lower.contains("checkpoint") { "checkpoint_invalid" }
    else if lower.contains("busy") { "session_busy" }
    else if lower.contains("manual name") || lower.contains("confirmed speaker") { "speaker_conflict" }
    else if lower.contains("permission") || lower.contains("tcc") || lower.contains("grant") { "permission" }
    else if lower.contains("model") || lower.contains("whisper") { "model" }
    else if lower.contains("archive") || lower.contains("db ") || lower.contains("sqlite") || lower.contains("persist") { "storage" }
    else if lower.contains("not found") || lower.contains("does not exist") || lower.contains("missing") { "not_found" }
    else if lower.contains("invalid") || lower.contains("usage") || lower.contains("requires") { "invalid_args" }
    else if lower.contains("device") || lower.contains("rebuild") { "capture_io" }
    else { "operation_failed" }
}

fn help(class: &str) -> &'static str {
    match class {
        "archive_identity" => "Restore the matching archive before --resume; original files were preserved.",
        "checkpoint_invalid" => "No trusted input prefix; preserve originals for manual inspection.",
        "session_busy" => "Another owner holds the session; retry after it finishes.",
        "child_launch" => "Check the transcription executable and permissions; the durable retry budget applies.",
        "permission" => "Check file permissions and macOS microphone/system-audio grants.",
        "model" => "Check the installed model and runtime; native diagnostic text is withheld.",
        "storage" => "Archive operation unavailable; check configuration and durable recovery state.",
        "speaker_conflict" => "Inspect confirmed speaker assignments before changing this merge.",
        "input_format" => "Input must be mono 32-bit float WAV; convert the supplied audio before transcribing.",
        "invalid_args" => "Check command options with --help.",
        "not_found" => "A required resource is absent; check command configuration.",
        "invalid_state" | "invalid_fields" => "Unrecognized diagnostic fields were withheld; inspect private recovery state.",
        _ => "Diagnostic content withheld; use the component, source line and recovery state to investigate.",
    }
}

fn safe_fields(fields: Map<String, Value>) -> Option<Map<String, Value>> {
    if !fields.contains_key("component") || !fields.contains_key("event") { return None; }
    let mut output = Map::new();
    let mut invalid_class = false;
    for (key, value) in fields {
        let valid = match key.as_str() {
            "component" => value.as_str().is_some_and(|value| matches!(value, "recovery" | "cli" | "pipeline" | "daemon" | "asr" | "control" | "retention" | "provision")),
            "event" => value.as_str().is_some_and(|value| matches!(value,
                "capture_started" | "roll_started" | "checkpoint_committed" | "capture_finalized" | "capture_failed" |
                "archive_bound" | "attempt_reserved" | "archive_committed" | "session_complete" | "attention_required" |
                "attempt_failed" | "committed_inputs_recovered" | "command_failed" | "source_unchanged" |
                "model_stage_ready" | "model_published" | "runtime_stage_ready" | "runtime_backup_retained" |
                "runtime_published" | "runtime_restored" | "stop_requested" | "stop_received" | "status_published" | "recovery_scan_failed" | "launch_failure_persist_failed")),
            "session_id" | "instance_id" => value.as_str().is_some_and(|value| value.len() == 32 && value.bytes().all(|byte| byte.is_ascii_hexdigit())),
            "phase" => value.as_str().is_some_and(|value| matches!(value, "Recording" | "Ready" | "Archived" | "Complete" | "NeedsAttention" | "SkippedShort")),
            "attempt" => value.as_u64().is_some_and(|value| value <= 4),
            "pending" | "attention" => value.as_u64().is_some(),
            "daemon_state" => value.as_str().is_some_and(|value| matches!(value, "idle" | "recording" | "transcribing" | "paused")),
            "elapsed_ms" => value.as_i64().is_some_and(|value| value >= 0),
            "outcome" => value.as_str().is_some_and(|value| matches!(value, "ok" | "attention" | "error" | "busy")),
            "error_class" => {
                if let Some(class) = value.as_str() {
                    if !approved_class(class) { invalid_class = true; }
                    true
                } else { value.is_null() }
            }
            _ => continue,
        };
        if !valid { return None; }
        output.insert(key, value);
    }
    if invalid_class {
        output.insert("error_class".into(), json!("invalid_state"));
        output.insert("outcome".into(), json!("attention"));
    }
    Some(output)
}

fn formatted(record: &log::Record<'_>, run_id: &str) -> Value {
    let component = match record.target().rsplit("::").next().unwrap_or("") {
        "meetscribe" => "cli",
        value @ ("asr" | "capture" | "daemon" | "db" | "detect" | "diagnostics" | "export" | "maintenance" | "pipeline" | "recovery" | "session" | "speakers" | "status" | "tray") => value,
        _ => "dependency",
    };
    let mut output = json!({
        "timestamp_ms": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis(),
        "level": record.level().as_str().to_lowercase(), "component": component,
        "run_id": run_id, "event": "diagnostic", "source_line": record.line(),
    });
    let message = record.args().to_string();
    if record.target() == "meetscribe::event" {
        let fields = serde_json::from_str::<Map<String, Value>>(&message).ok().and_then(safe_fields);
        if let Some(fields) = fields {
            for (key, value) in fields { output[&key] = value; }
        } else {
            output["component"] = json!("diagnostics"); output["event"] = json!("invalid_event");
            output["error_class"] = json!("invalid_fields");
        }
    } else {
        let native = if component == "asr" {
            [("component=asr event=native_warning count=", "native_warning"), ("component=asr event=native_error count=", "native_error")]
                .into_iter().find_map(|(prefix, event)| message.strip_prefix(prefix)?.parse::<u64>().ok().map(|count| (event, count)))
        } else { None };
        if let Some((event, count)) = native {
            output["event"] = json!(event); output["count"] = json!(count);
        } else {
            output["diagnostic_class"] = json!(classify(&message)); output["message_omitted"] = json!(true);
        }
    }
    if let Some(class) = output["error_class"].as_str() {
        let guidance = help(class); output["help"] = json!(guidance);
    }
    output
}

pub fn init() {
    let run_id = crate::recovery::new_identity();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(move |buffer, record| writeln!(buffer, "{}", formatted(record, &run_id)))
        .init();
}

/// The CLI's Result-error boundary must not bypass the same content-free policy.
pub fn command_failed(error: &anyhow::Error) {
    let class = classify(&format!("{error:#}"));
    log::error!(target: "meetscribe::event", "{}", json!({
        "component": "cli", "event": "command_failed", "outcome": "error", "error_class": class,
    }));
}

pub fn event(component: &'static str, event: &'static str, session: &str, phase: &str, attempt: u8, elapsed_ms: i64, error_class: Option<&str>) {
    let session = if session.len() == 32 && session.bytes().all(|byte| byte.is_ascii_hexdigit()) { session } else { "invalid" };
    log::info!(target: "meetscribe::event", "{}", json!({
        "component": component, "event": event, "session_id": session,
        "phase": phase, "attempt": attempt, "elapsed_ms": elapsed_ms.max(0),
        "outcome": if error_class.is_some() { "attention" } else { "ok" },
        "error_class": error_class,
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_formatter_omits_content_and_rejects_untyped_fields() {
        const MARKER: &str = "SYNTHETIC_PRIVATE_DIAGNOSTIC_BODY";
        if std::env::var_os("MEETSCRIBE_TEST_FORMATTER").is_some() {
            init();
            let source = format!("/outside-home/{MARKER}/meeting");
            log::warn!(target: "meetscribe::speakers", "meeting 41 skipped: source directory '{source}' is unavailable");
            log::info!(target: "meetscribe::event", "{}", json!({
                "component": {"private": MARKER}, "event": "attempt_failed", "phase": [MARKER],
                "attempt": [MARKER], "error_class": {"private": MARKER}, "outcome": true,
            }));
            let root = std::env::temp_dir().join(format!("meetscribe-log-{}", crate::recovery::new_identity()));
            crate::recovery::private_dir(&root).unwrap();
            let dir = root.join("recording");
            let mut state = crate::recovery::Session::begin(&dir, &root.join("archive.db"), &dir,
                "org.example.fixture", "Release readiness sync", 1_762_880_000).unwrap();
            state.state.error_class = Some(MARKER.into());
            state.start_roll(0, 16_000, 0.0).unwrap();
            drop(state);
            let mut loaded = crate::recovery::Session::open(&dir).unwrap().unwrap();
            loaded.start_roll(1, 16_000, 0.25).unwrap();
            drop(loaded);
            std::fs::remove_dir_all(root).unwrap();
            println!("{{\"event\":\"formatter_ack\"}}");
            return;
        }
        for filter in ["info", "debug"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "diagnostics::tests::production_formatter_omits_content_and_rejects_untyped_fields", "--nocapture"])
                .env("MEETSCRIBE_TEST_FORMATTER", "1").env("RUST_LOG", filter)
                .output().unwrap();
            assert!(output.status.success(), "formatter child failed: {output:?}");
            assert!(String::from_utf8_lossy(&output.stdout).contains("formatter_ack"));
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!stderr.contains(MARKER), "private diagnostic content leaked at {filter}");
            let events: Vec<Value> = stderr.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
            assert!(events.iter().all(|event| event["run_id"].is_string() && event["event"].is_string()));
            assert!(events.iter().any(|event| event["error_class"] == "invalid_state"));
            assert!(events.iter().any(|event| event["error_class"] == "invalid_fields"));
        }
    }

    #[test]
    fn launch_accounting_diagnostics_preserve_only_approved_fields() {
        for (event, class) in [("attempt_failed", "child_launch"), ("launch_failure_persist_failed", "storage")] {
            let payload = json!({"component":"recovery", "event":event, "outcome":"error", "error_class":class,
                "raw_text":"SYNTHETIC_PRIVATE_LAUNCH_DETAIL"}).to_string();
            let output = formatted(&log::Record::builder().target("meetscribe::event")
                .args(format_args!("{payload}")).build(), "synthetic-run");
            assert_eq!(output["event"], event); assert_eq!(output["error_class"], class);
            assert_eq!(output["outcome"], "error"); assert!(output["help"].is_string());
            assert!(!output.to_string().contains("SYNTHETIC_PRIVATE_LAUNCH_DETAIL"));
        }
    }

    #[test]
    fn valid_fields_and_native_counts_survive_content_filtering() {
        let payload = json!({"component":"recovery", "event":"attempt_reserved",
            "session_id":"a6e79ba0314c4cc2b647f4d25b327011", "phase":"Ready", "attempt":2,
            "elapsed_ms":17, "outcome":"ok", "error_class":null,
            "raw_text":"/synthetic/home/meeting token=synthetic-token Bearer synthetic-bearer"}).to_string();
        let output = formatted(&log::Record::builder().target("meetscribe::event")
            .args(format_args!("{payload}")).build(), "synthetic-run");
        assert_eq!(output["event"], "attempt_reserved");
        assert_eq!(output["attempt"], 2);
        assert_eq!(output["elapsed_ms"], 17);
        assert_eq!(output["session_id"], "a6e79ba0314c4cc2b647f4d25b327011");
        assert!(!output.to_string().contains("synthetic-token"));
        let native = formatted(&log::Record::builder().target("meetscribe::asr")
            .args(format_args!("component=asr event=native_warning count={}", 3)).build(), "synthetic-run");
        assert_eq!(native["event"], "native_warning");
        assert_eq!(native["count"], 3);
    }
}
