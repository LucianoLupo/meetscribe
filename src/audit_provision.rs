use super::*;
use std::cell::Cell;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use provisioning::{ModelStage, RuntimeStage};

fn root() -> PathBuf {
    let root = std::env::temp_dir().join(format!("meetscribe-provision-{}", crate::recovery::new_identity()));
    crate::recovery::private_dir(&root).unwrap(); fs::canonicalize(root).unwrap()
}
fn runtime(path: &Path, version: &str) {
    crate::recovery::private_dir(path).unwrap();
    fs::write(path.join("nemo-speech-diar"), format!("#!/bin/sh\nprintf '{version}\\n'\n")).unwrap();
    fs::set_permissions(path.join("nemo-speech-diar"), fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(path.join("libfixture.dylib"), format!("complete {version} library")).unwrap();
    fs::write(path.join("COMMIT"), format!("fixture-{version}-build")).unwrap();
}

#[test]
fn audit_f5_self_source_provisioning() {
    for _ in 0..5 {
        for copy in [false, true] {
            let root = root();
            for name in ["ggml-large-v3.bin", "speaker.onnx"] {
                let model = root.join(name);
                let bytes = b"synthetic nonzero installed model bytes";
                fs::write(&model, bytes).unwrap();
                provision_model(&model, &model, copy).unwrap();
                let alias = root.join(format!("{name}.alias"));
                std::os::unix::fs::symlink(&model, &alias).unwrap();
                provision_model(&alias, &model, copy).unwrap();
                let hardlink = root.join(format!("{name}.hardlink"));
                fs::hard_link(&model, &hardlink).unwrap();
                provision_model(&hardlink, &model, copy).unwrap();
                assert_eq!(fs::read(&model).unwrap(), bytes);
                assert!(!fs::symlink_metadata(&model).unwrap().file_type().is_symlink());
            }
            let installed = root.join("runtime"); runtime(&installed, "installed");
            let before = fs::read(installed.join("nemo-speech-diar")).unwrap();
            RuntimeStage::prepare(&installed, &installed).unwrap().publish().unwrap();
            assert_eq!(fs::read(installed.join("nemo-speech-diar")).unwrap(), before);
            fs::remove_dir_all(root).unwrap();
        }
    }
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f5_self_source\",\"iterations\":5}}");
}

#[test]
fn staging_failures_leave_artifacts_and_stop_sink_untouched() {
    for _ in 0..5 {
        let root = root(); let base = root.join("installed");
        let target = base.join("models/ggml-large-v3.bin");
        crate::recovery::private_dir(target.parent().unwrap()).unwrap();
        fs::write(&target, b"complete installed weights").unwrap();
        let source = root.join("source.bin"); fs::write(&source, b"complete replacement weights").unwrap();
        let missing = root.join("missing"); let stopped = Cell::new(false);
        let sources = || Sources { model: &source, speaker: &missing, diarizer_model: &missing, diarizer_bin: &missing };
        let failure = prepare_then_stop_with(&base, sources(), true, &mut |source, target, copy| {
            ModelStage::prepare_with(source, target, copy, &mut |_, partial| {
                fs::write(partial, b"incomplete stage")?;
                Err(std::io::Error::other("injected copy failure"))
            })
        }, || { stopped.set(true); Ok(()) });
        assert!(failure.is_err()); assert!(!stopped.get());
        assert_eq!(fs::read(&target).unwrap(), b"complete installed weights");
        assert!(prepare_then_stop(&base, Sources { model: &missing, ..sources() }, true, || {
            stopped.set(true); Ok(())
        }).is_err());
        assert!(!stopped.get());
        let stage = ModelStage::prepare(&source, &target, true).unwrap();
        assert!(stage.publish_with(&mut |_, _| Err(std::io::Error::other("injected publication failure"))).is_err());
        assert_eq!(fs::read(&target).unwrap(), b"complete installed weights");
        let assets = prepare_then_stop(&base, sources(), true, || {
            assert_eq!(fs::read(&target).unwrap(), b"complete installed weights");
            stopped.set(true); Ok(())
        }).unwrap();
        assert!(stopped.get()); assets.publish().unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"complete replacement weights");
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn runtime_swap_failures_preserve_recoverable_trees() {
    for _ in 0..5 {
        for restore_fails in [false, true] {
            let root = root(); let source = root.join("source"); let target = root.join("installed");
            runtime(&source, "new"); runtime(&target, "old");
            let before = fs::read(target.join("nemo-speech-diar")).unwrap();
            assert!(RuntimeStage::prepare_with(&source, &target, &mut |source, partial| {
                if source.extension().is_some_and(|extension| extension == "dylib") {
                    fs::write(partial, b"partial library")?;
                    Err(std::io::Error::other("injected dylib copy failure"))
                } else { fs::copy(source, partial) }
            }).is_err());
            assert_eq!(fs::read(target.join("nemo-speech-diar")).unwrap(), before);
            let stage = RuntimeStage::prepare(&source, &target).unwrap();
            let mut calls = 0;
            assert!(stage.publish_with(&mut |from, to| {
                calls += 1;
                if calls == 2 || restore_fails && calls == 3 { Err(std::io::Error::other("injected rename failure")) }
                else { fs::rename(from, to) }
            }).is_err());
            let journal: serde_json::Value = serde_json::from_reader(fs::File::open(root.join(".runtime-swap.json")).unwrap()).unwrap();
            let staged = PathBuf::from(journal["staged"].as_str().unwrap());
            assert_eq!(fs::read(staged.join("libfixture.dylib")).unwrap(), b"complete new library");
            if restore_fails {
                assert!(!target.exists());
                let backup = PathBuf::from(journal["backup"].as_str().unwrap());
                assert_eq!(fs::read(backup.join("nemo-speech-diar")).unwrap(), before);
            } else { assert_eq!(fs::read(target.join("nemo-speech-diar")).unwrap(), before); }
            RuntimeStage::reconcile(&target, &mut |from, to| fs::rename(from, to)).unwrap();
            assert_eq!(fs::read(target.join("nemo-speech-diar")).unwrap(), before);
            assert!(staged.exists(), "failed-swap evidence must survive reconciliation");
            fs::remove_dir_all(root).unwrap();
        }
    }
}

#[test]
fn absent_optional_sources_do_not_bypass_pending_runtime_reconciliation() {
    for _ in 0..5 {
        let root = root(); let base = root.join("base"); let source = root.join("runtime-source");
        let target = base.join(crate::DIARIZER_BIN_REL).parent().unwrap().to_path_buf();
        runtime(&source, "new"); runtime(&target, "old");
        let before = fs::read(target.join("nemo-speech-diar")).unwrap();
        let mut calls = 0;
        assert!(RuntimeStage::prepare(&source, &target).unwrap().publish_with(&mut |from, to| {
            calls += 1;
            if calls == 2 || calls == 3 { Err(std::io::Error::other("injected publication/restore failure")) }
            else { fs::rename(from, to) }
        }).is_err());
        assert!(!target.exists()); fs::remove_dir_all(&source).unwrap();
        let asr = root.join("asr.bin"); fs::write(&asr, b"complete supplied ASR weights").unwrap();
        let missing = root.join("optional-source-absent"); let restored_at_stop = Cell::new(false);
        let assets = prepare_then_stop(&base, Sources { model: &asr, speaker: &missing,
            diarizer_model: &missing, diarizer_bin: &missing }, true, || {
            restored_at_stop.set(fs::read(target.join("nemo-speech-diar")).is_ok_and(|bytes| bytes == before));
            Ok(())
        }).unwrap();
        assert!(restored_at_stop.get(), "optional absence bypassed unresolved runtime recovery");
        assets.publish().unwrap();
        assert_eq!(fs::read(target.join("nemo-speech-diar")).unwrap(), before);
        fs::remove_dir_all(root).unwrap();
    }
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f5_source_absence_reconcile\",\"iterations\":5}}");
}

#[test]
fn installer_preserves_supplied_diarizer_entry_identity() {
    for _ in 0..5 {
        for supplied in ["installed", "alias", "external"] {
            let root = root(); let base = root.join("base");
            let installed = base.join(crate::DIARIZER_MODEL_REL);
            crate::recovery::private_dir(installed.parent().unwrap()).unwrap();
            let external = root.join("external.gguf"); fs::write(&external, b"complete external diarizer weights").unwrap();
            std::os::unix::fs::symlink(&external, &installed).unwrap();
            let alias = root.join("installed-model-alias"); std::os::unix::fs::symlink(&installed, &alias).unwrap();
            let bin = base.join(crate::DIARIZER_BIN_REL); runtime(bin.parent().unwrap(), "installed");
            let asr = root.join("asr.bin"); fs::write(&asr, b"complete supplied ASR weights").unwrap();
            let absent = root.join("optional-speaker-absent");
            let model = match supplied { "installed" => &installed, "alias" => &alias, _ => &external };
            let assets = prepare_then_stop(&base, Sources { model: &asr, speaker: &absent,
                diarizer_model: model, diarizer_bin: &bin }, true, || Ok(())).unwrap();
            assets.publish().unwrap();
            assert_eq!(fs::symlink_metadata(&installed).unwrap().file_type().is_symlink(), supplied != "external",
                "the supplied entry identity was lost for {supplied}");
            assert_eq!(fs::read(&installed).unwrap(), b"complete external diarizer weights");
            fs::remove_dir_all(root).unwrap();
        }
    }
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f5_source_entry_identity\",\"iterations\":5}}");
}

#[test]
fn interrupted_runtime_swap_reconciles() {
    if let Some(root) = std::env::var_os("MEETSCRIBE_TEST_RUNTIME_CUTPOINT") {
        let root = PathBuf::from(root);
        let stage = RuntimeStage::prepare(&root.join("source"), &root.join("installed")).unwrap();
        let mut calls = 0;
        let _ = stage.publish_with(&mut |from, to| {
            calls += 1;
            if calls == 2 {
                fs::write(root.join("cutpoint.json"), b"{\"event\":\"old_runtime_backed_up\"}\n").unwrap();
                std::process::exit(84); // No Drop/finalization at the exact pre-publication cutpoint.
            }
            fs::rename(from, to)
        });
        panic!("cutpoint was not reached");
    }
    for _ in 0..5 {
        let root = root(); runtime(&root.join("source"), "new"); runtime(&root.join("installed"), "old");
        let before = fs::read(root.join("installed/nemo-speech-diar")).unwrap();
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "launchd::audit_provision::interrupted_runtime_swap_reconciles", "--nocapture"])
            .env("MEETSCRIBE_TEST_RUNTIME_CUTPOINT", &root).output().unwrap();
        assert_eq!(child.status.code(), Some(84));
        assert!(root.join("cutpoint.json").is_file()); assert!(!root.join("installed").exists());
        RuntimeStage::reconcile(&root.join("installed"), &mut |from, to| fs::rename(from, to)).unwrap();
        assert_eq!(fs::read(root.join("installed/nemo-speech-diar")).unwrap(), before);
        fs::remove_dir_all(root).unwrap();
    }
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f5_interrupted_swap\",\"iterations\":5}}");
}
