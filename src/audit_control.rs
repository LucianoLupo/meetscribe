use super::stop_action;
use crate::status::{self, DaemonState, Instance, Status};
use std::cell::Cell;

fn snapshot(instance: Option<Instance>) -> Status {
    Status { state: DaemonState::Idle, app: None, since_epoch: 1_762_880_000,
        updated_epoch: 1_762_880_001, pid: instance.as_ref().map_or(7070, |value| value.pid),
        instance, pending: 0, attention: 0 }
}

#[test]
fn audit_f6_stale_instance_control() {
    for _ in 0..5 {
        let root = std::env::temp_dir().join(format!("meetscribe-control-{}", crate::recovery::new_identity()));
        crate::recovery::private_dir(&root).unwrap();
        let a = Instance { pid: 7070, start_seconds: 1_762_880_000, start_micros: 123_456, nonce: crate::recovery::new_identity() };
        let b = Instance { start_seconds: a.start_seconds + 90, nonce: crate::recovery::new_identity(), ..a.clone() };
        let writes = Cell::new(0);
        snapshot(Some(a.clone())).write(&root).unwrap();
        assert!(stop_action(&root, Some(&a), |_| Some((a.start_seconds, a.start_micros)), |path, selected| {
            writes.set(writes.get() + 1); status::write_stop(path, selected)
        }).unwrap());
        assert!(status::stop_requested(&root, &a));
        snapshot(Some(b.clone())).write(&root).unwrap();
        assert!(!status::stop_requested(&root, &b), "old-instance request must not stop the replacement");
        assert!(!stop_action(&root, Some(&a), |_| panic!("stale selection must not be retargeted"), |_, _| {
            panic!("stale selection reached a control sink")
        }).unwrap());
        assert_eq!(writes.get(), 1);
        assert!(stop_action(&root, Some(&b), |_| Some((b.start_seconds, b.start_micros)), status::write_stop).unwrap());
        assert!(status::stop_requested(&root, &b));
        // Repeated current-instance requests are harmless and idempotent.
        assert!(stop_action(&root, Some(&b), |_| Some((b.start_seconds, b.start_micros)), status::write_stop).unwrap());
        for pid in [0, 1, i32::MAX as u32 + 1, u32::MAX] {
            let invalid = Instance { pid, ..b.clone() };
            assert!(!invalid.live(|_| panic!("invalid PID reached the kernel-read seam")));
            assert!(!status::stop_requested(&root, &invalid));
        }
        assert!(!b.live(|_| None));
        assert!(!b.live(|_| Some((b.start_seconds + 1, b.start_micros))));
        snapshot(None).write(&root).unwrap();
        assert!(!stop_action(&root, Some(&b), |_| panic!(), |_, _| panic!()).unwrap());
        std::fs::write(status::status_path(&root), b"not JSON").unwrap();
        assert!(!stop_action(&root, Some(&b), |_| panic!(), |_, _| panic!()).unwrap());
        std::fs::remove_file(status::status_path(&root)).unwrap();
        assert!(!stop_action(&root, Some(&b), |_| panic!(), |_, _| panic!()).unwrap());
        // Real, sacrificial sleep PID: stale start identity cannot authorize it. No signals sent.
        let mut sleeper = std::process::Command::new("/bin/sleep").arg("1").spawn().unwrap();
        let (seconds, micros) = status::process_start(sleeper.id()).expect("positive child identity read required");
        let stale = Instance { pid: sleeper.id(), start_seconds: seconds + 1, start_micros: micros, nonce: crate::recovery::new_identity() };
        assert!(!stale.live(status::process_start));
        assert!(sleeper.wait().unwrap().success());
        std::fs::remove_dir_all(root).unwrap();
    }
    // Legacy resident trays remain an explicit upgrade risk: these tests do not retire them.
    assert!(!include_str!("tray.rs").contains("libc::kill"));
    assert!(!include_str!("tray.rs").contains("[\"bootout\""));
    println!("{{\"event\":\"audit_case_complete\",\"case\":\"f6_stale_control\",\"iterations\":5}}");
}
