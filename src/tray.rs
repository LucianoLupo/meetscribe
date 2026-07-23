//! `meetscribe tray` — a menu-bar status/control app for the background daemon.
//!
//! SEPARATE PROCESS by design: the daemon stays single-threaded with no GUI run loop competing with
//! its Core Audio listeners, and this process runs the AppKit event loop but never touches audio.
//! The two talk only through `~/.meetscribe/{status.json, paused}` (see `crate::status`).
//!
//! It polls `status.json` once a second and reflects state in the icon + a menu line; menu actions
//! toggle the `paused` flag, open folders, and stop the daemon. macOS requires the tray to live on
//! the main thread, so everything happens inside the `tao` event loop callback.

#[cfg(not(target_os = "macos"))]
pub(crate) fn run_tray(_argv: &[String]) -> anyhow::Result<()> {
    anyhow::bail!("`meetscribe tray` is macOS-only");
}

#[cfg(target_os = "macos")]
pub(crate) fn run_tray(argv: &[String]) -> anyhow::Result<()> {
    imp::run(argv)
}

#[cfg(target_os = "macos")]
mod imp {
    use anyhow::{Context, Result};
    use std::path::Path;
    use std::process::Command;
    use std::time::{Duration, Instant};

    use tao::event::{Event, StartCause};
    use tao::event_loop::{ControlFlow, EventLoopBuilder};
    use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS};
    use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
    use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

    use crate::status::{self, DaemonState};

    /// How often the tray re-reads `status.json`.
    const REFRESH: Duration = Duration::from_secs(1);

    /// What the tray renders — derived from `status.json` + daemon liveness.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Display {
        Down,
        Idle,
        Recording(String),
        Transcribing(String),
        Paused,
    }

    enum UserEvent {
        MenuClick(MenuEvent),
    }

    pub(super) fn run(_argv: &[String]) -> Result<()> {
        let base = crate::base_dir().context("tray: cannot resolve $HOME")?;

        let mut event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
        // Accessory = a menu-bar app with no Dock icon (must be set before run()).
        event_loop.set_activation_policy(ActivationPolicy::Accessory);

        // Menu clicks arrive on a global channel; forward them into the loop so actions are instant
        // (the 1s WaitUntil below only drives the status refresh).
        let proxy = event_loop.create_proxy();
        MenuEvent::set_event_handler(Some(move |e| {
            let _ = proxy.send_event(UserEvent::MenuClick(e));
        }));

        // Built on StartCause::Init (must be on the main thread, which the loop callback is).
        let mut ui: Option<Ui> = None;

        log::info!("meetscribe tray — watching {}", status::status_path(&base).display());

        event_loop.run(move |event, _target, control_flow| {
            match event {
                Event::NewEvents(StartCause::Init) => {
                    match Ui::build() {
                        Ok(built) => {
                            ui = Some(built);
                            log::info!("tray: menu-bar item ready");
                        }
                        Err(e) => {
                            log::error!("tray: failed to build menu-bar item: {e:#}");
                            *control_flow = ControlFlow::Exit;
                            return;
                        }
                    }
                    if let Some(ui) = ui.as_mut() {
                        ui.refresh(&base);
                    }
                    *control_flow = ControlFlow::WaitUntil(Instant::now() + REFRESH);
                }
                Event::NewEvents(StartCause::ResumeTimeReached { .. }) => {
                    if let Some(ui) = ui.as_mut() {
                        ui.refresh(&base);
                    }
                    *control_flow = ControlFlow::WaitUntil(Instant::now() + REFRESH);
                }
                Event::UserEvent(UserEvent::MenuClick(ev)) => {
                    if let Some(ui) = ui.as_mut() {
                        if ui.handle_click(&ev.id, &base) == ClickResult::QuitTray {
                            *control_flow = ControlFlow::Exit;
                            return;
                        }
                        // Reflect the action (e.g. a toggled pause flag) right away.
                        ui.refresh(&base);
                    }
                    *control_flow = ControlFlow::WaitUntil(Instant::now() + REFRESH);
                }
                _ => {}
            }
        })
    }

    #[derive(PartialEq, Eq)]
    enum ClickResult {
        Continue,
        QuitTray,
    }

    /// The live menu-bar item plus the handles the refresh loop mutates.
    struct Ui {
        tray: TrayIcon,
        status_item: MenuItem,
        pause_item: MenuItem,
        open_recordings: MenuItem,
        open_config: MenuItem,
        stop_daemon: MenuItem,
        quit_tray: MenuItem,
        shown: Option<Display>,
        last_pid: Option<u32>,
    }

    impl Ui {
        fn build() -> Result<Ui> {
            let status_item = MenuItem::new("starting…", false, None);
            let pause_item = MenuItem::new("Pause", true, None);
            let open_recordings = MenuItem::new("Open recordings folder", true, None);
            let open_config = MenuItem::new("Open config file", true, None);
            let stop_daemon = MenuItem::new("Stop background daemon", true, None);
            let quit_tray = MenuItem::new("Quit tray", true, None);

            let menu = Menu::new();
            menu.append(&status_item)?;
            menu.append(&PredefinedMenuItem::separator())?;
            menu.append(&pause_item)?;
            menu.append(&open_recordings)?;
            menu.append(&open_config)?;
            menu.append(&PredefinedMenuItem::separator())?;
            menu.append(&stop_daemon)?;
            menu.append(&quit_tray)?;

            let tray = TrayIconBuilder::new()
                .with_menu(Box::new(menu))
                .with_tooltip("meetscribe")
                .with_icon(icon_for(&Display::Down))
                .build()
                .context("build tray icon")?;

            Ok(Ui {
                tray,
                status_item,
                pause_item,
                open_recordings,
                open_config,
                stop_daemon,
                quit_tray,
                shown: None,
                last_pid: None,
            })
        }

        /// Re-read status.json + liveness, update the icon (only on change) and the menu labels.
        fn refresh(&mut self, base: &Path) {
            let (display, pid) = read_display(base);
            self.last_pid = pid;

            self.status_item.set_text(status_line(&display));
            match display {
                Display::Paused => {
                    self.pause_item.set_text("Resume");
                    self.pause_item.set_enabled(true);
                }
                Display::Idle | Display::Recording(_) | Display::Transcribing(_) => {
                    self.pause_item.set_text("Pause");
                    self.pause_item.set_enabled(true);
                }
                Display::Down => {
                    self.pause_item.set_text("Pause");
                    self.pause_item.set_enabled(false);
                }
            }

            if self.shown.as_ref() != Some(&display) {
                if let Err(e) = self.tray.set_icon(Some(icon_for(&display))) {
                    log::warn!("tray: set_icon failed: {e}");
                }
                self.shown = Some(display);
            }
        }

        fn handle_click(&self, id: &tray_icon::menu::MenuId, base: &Path) -> ClickResult {
            if id == self.quit_tray.id() {
                log::info!("tray: quitting tray (daemon keeps running)");
                return ClickResult::QuitTray;
            }
            if id == self.pause_item.id() {
                toggle_pause(base);
            } else if id == self.open_recordings.id() {
                let dir = base.join("sessions");
                open_path(if dir.exists() { &dir } else { base });
            } else if id == self.open_config.id() {
                open_path(&crate::config::config_path(base));
            } else if id == self.stop_daemon.id() {
                stop_daemon(self.last_pid);
            }
            ClickResult::Continue
        }
    }

    /// Read status.json and confirm the daemon pid is actually alive.
    fn read_display(base: &Path) -> (Display, Option<u32>) {
        match status::Status::read(base) {
            None => (Display::Down, None),
            Some(s) => {
                if !pid_alive(s.pid) {
                    return (Display::Down, Some(s.pid));
                }
                let d = match s.state {
                    DaemonState::Idle => Display::Idle,
                    DaemonState::Paused => Display::Paused,
                    DaemonState::Recording => {
                        Display::Recording(s.app.unwrap_or_else(|| "meeting".to_string()))
                    }
                    DaemonState::Transcribing => {
                        Display::Transcribing(s.app.unwrap_or_else(|| "meeting".to_string()))
                    }
                };
                (d, Some(s.pid))
            }
        }
    }

    fn status_line(d: &Display) -> String {
        match d {
            Display::Down => "meetscribe: daemon stopped".to_string(),
            Display::Idle => "meetscribe: idle (watching)".to_string(),
            Display::Recording(app) => format!("meetscribe: ● recording — {app}"),
            Display::Transcribing(app) => format!("meetscribe: ◐ transcribing — {app}"),
            Display::Paused => "meetscribe: paused".to_string(),
        }
    }

    /// A 22×22 antialiased RGBA dot coloured by state (green=idle, red=recording, amber=paused,
    /// gray=stopped). Non-template on purpose — the colour IS the status.
    fn icon_for(d: &Display) -> Icon {
        let rgb = match d {
            Display::Down => (150u8, 150u8, 150u8),
            Display::Idle => (60, 180, 90),
            Display::Recording(_) => (220, 55, 50),
            Display::Transcribing(_) => (60, 130, 220),
            Display::Paused => (230, 170, 40),
        };
        let (w, h) = (22u32, 22u32);
        let (cx, cy, r) = (10.5f32, 10.5f32, 8.0f32);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let a = if dist <= r {
                    255u8
                } else if dist <= r + 1.0 {
                    ((r + 1.0 - dist) * 255.0) as u8
                } else {
                    0u8
                };
                rgba.extend_from_slice(&[rgb.0, rgb.1, rgb.2, a]);
            }
        }
        Icon::from_rgba(rgba, w, h).expect("valid 22x22 rgba icon")
    }

    fn pid_alive(pid: u32) -> bool {
        // kill(pid, 0) == 0 → the process exists and we can signal it (same user).
        // SAFETY: kill with signal 0 performs only an existence/permission check.
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
    }

    fn toggle_pause(base: &Path) {
        let p = status::pause_path(base);
        if p.exists() {
            match std::fs::remove_file(&p) {
                Ok(()) => log::info!("tray: resumed (removed {})", p.display()),
                Err(e) => log::warn!("tray: could not remove {}: {e}", p.display()),
            }
        } else {
            match std::fs::write(&p, b"paused by tray\n") {
                Ok(()) => log::info!("tray: paused (wrote {})", p.display()),
                Err(e) => log::warn!("tray: could not write {}: {e}", p.display()),
            }
        }
    }

    fn open_path(path: &Path) {
        if let Err(e) = Command::new("open").arg(path).status() {
            log::warn!("tray: `open {}` failed: {e}", path.display());
        }
    }

    /// Stop the daemon: bootout the LaunchAgent (the launchd-managed case), and SIGTERM the last
    /// known pid as a fallback for a manually-started daemon. Both best-effort.
    fn stop_daemon(pid: Option<u32>) {
        let uid = unsafe { libc::getuid() };
        let target = format!("gui/{uid}/{}", crate::launchd::LABEL);
        let _ = Command::new("launchctl").args(["bootout", &target]).status();
        if let Some(pid) = pid
            && pid_alive(pid)
        {
            // SAFETY: sending SIGTERM to a pid we own is safe; the daemon finalizes cleanly.
            unsafe {
                libc::kill(pid as libc::pid_t, libc::SIGTERM);
            }
        }
        log::info!("tray: requested daemon stop ({target})");
    }
}
