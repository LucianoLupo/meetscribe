//! One capture session: the route-hardened single-aggregate drain loop, extracted from the
//! Phase-0 `main()` so BOTH the interactive CLI and the Phase-4 daemon drive the SAME proven
//! loop (rebuild/watchdog/rate-roll). The only caller-specific bit — when to stop — is a
//! `stop` predicate polled each tick; the interactive path folds Enter/deadline into it, the
//! daemon folds "meeting ended OR signalled" into it.
//!
//! `run_capture` is a mechanical lift of the original loop: no change to the rebuild, mic-dry
//! watchdog, or rate-roll behavior (guarded by an unchanged `--seconds` capture in Batch 2).

use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::capture::DualCapture;

/// Outcome of a capture session — enough for the daemon to gate the pipeline (duration) and
/// for both callers to log what happened.
pub(crate) struct CaptureSummary {
    /// Wall-clock capture duration. The daemon skips transcription+store below MIN_MEETING_SECS.
    pub duration_secs: f64,
    /// Number of segment files written (1 in the common case; >1 only on rate-roll rebuilds).
    pub segments: u32,
    pub rebuilds: u32,
    pub dropped: u32,
    /// `number_buffers` seen on the first callback (2 = the expected mic+tap layout).
    pub num_buffers: u32,
}

/// Run one capture session into `out_dir`, stopping when `stop()` first returns true. Writes
/// `mic.wav`/`system.wav` (+ rolled segments on a cross-rate route change) exactly as the
/// Phase-0/1 loop did. `rebuild_after` is the `--rebuild-after` debug hook (force a rebuild N
/// seconds in); pass `None` in production.
pub(crate) fn run_capture(
    out_dir: &Path,
    mut stop: impl FnMut() -> bool,
    rebuild_after: Option<u64>,
    mut managed: Option<&mut crate::recovery::Session>,
) -> Result<CaptureSummary> {
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("create output dir {}", out_dir.display()))?;

    // Startup self-check: the classified StartError Display carries the actionable remedy (TCC
    // grant missing vs mic/route dropped). The caller decides fatal (CLI) vs retry (daemon).
    let mut cap = DualCapture::start().map_err(|e| anyhow::anyhow!("cannot start capture — {e}"))?;

    log::info!(
        "mic (\"You\"):   uid={} native={} Hz, {} ch",
        cap.mic_uid,
        cap.mic_native_rate,
        cap.mic_native_channels
    );
    log::info!(
        "tap (\"Others\"): native={} Hz, {} ch  (Teams typically 24000)",
        cap.tap_native_rate,
        cap.tap_native_channels
    );

    // The aggregate runs at the mic clock's rate; the tap's native rate is drift-resampled into
    // it, so one rate governs both WAVs. `rate`/output files can change if a rebuild can't hold
    // the original rate (roll to a new segment); in the common case both stay put.
    let mut rate = cap.aggregate_rate().max(1);
    log::info!("aggregate rate = {rate} Hz (both channels)");

    let mut segment: u32 = 0;
    if let Some(session) = managed.as_deref_mut() { session.start_roll(0, rate, 0.0)?; }
    let mut mic_path = segment_path(out_dir, "mic", segment);
    let mut sys_path = segment_path(out_dir, "system", segment);
    let mut mic_w = WavStream::create(&mic_path, rate).context("open mic.wav")?;
    let mut sys_w = WavStream::create(&sys_path, rate).context("open system.wav")?;

    let started_at = Instant::now();
    // One-shot forced-rebuild trigger (debug). None once it has fired.
    let mut rebuild_deadline = rebuild_after.map(|s| started_at + Duration::from_secs(s));
    let mut rebuilds = 0u32;
    // Consecutive failed rebuild attempts (transient route-transition failures). A rebuild
    // failure is NEVER fatal — we re-arm and retry rather than tear down a live recording.
    let mut rebuild_failures = 0u32;
    // A single physical route change emits a BURST of HAL notifications; wait for the route to
    // settle before rebuilding once, so we don't spawn empty micro-segments.
    const REBUILD_SETTLE: Duration = Duration::from_millis(500);
    let mut rebuild_pending_since: Option<Instant> = None;
    // Mic-dry watchdog: the mic is the always-on clock master, so once the device has fired its
    // first callback, a mic that stops delivering FRAMES (not just goes quiet) for this long
    // means the input route dropped — a case a default-device listener can miss.
    const MIC_DRY_TIMEOUT: Duration = Duration::from_secs(3);
    let mut mic_last_seen = Instant::now();
    // Small per-drain scratch buffers (reused) — only a ring's-worth is ever resident.
    let mut mic_raw: Vec<f32> = Vec::new();
    let mut tap_raw: Vec<f32> = Vec::new();
    let mut last_checkpoint = Instant::now();

    loop {
        mic_raw.clear();
        tap_raw.clear();
        cap.drain_into(&mut mic_raw, &mut tap_raw);
        mic_w.write(&mic_raw, cap.observed_mic_channels())?;
        sys_w.write(&tap_raw, cap.observed_tap_channels())?;
        if last_checkpoint.elapsed() >= Duration::from_secs(2) {
            let mic_frames = mic_w.checkpoint()?;
            let system_frames = sys_w.checkpoint()?;
            if let Some(session) = managed.as_deref_mut() { session.checkpoint(segment, mic_frames, system_frames)?; }
            last_checkpoint = Instant::now();
        }

        // Mic-dry watchdog. The mic delivers frames continuously once the device has started,
        // even during system silence, so a dry mic = a dropped input route. Gated on the device
        // having started, and paused while a rebuild is already pending/settling.
        if !mic_raw.is_empty() {
            mic_last_seen = Instant::now();
        }
        if cap.first_num_buffers() > 0
            && rebuild_pending_since.is_none()
            && mic_last_seen.elapsed() > MIC_DRY_TIMEOUT
        {
            log::warn!(
                "mic delivered no frames for >{}s while the device is running → requesting \
                 rebuild (input route dropped?)",
                MIC_DRY_TIMEOUT.as_secs()
            );
            cap.request_rebuild();
            mic_last_seen = Instant::now();
        }

        // Fire the one-shot forced rebuild once its deadline passes.
        if let Some(d) = rebuild_deadline
            && Instant::now() >= d
        {
            log::info!("--rebuild-after fired → requesting rebuild");
            cap.request_rebuild();
            rebuild_deadline = None;
        }

        // A requested rebuild (forced trigger, the route-change listener, or the mic-dry
        // watchdog) starts a settle timer; we only rebuild once the burst has quieted.
        if cap.rebuild_requested() {
            rebuild_pending_since.get_or_insert_with(Instant::now);
        }
        let do_rebuild = rebuild_pending_since
            .map(|since| since.elapsed() >= REBUILD_SETTLE)
            .unwrap_or(false);
        if do_rebuild {
            rebuild_pending_since = None;
            // Synchronized final flush of BOTH channels so they end at equal frame counts before
            // the old device is torn down.
            mic_raw.clear();
            tap_raw.clear();
            cap.drain_into(&mut mic_raw, &mut tap_raw);
            mic_w.write(&mic_raw, cap.observed_mic_channels())?;
            sys_w.write(&tap_raw, cap.observed_tap_channels())?;

            let mic_before = mic_w.written();
            let sys_before = sys_w.written();

            // A rebuild failure is NEVER fatal. `rebuild()` fails on exactly the transient
            // operations (default input / process tap / aggregate assembly) that are most likely
            // to hiccup DURING a route change — the moment we rebuild. Killing the session here
            // would make the recovery machinery destroy what it exists to save. So: log the
            // classified remedy, re-arm the settle timer, and retry on a later tick. `cap.device`
            // is left None by a failed rebuild and recovers on a retry.
            match cap.rebuild() {
                Err(e) => {
                    rebuild_failures += 1;
                    if rebuild_failures == 1 || rebuild_failures.is_multiple_of(20) {
                        log::warn!("rebuild failed — {e} (capture paused; retry #{rebuild_failures})");
                    }
                    rebuild_pending_since = Some(Instant::now());
                    mic_last_seen = Instant::now();
                }
                Ok(out) => {
                    rebuilds += 1;
                    rebuild_failures = 0;
                    mic_last_seen = Instant::now(); // fresh watchdog window for the new device

                    if out.rate_changed {
                        let mic_frames = mic_w.checkpoint()?;
                        let system_frames = sys_w.checkpoint()?;
                        if let Some(session) = managed.as_deref_mut() {
                            session.checkpoint(segment, mic_frames, system_frames)?;
                            session.start_roll(segment + 1, out.actual_rate, out.gap_frames as f64 / out.actual_rate as f64)?;
                        }
                        // The new device couldn't hold the original rate → the fixed-header WAV
                        // can't continue. Finalize the current segment, roll to a fresh pair at
                        // the new rate, and record the boundary + gap in the manifest.
                        let (mn, mr, mp) = std::mem::replace(
                            &mut mic_w,
                            WavStream::create(
                                &segment_path(out_dir, "mic", segment + 1),
                                out.actual_rate,
                            )
                            .context("open rolled mic segment")?,
                        )
                        .finalize()
                        .context("finalize rolled mic segment")?;
                        let (sn, sr, sp) = std::mem::replace(
                            &mut sys_w,
                            WavStream::create(
                                &segment_path(out_dir, "system", segment + 1),
                                out.actual_rate,
                            )
                            .context("open rolled system segment")?,
                        )
                        .finalize()
                        .context("finalize rolled system segment")?;
                        report("mic seg (rolled)   ", &mic_path, mn, mr, mp, rate);
                        report("system seg (rolled)", &sys_path, sn, sr, sp, rate);

                        segment += 1;
                        mic_path = segment_path(out_dir, "mic", segment);
                        sys_path = segment_path(out_dir, "system", segment);
                        let old_rate = rate;
                        rate = out.actual_rate;
                        log::warn!(
                            "GAP MARKER rebuild #{rebuilds}: rate {old_rate}→{rate} Hz, \
                             ~{} frame gap → ROLLED to segment {segment}",
                            out.gap_frames
                        );
                        append_manifest(
                            out_dir,
                            &format!(
                                "seg {segment} mic={} system={} rate={rate} gap_frames={} \
                                 prev_rate={old_rate} reason=route_change",
                                mic_path.display(),
                                sys_path.display(),
                                out.gap_frames,
                            ),
                            managed.is_some(),
                        );
                    } else {
                        // Rate held → the same files continue. A rate-held rebuild can still have
                        // SWAPPED the input device, so re-latch each WavStream's channel count
                        // before padding — otherwise a device with a different channel count would
                        // be downmixed with the stale divisor and misalign the channels. Then pad
                        // BOTH channels with equal silence for the gap.
                        mic_w.reset_channels();
                        sys_w.reset_channels();
                        mic_w.write_silence(out.gap_frames)?;
                        sys_w.write_silence(out.gap_frames)?;
                        log::info!(
                            "GAP MARKER rebuild #{rebuilds}: padded {} silence frames into both \
                             channels at ~{rate} Hz (mic {mic_before}→{}, system {sys_before}→{})",
                            out.gap_frames,
                            mic_w.written(),
                            sys_w.written(),
                        );
                    }
                }
            }
        }

        if stop() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    // Final flush of whatever is still buffered in the rings.
    mic_raw.clear();
    tap_raw.clear();
    cap.drain_into(&mut mic_raw, &mut tap_raw);
    mic_w.write(&mic_raw, cap.observed_mic_channels())?;
    sys_w.write(&tap_raw, cap.observed_tap_channels())?;

    let duration_secs = started_at.elapsed().as_secs_f64();
    let mic_ch = cap.observed_mic_channels();
    let tap_ch = cap.observed_tap_channels();
    let num_buffers = cap.first_num_buffers();
    let dropped = cap.dropped();

    log::info!("stopped after {duration_secs:.1}s ({rebuilds} rebuild(s))");
    log::info!(
        "OBSERVED IO-proc layout: number_buffers={num_buffers} (expect 2), \
         mic_buffer_channels={mic_ch}, tap_buffer_channels={tap_ch}"
    );
    if num_buffers != 2 {
        log::warn!(
            "number_buffers={num_buffers} != 2 — mic and tap did NOT arrive as two separate \
             buffers (or the last device hadn't fired yet post-rebuild). Report this number."
        );
    }
    if dropped > 0 {
        log::warn!(
            "{dropped} samples dropped (ring full — drain fell behind); increase RING_CAPACITY \
             or drain interval"
        );
    }

    let mic_frames = mic_w.checkpoint()?;
    let system_frames = sys_w.checkpoint()?;
    if let Some(session) = managed { session.checkpoint(segment, mic_frames, system_frames)?; }
    let (mic_n, mic_rms, mic_peak) = mic_w.finalize().context("finalize mic.wav")?;
    let (sys_n, sys_rms, sys_peak) = sys_w.finalize().context("finalize system.wav")?;
    report("mic.wav    (You)   ", &mic_path, mic_n, mic_rms, mic_peak, rate);
    report("system.wav (Others)", &sys_path, sys_n, sys_rms, sys_peak, rate);

    Ok(CaptureSummary {
        duration_secs,
        segments: segment + 1,
        rebuilds,
        dropped,
        num_buffers,
    })
}

/// Incremental mono WAV writer: downmixes interleaved frames to mono and streams them to disk as
/// they're drained, so only a ring's-worth of audio is ever in memory. Carries a per-source frame
/// remainder so a drain that splits mid-frame never misaligns channels.
struct WavStream {
    writer: hound::WavWriter<BufWriter<File>>,
    channels: usize,
    pending: Vec<f32>,
    written: usize,
    sum_sq: f64,
    peak: f32,
}

impl WavStream {
    fn create(path: &Path, sample_rate: u32) -> Result<Self> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        Ok(Self {
            writer: hound::WavWriter::create(path, spec)?,
            channels: 0,
            pending: Vec::new(),
            written: 0,
            sum_sq: 0.0,
            peak: 0.0,
        })
    }

    /// Append a freshly-drained interleaved chunk; write every complete mono frame, keep the
    /// remainder. `observed_channels` latches on the first non-zero value (fixed per device).
    fn write(&mut self, interleaved: &[f32], observed_channels: u32) -> Result<()> {
        if interleaved.is_empty() {
            return Ok(());
        }
        if self.channels == 0 && observed_channels > 0 {
            self.channels = observed_channels as usize;
        }
        let ch = self.channels.max(1);
        self.pending.extend_from_slice(interleaved);
        let full = self.pending.len() - self.pending.len() % ch;
        for frame in self.pending[..full].chunks_exact(ch) {
            let mono = frame.iter().sum::<f32>() / ch as f32;
            self.writer.write_sample(mono).context("write capture sample")?;
            self.written += 1;
            self.sum_sq += mono as f64 * mono as f64;
            self.peak = self.peak.max(mono.abs());
        }
        self.pending.drain(..full);
        Ok(())
    }

    /// Publish a readable prefix without finalizing or moving the write cursor.
    fn checkpoint(&mut self) -> Result<usize> {
        self.writer.flush().context("checkpoint capture WAV")?;
        Ok(self.written)
    }

    /// Forget the latched channel count so the next `write` re-latches from the (possibly new)
    /// device. Called on a rate-held rebuild in case the input device swapped to a different
    /// channel count. Drops any sub-frame remainder (< old channels) to avoid mixing the old and
    /// new channel layouts across the boundary.
    fn reset_channels(&mut self) {
        self.channels = 0;
        self.pending.clear();
    }

    /// Write `frames` mono zero-samples — used to pad a rebuild gap equally into both channels so
    /// they stay aligned to each other.
    fn write_silence(&mut self, frames: usize) -> Result<()> {
        for _ in 0..frames {
            self.writer.write_sample(0.0f32).context("write capture silence")?;
            self.written += 1;
        }
        Ok(())
    }

    /// Mono samples written to this segment so far.
    fn written(&self) -> usize {
        self.written
    }

    /// Returns (mono samples written, RMS, peak). Any trailing partial frame (< channels) is
    /// discarded — at most `channels - 1` samples.
    fn finalize(self) -> Result<(usize, f32, f32)> {
        let rms = if self.written > 0 {
            (self.sum_sq / self.written as f64).sqrt() as f32
        } else {
            0.0
        };
        self.writer.finalize()?;
        Ok((self.written, rms, self.peak))
    }
}

/// Segment 0 keeps the plain `mic.wav`/`system.wav` names; later segments (only produced when a
/// rebuild can't hold the rate) get a zero-padded suffix, e.g. `mic.001.wav`.
fn segment_path(dir: &Path, base: &str, segment: u32) -> PathBuf {
    if segment == 0 {
        dir.join(format!("{base}.wav"))
    } else {
        dir.join(format!("{base}.{segment:03}.wav"))
    }
}

/// Unmanaged recordings retain their legacy manifest. Managed captures already persist roll
/// boundaries under their stable lock; never create or rewrite a redundant manifest for them.
/// Best-effort for unmanaged capture (a manifest write failure must not abort capture).
fn append_manifest(dir: &Path, line: &str, managed: bool) {
    if managed { return; }
    let path = dir.join("segments.txt");
    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{line}") {
                log::warn!("could not append to manifest {}: {e}", path.display());
            }
        }
        Err(e) => log::warn!("could not open manifest {}: {e}", path.display()),
    }
}

fn report(label: &str, path: &Path, n: usize, rms: f32, peak: f32, rate: u32) {
    let secs = if rate > 0 { n as f32 / rate as f32 } else { 0.0 };
    let verdict = if peak < 1e-5 {
        "SILENT — check permissions / source"
    } else {
        "signal present"
    };
    log::info!(
        "{label}: {n} samples, {secs:.1}s, RMS={rms:.5}, peak={peak:.5} → {verdict}  [{}]",
        path.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_rate_rolls_are_prunable_without_deleting_legacy_manifests() {
        use crate::{db, recovery};
        for _ in 0..5 {
            for (short, existing) in [(false, false), (true, false), (false, true), (true, true)] {
                let root = std::env::temp_dir().join(format!("meetscribe-rolled-{}", recovery::new_identity()));
                let dir = root.join("sessions/recording");
                let mut owner = recovery::Session::begin(&dir, &root.join("archive.db"), &dir,
                    "org.example.synthetic-call", "Synthetic route-change review", 1_762_880_000).unwrap();
                for (index, rate, gap) in [(0, 16_000, 0.0), (1, 24_000, 0.1)] {
                    owner.start_roll(index, rate, gap).unwrap();
                    let mut mic = WavStream::create(&segment_path(&dir, "mic", index), rate).unwrap();
                    let mut system = WavStream::create(&segment_path(&dir, "system", index), rate).unwrap();
                    mic.write(&vec![0.25; rate as usize], 1).unwrap();
                    system.write(&vec![-0.5; rate as usize], 1).unwrap();
                    owner.checkpoint(index, mic.checkpoint().unwrap(), system.checkpoint().unwrap()).unwrap();
                    mic.finalize().unwrap(); system.finalize().unwrap();
                }
                let manifest = dir.join("segments.txt");
                let original = b"operator-retained legacy boundary\n";
                if existing { std::fs::write(&manifest, original).unwrap(); }
                // Same publication boundary used by run_capture's successful rate-change branch.
                append_manifest(&dir, "seg 1 mic=mic.001.wav system=system.001.wav rate=24000 gap_frames=2400", true);
                if existing { assert_eq!(std::fs::read(&manifest).unwrap(), original); }
                else { assert!(!manifest.exists(), "managed rolls must rely on persisted metadata, not segments.txt"); }
                owner.finish_capture(2.1, if short { 3.0 } else { 0.1 }).unwrap();
                if !short {
                    owner.reserve(crate::now_epoch(), false).unwrap();
                    let key = owner.state.key.clone();
                    let metadata = db::MeetingMeta { title: owner.state.title.clone(), source_dir: owner.dir.display().to_string(),
                        model: "synthetic-whisper-large-v3".into(), lang: "en".into(), started_at: owner.state.started_at,
                        duration_secs: 2.1, created_at: 1_762_880_010, capture_status: Some("complete".into()) };
                    let segments = [crate::transcript::TranscriptSegment { speaker: crate::transcript::Speaker::You,
                        text: "The route-change release check passed".into(), t_start: 0.1, t_end: 0.9, confidence: 0.94 }];
                    let (id, row, stored) = crate::new_runtime().unwrap().block_on(async {
                        let mut archive = owner.archive().await.unwrap();
                        let id = archive.insert_keyed_meeting_with_voices(&metadata, &segments, &[], &[], &key).await.unwrap();
                        let row = archive.get_meeting(id).await.unwrap().unwrap();
                        let stored = archive.load_segments(id).await.unwrap(); archive.close().await.unwrap(); (id, row, stored)
                    });
                    let (vocab, _) = crate::render::Vocab::compile(&[]);
                    let rendered = crate::render::render(&stored, &crate::render::IdentityMap::from_db(&[]), &vocab);
                    let (md, _) = crate::export::write_managed_exports(&dir, &key, crate::export::Publication::Initial, &row, &rendered).unwrap();
                    owner.committed(id).unwrap(); owner.complete(md.parent().unwrap()).unwrap();
                }
                drop(owner);
                let pruned = crate::maintenance::prune_sessions(&root.join("sessions"), 30, crate::now_epoch() + 100 * 86_400).unwrap();
                assert_eq!(pruned.removed, usize::from(!existing), "short={short}, existing={existing}");
                if existing { assert_eq!(std::fs::read(&manifest).unwrap(), original); }
                assert_eq!(root.join("archive.db").exists(), !short);
                std::fs::remove_dir_all(root).unwrap();
            }
            let dir = std::env::temp_dir().join(format!("meetscribe-unmanaged-roll-{}", recovery::new_identity()));
            std::fs::create_dir(&dir).unwrap();
            let line = "seg 1 mic=mic.001.wav system=system.001.wav rate=24000 gap_frames=2400";
            append_manifest(&dir, line, false);
            assert_eq!(std::fs::read_to_string(dir.join("segments.txt")).unwrap(), format!("{line}\n"));
            assert_eq!(crate::pipeline::read_segment_gaps(&dir).unwrap().get(&1), Some(&0.1));
            std::fs::remove_dir_all(dir).unwrap();
        }
        println!(r#"{{"event":"audit_case_complete","case":"f2_rolled_capture_retention","iterations":5}}"#);
    }

    #[test]
    fn interrupted_between_channel_flushes_keeps_the_previous_paired_checkpoint() {
        use crate::recovery;
        use std::io::{Read, Write};
        if let Some(root) = std::env::var_os("MEETSCRIBE_TEST_PAIRED_ROOT") {
            let root = PathBuf::from(root); let dir = root.join("sessions/recording");
            let mut owner = recovery::Session::begin(&dir, &root.join("archive.db"), &dir,
                "org.example.synthetic-call", "Paired checkpoint interruption", 1_762_880_000).unwrap();
            owner.start_roll(0, 16_000, 0.0).unwrap();
            let mut mic = WavStream::create(&dir.join("mic.wav"), 16_000).unwrap();
            let mut system = WavStream::create(&dir.join("system.wav"), 16_000).unwrap();
            mic.write(&vec![0.25; 16_000], 1).unwrap(); system.write(&vec![-0.5; 16_000], 1).unwrap();
            owner.checkpoint(0, mic.checkpoint().unwrap(), system.checkpoint().unwrap()).unwrap();
            recovery::atomic_json(&root.join("paired-ack.json"), &serde_json::json!({"event":"paired_checkpoint_held", "frames":16000})).unwrap();
            std::io::stdin().read_exact(&mut [0u8; 1]).unwrap();
            mic.write(&vec![0.75; 16_000], 1).unwrap(); system.write(&vec![-0.75; 16_000], 1).unwrap();
            assert_eq!(mic.checkpoint().unwrap(), 32_000);
            recovery::atomic_json(&root.join("interrupted-ack.json"), &serde_json::json!({"event":"only_mic_flushed", "frames":32000})).unwrap();
            std::process::exit(85); // Do not flush the other header or publish a paired receipt.
        }
        for _ in 0..5 {
            let root = std::env::temp_dir().join(format!("meetscribe-paired-cutpoint-{}", recovery::new_identity()));
            recovery::private_dir(&root).unwrap(); let dir = root.join("sessions/recording");
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "session::tests::interrupted_between_channel_flushes_keeps_the_previous_paired_checkpoint", "--nocapture"])
                .env("MEETSCRIBE_TEST_PAIRED_ROOT", &root).env("HOME", &root).stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while !root.join("paired-ack.json").exists() { assert!(Instant::now() < deadline); std::thread::sleep(Duration::from_millis(2)); }
            let ack: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("paired-ack.json")).unwrap()).unwrap();
            assert_eq!(ack["frames"], 16000); assert!(recovery::Session::open(&dir).unwrap().is_none());
            let metadata = std::fs::read(dir.join(recovery::STATE_FILE)).unwrap();
            assert!(crate::run_transcribe(&[dir.display().to_string(), "--resume".into(), "--model".into(), root.join("absent.bin").display().to_string()]).is_err());
            assert_eq!(std::fs::read(dir.join(recovery::STATE_FILE)).unwrap(), metadata);
            assert_eq!(crate::maintenance::prune_sessions(&root.join("sessions"), 1, crate::now_epoch() + 100 * 86_400).unwrap().removed, 0);
            child.stdin.take().unwrap().write_all(b"x").unwrap();
            let output = child.wait_with_output().unwrap(); assert_eq!(output.status.code(), Some(85));
            let ack: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("interrupted-ack.json")).unwrap()).unwrap(); assert_eq!(ack["frames"], 32000);
            let original_mic = std::fs::read(dir.join("mic.wav")).unwrap(); let original_system = std::fs::read(dir.join("system.wav")).unwrap();
            let mut session = recovery::Session::open(&dir).unwrap().unwrap();
            assert_eq!((session.state.rolls[0].mic_frames, session.state.rolls[0].system_frames), (16_000, 16_000));
            assert_eq!(session.state.attempts, 0); let input = session.inputs().unwrap();
            assert_eq!(session.state.capture_status, recovery::CaptureStatus::Partial);
            for (channel, expected) in [("mic", 0.25f32), ("system", -0.5f32)] {
                let reader = hound::WavReader::open(input.join(format!("{channel}.wav"))).unwrap(); assert_eq!(reader.len(), 16_000);
                assert!(reader.into_samples::<f32>().all(|sample| sample.unwrap() == expected));
            }
            assert_eq!(std::fs::read(dir.join("mic.wav")).unwrap(), original_mic);
            assert_eq!(std::fs::read(dir.join("system.wav")).unwrap(), original_system);
            drop(session); std::fs::remove_dir_all(root).unwrap();
        }
        println!(r#"{{"event":"audit_case_complete","case":"f1_paired_checkpoint_interrupt","iterations":5}}"#);
    }

    #[test]
    fn audit_f1_checkpoint_recovery() {
        const FRAMES: usize = 48_000;
        if let Some(dir) = std::env::var_os("MEETSCRIBE_TEST_CHECKPOINT_DIR") {
            let dir = PathBuf::from(dir);
            let mut mic = WavStream::create(&dir.join("mic.wav"), 16_000).unwrap();
            let mut system = WavStream::create(&dir.join("system.wav"), 16_000).unwrap();
            mic.write(&vec![0.25; FRAMES], 1).unwrap();
            system.write(&vec![-0.5; FRAMES], 1).unwrap();
            assert_eq!(mic.checkpoint().unwrap(), FRAMES);
            assert_eq!(system.checkpoint().unwrap(), FRAMES);
            println!("{{\"event\":\"checkpoint_ack\",\"frames\":{FRAMES}}}");
            // The acknowledged checkpoint must survive without running either destructor.
            std::process::exit(0);
        }

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        for iteration in 0..5 {
            let dir = std::env::temp_dir().join(format!(
                "meetscribe-checkpoint-{}-{nonce}-{iteration}", std::process::id()
            ));
            std::fs::create_dir(&dir).unwrap();
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "session::tests::audit_f1_checkpoint_recovery", "--nocapture"])
                .env("MEETSCRIBE_TEST_CHECKPOINT_DIR", &dir)
                .env("HOME", &dir)
                .output()
                .unwrap();
            assert!(output.status.success(), "checkpoint child failed: {output:?}");
            assert!(String::from_utf8_lossy(&output.stdout).contains("checkpoint_ack"));
            for (channel, expected) in [("mic", 0.25f32), ("system", -0.5f32)] {
                let reader = hound::WavReader::open(dir.join(format!("{channel}.wav"))).unwrap();
                assert_eq!(reader.len() as usize, FRAMES, "{channel}, iteration {iteration}");
                assert!(reader.into_samples::<f32>().all(|sample| sample.unwrap() == expected));
            }
            std::fs::remove_dir_all(&dir).unwrap();
        }
        println!("{{\"event\":\"audit_case_complete\",\"case\":\"f1_checkpoint\",\"iterations\":5}}");
    }
}
