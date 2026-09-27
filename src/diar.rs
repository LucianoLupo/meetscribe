//! Far-end speaker diarization (split-then-name): who spoke when on one system roll.
//!
//! Runs NVIDIA Nemotron 3 Diarization through the NeMo-Speech.cpp runtime as a subprocess — the
//! exact runtime the blind evaluation used (`plans/2026-09-26-nemotron-split-naming.md` §1). The
//! roll's 16 kHz audio is written as a mono PCM16 WAV under the session dir (the daemon's
//! environment is HOME-only, so no `TMPDIR`), diarized to RTTM, parsed, and both files deleted.
//!
//! Every failure — missing binary or model, non-zero exit, timeout, unparseable output — is an
//! `Err` the caller turns into "no turns for this roll", i.e. today's unsplit pipeline.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use crate::resample;

/// One speaker turn, in seconds relative to the start of the audio handed to [`Diarizer::turns`].
/// `speaker` is the diarizer's LOCAL label for this roll (0-based) — it carries no identity.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub t_start: f64,
    pub t_end: f64,
    pub speaker: u32,
}

pub struct Diarizer {
    bin: PathBuf,
    model: PathBuf,
    /// Overrides the default timeout (= the audio's duration). Tests only.
    timeout: Option<Duration>,
    seq: u32,
}

/// Poll interval while waiting for the subprocess.
const POLL: Duration = Duration::from_millis(100);

impl Diarizer {
    /// Check both artifacts exist; nothing is spawned until [`Diarizer::turns`].
    pub fn load(bin: &Path, model: &Path) -> Result<Self> {
        if !bin.is_file() {
            bail!("diarizer binary not found at {}", bin.display());
        }
        if !model.is_file() {
            bail!("diarizer model not found at {}", model.display());
        }
        Ok(Self { bin: bin.to_path_buf(), model: model.to_path_buf(), timeout: None, seq: 0 })
    }

    #[cfg(test)]
    fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = Some(t);
        self
    }

    /// Diarize 16 kHz mono audio. `work_dir` holds the temporary WAV + RTTM (deleted on return).
    /// Timeout = the audio's duration — ≥ 9× the slowest measured run — then the process is killed.
    pub fn turns(&mut self, audio16: &[f32], work_dir: &Path) -> Result<Vec<Turn>> {
        self.seq += 1;
        let stem = format!(".diar-{}-{}", std::process::id(), self.seq);
        let wav = work_dir.join(format!("{stem}.wav"));
        let rttm = work_dir.join(format!("{stem}.rttm"));
        let _cleanup = Cleanup(vec![wav.clone(), rttm.clone()]);

        write_pcm16(&wav, audio16)?;
        let secs = audio16.len() as f64 / resample::TARGET_RATE as f64;
        let timeout = self.timeout.unwrap_or_else(|| Duration::from_secs_f64(secs.max(1.0)));

        let mut child = Command::new(&self.bin)
            .arg("diarize")
            .arg(&wav)
            .arg("-m")
            .arg(&self.model)
            .args(["--format", "rttm", "--force", "-o"])
            .arg(&rttm)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawn {}", self.bin.display()))?;

        let started = Instant::now();
        let status = loop {
            if let Some(s) = child.try_wait().context("wait for diarizer")? {
                break s;
            }
            if started.elapsed() >= timeout {
                let _ = child.kill();
                let _ = child.wait();
                bail!("diarizer timed out after {:.0}s on {secs:.0}s of audio", timeout.as_secs_f64());
            }
            std::thread::sleep(POLL);
        };
        if !status.success() {
            let mut err = String::new();
            if let Some(mut e) = child.stderr.take() {
                use std::io::Read;
                let _ = e.read_to_string(&mut err);
            }
            bail!("diarizer exited with {status}: {}", err.trim());
        }
        let text = std::fs::read_to_string(&rttm).with_context(|| format!("read {}", rttm.display()))?;
        parse_rttm(&text)
    }
}

/// Removes the temporary files however `turns` returns.
struct Cleanup(Vec<PathBuf>);

impl Drop for Cleanup {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn write_pcm16(path: &Path, audio16: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: resample::TARGET_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).with_context(|| format!("create {}", path.display()))?;
    for &s in audio16 {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    w.finalize().with_context(|| format!("finalize {}", path.display()))?;
    Ok(())
}

/// `SPEAKER <file> 1 <start> <dur> <NA> <NA> speaker_<n> <NA> <NA>` → turns sorted by start.
/// Local labels are renumbered 0.. in order of first appearance.
fn parse_rttm(text: &str) -> Result<Vec<Turn>> {
    let mut labels: Vec<String> = Vec::new();
    let mut turns = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.is_empty() {
            continue;
        }
        if f.len() < 8 || f[0] != "SPEAKER" {
            bail!("RTTM line {}: unexpected {line:?}", n + 1);
        }
        let start: f64 = f[3].parse().with_context(|| format!("RTTM line {}: start", n + 1))?;
        let dur: f64 = f[4].parse().with_context(|| format!("RTTM line {}: duration", n + 1))?;
        let label = f[7];
        let speaker = match labels.iter().position(|l| l == label) {
            Some(i) => i,
            None => {
                labels.push(label.to_string());
                labels.len() - 1
            }
        };
        turns.push(Turn { t_start: start, t_end: start + dur, speaker: speaker as u32 });
    }
    turns.sort_by(|a, b| a.t_start.total_cmp(&b.t_start));
    Ok(turns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("meetscribe-diar-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A stub diarizer: a shell script with the given body. Argv is `diarize <wav> -m <model>
    /// --format rttm --force -o <rttm>`, so `$9` is the RTTM path.
    fn stub(dir: &Path, body: &str) -> (PathBuf, PathBuf) {
        let bin = dir.join("stub-diar");
        std::fs::write(&bin, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let model = dir.join("model.gguf");
        std::fs::write(&model, b"gguf").unwrap();
        (bin, model)
    }

    fn only_stub_files(dir: &Path) -> bool {
        std::fs::read_dir(dir).unwrap().all(|e| {
            let n = e.unwrap().file_name();
            n == "stub-diar" || n == "model.gguf"
        })
    }

    #[test]
    fn parses_rttm_and_renumbers_labels_by_first_appearance() {
        let t = parse_rttm(
            "SPEAKER x 1 3.200 1.000 <NA> <NA> speaker_3 <NA> <NA>\n\
             SPEAKER x 1 0.500 2.000 <NA> <NA> speaker_1 <NA> <NA>\n\n",
        )
        .unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0], Turn { t_start: 0.5, t_end: 2.5, speaker: 1 });
        assert_eq!(t[1].speaker, 0);
        assert!(parse_rttm("garbage line\n").is_err());
    }

    #[test]
    fn missing_binary_or_model_fails_to_load() {
        let d = tmp("load");
        let (bin, model) = stub(&d, "exit 0");
        assert!(Diarizer::load(&d.join("nope"), &model).is_err());
        assert!(Diarizer::load(&bin, &d.join("nope.gguf")).is_err());
        assert!(Diarizer::load(&bin, &model).is_ok());
    }

    #[test]
    fn success_parses_the_rttm_and_cleans_up() {
        let d = tmp("ok");
        let (bin, model) = stub(&d, "echo 'SPEAKER w 1 0.100 0.400 <NA> <NA> speaker_1 <NA> <NA>' > \"$9\"");
        let mut diar = Diarizer::load(&bin, &model).unwrap();
        let turns = diar.turns(&vec![0.0; 16_000], &d).unwrap();
        assert_eq!(turns, vec![Turn { t_start: 0.1, t_end: 0.5, speaker: 0 }]);
        assert!(only_stub_files(&d), "temporary WAV/RTTM must be deleted");
    }

    #[test]
    fn nonzero_exit_is_an_error_carrying_stderr() {
        let d = tmp("exit");
        let (bin, model) = stub(&d, "echo 'model load failed' >&2; exit 3");
        let err = Diarizer::load(&bin, &model).unwrap().turns(&vec![0.0; 16_000], &d).unwrap_err();
        assert!(format!("{err:#}").contains("model load failed"), "{err:#}");
        assert!(only_stub_files(&d));
    }

    #[test]
    fn a_hung_diarizer_is_killed_at_the_timeout() {
        let d = tmp("hang");
        let (bin, model) = stub(&d, "sleep 30");
        let mut diar = Diarizer::load(&bin, &model).unwrap().with_timeout(Duration::from_millis(300));
        let t0 = Instant::now();
        let err = diar.turns(&vec![0.0; 16_000], &d).unwrap_err();
        assert!(format!("{err:#}").contains("timed out"), "{err:#}");
        assert!(t0.elapsed() < Duration::from_secs(5));
        assert!(only_stub_files(&d));
    }

    /// Real runtime on two macOS `say` voices (needs `bash models/provision.sh` +
    /// `bash models/build-diarizer.sh`): `cargo test diar -- --ignored`.
    #[test]
    #[ignore]
    fn real_diarizer_finds_the_boundary_between_two_voices() {
        let d = tmp("real");
        let mut audio = Vec::new();
        for (voice, text) in [
            ("Samantha", "Hello everyone, thanks for joining the call today, let us look at the plan for the next release."),
            ("Daniel", "Thanks, I reviewed it this morning and I think the timeline for the second phase is too tight."),
        ] {
            let out = d.join(format!("{voice}.wav"));
            let ok = Command::new("say")
                .args(["-v", voice, "--data-format=LEI16@16000", "-o"])
                .arg(&out)
                .arg(text)
                .status()
                .unwrap()
                .success();
            assert!(ok, "say failed for {voice}");
            let mut r = hound::WavReader::open(&out).unwrap();
            audio.extend(r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0));
            if voice == "Samantha" {
                audio.extend(std::iter::repeat_n(0.0, 8_000)); // 0.5 s pause
            }
        }
        let boundary = {
            let r = hound::WavReader::open(d.join("Samantha.wav")).unwrap();
            r.duration() as f64 / 16_000.0 + 0.5
        };
        let mut diar =
            Diarizer::load(Path::new(crate::DIARIZER_BIN_REL), Path::new(crate::DIARIZER_MODEL_REL)).unwrap();
        let turns = diar.turns(&audio, &d).unwrap();
        let speakers: std::collections::BTreeSet<u32> = turns.iter().map(|t| t.speaker).collect();
        assert_eq!(speakers.len(), 2, "turns: {turns:?}");
        let second = turns.iter().find(|t| t.speaker != turns[0].speaker).unwrap();
        assert!((second.t_start - boundary).abs() <= 0.3, "boundary {boundary:.2} vs {second:?}");
    }
}
