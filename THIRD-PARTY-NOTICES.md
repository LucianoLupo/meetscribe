# Third-party notices

meetscribe's own source is MIT-licensed (see [`LICENSE`](LICENSE)).

This repository additionally **contains and redistributes** third-party source, and
**derives** from one of those projects. All are MIT-licensed; their copyright notices and
license texts are reproduced below as those licenses require.

| Path | Upstream | License | Relationship |
|---|---|---|---|
| `src/capture.rs` | [Zackriya-Solutions/meetily](https://github.com/Zackriya-Solutions/meetily) | MIT | **derivative work** |
| `docs/meetily-ref/` | [Zackriya-Solutions/meetily](https://github.com/Zackriya-Solutions/meetily) | MIT | verbatim copies, reference only |
| `docs/swift-ref/muesli/` | [Muesli-HQ/muesli](https://github.com/Muesli-HQ/muesli) | MIT | verbatim copies, reference only |
| `docs/swift-ref/pasrom/` | [pasrom/meeting-transcriber](https://github.com/pasrom/meeting-transcriber) | MIT | verbatim copies, reference only |

The `docs/*-ref/` directories are **reference material, not build inputs** — nothing in
them is compiled, linked, or executed. They are retained deliberately so the prior art
behind meetscribe's capture design stays readable alongside the code it produced. See each
directory's `README.md` for provenance.

**Scope note:** the MIT grant in [`LICENSE`](LICENSE) covers meetscribe's own work. It does
**not** relicense the contents of `docs/meetily-ref/` or `docs/swift-ref/` — those remain
under their respective upstream copyrights, reproduced below.

---

## 1. meetily — Zackriya Solutions

**Derivative work:** `src/capture.rs` was ported and adapted from meetily's
`core_audio.rs` — specifically its global process tap +
`AudioHardwareCreateAggregateDevice` recipe. meetscribe extends that recipe by adding the
microphone as an input sub-device of the same aggregate, so mic and system audio arrive on
**one clock** as two separately-addressable, frame-aligned channels. meetily mixes the two
into a single stream; keeping them apart is what makes meetscribe's channel-based speaker
attribution possible.

meetscribe's `cidre` dependency is pinned to rev `a9587fa` because that is the revision
meetily verified the recipe against; the ported code compiles only against that API surface.

**Verbatim copies:** `docs/meetily-ref/` contains 5 unmodified files from meetily
(`core_audio.rs`, `microphone.rs`, `system.rs`, `mod.rs`, `Cargo.toml`).

```
MIT License

Copyright (c) 2024 Zackriya Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. muesli — Muesli-HQ

**Verbatim copies:** `docs/swift-ref/muesli/` contains 4 unmodified Swift files
(`CoreAudioSystemRecorder.swift`, `SystemAudioRecorder.swift`,
`AudioProcessAttributionCollector.swift`, `MeetingNeuralAec.swift`).

Kept as the reference for the **ScreenCaptureKit** capture path — the fallback for apps a
Core Audio process tap cannot reach. No muesli code is present in meetscribe's own source.

```
MIT License

Copyright (c) 2026 Pranav Hari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 3. meeting-transcriber — pasrom

**Verbatim copies:** `docs/swift-ref/pasrom/` contains 4 unmodified Swift files
(`AppAudioCapture.swift`, `AppAudioCapture+PIDTranslation.swift`, `DualSourceRecorder.swift`,
`ProcessTreeEnumerator.swift`).

Kept for its **negative result**, which shaped meetscribe's design:
[issue #79](https://github.com/pasrom/meeting-transcriber/issues/79) is the evidence that a
*per-process* `CATapDescription` captures nothing from Microsoft Teams — the aggregate
builds without error and delivers zero-filled buffers. meetscribe uses a system-wide
mixdown tap because of it. No pasrom code is present in meetscribe's own source.

```
MIT License

Copyright (c) 2025 pasrom

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Fetched at provisioning time (not redistributed)

These are downloaded or built on your machine by the provisioning scripts into gitignored paths.
The repository does not contain or redistribute them; their own licenses apply.

| Artifact | Fetched by | Upstream | License |
|---|---|---|---|
| Whisper `ggml-large-v3.bin` + CoreML encoder | `models/provision.sh` | [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) (OpenAI Whisper weights) | MIT |
| 3D-Speaker CAM++ speaker-embedding ONNX | `models/provision.sh` | [csukuangfj/speaker-embedding-models](https://huggingface.co/csukuangfj/speaker-embedding-models), exported from [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker) | Apache-2.0 (upstream 3D-Speaker; the HF export repo declares none) |
| `Nemotron-3-Diarization.q8_0.gguf` | `models/provision.sh` | [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) | OpenMDW License Agreement v1.1 |
| `nemo-speech-diar` diarizer runtime (built from source at commit `97a15af`) | `models/build-diarizer.sh` | [NVIDIA/NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) | Apache-2.0 |
| ggml dylibs bundled with that runtime | `models/build-diarizer.sh` | [ggml-org/ggml](https://github.com/ggml-org/ggml) (NeMo-Speech.cpp submodule) | MIT |

---

## Dependencies

Dependency licenses are declared in `Cargo.toml` / `Cargo.lock` and are not vendored here.
meetscribe distributes source only — no binary artifact is published — so no dependency
license bundle is required. If binaries are ever released, generate that bundle with
[`cargo about`](https://github.com/EmbarkStudios/cargo-about) from `Cargo.lock` rather than
maintaining a list by hand.
