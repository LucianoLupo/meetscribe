#!/usr/bin/env bash
# meetscribe — model provisioning (Phase 1.5). DELIBERATE, LOGGED, one-time.
#
# This is NOT a runtime download. The shipped app never fetches models — zero
# telemetry / no silent runtime download is a locked design goal. Provisioning
# is an explicit, reproducible, integrity-checked step you run by hand once.
#
# Provisions the multilingual full `large-v3` ggml model + its pre-converted
# CoreML encoder from the official ggerganov/whisper.cpp HuggingFace repo
# (the same source the whisper.cpp README points to as the alternative to
# running generate-coreml-model.sh locally).
#
# Artifacts land side-by-side in this dir with the exact names whisper.cpp's
# CoreML path expects (ggml-large-v3.bin -> ggml-large-v3-encoder.mlmodelc).
# Both are gitignored — recordings and models never enter git.
set -euo pipefail

MODEL="large-v3"
WHISPER_REPO="ggerganov/whisper.cpp"
# Speaker-embedding model (speaker identity): 3D-Speaker CAM++ zh/en "common advanced", exported
# to bare ONNX by the sherpa-onnx maintainers. Input = 80-bin Kaldi fbank, mean-normalised;
# output = 192-d embedding. 28 MB.
SPEAKER_REPO="csukuangfj/speaker-embedding-models"
SPEAKER_MODEL="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
# Far-end diarizer model (split-then-name): NVIDIA Nemotron 3 Diarization, q8_0 GGUF for the
# NeMo-Speech.cpp runtime built by build-diarizer.sh. 107 MB, OpenMDW v1.1.
# sha256 is PINNED here, not read from the LFS pointer: these are the exact bytes the blind
# evaluation used, and a pointer read at fetch time would accept an upstream re-upload.
DIAR_REPO="nvidia/Nemotron-3-Diarization"
DIAR_MODEL="Nemotron-3-Diarization.q8_0.gguf"
DIAR_SHA="08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN="$DIR/ggml-${MODEL}.bin"
ENC_ZIP="$DIR/ggml-${MODEL}-encoder.mlmodelc.zip"
ENC_DIR="$DIR/ggml-${MODEL}-encoder.mlmodelc"

log() { printf '[provision] %s\n' "$*"; }

# Expected sha256 for an LFS file, read from its git-lfs pointer at /raw/main/.
# $1 = HF repo (owner/name), $2 = remote filename
expected_sha() {
  curl -sL --fail --max-time 30 "https://huggingface.co/$1/raw/main/$2" \
    | awk '/^oid sha256:/{sub("sha256:","",$2); print $2}'
}

sha_of() { shasum -a 256 "$1" | awk '{print $1}'; }

download_verified() {
  # $1 = HF repo (owner/name), $2 = remote filename, $3 = local path, $4 = pinned sha256 (optional;
  # without it the expected sha is read from the git-lfs pointer)
  local repo="$1" name="$2" out="$3" want="${4:-}" have
  [ -n "$want" ] || want="$(expected_sha "$repo" "$name")"
  [ -n "$want" ] || { log "ERROR: could not read expected sha256 for $name"; exit 1; }
  if [ -f "$out" ]; then
    have="$(sha_of "$out")"
    if [ "$have" = "$want" ]; then log "$name present + sha OK — skip"; return; fi
    log "$name sha mismatch (have $have) — re-downloading"
  fi
  log "downloading $name ..."
  curl -L --fail --progress-bar -o "$out.tmp" "https://huggingface.co/$repo/resolve/main/$name"
  have="$(sha_of "$out.tmp")"
  [ "$have" = "$want" ] || { log "ERROR: sha256 mismatch for $name (got $have want $want)"; rm -f "$out.tmp"; exit 1; }
  mv "$out.tmp" "$out"
  log "$name verified sha256=$want"
}

# --- model weights ---
download_verified "$WHISPER_REPO" "ggml-${MODEL}.bin" "$BIN"

# --- CoreML encoder (pre-converted; sha-verified zip, then unzip + structural check) ---
if [ -d "$ENC_DIR" ] && [ -f "$ENC_DIR/coremldata.bin" ]; then
  log "encoder $ENC_DIR present — skip"
else
  download_verified "$WHISPER_REPO" "ggml-${MODEL}-encoder.mlmodelc.zip" "$ENC_ZIP"
  log "unzipping encoder ..."
  rm -rf "$ENC_DIR"
  unzip -q -o "$ENC_ZIP" -x "__MACOSX/*" -d "$DIR"
  rm -f "$ENC_ZIP"
  [ -f "$ENC_DIR/coremldata.bin" ] || { log "ERROR: $ENC_DIR/coremldata.bin missing after unzip"; exit 1; }
  log "encoder unpacked + structurally verified"
fi

# --- speaker-embedding model (speaker identity) ---
mkdir -p "$DIR/speaker"
download_verified "$SPEAKER_REPO" "$SPEAKER_MODEL" "$DIR/speaker/$SPEAKER_MODEL"

# --- far-end diarizer model (split-then-name; the runtime comes from build-diarizer.sh) ---
mkdir -p "$DIR/diarizer"
download_verified "$DIAR_REPO" "$DIAR_MODEL" "$DIR/diarizer/$DIAR_MODEL" "$DIAR_SHA"

log "done."
log "diarizer model: $DIR/diarizer/$DIAR_MODEL ($(du -h "$DIR/diarizer/$DIAR_MODEL" | awk '{print $1}')) — build the runtime with: bash models/build-diarizer.sh"
log "speaker: $DIR/speaker/$SPEAKER_MODEL ($(du -h "$DIR/speaker/$SPEAKER_MODEL" | awk '{print $1}'))"
log "model:   $BIN ($(du -h "$BIN" | awk '{print $1}'))"
log "encoder: $ENC_DIR ($(du -sh "$ENC_DIR" | awk '{print $1}'))"
