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
HF="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
HF_RAW="https://huggingface.co/ggerganov/whisper.cpp/raw/main"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN="$DIR/ggml-${MODEL}.bin"
ENC_ZIP="$DIR/ggml-${MODEL}-encoder.mlmodelc.zip"
ENC_DIR="$DIR/ggml-${MODEL}-encoder.mlmodelc"

log() { printf '[provision] %s\n' "$*"; }

# Expected sha256 for an LFS file, read from its git-lfs pointer at /raw/main/.
expected_sha() {
  curl -sL --fail --max-time 30 "$HF_RAW/$1" \
    | awk '/^oid sha256:/{sub("sha256:","",$2); print $2}'
}

sha_of() { shasum -a 256 "$1" | awk '{print $1}'; }

download_verified() {
  # $1 = remote filename, $2 = local path
  local name="$1" out="$2" want have
  want="$(expected_sha "$name")"
  [ -n "$want" ] || { log "ERROR: could not read expected sha256 for $name"; exit 1; }
  if [ -f "$out" ]; then
    have="$(sha_of "$out")"
    if [ "$have" = "$want" ]; then log "$name present + sha OK — skip"; return; fi
    log "$name sha mismatch (have $have) — re-downloading"
  fi
  log "downloading $name ..."
  curl -L --fail --progress-bar -o "$out.tmp" "$HF/$name"
  have="$(sha_of "$out.tmp")"
  [ "$have" = "$want" ] || { log "ERROR: sha256 mismatch for $name (got $have want $want)"; rm -f "$out.tmp"; exit 1; }
  mv "$out.tmp" "$out"
  log "$name verified sha256=$want"
}

# --- model weights ---
download_verified "ggml-${MODEL}.bin" "$BIN"

# --- CoreML encoder (pre-converted; sha-verified zip, then unzip + structural check) ---
if [ -d "$ENC_DIR" ] && [ -f "$ENC_DIR/coremldata.bin" ]; then
  log "encoder $ENC_DIR present — skip"
else
  download_verified "ggml-${MODEL}-encoder.mlmodelc.zip" "$ENC_ZIP"
  log "unzipping encoder ..."
  rm -rf "$ENC_DIR"
  unzip -q -o "$ENC_ZIP" -x "__MACOSX/*" -d "$DIR"
  rm -f "$ENC_ZIP"
  [ -f "$ENC_DIR/coremldata.bin" ] || { log "ERROR: $ENC_DIR/coremldata.bin missing after unzip"; exit 1; }
  log "encoder unpacked + structurally verified"
fi

log "done."
log "model:   $BIN ($(du -h "$BIN" | awk '{print $1}'))"
log "encoder: $ENC_DIR ($(du -sh "$ENC_DIR" | awk '{print $1}'))"
