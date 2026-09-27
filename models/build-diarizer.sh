#!/usr/bin/env bash
# meetscribe — build the far-end speaker diarizer (NeMo-Speech.cpp, Nemotron 3 Diarization runtime).
#
# Like provision.sh this is DELIBERATE and one-time: the app never downloads or builds anything at
# runtime. The diarizer is the exact runtime the split-then-name evaluation used
# (plans/2026-09-26-nemotron-split-naming.md §1): NeMo-Speech.cpp built from source at a pinned
# commit, Metal backend, diarization only. The v0.1.0 release cannot load Nemotron 3 Diarization
# ("pre_ln transformer variant is not supported"), so a source build is required.
#
# Output (gitignored, /models/diarizer/):
#   models/diarizer/bin/nemo-speech-diar   + the dylibs it links, rpath = @executable_path
#
# We pin the COMMIT, not a binary sha: builds are not byte-reproducible. Verify by behaviour:
#   bash models/build-diarizer.sh --verify <16k-mono-pcm16.wav> <expected.rttm>
# re-diarizes the WAV and diffs the RTTM (recording-id column ignored).
#
# Needs: git, cmake >= 3.26, ninja, Xcode command line tools (brew install cmake ninja).
set -euo pipefail

REPO="https://github.com/NVIDIA/NeMo-Speech.cpp"
COMMIT="97a15af"   # "feat(diar): make Nemotron 3 Diarization the default diarizer (#52)", 2026-09-24
PRESET="metal-diar"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/diarizer"
SRC="$OUT/src"
BIN="$OUT/bin"
EXE="$BIN/nemo-speech-diar"

log() { printf '[build-diarizer] %s\n' "$*"; }
die() { log "ERROR: $*"; exit 1; }

verify() {
  local wav="$1" want="$2" got
  [ -x "$EXE" ] || die "$EXE missing — run without --verify first"
  got="$(mktemp -t diar-verify).rttm"
  log "verifying: diarize $wav"
  "$EXE" diarize "$wav" -m "$OUT/Nemotron-3-Diarization.q8_0.gguf" --device metal --format rttm \
    --force -o "$got" >/dev/null 2>&1 || die "diarize failed"
  if diff -q <(awk '{$2="x"; print}' "$got") <(awk '{$2="x"; print}' "$want") >/dev/null; then
    log "OK — output matches $want"
  else
    die "output differs from $want (kept at $got)"
  fi
  rm -f "$got"
}

if [ "${1:-}" = "--verify" ]; then
  [ $# -eq 3 ] || die "usage: $0 --verify <16k-mono-pcm16.wav> <expected.rttm>"
  verify "$2" "$3"
  exit 0
fi

for t in git cmake ninja; do command -v "$t" >/dev/null || die "$t not found (brew install cmake ninja)"; done

if [ -x "$EXE" ] && [ "$(cat "$OUT/COMMIT" 2>/dev/null)" = "$COMMIT" ]; then
  log "$EXE already built at $COMMIT — skip (delete $BIN to rebuild)"
  exit 0
fi

mkdir -p "$OUT"
if [ ! -d "$SRC/.git" ]; then
  log "cloning $REPO ..."
  git clone --quiet "$REPO" "$SRC"
fi
git -C "$SRC" fetch --quiet origin
git -C "$SRC" checkout --quiet "$COMMIT"
git -C "$SRC" submodule update --init --depth 1 ggml

log "configuring ($PRESET) ..."
(cd "$SRC" && cmake --preset "$PRESET" >/dev/null)
log "building ..."
(cd "$SRC" && cmake --build --preset "$PRESET" >/dev/null)

built="$SRC/build/$PRESET/bin"
[ -x "$built/nemo-speech" ] || die "build produced no $built/nemo-speech"

# Stage the executable + every @rpath dylib it links.
rm -rf "$BIN" "$OUT/COMMIT" && mkdir -p "$BIN"
cp "$built/nemo-speech" "$EXE"
for lib in $(otool -L "$EXE" | awk '/@rpath\//{print $1}' | sed 's#@rpath/##'); do
  cp -L "$built/$lib" "$BIN/$lib" || die "missing dylib $lib"
done

# On macOS NeMo-Speech.cpp links SentencePiece (and through it abseil) as SHARED Homebrew
# libraries — its static path is ELF-only. Diarization never calls SentencePiece, but the library
# is linked, so bundle every non-system absolute dependency here and point it at @rpath. Otherwise
# a `brew upgrade` (abseil dylib names are versioned) would silently switch splitting off.
deps() { otool -L "$1" | awk 'NR>1{print $1}' | grep '^/' | grep -v -e '^/usr/lib/' -e '^/System/' || true; }
changed=1
while [ "$changed" = 1 ]; do
  changed=0
  for f in "$EXE" "$BIN"/*.dylib; do
    for d in $(deps "$f"); do
      name="$(basename "$d")"
      if [ ! -f "$BIN/$name" ]; then
        cp -L "$d" "$BIN/$name" || die "cannot bundle $d"
        chmod u+w "$BIN/$name"
        install_name_tool -id "@rpath/$name" "$BIN/$name" 2>/dev/null
        changed=1
      fi
      install_name_tool -change "$d" "@rpath/$name" "$f" 2>/dev/null
    done
  done
done

# Only @executable_path may resolve @rpath: drop build-dir / Homebrew rpaths everywhere.
for f in "$EXE" "$BIN"/*.dylib; do
  otool -l "$f" | awk '/LC_RPATH/{r=1} r&&/ path /{print $2; r=0}' | while read -r p; do
    install_name_tool -delete_rpath "$p" "$f" 2>/dev/null
  done
done
install_name_tool -add_rpath @executable_path "$EXE"
codesign -f -s - "$EXE" "$BIN"/*.dylib 2>/dev/null

# Self-containment check: every dependency is @rpath (present in $BIN) or a system library.
for f in "$EXE" "$BIN"/*.dylib; do
  [ -z "$(deps "$f")" ] || die "$f still links $(deps "$f" | head -1)"
  for lib in $(otool -L "$f" | awk 'NR>1{print $1}' | grep '^@rpath/' | sed 's#@rpath/##'); do
    [ -f "$BIN/$lib" ] || die "$f links @rpath/$lib, not staged"
  done
done
"$EXE" --version >/dev/null || die "staged binary does not run"
if DYLD_PRINT_LIBRARIES=1 "$EXE" --version 2>&1 | grep -q -e '/opt/homebrew/' -e "$SRC/"; then
  die "staged binary still loads a library from Homebrew or the build dir"
fi
echo "$COMMIT" > "$OUT/COMMIT"
log "done: $EXE ($(du -sh "$BIN" | awk '{print $1}')) at $COMMIT"
log "GGUF is provisioned by provision.sh into $OUT/"
