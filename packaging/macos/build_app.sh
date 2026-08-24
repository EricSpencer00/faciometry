#!/usr/bin/env bash
#
# Build Vitruve.app and a .dmg that a non-technical user can install by
# dragging.
#
# The shape of the result, and why:
#
#   Vitruve.app/Contents/
#     MacOS/Vitruve                  the Swift launcher, the only thing macOS runs
#     Resources/runtime/             a python-build-standalone interpreter
#     Resources/runtime/lib/python3.11/site-packages/
#                                    vitruve plus the [permissive] and [api] stacks
#     Resources/app_main.py          the process the launcher supervises
#     Resources/assets/              weights.lock.json, the sha256 pins
#
# There is no virtualenv anywhere in that tree, on purpose. A venv records the
# absolute path of its base interpreter in pyvenv.cfg and in every console
# script's shebang, and an .app is relocated by definition: it is built in one
# directory, dragged to /Applications, and sometimes run from a Downloads
# folder. Installing straight into the bundled interpreter's own site-packages
# removes the class of bug rather than patching it up afterwards. The one
# remaining absolute path, the shebang on the console scripts uv writes into
# runtime/bin, is rewritten to a relative trampoline below.
#
# Usage:
#   ./build_app.sh                       build, sign, notarise if possible, make a dmg
#   ./build_app.sh --no-sign             skip signing entirely (fast local iteration)
#   ./build_app.sh --skip-deps           reuse the staged runtime from a previous run
#   ./build_app.sh --no-dmg              stop after the .app
#   ./build_app.sh --no-notarize         sign but do not submit to Apple
#
# Environment:
#   SIGN_IDENTITY       default "Developer ID Application: Eric Spencer (QAWD9U9CF6)"
#   NOTARY_PROFILE      notarytool keychain profile name, default "vitruve-notary"
#   MACOS_NOTARY_KEY / _KEY_ID / _ISSUER_ID   App Store Connect key, used by CI

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths and options
# ---------------------------------------------------------------------------

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
BUILD="$HERE/build"
DIST="$HERE/dist"
APP="$BUILD/Vitruve.app"
SNAPSHOT="$BUILD/source"

PY_VERSION="${PY_VERSION:-3.11}"
PY_TAG="python3.11"
BUNDLE_ID="${BUNDLE_ID:-us.ericspencer.vitruve}"
MIN_MACOS="${MIN_MACOS:-12.0}"
SIGN_IDENTITY="${SIGN_IDENTITY:-Developer ID Application: Eric Spencer (QAWD9U9CF6)}"
NOTARY_PROFILE="${NOTARY_PROFILE:-vitruve-notary}"
SIGN_JOBS="${SIGN_JOBS:-6}"
export NOTARY_PROFILE

DO_SIGN=1
DO_DMG=1
DO_NOTARIZE=1
SKIP_DEPS=0

while [ $# -gt 0 ]; do
  case "$1" in
    --no-sign) DO_SIGN=0; DO_NOTARIZE=0 ;;
    --no-dmg) DO_DMG=0 ;;
    --no-notarize) DO_NOTARIZE=0 ;;
    --skip-deps) SKIP_DEPS=1 ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

ARCH="$(uname -m)"
# python-build-standalone names Apple silicon "aarch64" while uname says
# "arm64". Both spellings are needed: one for the directory glob, one for the
# Swift target triple and the artifact name.
case "$ARCH" in
  arm64) PY_ARCH="aarch64" ;;
  *) PY_ARCH="$ARCH" ;;
esac

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() { printf '\n\033[1;31mbuild_app.sh: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

say "Preflight"
for tool in uv xcrun iconutil hdiutil ditto; do
  command -v "$tool" >/dev/null 2>&1 || die "$tool is not on PATH"
done
xcrun -f swiftc >/dev/null 2>&1 || die "swiftc is not available; install the Xcode command line tools"
[ "$ARCH" = "arm64" ] || note "building on $ARCH: the bundle will be $ARCH-only"

# The version is single-sourced from pyproject.toml. Parsed with a narrow
# regex rather than a TOML library so this script has no Python dependency of
# its own before the runtime is staged.
VERSION="$(sed -n 's/^version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$REPO/pyproject.toml" | head -1)"
[ -n "$VERSION" ] || die "could not read version from $REPO/pyproject.toml"
BUILD_NUMBER="$(date -u +%Y%m%d%H%M)"
note "vitruve $VERSION (build $BUILD_NUMBER), arch $ARCH"

if [ "$DO_SIGN" = 1 ]; then
  if ! security find-identity -v -p codesigning | grep -qF "$SIGN_IDENTITY"; then
    die "signing identity not found in the keychain: $SIGN_IDENTITY
    Re-run with --no-sign to produce an unsigned bundle for local testing."
  fi
  note "signing as: $SIGN_IDENTITY"
else
  note "signing disabled (--no-sign)"
fi

# ---------------------------------------------------------------------------
# Source snapshot
# ---------------------------------------------------------------------------
#
# The wheel is built from a copy, not from the working tree. Two reasons: a
# build must not pick up whatever a `git status` happens to show, and hatchling
# resolves the force-include of web/ relative to the project root, so the copy
# has to carry the same shape.

say "Snapshotting the source"
mkdir -p "$BUILD" "$DIST"
rm -rf "$SNAPSHOT"
mkdir -p "$SNAPSHOT"
for item in src web assets pyproject.toml README.md LICENSE; do
  [ -e "$REPO/$item" ] || die "missing from the checkout: $item"
  cp -R "$REPO/$item" "$SNAPSHOT/"
done
find "$SNAPSHOT" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
note "$(du -sh "$SNAPSHOT" | cut -f1) staged in $SNAPSHOT"

# ---------------------------------------------------------------------------
# The interpreter
# ---------------------------------------------------------------------------

RUNTIME="$APP/Contents/Resources/runtime"
BUNDLED_PY="$RUNTIME/bin/$PY_TAG"

if [ "$SKIP_DEPS" = 1 ] && [ -x "$BUNDLED_PY" ]; then
  say "Reusing the staged runtime (--skip-deps)"
else
  say "Staging a relocatable CPython $PY_VERSION"
  uv python install "$PY_VERSION" >/dev/null
  UV_PY_DIR="$(uv python dir)"
  # Glob rather than `uv python find`, which happily returns a venv in the
  # current directory. We want the managed standalone build and nothing else.
  SRC_PY=""
  for candidate in "$UV_PY_DIR"/cpython-"$PY_VERSION".*-macos-"$PY_ARCH"-none; do
    [ -x "$candidate/bin/$PY_TAG" ] && SRC_PY="$candidate"
  done
  [ -n "$SRC_PY" ] || die "no managed cpython-$PY_VERSION-macos-$PY_ARCH-none under $UV_PY_DIR"
  note "from $SRC_PY"

  rm -rf "$APP"
  mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
  # -R and not a symlink: the app has to carry its own interpreter or it is
  # not an app, it is a shortcut to one machine's home directory.
  cp -R "$SRC_PY" "$RUNTIME"
  chmod -R u+w "$RUNTIME"

  # This copy is ours now, not uv's, and PEP 668 would otherwise refuse every
  # install into it.
  find "$RUNTIME" -name EXTERNALLY-MANAGED -delete

  # Bulk that cannot be reached from `vitruve serve`. Tcl/Tk is the big one:
  # about 40 MB of GUI toolkit for a program whose entire interface is a web
  # page. The stdlib test suite and the C headers are the other two.
  say "Pruning the interpreter"
  BEFORE="$(du -sm "$RUNTIME" | cut -f1)"
  rm -rf "$RUNTIME/include" "$RUNTIME/share"
  rm -rf "$RUNTIME/lib/tcl"* "$RUNTIME/lib/tk"* "$RUNTIME/lib/itcl"* "$RUNTIME/lib/thread"*
  rm -f  "$RUNTIME/lib/libtcl"*.dylib "$RUNTIME/lib/libtk"*.dylib
  rm -rf "$RUNTIME/lib/$PY_TAG/test" "$RUNTIME/lib/$PY_TAG/idlelib" \
         "$RUNTIME/lib/$PY_TAG/turtledemo" "$RUNTIME/lib/$PY_TAG/tkinter" \
         "$RUNTIME/lib/$PY_TAG/config-"*
  rm -f  "$RUNTIME/lib/$PY_TAG/lib-dynload/_tkinter"*.so
  note "interpreter ${BEFORE} MB -> $(du -sm "$RUNTIME" | cut -f1) MB"

  # -------------------------------------------------------------------------
  # Dependencies
  # -------------------------------------------------------------------------
  #
  # [permissive] is the default backend stack and [api] is the server the app
  # exists to run. [pdf] is deliberately absent: WeasyPrint links against Pango
  # and Cairo from a system package manager, which a self-contained .app has no
  # way to guarantee, so bundling it would ship a report format that fails at
  # the moment a user asks for it.
  # --link-mode=copy is not a performance knob, it is a correctness one. uv
  # hardlinks wheel contents out of its cache by default, and codesign rewrites
  # a Mach-O in place: signing a hardlinked .so would write the app's signature
  # into the shared cache entry, and the next project to install that wheel
  # would get a file signed with this app's identity.
  say "Installing vitruve[permissive,api] into the bundle"
  uv pip install \
    --python "$BUNDLED_PY" \
    --link-mode=copy \
    "$SNAPSHOT[permissive,api]"

  say "Pruning site-packages"
  SP="$RUNTIME/lib/$PY_TAG/site-packages"
  BEFORE="$(du -sm "$SP" | cut -f1)"
  # torch ships its C++ headers and static archives for people compiling
  # extensions against it. Nothing in a running app reads them.
  #
  # torch/bin is NOT in this list, and the first version of it was: removing
  # torch/bin deletes torch_shm_manager, which torch looks for at import time,
  # and the only symptom is `vitruve doctor` and /health reporting the device
  # as "unknown" with the failure buried in device_detail. Prune by what a
  # running app reads, and check /health afterwards.
  rm -rf "$SP/torch/include" "$SP/torch/test" "$SP/torch/utils/bottleneck"
  # pip cannot be usefully run inside a signed bundle: anything it installed
  # would break the seal on Resources.
  rm -rf "$SP/pip" "$SP"/pip-*.dist-info
  find "$SP" -name "*.a" -delete
  find "$SP" -type d -name "tests" -path "*/numpy/*" -prune -exec rm -rf {} + 2>/dev/null || true
  find "$SP" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
  note "site-packages ${BEFORE} MB -> $(du -sm "$SP" | cut -f1) MB"
fi

SP="$RUNTIME/lib/$PY_TAG/site-packages"

# ---------------------------------------------------------------------------
# Relocatable shebangs
# ---------------------------------------------------------------------------
#
# uv writes console scripts whose shebang is the absolute path of the
# interpreter at install time, which here is a path inside packaging/macos/build
# on the machine that ran this script. Nothing in the app calls them (the
# launcher runs `python -m`), but a broken shebang in a shipped bundle is a
# trap for anyone who opens a terminal inside it, and it leaks a build path. The
# replacement is the standard sh/python polyglot: sh runs the exec line, Python
# sees a string literal.

say "Rewriting console script shebangs to a relative trampoline"
rewritten=0
for script in "$RUNTIME"/bin/*; do
  [ -f "$script" ] || continue
  head -c2 "$script" 2>/dev/null | grep -q '#!' || continue
  grep -q "^#!$BUILD" "$script" 2>/dev/null || continue
  tail -n +2 "$script" > "$script.body"
  {
    printf '#!/bin/sh\n'
    printf "'''exec' \"\$(dirname -- \"\$(realpath -- \"\$0\")\")/%s\" \"\$0\" \"\$@\"\n" "$PY_TAG"
    printf "' '''\n"
    cat "$script.body"
  } > "$script.new"
  mv "$script.new" "$script"
  rm -f "$script.body"
  chmod +x "$script"
  rewritten=$((rewritten + 1))
done
note "$rewritten script(s) rewritten"

if grep -rlI "$BUILD" "$RUNTIME/bin" 2>/dev/null | grep -q .; then
  die "a build path survives in $RUNTIME/bin; the bundle would not relocate"
fi

# ---------------------------------------------------------------------------
# Bundle resources
# ---------------------------------------------------------------------------

say "Assembling the bundle"
RES="$APP/Contents/Resources"
cp "$HERE/resources/app_main.py" "$RES/app_main.py"

# weights.lock.json is not in the wheel, and vitruve.models.weights looks for
# it four directories above its own source file, which is a checkout layout.
# app_main.py points VITRUVE_WEIGHTS_LOCK at this copy.
mkdir -p "$RES/assets"
cp "$REPO/assets/weights.lock.json" "$RES/assets/weights.lock.json"
cp "$REPO/LICENSE" "$RES/LICENSE"

sed -e "s|__BUNDLE_ID__|$BUNDLE_ID|g" \
    -e "s|__VERSION__|$VERSION|g" \
    -e "s|__BUILD__|$BUILD_NUMBER|g" \
    -e "s|__MIN_MACOS__|$MIN_MACOS|g" \
    "$HERE/templates/Info.plist" > "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"

say "Drawing the icon"
uv run --quiet --with "pillow>=10.3" python "$HERE/make_icon.py" \
  "$RES/Vitruve.icns" --png "$BUILD/icon-1024.png"

say "Compiling the launcher"
xcrun swiftc \
  -swift-version 5 -O \
  -target "$ARCH-apple-macos$MIN_MACOS" \
  -o "$APP/Contents/MacOS/Vitruve" \
  "$HERE/launcher/main.swift"

# ---------------------------------------------------------------------------
# Byte-compile before signing
# ---------------------------------------------------------------------------
#
# Two reasons, and the second is the one that matters. First, a cold import of
# torch and mediapipe from source is several seconds slower than from .pyc.
# Second, the Resources directory is sealed by the app's signature: a .pyc
# written after signing changes the sealed contents and invalidates it. The app
# runs with PYTHONDONTWRITEBYTECODE=1 so nothing is written at runtime, which
# is only tolerable if everything is compiled now.

say "Byte-compiling the bundled Python"
"$BUNDLED_PY" -m compileall -q -j 0 "$RUNTIME/lib/$PY_TAG" >/dev/null 2>&1 || \
  note "compileall reported errors on some files, which is normal for a tree that ships py2 fixtures"
"$BUNDLED_PY" -m compileall -q "$RES/app_main.py" >/dev/null 2>&1 || true

say "Smoke-testing the bundled interpreter"
"$BUNDLED_PY" -I -c '
import sys
import vitruve
from vitruve.api.app import web_root
print("    vitruve", vitruve.__version__, "on python", sys.version.split()[0])
print("    web ui:", web_root())
assert web_root() is not None, "the web UI is not in the wheel"
' || die "the bundled interpreter cannot import vitruve"

# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

ENTITLEMENTS="$HERE/templates/entitlements.plist"

# Every Mach-O in the bundle, deepest first. A single unsigned .so anywhere
# under Resources invalidates the whole bundle at Gatekeeper time and is
# rejected outright by the notary service, and this tree has thousands of them
# across numpy, opencv, mediapipe and torch. The list is built by reading magic
# numbers rather than by trusting file extensions, because some wheels ship
# extensionless helper binaries and .dylib content under a .so name.
machos() {
  "$BUNDLED_PY" - "$1" <<'PYEOF'
import os, sys

MAGIC = {
    b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",   # 32-bit Mach-O, both endians
    b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",   # 64-bit
    b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",   # universal
}
root = sys.argv[1]
found = []
for dirpath, dirnames, filenames in os.walk(root):
    for name in filenames:
        p = os.path.join(dirpath, name)
        if os.path.islink(p):
            continue
        try:
            with open(p, "rb") as fh:
                if fh.read(4) in MAGIC:
                    found.append(p)
        except OSError:
            pass
# Deepest first: codesign seals a directory's contents, so anything nested has
# to be final before its container is signed.
found.sort(key=lambda p: (-p.count(os.sep), p))
sys.stdout.write("\0".join(found))
PYEOF
}

if [ "$DO_SIGN" = 1 ]; then
  say "Signing"

  # Extended attributes and .DS_Store files make codesign refuse the bundle
  # with "resource fork, Finder information, or similar detritus not allowed".
  xattr -cr "$APP"
  find "$APP" -name '.DS_Store' -delete

  MACHO_LIST="$BUILD/macho-list"
  machos "$APP" > "$MACHO_LIST"
  COUNT="$(tr '\0' '\n' < "$MACHO_LIST" | grep -c . || true)"
  note "$COUNT Mach-O files to sign"

  # --timestamp is one network round trip per invocation and notarisation
  # requires it on everything, so this is the slow step. Batching (-n) and
  # parallelism (-P) both help; the retry is there because Apple's timestamp
  # service rate-limits, and a single 500 would otherwise fail the build after
  # twenty minutes of work.
  export SIGN_IDENTITY
  tr '\0' '\n' < "$MACHO_LIST" | grep -v "^$APP/Contents/MacOS/Vitruve$" | tr '\n' '\0' | \
  xargs -0 -P "$SIGN_JOBS" -n 12 /bin/sh -c '
    for attempt in 1 2 3 4; do
      if codesign --force --sign "$SIGN_IDENTITY" --timestamp --options runtime "$@" 2>/tmp/vitruve-codesign.$$; then
        rm -f /tmp/vitruve-codesign.$$
        exit 0
      fi
      cat /tmp/vitruve-codesign.$$ >&2
      sleep $((attempt * 4))
    done
    echo "codesign failed after 4 attempts on: $*" >&2
    exit 1
  ' sh

  # The interpreter is signed a second time, with entitlements. Entitlements
  # are a property of a Mach-O and not of a process tree: the launcher is one
  # executable and python3.11 is another, spawned as a separate process with
  # its own signature. Putting allow-jit only on the launcher would leave the
  # process that actually runs libffi without it.
  codesign --force --sign "$SIGN_IDENTITY" --timestamp --options runtime \
    --entitlements "$ENTITLEMENTS" "$BUNDLED_PY"

  codesign --force --sign "$SIGN_IDENTITY" --timestamp --options runtime \
    --entitlements "$ENTITLEMENTS" "$APP/Contents/MacOS/Vitruve"

  # Outermost last. This seals Resources, so nothing may be added to the
  # bundle after this line.
  codesign --force --sign "$SIGN_IDENTITY" --timestamp --options runtime \
    --entitlements "$ENTITLEMENTS" "$APP"

  say "Verifying the signature"
  codesign --verify --deep --strict --verbose=2 "$APP" 2>&1 | sed 's/^/    /'
else
  say "Ad-hoc signing (unsigned build)"
  # Even an unsigned build needs ad-hoc signatures on arm64: macOS refuses to
  # execute an arm64 Mach-O that carries no signature at all.
  xattr -cr "$APP"
  machos "$APP" | xargs -0 -P "$SIGN_JOBS" -n 20 codesign -s - -f 2>/dev/null || true
  codesign -s - -f --entitlements "$ENTITLEMENTS" "$APP" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Notarisation
# ---------------------------------------------------------------------------

NOTARISED="no"
if [ "$DO_NOTARIZE" = 1 ] && [ "$DO_SIGN" = 1 ]; then
  if "$HERE/notarize.sh" "$APP"; then
    NOTARISED="yes"
  fi
fi

# ---------------------------------------------------------------------------
# Disk image
# ---------------------------------------------------------------------------

DMG=""
if [ "$DO_DMG" = 1 ]; then
  say "Building the disk image"
  STAGE="$BUILD/dmg-stage"
  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  # ditto rather than cp: it preserves the signature's extended attributes and
  # the bundle's symlinks exactly, and cp -R has historically mangled both.
  ditto "$APP" "$STAGE/Vitruve.app"
  ln -s /Applications "$STAGE/Applications"

  DMG="$DIST/Vitruve-$VERSION-$ARCH.dmg"
  rm -f "$DMG"
  # ULFO is LZFSE. It was chosen by measuring both on this bundle rather than
  # by reputation: UDZO at zlib-level=9 took 12 minutes and produced 358 MB,
  # ULFO took 36 seconds and produced 330 MB. LZFSE images mount on macOS
  # 10.11 and later, and this app already requires 12.
  hdiutil create \
    -volname "Vitruve $VERSION" \
    -srcfolder "$STAGE" \
    -fs HFS+ \
    -format ULFO \
    -ov -quiet \
    "$DMG"

  if [ "$DO_SIGN" = 1 ]; then
    codesign --force --sign "$SIGN_IDENTITY" --timestamp "$DMG"
    if [ "$NOTARISED" = "yes" ]; then
      "$HERE/notarize.sh" "$DMG" || true
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

say "Result"
note "app:  $APP  ($(du -sh "$APP" | cut -f1))"
if [ -n "$DMG" ]; then
  note "dmg:  $DMG  ($(du -h "$DMG" | cut -f1))"
fi

if [ "$DO_SIGN" = 1 ]; then
  echo
  echo "    codesign --verify --deep --strict:"
  codesign --verify --deep --strict --verbose=2 "$APP" 2>&1 | sed 's/^/      /' || true
  echo
  echo "    spctl -a -vvv -t exec:"
  spctl -a -vvv -t exec "$APP" 2>&1 | sed 's/^/      /' || true
fi

# The state of a given build is a fact about that artifact, so it is written
# next to it rather than left in a terminal scrollback.
if [ -n "$DMG" ]; then
  {
    echo "vitruve $VERSION, build $BUILD_NUMBER, $ARCH"
    echo "signed:     $([ "$DO_SIGN" = 1 ] && echo "yes, $SIGN_IDENTITY" || echo no)"
    echo "notarised:  $NOTARISED"
    if [ "$NOTARISED" = "yes" ]; then
      echo "gatekeeper: opens on double-click on any Mac"
    else
      echo "gatekeeper: BLOCKED on a Mac other than the one that built it."
      echo "            A user must right-click and choose Open, or the app must be"
      echo "            notarised before release. See docs/INSTALL-MACOS.md."
    fi
  } > "${DMG%.dmg}.build.txt"
  sed 's/^/    /' "${DMG%.dmg}.build.txt"
fi

echo
if [ "$NOTARISED" != "yes" ] && [ "$DO_SIGN" = 1 ]; then
  printf '\033[1;33m    This build is signed but NOT notarised. Do not publish it as-is.\033[0m\n'
fi
