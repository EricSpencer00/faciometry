#!/usr/bin/env bash
#
# Submit an artifact to Apple's notary service and staple the ticket to it.
#
# Called with a .app or a .dmg. Exits 0 when the artifact came back notarised
# and stapled, and non-zero otherwise. build_app.sh treats a non-zero exit as
# "this build is not notarised" and carries on, because a build that cannot be
# published is still a build that can be tested, and failing the whole thing
# would mean nobody can produce an artifact without Apple credentials.
#
# What notarisation buys, so the decision to skip it is an informed one:
# without a stapled ticket, Gatekeeper on any Mac other than the one that
# signed the app shows "Vitruve cannot be opened because Apple cannot check it
# for malicious software", and the user has to right-click, choose Open, and
# confirm a scary dialog. For a tool whose entire pitch is that it is careful
# with your face, teaching users to click through that dialog is the wrong
# lesson. This step is not optional for a release.
#
# Two credential paths, checked in this order:
#
#   1. An App Store Connect API key in the environment. This is what CI uses,
#      because a .p8 file and two identifiers travel through repository secrets
#      and an app-specific password does not travel well at all.
#        MACOS_NOTARY_KEY        path to the .p8, or its base64 contents
#        MACOS_NOTARY_KEY_ID     the key ID
#        MACOS_NOTARY_ISSUER_ID  the issuer UUID
#
#   2. A notarytool keychain profile, which is what a laptop uses. Created once
#      with `xcrun notarytool store-credentials`; the exact command is printed
#      below when the profile is missing.

set -euo pipefail

ARTIFACT="${1:?usage: notarize.sh PATH_TO_APP_OR_DMG}"
NOTARY_PROFILE="${NOTARY_PROFILE:-vitruve-notary}"
TEAM_ID="${TEAM_ID:-QAWD9U9CF6}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*"; }

instructions() {
  cat >&2 <<EOF

    Notarisation was skipped: no credentials are available on this machine.

    To set them up once, with an app-specific password generated at
    https://appleid.apple.com under Sign-In and Security:

        xcrun notarytool store-credentials "$NOTARY_PROFILE" \\
            --apple-id "YOUR_APPLE_ID@example.com" \\
            --team-id "$TEAM_ID" \\
            --password "abcd-efgh-ijkl-mnop"

    That writes the credentials into the login keychain under the profile name
    "$NOTARY_PROFILE" and this script picks them up on the next run. Nothing
    else in the build changes.

    Alternatively, export an App Store Connect API key:

        export MACOS_NOTARY_KEY=/path/to/AuthKey_XXXXXXXXXX.p8
        export MACOS_NOTARY_KEY_ID=XXXXXXXXXX
        export MACOS_NOTARY_ISSUER_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

EOF
}

# ---------------------------------------------------------------------------
# Which credential path, if any
# ---------------------------------------------------------------------------

CREDS=()
CLEANUP_KEY=""

if [ -n "${MACOS_NOTARY_KEY:-}" ] && [ -n "${MACOS_NOTARY_KEY_ID:-}" ] && [ -n "${MACOS_NOTARY_ISSUER_ID:-}" ]; then
  KEY_PATH="$MACOS_NOTARY_KEY"
  if [ ! -f "$KEY_PATH" ]; then
    # A .p8 cannot be put in a GitHub secret as a file, so CI passes it
    # base64-encoded and it is materialised here into a file that is removed on
    # exit whatever happens.
    KEY_PATH="$(mktemp -t vitruve-notary-key)"
    CLEANUP_KEY="$KEY_PATH"
    printf '%s' "$MACOS_NOTARY_KEY" | base64 --decode > "$KEY_PATH" 2>/dev/null || {
      warn "MACOS_NOTARY_KEY is neither a readable path nor valid base64"
      instructions
      exit 1
    }
  fi
  CREDS=(--key "$KEY_PATH" --key-id "$MACOS_NOTARY_KEY_ID" --issuer "$MACOS_NOTARY_ISSUER_ID")
  note "credentials: App Store Connect API key $MACOS_NOTARY_KEY_ID"

elif security find-generic-password -s "com.apple.gke.notary.tool" -a "$NOTARY_PROFILE" >/dev/null 2>&1; then
  # The keychain item is checked directly rather than by calling notarytool,
  # because every notarytool subcommand is a network round trip and a missing
  # profile should be an instant, offline answer.
  CREDS=(--keychain-profile "$NOTARY_PROFILE")
  note "credentials: notarytool keychain profile \"$NOTARY_PROFILE\""

else
  say "Notarisation"
  instructions
  exit 1
fi

# Guarded with if rather than &&: an EXIT trap whose last command fails can
# change the script's exit status, and this script's exit status is the only
# thing build_app.sh reads.
cleanup() { if [ -n "$CLEANUP_KEY" ]; then rm -f "$CLEANUP_KEY"; fi; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

say "Notarising $(basename "$ARTIFACT")"

SUBMIT="$ARTIFACT"
TMPZIP=""
case "$ARTIFACT" in
  *.app)
    # The notary service does not accept a bare .app: it takes a zip, a dmg or
    # a pkg. ditto -c -k --keepParent is the only zip tool that preserves the
    # bundle's symlinks and signature xattrs; /usr/bin/zip does not.
    TMPZIP="$(mktemp -d -t vitruve-notary)/Vitruve.zip"
    ditto -c -k --keepParent "$ARTIFACT" "$TMPZIP"
    SUBMIT="$TMPZIP"
    ;;
esac

if ! xcrun notarytool submit "$SUBMIT" "${CREDS[@]}" --wait --timeout 45m; then
  warn "the notary service rejected the submission or timed out"
  warn "read the log with: xcrun notarytool log <submission-id> ${CREDS[*]}"
  if [ -n "$TMPZIP" ]; then rm -rf "$(dirname "$TMPZIP")"; fi
  exit 1
fi
if [ -n "$TMPZIP" ]; then rm -rf "$(dirname "$TMPZIP")"; fi

# ---------------------------------------------------------------------------
# Staple
# ---------------------------------------------------------------------------
#
# Stapling attaches the ticket to the artifact itself, so Gatekeeper can
# validate it without a network call. Without it a user who is offline, or
# behind a firewall that blocks Apple's OCSP endpoints, gets the same refusal
# as an unnotarised app. The ticket is attached to the artifact that was
# submitted AND, for a dmg, to the app inside it, which is why build_app.sh
# notarises the .app first and builds the dmg from the stapled copy.

say "Stapling"
if ! xcrun stapler staple "$ARTIFACT"; then
  warn "stapling failed; the artifact is notarised but needs a network check to open"
  exit 1
fi
xcrun stapler validate "$ARTIFACT" | sed 's/^/    /'

note "notarised and stapled"
