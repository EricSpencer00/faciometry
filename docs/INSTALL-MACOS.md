# Vitruve on macOS, without a terminal

Vitruve ships as an ordinary Mac app. Download the disk image, drag Vitruve to
Applications, open it. The web interface appears in your browser. There is no
Python to install and no command to type.

This page is in two halves. The first is for someone installing the app. The
second is for whoever builds and releases it.

---

## Installing

1. Download `Vitruve-<version>-arm64.dmg`.
2. Open it and drag **Vitruve** onto the **Applications** folder.
3. Open Vitruve from Applications or Spotlight.

A small window appears. On the first launch only, it downloads about 415 MB of
model weights and shows the progress. Every file is checked against a sha256
recorded in the repository before it is used, and a mismatch stops the download
rather than being repaired quietly. After that the app opens your browser at
`http://127.0.0.1:<port>/` and you can take or upload a photograph.

The port changes between launches. That is deliberate: the documented default
for `vitruve serve` is 8731, and an app that insisted on it would fail to start
for anyone who already had one running.

**Quitting.** Closing the browser tab does not stop Vitruve. Quit from the
Vitruve window, from the menu bar item, or with Command-Q. Any of those stops
the server. Force-quitting the app also stops it: the server watches the pipe
to the app and exits when it closes, so there is no way to leave one running by
accident.

**Requirements.** An Apple silicon Mac running macOS 12 or later, about 1.6 GB
of disk, and a network connection for the first launch. There is no Intel
build; the bundle carries an arm64 interpreter and arm64 builds of torch,
OpenCV and MediaPipe. The disk image is around 330 MB, the installed app around
1.2 GB, and the weights another 416 MB in your home directory.

**What the app is allowed to do.** It binds a loopback socket, reads the photo
you give it, and writes model weights to `~/.cache/vitruve/weights`. It writes
a log to `~/Library/Logs/Vitruve/vitruve.log`. It is not sandboxed, which is
normal for a directly downloaded Mac app, so the guarantee that nothing leaves
the machine comes from the code and its tests rather than from the sandbox.
`docs/PRIVACY.md` says which test asserts which claim.

**The camera.** The capture UI runs in your browser, so it is the browser that
asks for camera permission and the browser that appears in System Settings
under Privacy and Security. The app carries a camera entitlement and a usage
description for the paths that reach the camera from inside the bundle.

### If macOS refuses to open it

> "Vitruve" cannot be opened because Apple cannot check it for malicious
> software.

That message means the build you downloaded was **not notarised**. Check the
`.build.txt` file next to the disk image: it records, for that exact artifact,
whether it was signed and whether it was notarised.

A notarised build opens on a double-click with no dialog at all. If you have a
build that is not notarised, the workaround is to right-click the app and
choose Open, then confirm. Do that only if you trust where the file came from.
Teaching yourself to click through that dialog is a bad habit and the reason
the release process treats notarisation as mandatory rather than optional.

### Uninstalling

Drag Vitruve out of Applications. Then, if you want the rest:

```
rm -rf ~/.cache/vitruve ~/Library/Logs/Vitruve
```

Nothing else is written anywhere.

---

## Building a release

```
packaging/macos/build_app.sh
```

That produces `packaging/macos/dist/Vitruve-<version>-arm64.dmg` and a
`.build.txt` beside it recording the signing and notarisation state.

Options: `--no-sign` for a fast unsigned build, `--skip-deps` to reuse the
staged runtime from a previous run, `--no-dmg`, `--no-notarize`.

A full build takes about twelve minutes on an M1 Max: three to resolve and
install the stack, four to sign 429 Mach-O files with a timestamp on each, and
forty seconds to compress the image. `--skip-deps` removes the first three.

The image format is ULFO, which is LZFSE, and that was measured rather than
assumed: UDZO at `zlib-level=9` took twelve minutes and produced 358 MB, ULFO
took thirty-six seconds and produced 330 MB.

### What goes into the bundle

```
Vitruve.app/Contents/
  MacOS/Vitruve                 a Swift launcher, the only thing macOS executes
  Resources/runtime/            CPython 3.11, python-build-standalone
  Resources/runtime/lib/python3.11/site-packages/
                                vitruve, plus the [permissive] and [api] extras
  Resources/app_main.py         what the launcher supervises
  Resources/assets/             weights.lock.json
  Resources/Vitruve.icns
```

There is no virtualenv in there. A venv writes the absolute path of its base
interpreter into `pyvenv.cfg` and into every console script, and an app bundle
is relocated by definition, so the build installs straight into the bundled
interpreter's own `site-packages`. The relocation test in the acceptance list
below is what proves it.

The `[pdf]` extra is deliberately not bundled. WeasyPrint links against Pango
and Cairo from a system package manager, and a self-contained app cannot
promise those exist.

### Weights are fetched on first run, not bundled

The four pinned artifacts total about 415 MB, of which SPIGA alone is 254 MB.
They are downloaded on first launch, with a progress bar, rather than shipped
inside the disk image.

The reasons, in the order they mattered:

- **The disk image stays a download people finish.** Bundling would push it
  past a gigabyte, on top of a runtime that is already large because torch is.
- **`vitruve fetch-weights` already exists and already verifies.** The app
  calls the same `vitruve.models.weights.download`, so the sha256 check, the
  atomic rename and the hard failure on a mismatch are the shipped code path
  and not a second implementation that could drift from it.
- **Redistribution is a licence question, and not fetching it is the cleaner
  answer.** The permissive tier is clean, but the moment weights are copied
  into a signed artifact the project is redistributing them rather than
  pointing at them, and the tier system exists precisely to keep that boundary
  visible. 6DRepNet's checkpoint is trained on 300W-LP, which is rendered from
  the Basel Face Model, and that inherited obligation is recorded on the
  provenance rather than on the file.
- **Weights outlive a version.** A user who installs three releases downloads
  them once, because the cache is `~/.cache/vitruve/weights` and not inside the
  app.

The cost is real and is stated where a user sees it: the app needs the network
once, and the first launch takes as long as the download takes. If the download
fails the app still opens, the catalogue and licence pages work, and analysis
reports what is missing.

### Signing

Signing runs inner to outer. Every Mach-O in the bundle is signed first, then
the interpreter and the launcher with entitlements, then the app itself. The
list of Mach-O files is built by reading magic numbers rather than by matching
extensions, because some wheels ship extensionless helper binaries.

Order matters and so does completeness: one unsigned `.so` under
`site-packages/torch/lib` invalidates the whole bundle at Gatekeeper time and
is rejected outright by the notary service. This bundle has roughly two
thousand of them.

The entitlements are two lines, and the file says why each is there. The app is
not sandboxed, so the `com.apple.security.network.*` keys would be inert and
are deliberately absent; the loopback socket needs no entitlement.
`com.apple.security.cs.allow-jit` is applied to the **interpreter** as well as
the launcher, because entitlements attach to a Mach-O and not to a process
tree, and the Python process is the one that runs libffi.

Verify by hand:

```
codesign --verify --deep --strict --verbose=2 packaging/macos/build/Vitruve.app
spctl -a -vvv -t exec packaging/macos/build/Vitruve.app
```

### Notarisation, the one manual step

`build_app.sh` runs notarisation and stapling automatically **when credentials
exist**, and prints the exact command to create them when they do not. It does
not fail the build in that case, because a build that cannot be published is
still a build that can be tested.

Set the credentials up once, using an app-specific password generated at
<https://appleid.apple.com> under Sign-In and Security:

```
xcrun notarytool store-credentials "vitruve-notary" \
    --apple-id "YOUR_APPLE_ID@example.com" \
    --team-id "QAWD9U9CF6" \
    --password "abcd-efgh-ijkl-mnop"
```

That writes the credentials into the login keychain under the profile name
`vitruve-notary`, which is what `notarize.sh` looks for. Nothing else in the
build changes; the next run notarises and staples on its own.

CI uses the other path, an App Store Connect API key, because a `.p8` and two
identifiers travel through repository secrets and an app-specific password does
not:

```
export MACOS_NOTARY_KEY=/path/to/AuthKey_XXXXXXXXXX.p8
export MACOS_NOTARY_KEY_ID=XXXXXXXXXX
export MACOS_NOTARY_ISSUER_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

**A release must be notarised.** An app that makes a user right-click and
confirm a malware warning is a materially worse product, and it is a worse
product specifically for this one, whose claim is that it is careful with your
face. The `.build.txt` beside every disk image records which state that
artifact is in, and the build prints a warning when it is not notarised.

### Continuous integration

`.github/workflows/macos-app.yml` builds the same artifact on a `macos-14`
runner. Signing is driven by repository secrets and every signing step is
gated on them, so a fork with no secrets gets a green run and an unsigned disk
image whose summary says it is unsigned.

| Secret | What it is |
|---|---|
| `MACOS_CERT_P12` | base64 of a `.p12` holding the Developer ID Application certificate and its key |
| `MACOS_CERT_PASSWORD` | the password that `.p12` was exported with |
| `MACOS_SIGN_IDENTITY` | e.g. `Developer ID Application: Eric Spencer (QAWD9U9CF6)` |
| `MACOS_NOTARY_KEY` | base64 of an App Store Connect `.p8` |
| `MACOS_NOTARY_KEY_ID` | that key's ID |
| `MACOS_NOTARY_ISSUER_ID` | the issuer UUID |

### Acceptance checks for a build

1. The app launches, the browser opens, and `/health` returns 200.
2. **Move the `.app` to a different directory and launch it again.** This is
   the check that catches a path baked into the bundle at build time, which is
   the usual way a bundled interpreter breaks, and it is why there is no venv
   in the bundle.
3. `codesign --verify --deep --strict --verbose=2` reports the bundle valid on
   disk and satisfying its designated requirement.
4. `spctl -a -vvv -t exec` accepts it. Before notarisation this command
   **rejects** with `source=Unnotarized Developer ID`, which is the correct
   result for an unnotarised build and not a signing failure.
5. Quit the app, then check that no `python3.11` process from the bundle
   survives.
