"""What the .app actually runs.

The Swift launcher does not know anything about Python. It spawns this file
once, reads a line-oriented event stream from its stdout, and sends it SIGTERM
when the user quits. Everything Python-shaped happens here.

Three things this file exists to do, none of which belong in `vitruve serve`:

1. **Point the library at the bundled lock file.** A wheel-installed
   `vitruve.models.weights` looks for `assets/weights.lock.json` four
   directories above its own source file, which is a repository checkout
   layout and is not where anything lives inside an .app. The lock is copied
   into `Contents/Resources/assets/` at build time and pointed at with
   `VITRUVE_WEIGHTS_LOCK` here.

2. **Fetch weights on first run, visibly.** `vitruve fetch-weights` prints one
   line per artifact and nothing in between, which for 415 MB over a domestic
   connection is several minutes of an app that looks hung. This calls the
   same pinned, sha256-verified `weights.download` underneath and emits byte
   counts, so the launcher can draw a real progress bar. The verification and
   the atomic rename are the library's, not reimplemented here.

3. **Die when the launcher dies.** The launcher's stdin pipe is held open for
   the life of the launcher process. If the launcher is killed outright, the
   pipe closes, the watchdog thread sees EOF and takes the process down. That
   is what stops a user force-quitting the app from leaving a server holding a
   port and a loaded copy of torch.

The event stream is one JSON object per line behind a sentinel prefix, because
uvicorn's access log also writes to stdout and the launcher has to be able to
tell the two apart.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

SENTINEL = "@@VITRUVE@@ "

#: Progress events are throttled to this interval. A 254 MB file in 1 MB
#: chunks is 254 events per artifact if unthrottled, which is fine, but the
#: chunk size is not ours to depend on.
PROGRESS_INTERVAL_S = 0.15


def emit(event: str, **fields: object) -> None:
    """One event to the launcher. Never raises: a broken pipe means the
    launcher is already gone, and the watchdog is the thing that handles that."""
    try:
        sys.stdout.write(SENTINEL + json.dumps({"event": event, **fields}) + "\n")
        sys.stdout.flush()
    except (BrokenPipeError, ValueError):
        pass


def watchdog(interval: float = 1.0) -> None:
    """Exit when the launcher does.

    Two independent signals, because neither one alone covers every way a
    parent can vanish. stdin EOF fires immediately when the launcher's write
    end closes. The getppid() check catches the case where the pipe was
    inherited by something else. os._exit rather than sys.exit: this runs on a
    daemon thread while uvicorn owns the main thread, and an orphaned server is
    exactly what we are preventing.
    """

    def on_eof() -> None:
        try:
            sys.stdin.read()
        except Exception:
            pass
        os._exit(0)

    threading.Thread(target=on_eof, daemon=True, name="vitruve-stdin-watchdog").start()

    def on_reparent() -> None:
        while True:
            time.sleep(interval)
            if os.getppid() == 1:
                os._exit(0)

    threading.Thread(target=on_reparent, daemon=True, name="vitruve-ppid-watchdog").start()


def configure_paths(resources: Path) -> None:
    """Environment the bundled layout needs, without overriding a user's own.

    Each of these is set only if it is absent, so someone debugging the app
    from a terminal can still point it at a different cache or lock file.
    """
    lock = resources / "assets" / "weights.lock.json"
    if lock.exists():
        os.environ.setdefault("VITRUVE_WEIGHTS_LOCK", str(lock))

    # The bundle is read-only once it is in /Applications and, more to the
    # point, writing .pyc files into it after signing would break the sealed
    # resource directory. Everything is compiled at build time instead.
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    # MPS is the right device on this hardware, but a fallback keeps an
    # unimplemented op from being a crash in a GUI app with no console.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def ensure_weights(timeout: float = 60.0) -> bool:
    """Fetch anything missing at the permissive tier. Returns False on failure.

    Failure is not fatal. The catalogue, the licence pages and the whole UI
    work without a single weight; only `POST /analyze` does not. An app that
    refuses to open because a mirror is down is worse than one that opens and
    says what is missing.
    """
    from vitruve.cli.weights import split_by_tier
    from vitruve.models import weights as W
    from vitruve.models.licensing import Tier

    permitted, _refused = split_by_tier(Tier.PERMISSIVE)
    missing = [p for p in permitted if not W.verify_only(p.key).ok]
    if not missing:
        emit("weights_ok", cached=len(permitted), fetched=0)
        return True

    total = sum(p.size_bytes for p in missing)
    emit(
        "weights_start",
        count=len(missing),
        total_bytes=total,
        cache_dir=str(W.cache_dir()),
        names=[p.provenance for p in missing],
    )

    done_before = 0
    last = 0.0

    def progress(key: str, done: int, size: int) -> None:
        nonlocal last
        now = time.monotonic()
        if now - last < PROGRESS_INTERVAL_S and done < size:
            return
        last = now
        emit("weights_progress", key=key, done=done_before + done, total=total)

    failed: list[str] = []
    for p in missing:
        emit("weights_item", key=p.key, name=p.provenance, size_bytes=p.size_bytes)
        try:
            W.download(p.key, force=True, progress=progress)
        except Exception as exc:  # any network or hash failure, reported not raised
            failed.append(f"{p.provenance}: {exc}")
        done_before += p.size_bytes

    if failed:
        emit("weights_failed", messages=failed)
        return False
    emit("weights_ok", cached=len(permitted) - len(missing), fetched=len(missing))
    return True


def run_server(port: int) -> int:
    """Hand off to the real `vitruve serve`.

    Deliberately the library function and not a reimplementation with uvicorn:
    `serve()` is where the loopback refusal lives, and a second code path that
    binds a socket is a second code path that can bind the wrong one.
    """
    from vitruve.api.serve import serve

    emit("serving", port=port, url=f"http://127.0.0.1:{port}/")
    return serve(host="127.0.0.1", port=port, allow_remote=False, store=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vitruve-app")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument(
        "--resources",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Contents/Resources of the bundle (default: this file's directory)",
    )
    ap.add_argument("--no-watchdog", action="store_true", help="for running this by hand")
    args = ap.parse_args(argv)

    if not args.no_watchdog:
        watchdog()
    configure_paths(args.resources)

    # SIGTERM is what the launcher sends on Quit. uvicorn installs its own
    # handler once it is running; this one covers the window before that,
    # which is the whole weights download and can be minutes long.
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

    emit("starting", pid=os.getpid(), python=sys.version.split()[0])

    try:
        ensure_weights()
    except Exception as exc:
        emit("weights_failed", messages=[str(exc)])

    try:
        return run_server(args.port)
    except Exception as exc:
        emit("fatal", message=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
