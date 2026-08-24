"""Binding a socket, carefully.

The whole of this module is one decision: which address the server is allowed
to listen on. Vitruve handles photographs of faces and runs with no
authentication, because it is a local instrument. Those two facts are only
compatible while it is unreachable from the network, so a non-loopback bind
has to be an explicit act by the operator with the consequence printed in
front of them.

``--allow-remote`` is not a hidden flag and it is not refused outright. An
operator who genuinely wants Vitruve on a lab subnet behind their own auth
proxy has a real use for it. What is refused is getting there by accident.
"""

from __future__ import annotations

import ipaddress
import sys
from pathlib import Path

from ..models.licensing import Tier

LOOPBACK_NAMES = frozenset({"localhost", "localhost.", "127.0.0.1", "::1"})

WARNING = """\
==============================================================================
  vitruve is binding {host}, which is reachable from the network.

  There is no authentication. Anyone who can route to this address can post a
  photograph to /analyze and read the report. Uploads are held in memory and
  {storing}.

  Put it behind something that authenticates, or bind 127.0.0.1 instead.
==============================================================================
"""


class RemoteBindRefused(ValueError):
    """A non-loopback bind was requested without ``--allow-remote``."""


def is_loopback(host: str) -> bool:
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_bind(host: str, *, allow_remote: bool) -> str:
    """Return the host to bind, or refuse.

    Split out from :func:`serve` so the refusal is testable without opening a
    socket, which is the only way to test it in an offline test suite.
    """
    if is_loopback(host):
        return host
    if not allow_remote:
        raise RemoteBindRefused(
            f"refusing to bind {host}: it is reachable from the network and Vitruve "
            "has no authentication. Bind 127.0.0.1, or pass --allow-remote if you "
            "have your own access control in front of it."
        )
    return host


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8731,
    allow_remote: bool = False,
    store: bool = False,
    store_dir: Path | None = None,
    license_tier: Tier = Tier.PERMISSIVE,
) -> int:
    bind = resolve_bind(host, allow_remote=allow_remote)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError("uvicorn is not installed") from exc

    from .app import create_app

    if not is_loopback(bind):
        print(
            WARNING.format(
                host=bind,
                storing=(
                    "written to disk because --store was passed"
                    if store
                    else "never written to disk"
                ),
            ),
            file=sys.stderr,
        )
    if store:
        print(
            f"vitruve serve: --store is on, so uploaded images are written to "
            f"{Path(store_dir) if store_dir else Path('vitruve-store')} with their "
            "metadata stripped.",
            file=sys.stderr,
        )

    app = create_app(store=store, store_dir=store_dir, license_tier=license_tier)
    print(f"vitruve serve: http://{bind}:{port}", file=sys.stderr)
    uvicorn.run(app, host=bind, port=port, log_level="info")
    return 0
