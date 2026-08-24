"""Local HTTP transport.

Importing this package does not import FastAPI. `create_app` does, so a
machine without ``vitruve[api]`` installed can still run every other command.
"""

from __future__ import annotations

__all__ = ["create_app", "serve"]


def create_app(**kwargs):
    from .app import create_app as _create_app

    return _create_app(**kwargs)


def serve(**kwargs) -> int:
    from .serve import serve as _serve

    return _serve(**kwargs)
