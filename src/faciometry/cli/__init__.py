"""Command line entry points.

Importing this package costs a parser and nothing else. Each subcommand
imports its own dependencies inside its handler, so `faciometry catalogue` never
pays for FastAPI and `faciometry licenses` never pays for numpy.
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    from .main import main as _main

    return _main(argv)
