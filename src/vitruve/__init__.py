"""Vitruve: facial morphometrics that reports its own uncertainty.

Importing this package pulls in nothing heavier than the standard library. The
measurement layer (``vitruve.core``, ``vitruve.measure``, ``vitruve.norms``)
needs numpy and nothing else; the model backends need torch or ONNX and live
behind optional extras; the transports need FastAPI. Keeping the root module
empty is what lets ``vitruve catalogue`` run in a tenth of a second on a
machine with no weights and no GPU.

The public entry points are the CLI (``vitruve.cli.main:main``) and the local
HTTP API (``vitruve.api.app:create_app``).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
