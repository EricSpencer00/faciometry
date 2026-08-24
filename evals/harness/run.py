"""Shim so the repository-root ``make evals`` reaches the harness.

The root Makefile looks for ``evals/harness/run.py``; the harness itself lives
at ``evals/run_all.py``. Rather than edit a file this package does not own,
this forwards. Arm numbers may be passed through:

    python evals/harness/run.py 2 3
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.run_all import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
