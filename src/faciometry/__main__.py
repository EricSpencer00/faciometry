"""``python -m faciometry`` for anyone who has not put the script on their PATH."""

from __future__ import annotations

import sys

from .cli.main import main

if __name__ == "__main__":
    sys.exit(main())
