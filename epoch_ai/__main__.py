"""Enable ``python -m epoch_ai`` to invoke the CLI."""

from __future__ import annotations

from epoch_ai.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
