"""PyInstaller entrypoint for the packaged Windows app."""

from __future__ import annotations

import sys


def main() -> int:
    from backend.windows_patch import apply_pending_patch_and_maybe_exit

    exit_code = apply_pending_patch_and_maybe_exit()
    if exit_code is not None:
        return exit_code

    from backend.launcher import main as run_app

    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
