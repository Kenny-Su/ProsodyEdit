from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / "bin" / "python"


def ensure_project_environment() -> None:
    if Path(sys.prefix) == VENV:
        return
    if not VENV_PYTHON.is_file():
        raise RuntimeError(
            "ProsodyEdit's .venv is missing. Create it with: "
            "python3.12 -m venv .venv && .venv/bin/python -m pip install qwen-asr"
        )
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


ensure_project_environment()

from prosody_gui.server import main  # noqa: E402 - import after environment handoff


if __name__ == "__main__":
    main()
