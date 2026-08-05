from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"


@dataclass
class Config:
    ffmpeg: str = shutil.which("ffmpeg") or "ffmpeg"
    openai_api_key: str = ""
    openai_base_url: str = "https://portal.qwen.ai/v1"
    openai_model: str = "qwen3-max"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    config = Config()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
    if not config.openai_api_key:
        config.openai_api_key = os.environ.get("PROSODYEDIT_OPENAI_API_KEY", "")
    return config
