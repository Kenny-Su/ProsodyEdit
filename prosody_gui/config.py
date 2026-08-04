from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AppConfig:
    ffmpeg: str = shutil.which("ffmpeg") or "ffmpeg"
    qwen_asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    qwen_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    qwen_device: str = "mps"
    qwen_language: str = "English"
    qwen_max_new_tokens: int = 2048
    openai_api_key: str = ""
    openai_base_url: str = "https://portal.qwen.ai/v1"
    openai_model: str = "qwen3-max"


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config() -> AppConfig:
    config = AppConfig()
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
    save_config(config)
    return config


def save_config(config: AppConfig) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
