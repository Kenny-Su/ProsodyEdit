from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass
class AppConfig:
    input_dir: str = str(Path.home() / "Downloads" / "ProsodyEdit-audio")
    output_dir: str = str(Path.home() / "Downloads" / "ProsodyEdit-output")
    ffmpeg: str = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe: str = shutil.which("ffprobe") or "ffprobe"
    server_python: str = sys.executable
    whisperx_python: str = str(ROOT / "whisperx" / ".venv" / "bin" / "python")
    cosyvoice_python: str = str(ROOT / "CosyVoice" / ".venv" / "bin" / "python")
    whisperx_model: str = "small"
    whisperx_device: str = "cpu"
    whisperx_compute_type: str = "int8"
    whisperx_batch_size: int = 8
    cosyvoice_model_dir: str = str(ROOT / "CosyVoice" / "pretrained_models" / "Fun-CosyVoice3-0.5B")
    speed: float = 0.95
    crossfade: float = 0.150
    silence_threshold_db: float = -50.0
    min_silence: float = 0.05
    min_sound: float = 0.10
    gain_clamp_db: float = 3.0
    enable_gain_ramp: bool = False


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config() -> AppConfig:
    config = AppConfig()
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
    config.input_dir = str(Path(config.input_dir).expanduser().resolve())
    config.output_dir = str(Path(config.output_dir).expanduser().resolve())
    defaults = AppConfig()
    for key in ("whisperx_python", "cosyvoice_python", "cosyvoice_model_dir"):
        configured = Path(getattr(config, key)).expanduser()
        fallback = Path(getattr(defaults, key)).expanduser()
        if not configured.exists() and fallback.exists():
            setattr(config, key, str(fallback))
    save_config(config)
    return config


def save_config(config: AppConfig) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
