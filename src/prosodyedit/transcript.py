from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)

# Must be set before the lazy torch import performed during transcription.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass
class Word:
    index: int
    start: float
    end: float
    text: str


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _timestamp_items(result: Any) -> list[Any]:
    timestamps = _field(result, "time_stamps")
    if timestamps is None:
        return []
    items = getattr(timestamps, "items", None)
    if items is not None and not callable(items):
        return list(items)
    if isinstance(timestamps, dict):
        raw = timestamps.get("items", [])
        return list(raw) if isinstance(raw, (list, tuple)) else []
    if isinstance(timestamps, (list, tuple)):
        if len(timestamps) == 1 and hasattr(timestamps[0], "items"):
            return list(timestamps[0].items)
        return list(timestamps)
    try:
        return list(timestamps)
    except TypeError:
        return []


def _timed_units(result: Any) -> list[tuple[str, float, float]]:
    units: list[tuple[str, float, float]] = []
    for item in _timestamp_items(result):
        text = str(_field(item, "text", "") or "").strip()
        try:
            start = float(_field(item, "start_time"))
            end = float(_field(item, "end_time"))
        except (TypeError, ValueError):
            continue
        if not text or not math.isfinite(start) or not math.isfinite(end):
            continue
        if start < 0 or end <= start:
            continue
        units.append((text, start, end))
    return units


def parse_words(result: Any) -> list[Word]:
    return [
        Word(index=index, start=start, end=end, text=text)
        for index, (text, start, end) in enumerate(_timed_units(result), 1)
    ]


def _load_qwen_model(config: Config, device: str) -> Any:
    import torch
    from qwen_asr import Qwen3ASRModel

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this PyTorch runtime")
    dtype = torch.float16 if device == "mps" else torch.float32
    logger.info("Loading Qwen3-ASR and forced aligner on %s.", device)
    return Qwen3ASRModel.from_pretrained(
        config.qwen_asr_model,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=config.qwen_max_new_tokens,
        forced_aligner=config.qwen_aligner_model,
        forced_aligner_kwargs={"dtype": dtype, "device_map": device},
    )


def _transcribe_native(wav: Path, config: Config, device: str) -> Any:
    model = _load_qwen_model(config, device)
    return model.transcribe(
        audio=str(wav),
        language=config.qwen_language or None,
        return_time_stamps=True,
    )[0]


def transcribe(wav: Path, config: Config) -> Any:
    """Run Qwen3-ASR and forced alignment, falling back to CPU if MPS fails."""
    device = config.qwen_device
    try:
        return _transcribe_native(wav, config, device)
    except (RuntimeError, NotImplementedError) as exc:
        if device != "mps":
            raise
        logger.warning("MPS inference failed (%s); retrying the transcription on CPU.", exc)
        return _transcribe_native(wav, config, "cpu")
