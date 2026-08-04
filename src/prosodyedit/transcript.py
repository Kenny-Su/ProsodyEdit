from __future__ import annotations

import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import Config

logger = logging.getLogger(__name__)

# Must be set before the lazy torch import performed during transcription.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SENTENCE_END_RE = re.compile(r"[.!?。！？][\"'”’）】》]*$")
NO_LEADING_SPACE_RE = re.compile(r"^[,.;:!?%。，、；：！？）】》”’]")
NO_TRAILING_SPACE_RE = re.compile(r"[（【《“‘/]$")
CJK_RE = re.compile(r"[㐀-鿿぀-ヿ가-힯]")
NON_TERMINAL_ABBREVIATIONS = {"dr", "jr", "mr", "mrs", "ms", "prof", "sr", "st"}
LIKELY_SENTENCE_STARTERS = {
    "a", "after", "although", "an", "before", "but", "he", "however", "it",
    "meanwhile", "mr", "mrs", "she", "that", "the", "they", "this", "when",
}


@dataclass
class Word:
    index: int
    sentence_index: int
    start: float
    end: float
    text: str


@dataclass
class Sentence:
    index: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def join_units(units: Iterable[str]) -> str:
    text = ""
    for raw in units:
        unit = raw.strip()
        if not unit:
            continue
        if not text:
            text = unit
        elif NO_LEADING_SPACE_RE.search(unit) or NO_TRAILING_SPACE_RE.search(text):
            text += unit
        elif CJK_RE.search(unit) or CJK_RE.search(text[-1:]):
            text += unit
        else:
            text += " " + unit
    return text


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


def _is_aligner_character(char: str) -> bool:
    return char == "'" or unicodedata.category(char)[0] in {"L", "N"}


def _aligner_token_count(text: str) -> int:
    """Count tokens the same way as Qwen's space-language aligner processor."""
    count = 0
    for segment in text.split():
        cleaned = "".join(char for char in segment if _is_aligner_character(char))
        in_non_cjk_token = False
        for char in cleaned:
            if CJK_RE.match(char):
                if in_non_cjk_token:
                    count += 1
                    in_non_cjk_token = False
                count += 1
            else:
                in_non_cjk_token = True
        if in_non_cjk_token:
            count += 1
    return count


def _source_sentence_groups(result: Any, unit_count: int) -> list[tuple[int, str]]:
    """Return cumulative aligned-token boundaries derived from ASR punctuation."""
    source = str(_field(result, "text", "") or "").strip()
    if not source:
        return []

    groups: list[tuple[int, str]] = []
    start = 0
    token_total = 0
    index = 0
    while index < len(source):
        if source[index] not in ".!?。！？":
            index += 1
            continue
        if source[index] == ".":
            before = re.search(r"([A-Za-z]+)$", source[:index])
            previous_word = before.group(1) if before else ""
            immediate_next = source[index + 1:index + 2]
            if (
                (immediate_next and immediate_next.isalnum())
                or previous_word.lower() in NON_TERMINAL_ABBREVIATIONS
                or len(previous_word) == 1
            ):
                index += 1
                continue
        end = index + 1
        while end < len(source) and source[end] in "\"'”’）】》":
            end += 1
        sentence_text = source[start:end].strip()
        count = _aligner_token_count(sentence_text)
        if count:
            token_total += count
            groups.append((token_total, sentence_text))
        start = end
        index = end

    trailing = source[start:].strip()
    if trailing:
        count = _aligner_token_count(trailing)
        if count:
            token_total += count
            groups.append((token_total, trailing))

    if len(groups) < 2:
        return []
    if token_total == unit_count:
        return groups

    # The aligner can occasionally merge or discard a token. Preserve the ASR
    # sentence proportions instead of throwing away every boundary because of
    # a small count mismatch; the final sentence always absorbs the remainder.
    adjusted: list[tuple[int, str]] = []
    previous = 0
    for boundary, sentence_text in groups[:-1]:
        mapped = round(boundary * unit_count / token_total)
        mapped = max(previous + 1, min(mapped, unit_count - 1))
        adjusted.append((mapped, sentence_text))
        previous = mapped
    adjusted.append((unit_count, groups[-1][1]))
    return adjusted


def _punctuation_free_groups(
    units: list[tuple[str, float, float]],
) -> list[list[tuple[str, float, float]]]:
    """Create readable utterances when the ASR supplies no sentence punctuation."""
    if len(units) <= 45:
        return [units] if units else []
    groups: list[list[tuple[str, float, float]]] = []
    start = 0
    while len(units) - start > 45:
        low = start + 20
        high = min(start + 36, len(units) - 20)
        target = start + 30
        candidates: list[tuple[float, int]] = []
        for boundary in range(low, high + 1):
            previous = units[boundary - 1]
            following = units[boundary]
            pause = max(0.0, following[1] - previous[2])
            starter = re.sub(r"[^A-Za-z]", "", following[0]).lower()
            starter_bonus = 1.0 if starter in LIKELY_SENTENCE_STARTERS else 0.0
            distance_penalty = abs(boundary - target) * 0.015
            candidates.append((starter_bonus + pause - distance_penalty, boundary))
        boundary = max(candidates)[1]
        groups.append(units[start:boundary])
        start = boundary
    groups.append(units[start:])
    return groups


def parse_sentences(result: Any) -> list[Sentence]:
    sentences: list[Sentence] = []
    word_index = 0
    units = _timed_units(result)
    grouped: list[tuple[list[tuple[str, float, float]], str | None]] = []
    source_groups = _source_sentence_groups(result, len(units))
    if source_groups:
        group_start = 0
        for group_end, source_text in source_groups:
            grouped.append((units[group_start:group_end], source_text))
            group_start = group_end
    else:
        pending: list[tuple[str, float, float]] = []
        for unit in units:
            pending.append(unit)
            if SENTENCE_END_RE.search(unit[0]):
                grouped.append((pending, None))
                pending = []
        if pending:
            grouped.append((pending, None))
        if len(grouped) == 1 and len(units) > 45:
            grouped = [(group, None) for group in _punctuation_free_groups(units)]

    for idx, (group, source_text) in enumerate(grouped, 1):
        words: list[Word] = []
        for text, start, end in group:
            word_index += 1
            words.append(Word(index=word_index, sentence_index=idx, start=start, end=end, text=text))
        sentences.append(
            Sentence(
                index=idx,
                start=group[0][1],
                end=group[-1][2],
                text=source_text or join_units(unit[0] for unit in group),
                words=words,
            )
        )
    return sentences


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
