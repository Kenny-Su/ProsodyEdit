from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Word:
    index: int
    start: float
    end: float
    text: str


def load_words(timestamps_path: Path) -> list[Word]:
    """Load a word list from a forced-aligner timestamp JSON file.

    Expects a flat list of {"text", "start_time", "end_time"} objects in
    timeline order. Items with empty text, non-finite timestamps, negative
    starts, or end <= start are excluded.
    """
    raw = json.loads(Path(timestamps_path).read_text(encoding="utf-8"))
    words: list[Word] = []
    for item in raw:
        text = str(item.get("text", "") or "").strip()
        try:
            start = float(item.get("start_time"))
            end = float(item.get("end_time"))
        except (TypeError, ValueError):
            continue
        if not text or not math.isfinite(start) or not math.isfinite(end):
            continue
        if start < 0 or end <= start:
            continue
        words.append(Word(index=len(words) + 1, start=start, end=end, text=text))
    return words
