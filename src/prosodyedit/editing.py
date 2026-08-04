from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .config import Config
from .transcript import Word

logger = logging.getLogger(__name__)


def _run_ffmpeg(config: Config, args: list[str]) -> None:
    command = [config.ffmpeg, "-hide_banner", "-y", *args]
    logger.info("$ %s", " ".join(command))
    subprocess.run(command, check=True)


def apply_effect_groups(
    wav: Path,
    output: Path,
    words: list[Word],
    groups: Iterable[dict[str, Any]],
    config: Config,
) -> None:
    group_list = list(groups)
    if not group_list:
        raise ValueError("Add at least one word effect group.")
    available = {word.index: word for word in words}
    assigned: set[int] = set()
    chunks: list[dict[str, Any]] = []

    for group_number, group in enumerate(group_list, 1):
        try:
            speed = float(group.get("speed", 0.95))
            gain_db = float(group.get("gain_db", 0.0))
            requested = sorted(set(int(value) for value in group.get("word_ids", [])))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid settings in effect group {group_number}.") from exc
        if not math.isfinite(speed) or not 0.5 <= speed < 1.0:
            raise ValueError("Speed must be at least 0.50 and less than 1.00.")
        if not math.isfinite(gain_db) or not 0.0 <= gain_db <= 6.0:
            raise ValueError("Volume boost must be between 0 and 6 dB.")
        if not requested:
            raise ValueError(f"Effect group {group_number} has no selected words.")
        duplicates = assigned.intersection(requested)
        if duplicates:
            raise ValueError(f"Words cannot belong to multiple effect groups: {sorted(duplicates)}.")
        missing = [word_id for word_id in requested if word_id not in available]
        if missing:
            raise ValueError(f"Unknown word indexes: {', '.join(map(str, missing))}.")
        assigned.update(requested)
        selected_words = [available[word_id] for word_id in requested]
        runs: list[list[Word]] = []
        for word in selected_words:
            if runs and word.index == runs[-1][-1].index + 1:
                runs[-1].append(word)
            else:
                runs.append([word])
        for run in runs:
            chunks.append(
                {
                    "start": run[0].start,
                    "end": run[-1].end,
                    "speed": speed,
                    "gain_db": gain_db,
                }
            )

    chunks.sort(key=lambda chunk: (chunk["start"], chunk["end"]))
    for previous, current in zip(chunks, chunks[1:]):
        if current["start"] < previous["end"]:
            raise ValueError("Effect group intervals overlap.")

    filters: list[str] = []
    labels: list[str] = []
    cursor = 0.0

    def add_segment(expression: str) -> None:
        label = f"part{len(labels)}"
        filters.append(f"[0:a]{expression},asetpts=PTS-STARTPTS[{label}]")
        labels.append(f"[{label}]")

    for chunk in chunks:
        start, end = chunk["start"], chunk["end"]
        if start > cursor + 0.000001:
            add_segment(f"atrim=start={cursor:.6f}:end={start:.6f}")
        effects = [f"atrim=start={start:.6f}:end={end:.6f}", f"atempo={chunk['speed']:.6f}"]
        if chunk["gain_db"]:
            effects.append(f"volume={chunk['gain_db']:.3f}dB")
        add_segment(",".join(effects))
        cursor = end
    add_segment(f"atrim=start={cursor:.6f}")
    filter_complex = ";".join(filters + [f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]"])

    logger.info("Applying %d effect group(s) to %d word(s) in %d chunk(s).", len(group_list), len(assigned), len(chunks))
    _run_ffmpeg(
        config,
        [
            "-i",
            str(wav),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
    )
    logger.info("Edited WAV ready: %s", output)
