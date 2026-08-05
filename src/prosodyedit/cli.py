from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .ai import suggest_effects
from .config import DEFAULT_CONFIG_PATH, load_config
from .editing import apply_effect_groups
from .transcript import Word, join_units, parse_words, transcribe

logger = logging.getLogger(__name__)


def _build_edit_log(words: list[Word], groups: list[dict]) -> list[dict]:
    words_by_id = {word.index: word for word in words}
    entries = []
    for group in groups:
        matched = [words_by_id[i] for i in group["word_ids"] if i in words_by_id]
        entries.append(
            {
                "text": join_units(word.text for word in matched),
                "word_ids": group["word_ids"],
                "speed": group["speed"],
                "gain_db": group["gain_db"],
            }
        )
    return entries


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Transcribe a WAV and apply AI-suggested prosody edits."
    )
    parser.add_argument("audio", type=Path, help="Path to a source .wav file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for the rendered files (default: alongside the input file).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config.json (default: {DEFAULT_CONFIG_PATH}).",
    )
    args = parser.parse_args(argv)

    if args.audio.suffix.lower() != ".wav":
        parser.error("Input must be a .wav file.")
    if not args.audio.exists():
        parser.error(f"No such file: {args.audio}")

    config = load_config(args.config)
    output_dir = args.output_dir or args.audio.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.audio.stem

    logger.info("Transcribing %s", args.audio)
    result = transcribe(args.audio, config)
    words = parse_words(result)

    groups = suggest_effects(words, config)
    if not groups:
        raise SystemExit("The AI did not suggest any effect groups for this recording.")

    edited_wav = output_dir / f"{stem}.edited.wav"
    apply_effect_groups(args.audio, edited_wav, words, groups, config)

    log_path = output_dir / f"{stem}.edit_log.json"
    log_path.write_text(
        json.dumps(_build_edit_log(words, groups), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Edited audio: %s", edited_wav)
    logger.info("Edit log: %s", log_path)


if __name__ == "__main__":
    main()
