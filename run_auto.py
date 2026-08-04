from __future__ import annotations

import argparse
import json
import logging
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

from prosody_gui import pipeline  # noqa: E402 - import after environment handoff
from prosody_gui.config import load_config  # noqa: E402


def build_edit_log(ep_dir: Path, groups: list[dict]) -> list[dict]:
    words_by_id = {word.index: word for word in pipeline.parse_words(ep_dir)}
    entries = []
    for group in groups:
        matched = [words_by_id[i] for i in group["word_ids"] if i in words_by_id]
        entries.append(
            {
                "text": pipeline.join_units(word.text for word in matched),
                "word_ids": group["word_ids"],
                "speed": group["speed"],
                "gain_db": group["gain_db"],
            }
        )
    return entries


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Transcribe a WAV and apply AI-suggested prosody edits, fully automated."
    )
    parser.add_argument("audio", type=Path, help="Path to a source .wav file.")
    parser.add_argument("--name", help="Episode name (defaults to the input filename).")
    args = parser.parse_args()

    config = load_config()
    episode = pipeline.import_wav(args.audio, args.name, print)
    name = episode["name"]
    pipeline.transcribe_episode(name, config, print)
    result = pipeline.ai_auto_edit(name, config, print)

    ep_dir = pipeline.episode_dir(name)
    log_path = ep_dir / "edit_log.json"
    log_path.write_text(
        json.dumps(build_edit_log(ep_dir, result["ai_effect_groups"]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Edited audio: {pipeline.edited_wav(ep_dir)}")
    print(f"Edit log: {log_path}")


if __name__ == "__main__":
    main()
