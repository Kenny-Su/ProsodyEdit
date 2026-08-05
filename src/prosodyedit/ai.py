from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.request
from typing import Any

from .config import Config
from .transcript import Word

logger = logging.getLogger(__name__)

WORDS_PER_LINE = 40

SYSTEM_PROMPT = """You are directing expressive prosody edits for a spoken-word \
recording. You are given a transcript as "<word_id>:<word text>" tokens in \
timeline order, wrapped to one line per several dozen words purely for \
readability; the line breaks carry no meaning. Sentence-ending punctuation is \
attached to the word it follows, so use it to read sentence and phrase \
boundaries yourself. Decide which words deserve an effect group so the \
delivery sounds more expressive: emphasis on important words or phrases by \
slowing them down, optionally with a volume boost.

Use effects sparingly. Most words should be left alone; only mark words where the \
effect clearly improves delivery.

Each effect group applies to one or more word ids and always has:
- "word_ids": a list of integers from the transcript. A word id may appear in at \
most one group. Consecutive ids in a group are treated as one continuous phrase; \
non-consecutive ids in the same group are treated as separate occurrences that \
share the same settings.
- "speed": required float, 0.50 to 0.99 (exclusive of 1.0). This is how much to \
slow the word(s) down for emphasis. Use a lower value (0.6-0.85) for strong \
emphasis, and a value close to 0.99 when you only want the boost below without \
noticeably slowing the word down.
- "gain_db": optional float, 0.0 to 6.0, a volume boost for emphasis. 0 means no \
boost. Omit or set to 0 when you don't want a boost.

Respond with strict JSON only, no prose, matching exactly:
{"groups": [{"word_ids": [int, ...], "speed": float, "gain_db": float}, ...]}

Return {"groups": []} if no edits are warranted."""


def _transcript_prompt(words: list[Word]) -> str:
    tokens = [f"{word.index}:{word.text}" for word in words]
    lines = [" ".join(tokens[i : i + WORDS_PER_LINE]) for i in range(0, len(tokens), WORDS_PER_LINE)]
    return "\n".join(lines)


def _call_chat(config: Config, messages: list[dict[str, str]]) -> str:
    if not config.openai_api_key:
        raise ValueError(
            "Set openai_api_key (your Qwen Token Plan key, with openai_base_url "
            "if not using the default) in the config file before requesting an "
            "AI edit, or set the PROSODYEDIT_OPENAI_API_KEY environment variable."
        )
    url = config.openai_base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": config.openai_model,
            "messages": messages,
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.openai_api_key}",
            "User-Agent": "Mozilla/5.0",
        },
    )
    logger.info("Requesting AI effect suggestions from %s.", config.openai_model)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AI request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"AI request failed: {exc.reason}") from exc
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected AI response shape: {payload}") from exc


def _parse_groups(content: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AI response was not valid JSON: {content}") from exc
    groups = parsed.get("groups") if isinstance(parsed, dict) else None
    if not isinstance(groups, list):
        raise ValueError(f"AI response was missing a 'groups' list: {content}")
    if not all(isinstance(group, dict) for group in groups):
        raise ValueError("Each AI effect group must be an object.")
    return groups


def _normalize_group(group: dict[str, Any], position: int) -> dict[str, Any]:
    try:
        word_ids = sorted({int(value) for value in group.get("word_ids", [])})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"group {position} has invalid word_ids") from exc
    if not word_ids:
        raise ValueError(f"group {position} has no word_ids")
    try:
        speed = float(group.get("speed", 0.9))
        gain_db = float(group.get("gain_db", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"group {position} has invalid numeric settings") from exc
    if not math.isfinite(speed):
        raise ValueError(f"group {position} has a non-finite speed")
    return {
        "word_ids": word_ids,
        "speed": round(min(0.99, max(0.5, speed)), 3),
        "gain_db": round(min(6.0, max(0.0, gain_db)), 2),
    }


def _dedupe_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assigned: set[int] = set()
    deduped: list[dict[str, Any]] = []
    for group in groups:
        word_ids = [word_id for word_id in group["word_ids"] if word_id not in assigned]
        dropped = len(group["word_ids"]) - len(word_ids)
        if dropped:
            logger.warning("Dropped %d word id(s) already claimed by an earlier AI group.", dropped)
        if not word_ids:
            continue
        assigned.update(word_ids)
        deduped.append({**group, "word_ids": word_ids})
    return deduped


def suggest_effects(words: list[Word], config: Config) -> list[dict[str, Any]]:
    transcript = _transcript_prompt(words)
    if not transcript:
        raise ValueError("No transcript available; nothing to suggest edits for.")
    valid_ids = {word.index for word in words}
    content = _call_chat(
        config,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
    )
    raw_groups = _parse_groups(content)

    normalized: list[dict[str, Any]] = []
    for position, group in enumerate(raw_groups, 1):
        try:
            normalized.append(_normalize_group(group, position))
        except ValueError as exc:
            logger.warning("Skipping AI effect: %s", exc)

    for group in normalized:
        before = len(group["word_ids"])
        group["word_ids"] = [word_id for word_id in group["word_ids"] if word_id in valid_ids]
        dropped = before - len(group["word_ids"])
        if dropped:
            logger.warning("Dropped %d word id(s) not present in this transcript.", dropped)
    normalized = [group for group in normalized if group["word_ids"]]

    normalized = _dedupe_groups(normalized)
    logger.info("AI suggested %d usable effect group(s) from %d raw group(s).", len(normalized), len(raw_groups))
    return normalized
