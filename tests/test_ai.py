from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from prosodyedit import ai
from prosodyedit.config import Config
from prosodyedit.transcript import Word

WORDS = [
    Word(index=1, start=1.0, end=1.3, text="Make"),
    Word(index=2, start=1.35, end=1.65, text="this"),
    Word(index=3, start=1.7, end=2.2, text="count."),
    Word(index=4, start=3.1, end=3.7, text="Again."),
]


class TranscriptPromptTests(unittest.TestCase):
    def test_includes_word_ids_in_timeline_order(self) -> None:
        prompt = ai._transcript_prompt(WORDS)
        self.assertEqual(prompt, "1:Make 2:this 3:count. 4:Again.")

    def test_wraps_long_transcripts_across_lines(self) -> None:
        words = [Word(index=i, start=float(i), end=float(i) + 0.5, text=f"word{i}") for i in range(1, 85)]
        prompt = ai._transcript_prompt(words)
        lines = prompt.split("\n")
        self.assertEqual(len(lines), 3)
        self.assertEqual(len(lines[0].split()), ai.WORDS_PER_LINE)
        self.assertEqual(len(lines[-1].split()), 4)


class NormalizeGroupTests(unittest.TestCase):
    def test_fills_defaults_and_sorts_word_ids(self) -> None:
        group = ai._normalize_group({"word_ids": [2, 1]}, 1)
        self.assertEqual(group, {"word_ids": [1, 2], "speed": 0.9, "gain_db": 0.0})

    def test_clamps_out_of_range_values(self) -> None:
        clamped = ai._normalize_group({"word_ids": [1], "speed": 1.5, "gain_db": 50}, 2)
        self.assertEqual(clamped["speed"], 0.99)
        self.assertEqual(clamped["gain_db"], 6.0)

    def test_rejects_empty_or_invalid_word_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "no word_ids"):
            ai._normalize_group({"word_ids": []}, 1)
        with self.assertRaisesRegex(ValueError, "invalid word_ids"):
            ai._normalize_group({"word_ids": ["not-an-int"]}, 1)


class DedupeGroupsTests(unittest.TestCase):
    def test_drops_words_already_claimed(self) -> None:
        groups = [
            {"word_ids": [1, 2], "speed": 0.9, "gain_db": 0.0},
            {"word_ids": [2, 3], "speed": 0.8, "gain_db": 0.0},
        ]
        deduped = ai._dedupe_groups(groups)
        self.assertEqual([group["word_ids"] for group in deduped], [[1, 2], [3]])


class SuggestEffectsTests(unittest.TestCase):
    def test_drops_hallucinated_word_ids(self) -> None:
        content = json.dumps({"groups": [{"word_ids": [1, 99], "speed": 0.8, "gain_db": 2.0}]})
        with patch.object(ai, "_call_chat", return_value=content) as mock_call:
            groups = ai.suggest_effects(WORDS, Config(openai_api_key="key"))

        mock_call.assert_called_once()
        self.assertEqual(groups, [{"word_ids": [1], "speed": 0.8, "gain_db": 2.0}])

    def test_requires_a_transcript(self) -> None:
        with self.assertRaisesRegex(ValueError, "No transcript available"):
            ai.suggest_effects([], Config())

    def test_call_chat_requires_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "openai_api_key"):
            ai._call_chat(Config(openai_api_key=""), [])


if __name__ == "__main__":
    unittest.main()
