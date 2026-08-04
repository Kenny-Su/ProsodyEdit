from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from prosodyedit import editing
from prosodyedit.config import Config
from prosodyedit.transcript import Word

WORDS = [
    Word(index=1, sentence_index=1, start=1.0, end=1.3, text="Make"),
    Word(index=2, sentence_index=1, start=1.35, end=1.65, text="this"),
    Word(index=3, sentence_index=1, start=1.7, end=2.2, text="count."),
    Word(index=4, sentence_index=2, start=3.1, end=3.7, text="Again."),
]


class ApplyEffectGroupsTests(unittest.TestCase):
    def _run(self, groups):
        commands: list[list[str]] = []
        with patch("subprocess.run", side_effect=lambda args, **_kwargs: commands.append(args)):
            editing.apply_effect_groups(Path("in.wav"), Path("out.wav"), WORDS, groups, Config())
        return commands[0]

    def _graph(self, command: list[str]) -> str:
        return command[command.index("-filter_complex") + 1]

    def test_slows_selected_words_and_preserves_unselected_timeline(self) -> None:
        graph = self._graph(self._run([{"word_ids": [3, 1], "speed": 0.95}]))
        self.assertIn("atrim=start=0.000000:end=1.000000", graph)
        self.assertIn("atrim=start=1.000000:end=1.300000,atempo=0.950000", graph)
        self.assertIn("atrim=start=1.300000:end=1.700000", graph)
        self.assertIn("atrim=start=1.700000:end=2.200000,atempo=0.950000", graph)
        self.assertIn("atrim=start=2.200000", graph)

    def test_consecutive_words_are_merged_into_one_chunk(self) -> None:
        graph = self._graph(self._run([{"word_ids": [3, 1, 2], "speed": 0.9}]))
        self.assertIn("atrim=start=1.000000:end=2.200000,atempo=0.900000", graph)
        self.assertNotIn("atrim=start=1.000000:end=1.300000,atempo", graph)

    def test_applies_volume_boost(self) -> None:
        graph = self._graph(self._run([{"word_ids": [1, 2], "speed": 0.9, "gain_db": 2.5}]))
        self.assertIn("atrim=start=1.000000:end=1.650000,atempo=0.900000,volume=2.500dB", graph)

    def test_applies_distinct_settings_to_multiple_groups(self) -> None:
        graph = self._graph(
            self._run(
                [
                    {"word_ids": [1, 2], "speed": 0.9, "gain_db": 2},
                    {"word_ids": [4], "speed": 0.8, "gain_db": 4},
                ]
            )
        )
        self.assertIn("atempo=0.900000,volume=2.000dB", graph)
        self.assertIn("atempo=0.800000,volume=4.000dB", graph)

    def test_rejects_empty_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            editing.apply_effect_groups(Path("in.wav"), Path("out.wav"), WORDS, [], Config())

    def test_rejects_out_of_range_speed(self) -> None:
        with self.assertRaisesRegex(ValueError, "less than 1.00"):
            editing.apply_effect_groups(
                Path("in.wav"), Path("out.wav"), WORDS, [{"word_ids": [1], "speed": 1.0}], Config()
            )

    def test_rejects_out_of_range_gain(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 6"):
            editing.apply_effect_groups(
                Path("in.wav"),
                Path("out.wav"),
                WORDS,
                [{"word_ids": [1], "speed": 0.9, "gain_db": 7}],
                Config(),
            )

    def test_rejects_words_assigned_to_multiple_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple effect groups"):
            editing.apply_effect_groups(
                Path("in.wav"),
                Path("out.wav"),
                WORDS,
                [{"word_ids": [1, 2]}, {"word_ids": [2, 3]}],
                Config(),
            )

    def test_rejects_unknown_word_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown word indexes"):
            editing.apply_effect_groups(
                Path("in.wav"), Path("out.wav"), WORDS, [{"word_ids": [99]}], Config()
            )


if __name__ == "__main__":
    unittest.main()
