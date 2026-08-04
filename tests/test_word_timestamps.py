from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prosody_gui import pipeline
from prosody_gui.config import AppConfig


class WordTimestampTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        pipeline.configure_output_dir(self.output_dir)
        self.episode_dir = self.output_dir / "episode"
        self.episode_dir.mkdir()
        pipeline.sentences_dir(self.episode_dir).mkdir()
        pipeline.words_dir(self.episode_dir).mkdir()
        pipeline.original_wav(self.episode_dir).touch()
        self.timestamp_items = [
            SimpleNamespace(text="Make", start_time=1.0, end_time=1.3),
            SimpleNamespace(text="this", start_time=1.35, end_time=1.65),
            SimpleNamespace(text="count.", start_time=1.7, end_time=2.2),
            SimpleNamespace(text="Again.", start_time=3.1, end_time=3.7),
        ]
        self.native_result = SimpleNamespace(
            language="English",
            text="Make this count. Again.",
            time_stamps=SimpleNamespace(items=self.timestamp_items),
        )
        pipeline.set_transcript("episode", self.native_result)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_parses_native_word_timestamps(self) -> None:
        sentences = pipeline.parse_sentences(self.episode_dir)
        words = pipeline.parse_words(self.episode_dir)

        self.assertEqual([word.text for word in words], ["Make", "this", "count.", "Again."])
        self.assertEqual([word.index for word in words], [1, 2, 3, 4])
        self.assertEqual([word.sentence_index for word in words], [1, 1, 1, 2])
        self.assertEqual((sentences[1].words[0].start, sentences[1].words[0].end), (3.1, 3.7))
        self.assertIsNone(sentences[1].words[0].score)

    def test_rejects_invalid_word_intervals(self) -> None:
        self.timestamp_items.extend(
            [
                SimpleNamespace(text="backwards", start_time=2.3, end_time=2.2),
                SimpleNamespace(text="", start_time=2.2, end_time=2.3),
            ]
        )

        self.assertEqual([word.text for word in pipeline.parse_words(self.episode_dir)], ["Make", "this", "count.", "Again."])

    def test_cut_audio_uses_exact_word_timestamps(self) -> None:
        commands: list[list[str]] = []
        with (
            patch.object(pipeline, "run_command", side_effect=lambda args, *_args, **_kwargs: commands.append(args)),
            patch.object(pipeline, "get_episode", return_value={"name": "episode"}),
        ):
            pipeline.cut_transcript_audio("episode", AppConfig(), lambda _message: None)

        word_commands = [command for command in commands if "/words/word_" in command[-1]]
        self.assertEqual(len(word_commands), 4)
        self.assertEqual(word_commands[0][word_commands[0].index("-ss") + 1], "1.000")
        self.assertEqual(word_commands[0][word_commands[0].index("-t") + 1], "0.300")
        self.assertEqual(word_commands[-1][word_commands[-1].index("-ss") + 1], "3.100")
        self.assertEqual(word_commands[-1][word_commands[-1].index("-t") + 1], "0.600")

    def test_slows_selected_words_and_preserves_unselected_timeline(self) -> None:
        commands: list[list[str]] = []
        with (
            patch.object(pipeline, "run_command", side_effect=lambda args, *_args, **_kwargs: commands.append(args)),
            patch.object(pipeline, "get_episode", return_value={"name": "episode", "edited": "episode/edited.wav"}),
        ):
            result = pipeline.slow_words("episode", [3, 1], 0.95, AppConfig(), lambda _message: None)

        self.assertEqual(result["edited"], "episode/edited.wav")
        command = commands[0]
        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("atrim=start=0.000000:end=1.000000", graph)
        self.assertIn("atrim=start=1.000000:end=1.300000,atempo=0.950000", graph)
        self.assertIn("atrim=start=1.300000:end=1.700000", graph)
        self.assertIn("atrim=start=1.700000:end=2.200000,atempo=0.950000", graph)
        self.assertIn("atrim=start=2.200000", graph)
        self.assertEqual(Path(command[-1]).resolve(), pipeline.edited_wav(self.episode_dir).resolve())

    def test_word_slowdown_validates_selection_and_speed(self) -> None:
        with self.assertRaisesRegex(ValueError, "no selected words"):
            pipeline.slow_words("episode", [], 0.95, AppConfig(), lambda _message: None)
        with self.assertRaisesRegex(ValueError, "less than 1.00"):
            pipeline.slow_words("episode", [1], 1.0, AppConfig(), lambda _message: None)

    def test_consecutive_words_are_slowed_as_one_chunk(self) -> None:
        commands: list[list[str]] = []
        logs: list[str] = []
        with (
            patch.object(pipeline, "run_command", side_effect=lambda args, *_args, **_kwargs: commands.append(args)),
            patch.object(pipeline, "get_episode", return_value={"name": "episode"}),
        ):
            pipeline.slow_words("episode", [3, 1, 2], 0.9, AppConfig(), logs.append)

        graph = commands[0][commands[0].index("-filter_complex") + 1]
        self.assertIn("atrim=start=1.000000:end=2.200000,atempo=0.900000", graph)
        self.assertNotIn("atrim=start=1.000000:end=1.300000,atempo", graph)
        self.assertTrue(any("1 effect group(s) to 3 word(s) in 1 chunk(s)" in line for line in logs))

    def test_applies_volume_boost_to_each_selected_chunk(self) -> None:
        commands: list[list[str]] = []
        with (
            patch.object(pipeline, "run_command", side_effect=lambda args, *_args, **_kwargs: commands.append(args)),
            patch.object(pipeline, "get_episode", return_value={"name": "episode"}),
        ):
            pipeline.slow_words(
                "episode",
                [1, 2],
                0.9,
                AppConfig(),
                lambda _message: None,
                gain_db=2.5,
            )

        graph = commands[0][commands[0].index("-filter_complex") + 1]
        self.assertIn(
            "atrim=start=1.000000:end=1.650000,atempo=0.900000,volume=2.500dB",
            graph,
        )
        self.assertNotIn("adelay", graph)
        self.assertNotIn("apad", graph)

    def test_word_slowdown_validates_gain(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 6"):
            pipeline.slow_words(
                "episode", [1], 0.95, AppConfig(), lambda _message: None, gain_db=7
            )

    def test_applies_distinct_settings_to_multiple_effect_groups(self) -> None:
        commands: list[list[str]] = []
        edits = [
            {"word_ids": [1, 2], "speed": 0.9, "gain_db": 2},
            {"word_ids": [4], "speed": 0.8, "gain_db": 4},
        ]
        with (
            patch.object(pipeline, "run_command", side_effect=lambda args, *_args, **_kwargs: commands.append(args)),
            patch.object(pipeline, "get_episode", return_value={"name": "episode"}),
        ):
            pipeline.edit_word_groups("episode", edits, AppConfig(), lambda _message: None)
        graph = commands[0][commands[0].index("-filter_complex") + 1]
        self.assertIn("atempo=0.900000,volume=2.000dB", graph)
        self.assertIn("atempo=0.800000,volume=4.000dB", graph)

    def test_rejects_words_assigned_to_multiple_effect_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple effect groups"):
            pipeline.edit_word_groups(
                "episode",
                [{"word_ids": [1, 2]}, {"word_ids": [2, 3]}],
                AppConfig(),
                lambda _message: None,
            )

    def test_episode_payload_contains_alignment_data_only(self) -> None:
        episode = pipeline.get_episode("episode")

        self.assertEqual(episode["word_count"], 4)
        self.assertIsNone(episode["sentences"][0]["words"][2]["score"])
        self.assertNotIn("generated_groups", episode)
        self.assertNotIn("final", episode)

    def test_import_wav_sanitizes_name_and_keeps_display_name(self) -> None:
        source = self.output_dir / "source.wav"
        source.write_bytes(b"wav")
        imported = pipeline.import_wav(source, "friendly_internal", lambda _message: None, display_name="Friendly Episode")
        self.assertEqual(imported["name"], "friendly_internal")
        self.assertEqual(imported["display_name"], "Friendly Episode")

    def test_workspace_reset_removes_previous_audio(self) -> None:
        stale_file = pipeline.words_dir(self.episode_dir) / "old-result.wav"
        stale_file.write_bytes(b"preview wav")

        pipeline.reset_workspace()

        self.assertEqual(list(self.output_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
