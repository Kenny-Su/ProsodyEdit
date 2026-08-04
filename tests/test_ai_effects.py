from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prosody_gui import pipeline
from prosody_gui.config import AppConfig


class AiEffectsTests(unittest.TestCase):
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

    def test_transcript_prompt_includes_sentence_markers_and_word_ids(self) -> None:
        prompt = pipeline._transcript_prompt(self.episode_dir)
        self.assertEqual(
            prompt,
            "[S1] 1:Make 2:this 3:count.\n[S2] 4:Again.",
        )

    def test_normalize_ai_group_fills_defaults_and_clamps_ranges(self) -> None:
        group = pipeline._normalize_ai_group({"word_ids": [2, 1]}, 1)
        self.assertEqual(group, {"word_ids": [1, 2], "speed": 0.9, "gain_db": 0.0})

        clamped = pipeline._normalize_ai_group(
            {"word_ids": [1], "speed": 1.5, "gain_db": 50},
            2,
        )
        self.assertEqual(clamped["speed"], 0.99)
        self.assertEqual(clamped["gain_db"], 6.0)

    def test_normalize_ai_group_rejects_empty_or_invalid_word_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "no word_ids"):
            pipeline._normalize_ai_group({"word_ids": []}, 1)
        with self.assertRaisesRegex(ValueError, "invalid word_ids"):
            pipeline._normalize_ai_group({"word_ids": ["not-an-int"]}, 1)

    def test_dedupe_ai_groups_drops_words_already_claimed(self) -> None:
        logs: list[str] = []
        groups = [
            {"word_ids": [1, 2], "speed": 0.9, "gain_db": 0.0},
            {"word_ids": [2, 3], "speed": 0.8, "gain_db": 0.0},
        ]
        deduped = pipeline._dedupe_ai_groups(groups, logs.append)
        self.assertEqual([group["word_ids"] for group in deduped], [[1, 2], [3]])
        self.assertTrue(any("already claimed" in line for line in logs))

    def test_ai_suggest_effects_drops_hallucinated_word_ids(self) -> None:
        content = json.dumps(
            {
                "groups": [
                    {"word_ids": [1, 99], "speed": 0.8, "gain_db": 2.0},
                ]
            }
        )
        logs: list[str] = []
        with patch.object(pipeline, "_call_openai_chat", return_value=content) as mock_call:
            groups = pipeline.ai_suggest_effects("episode", AppConfig(), logs.append)

        mock_call.assert_called_once()
        self.assertEqual(groups, [{"word_ids": [1], "speed": 0.8, "gain_db": 2.0}])
        self.assertTrue(any("not present in this transcript" in line for line in logs))

    def test_ai_suggest_effects_requires_a_transcript(self) -> None:
        pipeline.set_transcript("episode", SimpleNamespace(text="", time_stamps=SimpleNamespace(items=[])))
        with self.assertRaisesRegex(ValueError, "Transcribe the episode"):
            pipeline.ai_suggest_effects("episode", AppConfig(), lambda _message: None)

    def test_call_openai_chat_requires_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "openai_api_key"):
            pipeline._call_openai_chat(AppConfig(openai_api_key=""), [], lambda _message: None)

    def test_ai_auto_edit_applies_suggested_groups(self) -> None:
        commands: list[list[str]] = []
        content = json.dumps(
            {
                "groups": [
                    {"word_ids": [1, 2], "speed": 0.8, "gain_db": 1.5},
                ]
            }
        )
        with (
            patch.object(pipeline, "_call_openai_chat", return_value=content),
            patch.object(pipeline, "run_command", side_effect=lambda args, *_args, **_kwargs: commands.append(args)),
            patch.object(pipeline, "get_episode", return_value={"name": "episode"}),
        ):
            result = pipeline.ai_auto_edit("episode", AppConfig(), lambda _message: None)

        self.assertEqual(
            result["ai_effect_groups"],
            [{"word_ids": [1, 2], "speed": 0.8, "gain_db": 1.5}],
        )
        graph = commands[0][commands[0].index("-filter_complex") + 1]
        self.assertIn("atempo=0.800000", graph)
        self.assertIn("volume=1.500dB", graph)
        self.assertNotIn("apad", graph)
        self.assertNotIn("adelay", graph)

    def test_ai_auto_edit_raises_when_no_groups_survive(self) -> None:
        content = json.dumps({"groups": []})
        with patch.object(pipeline, "_call_openai_chat", return_value=content):
            with self.assertRaisesRegex(ValueError, "did not suggest any effect groups"):
                pipeline.ai_auto_edit("episode", AppConfig(), lambda _message: None)


if __name__ == "__main__":
    unittest.main()
