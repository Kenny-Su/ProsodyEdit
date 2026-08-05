from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from prosodyedit import transcript
from prosodyedit.config import Config


def item(text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(text=text, start_time=start, end_time=end)


def result(items) -> SimpleNamespace:
    return SimpleNamespace(time_stamps=SimpleNamespace(items=list(items)))


class ParseWordsTests(unittest.TestCase):
    def test_assigns_sequential_ids_in_timeline_order(self) -> None:
        words = transcript.parse_words(
            result([item("Make", 1.0, 1.3), item("this", 1.35, 1.65), item("count.", 1.7, 2.2)])
        )
        self.assertEqual([word.index for word in words], [1, 2, 3])
        self.assertEqual([word.text for word in words], ["Make", "this", "count."])
        self.assertEqual((words[0].start, words[0].end), (1.0, 1.3))

    def test_filters_invalid_timestamp_items(self) -> None:
        native = result(
            [
                item("", 0.0, 0.1),
                item("negative", -0.1, 0.1),
                item("backwards", 1.0, 0.9),
                item("infinite", 1.0, math.inf),
                item("valid.", 2.0, 2.4),
            ]
        )
        words = transcript.parse_words(native)
        self.assertEqual([word.text for word in words], ["valid."])
        self.assertEqual(words[0].index, 1)

    def test_preserves_long_audio_offsets(self) -> None:
        native = result([item("First.", 299.2, 299.8), item("Second.", 300.4, 301.0)])
        words = transcript.parse_words(native)
        self.assertEqual([(word.start, word.end) for word in words], [(299.2, 299.8), (300.4, 301.0)])


class JoinUnitsTests(unittest.TestCase):
    def test_joins_with_spaces_and_respects_punctuation(self) -> None:
        self.assertEqual(transcript.join_units(["Make", "this", "count."]), "Make this count.")

    def test_joins_cjk_without_spaces(self) -> None:
        self.assertEqual(transcript.join_units(["你好", "世界。"]), "你好世界。")


class TranscribeTests(unittest.TestCase):
    def test_mps_failure_retries_on_cpu(self) -> None:
        calls: list[str] = []
        native = result([item("Done.", 0.0, 0.4)])

        def fake_transcribe(_wav, _config, device):
            calls.append(device)
            if device == "mps":
                raise RuntimeError("MPS operation is not implemented")
            return native

        config = Config(qwen_device="mps")
        with patch.object(transcript, "_transcribe_native", side_effect=fake_transcribe):
            output = transcript.transcribe("audio.wav", config)

        self.assertIs(output, native)
        self.assertEqual(calls, ["mps", "cpu"])


if __name__ == "__main__":
    unittest.main()
