from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prosody_gui import pipeline
from prosody_gui.config import AppConfig


def item(text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(text=text, start_time=start, end_time=end)


def result(items, language: str = "English", text: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        language=language,
        text=text,
        time_stamps=SimpleNamespace(items=list(items)),
    )


class QwenNativeTests(unittest.TestCase):
    def test_groups_native_english_timestamps_and_trailing_fragment(self) -> None:
        pipeline.set_transcript(
            "episode",
            result(
                [item("Hello", 0.1, 0.4), item("world!", 0.5, 0.9), item("Still", 1.1, 1.4)],
                text="Hello world! Still",
            ),
        )
        with patch.object(pipeline, "words_dir", return_value=Path("/tmp/words")), patch.object(
            pipeline, "sentences_dir", return_value=Path("/tmp/sentences")
        ):
            sentences = pipeline.parse_sentences(Path("/tmp/episode"))

        self.assertEqual([sentence.text for sentence in sentences], ["Hello world!", "Still"])
        self.assertEqual((sentences[1].start, sentences[1].end), (1.1, 1.4))

    def test_groups_cjk_without_spaces(self) -> None:
        pipeline.set_transcript(
            "cjk",
            result([item("你好", 0.0, 0.2), item("世界。", 0.2, 0.6), item("再见！", 0.8, 1.2)], "Chinese"),
        )
        with patch.object(pipeline, "words_dir", return_value=Path("/tmp/words")), patch.object(
            pipeline, "sentences_dir", return_value=Path("/tmp/sentences")
        ):
            sentences = pipeline.parse_sentences(Path("/tmp/cjk"))
        self.assertEqual([sentence.text for sentence in sentences], ["你好世界。", "再见！"])

    def test_filters_invalid_native_timestamp_items(self) -> None:
        native = result(
            [
                item("", 0.0, 0.1),
                item("negative", -0.1, 0.1),
                item("backwards", 1.0, 0.9),
                item("infinite", 1.0, math.inf),
                item("valid.", 2.0, 2.4),
            ]
        )
        self.assertEqual(pipeline._timed_units(native), [("valid.", 2.0, 2.4)])

    def test_preserves_long_audio_offsets(self) -> None:
        native = result([item("First.", 299.2, 299.8), item("Second.", 300.4, 301.0)])
        self.assertEqual(
            pipeline._timed_units(native),
            [("First.", 299.2, 299.8), ("Second.", 300.4, 301.0)],
        )

    def test_mps_failure_retries_on_cpu_and_keeps_native_result(self) -> None:
        calls: list[str] = []
        native = result([item("Done.", 0.0, 0.4)])

        def fake_transcribe(_wav, _config, device, _log):
            calls.append(device)
            if device == "mps":
                raise RuntimeError("MPS operation is not implemented")
            return native

        config = AppConfig(qwen_device="mps")
        pipeline._MPS_DISABLED_REASON = None
        with (
            patch.object(pipeline, "_transcribe_native", side_effect=fake_transcribe),
            patch.object(pipeline, "_clear_qwen_model"),
            patch.object(pipeline, "original_wav", return_value=Path(__file__)),
            patch.object(pipeline, "cut_transcript_audio"),
            patch.object(pipeline, "get_episode", return_value={"name": "fallback"}),
        ):
            output = pipeline.transcribe_episode("fallback", config, lambda _message: None)

        self.assertEqual(output, {"name": "fallback"})
        self.assertEqual(calls, ["mps", "cpu"])
        self.assertIs(pipeline.TRANSCRIPTS["fallback"], native)


if __name__ == "__main__":
    unittest.main()
