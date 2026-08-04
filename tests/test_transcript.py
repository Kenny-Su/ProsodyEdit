from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from prosodyedit import transcript
from prosodyedit.config import Config


def item(text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(text=text, start_time=start, end_time=end)


def result(items, language: str = "English", text: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        language=language,
        text=text,
        time_stamps=SimpleNamespace(items=list(items)),
    )


class ParseSentencesTests(unittest.TestCase):
    def test_groups_punctuation_free_aligner_words_from_full_transcript(self) -> None:
        sentences = transcript.parse_sentences(
            result(
                [
                    item("First", 0.1, 0.3),
                    item("sentence", 0.3, 0.7),
                    item("Second", 0.9, 1.2),
                    item("sentence", 1.2, 1.6),
                ],
                text="First sentence. Second sentence!",
            )
        )

        self.assertEqual([sentence.text for sentence in sentences], ["First sentence.", "Second sentence!"])
        self.assertEqual(
            [[word.text for word in sentence.words] for sentence in sentences],
            [["First", "sentence"], ["Second", "sentence"]],
        )
        self.assertEqual([(sentence.start, sentence.end) for sentence in sentences], [(0.1, 0.7), (0.9, 1.6)])

    def test_does_not_split_titles_or_initials(self) -> None:
        words = ["Mr", "Greenspan", "served", "George", "HW", "Bush", "He", "retired"]
        sentences = transcript.parse_sentences(
            result(
                [item(word, index * 0.2, index * 0.2 + 0.1) for index, word in enumerate(words)],
                text="Mr. Greenspan served George H.W. Bush. He retired.",
            )
        )

        self.assertEqual(
            [sentence.text for sentence in sentences],
            ["Mr. Greenspan served George H.W. Bush.", "He retired."],
        )
        self.assertEqual([len(sentence.words) for sentence in sentences], [6, 2])

    def test_groups_long_punctuation_free_transcript_into_utterances(self) -> None:
        words = [f"word{index}" for index in range(1, 30)] + ["Mr"]
        words += [f"word{index}" for index in range(31, 55)] + ["He"]
        words += [f"word{index}" for index in range(56, 96)]
        aligned = [item(word, index * 0.2, index * 0.2 + 0.1) for index, word in enumerate(words)]
        sentences = transcript.parse_sentences(result(aligned, text=" ".join(words)))

        self.assertEqual([len(sentence.words) for sentence in sentences], [29, 25, 41])
        self.assertEqual(sentences[1].words[0].text, "Mr")
        self.assertEqual(sentences[2].words[0].text, "He")

    def test_groups_native_english_timestamps_and_trailing_fragment(self) -> None:
        sentences = transcript.parse_sentences(
            result(
                [item("Hello", 0.1, 0.4), item("world!", 0.5, 0.9), item("Still", 1.1, 1.4)],
                text="Hello world! Still",
            )
        )

        self.assertEqual([sentence.text for sentence in sentences], ["Hello world!", "Still"])
        self.assertEqual((sentences[1].start, sentences[1].end), (1.1, 1.4))
        self.assertEqual([word.index for sentence in sentences for word in sentence.words], [1, 2, 3])

    def test_groups_cjk_without_spaces(self) -> None:
        sentences = transcript.parse_sentences(
            result([item("你好", 0.0, 0.2), item("世界。", 0.2, 0.6), item("再见！", 0.8, 1.2)], "Chinese")
        )
        self.assertEqual([sentence.text for sentence in sentences], ["你好世界。", "再见！"])

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
        self.assertEqual(transcript._timed_units(native), [("valid.", 2.0, 2.4)])

    def test_preserves_long_audio_offsets(self) -> None:
        native = result([item("First.", 299.2, 299.8), item("Second.", 300.4, 301.0)])
        self.assertEqual(
            transcript._timed_units(native),
            [("First.", 299.2, 299.8), ("Second.", 300.4, 301.0)],
        )


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
