from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from prosodyedit import transcript


def write_timestamps(tmp: str, items: list[dict]) -> Path:
    path = Path(tmp) / "timestamp.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def item(text: str, start: float, end: float) -> dict:
    return {"text": text, "start_time": start, "end_time": end}


class LoadWordsTests(unittest.TestCase):
    def test_assigns_sequential_ids_in_timeline_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_timestamps(
                tmp, [item("Make", 1.0, 1.3), item("this", 1.35, 1.65), item("count.", 1.7, 2.2)]
            )
            words = transcript.load_words(path)
        self.assertEqual([word.index for word in words], [1, 2, 3])
        self.assertEqual([word.text for word in words], ["Make", "this", "count."])
        self.assertEqual((words[0].start, words[0].end), (1.0, 1.3))

    def test_filters_invalid_timestamp_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_timestamps(
                tmp,
                [
                    item("", 0.0, 0.1),
                    item("negative", -0.1, 0.1),
                    item("backwards", 1.0, 0.9),
                    item("infinite", 1.0, math.inf),
                    item("valid.", 2.0, 2.4),
                ],
            )
            words = transcript.load_words(path)
        self.assertEqual([word.text for word in words], ["valid."])
        self.assertEqual(words[0].index, 1)

    def test_preserves_long_audio_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_timestamps(tmp, [item("First.", 299.2, 299.8), item("Second.", 300.4, 301.0)])
            words = transcript.load_words(path)
        self.assertEqual([(word.start, word.end) for word in words], [(299.2, 299.8), (300.4, 301.0)])


if __name__ == "__main__":
    unittest.main()
