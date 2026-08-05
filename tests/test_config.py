from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prosodyedit.config import Config, load_config


class LoadConfigTests(unittest.TestCase):
    def test_returns_defaults_when_no_file_exists(self) -> None:
        config = load_config(Path("/nonexistent/config.json"))
        self.assertEqual(config, Config(ffmpeg=config.ffmpeg))

    def test_overrides_defaults_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"openai_model": "custom"}), encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.openai_model, "custom")

    def test_falls_back_to_environment_variable_for_api_key(self) -> None:
        with patch.dict("os.environ", {"PROSODYEDIT_OPENAI_API_KEY": "sk-from-env"}):
            config = load_config(Path("/nonexistent/config.json"))
        self.assertEqual(config.openai_api_key, "sk-from-env")


if __name__ == "__main__":
    unittest.main()
