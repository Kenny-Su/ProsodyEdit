from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torchaudio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--speed", type=float, default=0.95)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1] / "CosyVoice"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel  # noqa: PLC0415

    model = AutoModel(model_dir=args.model_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks = list(
        model.inference_zero_shot(
            args.text,
            args.prompt_text,
            args.prompt_wav,
            stream=False,
            speed=args.speed,
        )
    )
    if not chunks:
        raise RuntimeError("CosyVoice produced no audio chunks.")
    speech = chunks[-1]["tts_speech"]
    torchaudio.save(str(output), speech, model.sample_rate)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
