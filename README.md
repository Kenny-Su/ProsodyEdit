# ProsodyEdit

ProsodyEdit is a local, fully automated command-line pipeline: give it a WAV,
it transcribes the audio with Qwen3-ASR and Qwen3-ForcedAligner, asks an AI
model which words deserve expressive emphasis, and renders an edited WAV plus
a JSON edit log describing exactly what changed. The original file is never
modified; a bad AI call just produces a render you can discard.

## Requirements

- An Apple Silicon Mac. MPS is used by default, with a CPU fallback.
- Python 3.12 for the model environment.
- FFmpeg on `PATH`.
- Enough free disk space and unified memory for the 1.7B ASR and 0.6B aligner
  checkpoints. The first transcription downloads both models from Hugging Face.

Install the system prerequisites with Homebrew:

```bash
brew install python@3.12 ffmpeg
```

## Model environment

Create a dedicated environment for Qwen inference.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade qwen-asr
deactivate
```

Do not install the optional vLLM or FlashAttention packages on macOS; those
paths target CUDA. Model weights are cached by Hugging Face and are not stored
in this repository.

## Run

```bash
.venv/bin/python run_auto.py path/to/episode.wav
```

You may also run `python run_auto.py ...`; the launcher detects an interpreter
outside the project environment and restarts itself with `.venv/bin/python`.
Pass `--name` to control the output folder name; it defaults to the input
file's stem.

Progress, command output, and full error details are printed to the terminal.
Output is written under `~/Downloads/ProsodyEdit-output/<name>/`:

- `original.wav` — an untouched copy of the input
- `edited.wav` — the rendered result
- `edit_log.json` — the effect groups the AI chose, with the affected word
  text, speed, and gain for each

The first run creates the ignored `prosody_gui/config.json`. Models load
lazily on first use and are reused for later runs in the same process.
Defaults are:

- ASR: `Qwen/Qwen3-ASR-1.7B`
- Forced aligner: `Qwen/Qwen3-ForcedAligner-0.6B`
- Device: MPS with unsupported-operation CPU fallback
- Language: English; set `qwen_language` to an empty string for detection
- Generation limit: 2048 tokens

See `prosody_gui/config.example.json` for all settings. When MPS initialization
or inference fails, ProsodyEdit logs the error, releases the MPS model, and
retries the complete job on CPU with float32. CPU inference can be substantially
slower.

## Word-level slowdown

Emphasis is applied per selected word run: a speed from `0.50` to `0.99` and
an optional `0–6 dB` volume boost. ProsodyEdit applies FFmpeg `atempo` and
`volume`, then concatenates the result in timeline order. Consecutive selected
words are merged from the first word's start to the last word's end.
Nonconsecutive selections remain separate chunks, and all audio between those
chunks is unchanged. A word may belong to only one effect group.
The edited episode becomes longer by the added duration of the slowed words.

The official Qwen wrapper automatically splits long audio to the forced
aligner's supported chunk length and restores absolute offsets when merging
the results. Test a short recording first, then verify a recording longer than
five minutes on the target machine before relying on a long batch.

## AI auto-edit

ProsodyEdit sends the transcript (sentence markers plus each word's id) to an
OpenAI-compatible chat completions endpoint and asks it to choose which words
get emphasis (slow down + volume boost), using the same two knobs as a manual
effect group. The response is validated, clamped to the same ranges described
above (speed `0.50–0.99`, boost `0–6 dB`), deduplicated so no word is claimed
twice, and then applied in one FFmpeg pass.

This requires a Qwen Token Plan API key. Add it to the gitignored
`prosody_gui/config.json` (see `prosody_gui/config.example.json`):

```json
{
  "openai_api_key": "sk-...",
  "openai_base_url": "https://portal.qwen.ai/v1",
  "openai_model": "qwen3-max"
}
```

`openai_base_url` defaults to the Qwen Token Plan's OpenAI-compatible
endpoint, `https://portal.qwen.ai/v1`; point it at any other
OpenAI-compatible endpoint to use a different provider. The key is only ever
read from this local config file and sent as a bearer token to that endpoint.
