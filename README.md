# ProsodyEdit

ProsodyEdit is a local CLI: give it a WAV, it transcribes the audio with
Qwen3-ASR and Qwen3-ForcedAligner, asks an AI model which words deserve
expressive emphasis, and renders an edited WAV plus a JSON edit log describing
exactly what changed. The input file is never modified; a bad AI call just
produces a render you can discard.

## Requirements

- An Apple Silicon Mac. MPS is used by default, with a CPU fallback.
- Python 3.12.
- FFmpeg on `PATH`.
- Enough free disk space and unified memory for the 1.7B ASR and 0.6B aligner
  checkpoints. The first transcription downloads both models from Hugging Face.

```bash
brew install python@3.12 ffmpeg
```

## Install

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

This installs the `prosodyedit` command into `.venv/bin/`. Do not install the
optional vLLM or FlashAttention extras for `qwen-asr` on macOS; those paths
target CUDA. Model weights are cached by Hugging Face and are not stored in
this repository.

## Configure

Copy `config.example.json` to `config.json` in the project root and fill in
`openai_api_key` to enable AI-suggested edits. You can also set the key via
the `PROSODYEDIT_OPENAI_API_KEY` environment variable instead of writing it
to disk. `config.json` is gitignored. Built-in defaults (used for any field
you don't set):

- ASR: `Qwen/Qwen3-ASR-1.7B`
- Forced aligner: `Qwen/Qwen3-ForcedAligner-0.6B`
- Device: MPS with unsupported-operation CPU fallback
- Language: English; set `qwen_language` to an empty string for detection
- Generation limit: 2048 tokens
- `openai_base_url`: `https://portal.qwen.ai/v1` (point it at any other
  OpenAI-compatible endpoint to use a different provider)

## Run

```bash
.venv/bin/prosodyedit path/to/episode.wav
```

Pass `--output-dir` to control where the output lands (default: alongside the
input file). Pass `--config` to use a config file at a non-default path.

Output, next to the input file by default:

- `episode.edited.wav` — the rendered result
- `episode.edit_log.json` — the full transcribed word list plus the effect
  groups the AI chose (word ids, speed, and gain for each)

## How it works

1. Transcribe the WAV with `Qwen/Qwen3-ASR-1.7B`.
2. Align the recognized text with `Qwen/Qwen3-ForcedAligner-0.6B`. The native
   `ForcedAlignResult` items (`text`, `start_time`, `end_time`) are the
   timestamp source of truth; items with empty text, non-finite timestamps,
   negative starts, or `end <= start` are excluded.
3. Send the transcript as `<word_id>:<text>` tokens in timeline order (wrapped
   to one line per 40 words purely for readability) to the OpenAI-compatible
   endpoint, asking for `{"groups": [...]}` — each group a list of word ids
   plus a `speed` (`0.50`-`0.99`) and optional `gain_db` (`0`-`6`) for
   emphasis. Sentence-ending punctuation stays attached to each word, so the
   model reads phrase and sentence boundaries itself; there's no separate
   sentence-detection step. The response is validated, clamped to those
   ranges, and deduplicated so no word id is claimed by more than one group.
4. Render the edit: consecutive selected word indexes are merged into one
   interval from the first word's start through the last word's end, then
   processed with FFmpeg `atempo` and `volume`. Nonconsecutive selections
   remain separate chunks; audio outside any chunk is untouched. All chunks
   are rendered together in one FFmpeg pass to `episode.edited.wav`
   (`pcm_s16le`). The edited episode becomes longer by the added duration of
   the slowed words.

The official `qwen-asr` wrapper chunks long timestamped inputs and merges the
aligned units back onto the original timeline, restoring absolute offsets.
When MPS initialization or inference fails, ProsodyEdit logs the error and
retries the complete job on CPU with float32 — substantially slower, but
functional. Test a short recording first, then verify a recording longer than
five minutes on the target machine before relying on a long batch.

## Tests

```bash
.venv/bin/python -m unittest
```
