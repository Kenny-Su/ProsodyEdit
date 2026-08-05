# ProsodyEdit

ProsodyEdit is a local CLI: give it a folder with a pre-transcribed
recording, it asks an AI model which words deserve expressive emphasis, and
renders an edited WAV plus a JSON edit log describing exactly what changed.
The input file is never modified; a bad AI call just produces a render you
can discard.

Transcription and forced alignment are done separately (e.g. with the
`qwen-asr-demo` GUI) and copied into the input folder as `timestamp.json`.

## Requirements

- Python 3.12.
- FFmpeg on `PATH`.

```bash
brew install python@3.12 ffmpeg
```

## Install

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

This installs the `prosodyedit` command into `.venv/bin/`.

## Configure

Copy `config.example.json` to `config.json` in the project root and fill in
`openai_api_key` to enable AI-suggested edits. You can also set the key via
the `PROSODYEDIT_OPENAI_API_KEY` environment variable instead of writing it
to disk. `config.json` is gitignored. Built-in default (used if you don't
set it):

- `openai_base_url`: `https://portal.qwen.ai/v1` (point it at any other
  OpenAI-compatible endpoint to use a different provider)

## Run

```bash
.venv/bin/prosodyedit path/to/input_dir
```

`input_dir` must contain:

- `audio.wav` — the source recording
- `timestamp.json` — a flat list of `{"text", "start_time", "end_time"}`
  word objects in timeline order (the forced aligner's native output)

Pass `--output-dir` to control where the output lands (default: the input
directory). Pass `--config` to use a config file at a non-default path.

Output:

- `audio.edited.wav` — the rendered result
- `edit_log.json` — the full word list plus the effect groups the AI chose
  (word ids, speed, and gain for each)

## How it works

1. Load `timestamp.json`. Items with empty text, non-finite timestamps,
   negative starts, or `end <= start` are excluded.
2. Send the transcript as `<word_id>:<text>` tokens in timeline order (wrapped
   to one line per 40 words purely for readability) to the OpenAI-compatible
   endpoint, asking for `{"groups": [...]}` — each group a list of word ids
   plus a `speed` (`0.50`-`0.99`) and optional `gain_db` (`0`-`6`) for
   emphasis. Sentence-ending punctuation stays attached to each word, so the
   model reads phrase and sentence boundaries itself; there's no separate
   sentence-detection step. The response is validated, clamped to those
   ranges, and deduplicated so no word id is claimed by more than one group.
3. Render the edit: consecutive selected word indexes are merged into one
   interval from the first word's start through the last word's end, then
   processed with FFmpeg `atempo` and `volume`. Nonconsecutive selections
   remain separate chunks; audio outside any chunk is untouched. All chunks
   are rendered together in one FFmpeg pass to `audio.edited.wav`
   (`pcm_s16le`). The edited recording becomes longer by the added duration
   of the slowed words.

## Tests

```bash
.venv/bin/python -m unittest
```
