# ProsodyEdit

ProsodyEdit asks an LLM which words in a spoken recording deserve emphasis, then
renders those words slower and louder. It is a single dependency-free Python
script, written as the software attachment to an HCI paper.

## Requirements

Python 3.12 and FFmpeg on `PATH`:

```bash
brew install python@3.12 ffmpeg
```

Copy `config.example.json` to `config.json` and fill in `openai_api_key`.
`openai_base_url` may point at any OpenAI-compatible endpoint. `config.json` is
gitignored.

## Run

```bash
python3 prosodyedit.py path/to/input_dir
```

The input directory holds:

- `audio.wav` — the source recording
- `timestamp.json` — a forced aligner's flat list of
  `{"text", "start_time", "end_time"}` words in timeline order

ProsodyEdit writes `audio.edited.wav` and `edit_log.json` beside them, and never
modifies the originals.

## How it works

1. The transcript is sent as `<word_id>:<text>` tokens in timeline order.
   Sentence-ending punctuation stays attached to each word, so the model reads
   phrase and sentence boundaries itself, and replies with `{"word_ids": [...]}`
   — the words worth emphasizing.
2. Emphasis is binary: every selected word is slowed to `0.9x` and boosted by
   `5 dB` (`SPEED` and `GAIN_DB` at the top of the script). Runs of consecutive
   selected words become one continuous span, so a phrase is stretched as a
   unit rather than word by word.
3. The spans and the untouched audio between them are rendered in one FFmpeg
   pass. The edited recording is longer than the input by the time added to the
   slowed words.

`edit_log.json` is the full word list with an `emphasized` flag on each word.
