# ProsodyEdit

ProsodyEdit compares two ways to emphasize the same words in an English narration:

- `audio.edited.wav`: slow selected spans to 0.95× and add 2 dB.
- `audio.cosyvoice.wav`: regenerate selected sentences with Fun-CosyVoice 3.

The pipeline is deliberately small: MFA aligns the transcript, an
OpenAI-compatible model selects `word_ids`, and both interventions consume that
same plan.

## Environment

Install `~/CosyVoice` and its requirements following the upstream repository,
then add ProsodyEdit's alignment dependencies to that environment:

```bash
conda env update -n cosyvoice -f environment.yml
conda activate cosyvoice
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa
```

Copy `config.example.json` to `config.json` and fill in the API key. Change
`cosyvoice_root` only if the repository is not at `~/CosyVoice`. `config.json`
is ignored by Git.

## Input

Each input directory contains:

```text
audio.wav   clean mono English narration, at least 16 kHz
script.txt  verbatim transcript
```

Punctuation defines sentence boundaries. The tokenizer intentionally assumes
controlled research transcripts: use whitespace between words and spell numbers
as they are spoken. Each sentence must be at most 30 seconds because it becomes
a CosyVoice reference clip.

## Run

```bash
python prosodyedit.py INPUT_DIR
```

The first run writes `alignment.json` and `emphasis_plan.json`; later runs reuse
them so the experimental condition stays fixed. When the input changes, delete
both files. To repeat only planning, delete `emphasis_plan.json`.
The saved plan is the experimental manipulation and should be inspected before
analysis.

For every selected sentence, CosyVoice receives that sentence's original audio
as its voice reference. Selected phrases are wrapped in `<strong>` and repeated
in an `inference_instruct2` instruction asking for slower, louder, more prominent
delivery. The sampling seed is fixed at 1986. Generated audio is resampled and
spliced into the original recording without loudness normalization.
