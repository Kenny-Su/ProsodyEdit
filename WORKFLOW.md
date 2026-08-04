# Word Timestamp Workflow

## Overview

For an input WAV, `run_auto.py` runs the whole pipeline end to end:

1. Import the WAV into `~/Downloads/ProsodyEdit-output/<name>/`.
2. Transcribe it with `Qwen/Qwen3-ASR-1.7B`.
3. Align the recognized text with `Qwen/Qwen3-ForcedAligner-0.6B`.
4. Group aligned units into sentences at `.`, `?`, `!`, `。`, `？`, and `！`.
5. Cut exact sentence and word intervals for listening-based verification.
6. Ask the configured AI model which words deserve emphasis.
7. Render `edited.wav` and write `edit_log.json`.

## Timestamp source of truth

The native Qwen `ASRTranscription` result is retained in process memory. Its
`ForcedAlignResult.items` collection is the timestamp source of truth; each
item provides `text`, `start_time`, and `end_time`. No transcript JSON is
exported.

ProsodyEdit derives its sentence and word views from those native items.
Items with empty text, non-finite timestamps, negative starts, or `end <=
start` are excluded. The aligner does not return confidence scores, so
ProsodyEdit does not invent or display them.

## Verification audio

For every sentence, ProsodyEdit creates:

```text
sentences/sentence_01.wav
```

For every timed unit, it creates the exact `[start, end]` interval:

```text
words/word_0001.wav
words/word_0002.wav
```

Sentence audio provides surrounding context for checking a reported boundary.

## Word-level slowdown

`pipeline.edit_word_groups` takes one or more effect groups, each a speed
between `0.50` and `0.99` and a volume boost between `0` and `6 dB` applied to
a set of word ids. Consecutive selected word indexes are merged into one
interval from the first word's start through the last word's end, then
processed with FFmpeg `atempo` and `volume`. Nonconsecutive selections remain
separate chunks; audio between them and the remaining tail is unchanged.

Groups cannot share a word; all groups are sorted by source timestamp and
rendered together in one FFmpeg pass. The result is exported as PCM `s16le`
audio at `edited.wav`.

## Long audio and fallback

The official `qwen-asr` wrapper chunks long timestamped inputs and merges the
aligned units back onto the original timeline. ProsodyEdit requests MPS with
float16 first. If MPS is unavailable or raises an unsupported runtime error,
it retries the entire transcription on CPU with float32 and retains that CPU
model for subsequent jobs in the same process.

## AI-suggested effect groups

`pipeline.ai_auto_edit` sends a compact transcript (`[S<n>] <word_id>:<text>`
per sentence) to the OpenAI-compatible endpoint configured by `openai_api_key`,
`openai_base_url`, and `openai_model` in `prosody_gui/config.json`, asking it to
return `{"groups": [...]}` using the same `word_ids` / `speed` / `gain_db`
shape as a manually built effect group.
`pipeline._normalize_ai_group` coerces types and clamps every value to the
manual-edit ranges, invalid or empty groups are dropped, and
`pipeline._dedupe_ai_groups` drops any word id already claimed by an earlier
group so the result always satisfies the same one-group-per-word invariant as
`edit_word_groups`. The surviving groups are applied immediately with
`edit_word_groups`; `run_auto.py` also writes them to `edit_log.json` with
each group's resolved word text for review.

## Current scope

The project supports timestamp inspection and direct word-level time
stretching, driven fully automatically end to end via `run_auto.py`. It does
not synthesize replacement speech, and there is no interactive review step
before a render — the original WAV is left untouched, so a bad render can
simply be discarded and rerun.
