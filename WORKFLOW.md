# Word Timestamp Workflow

## Overview

For the current WAV:

1. Upload the WAV in the browser.
2. Transcribe it with `Qwen/Qwen3-ASR-1.7B`.
3. Align the recognized text with `Qwen/Qwen3-ForcedAligner-0.6B`.
4. Group aligned units into sentences at `.`, `?`, `!`, `。`, `？`, and `！`.
5. Cut exact sentence and word intervals for listening-based verification.

## Timestamp source of truth

The native Qwen `ASRTranscription` result is retained in server memory. Its
`ForcedAlignResult.items` collection is the timestamp source of truth; each
item provides `text`, `start_time`, and `end_time`. No transcript JSON is
exported.

ProsodyEdit derives its sentence and browser views from those native items.
Items with empty text, non-finite timestamps, negative starts, or `end <=
start` are excluded. The aligner does not return confidence scores, so the GUI
does not invent or display them.

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

Clicking a tile plays that interval. Sentence audio provides surrounding
context for checking the reported boundary.

## Word-level slowdown

Select one or more aligned words in the GUI and choose a speed between `0.50`
and `0.99`, a volume boost between `0` and `6 dB`, and optional pauses of up to
`500 ms` before and after. Consecutive selected word indexes are merged into one
interval from the first word's start through the last word's end, then processed
with FFmpeg `atempo` and `volume`. `adelay` and `apad` insert silence before and
after the run while preserving the source format. Nonconsecutive selections
remain separate chunks; audio between them and the remaining tail is unchanged.

Each selection and its settings are saved as an effect group. Additional words
can be selected with different speed, gain, and pause settings. Groups cannot
share a word; all groups are sorted by source timestamp and rendered together
when the edited WAV is created.

The result is exported as temporary PCM `s16le` audio at `edited.wav` and is
available through the Edited player and Download WAV link.

## Long audio and fallback

The official `qwen-asr` wrapper chunks long timestamped inputs and merges the
aligned units back onto the original timeline. ProsodyEdit requests MPS with
float16 first. If MPS is unavailable or raises an unsupported runtime error,
the server retries the entire transcription on CPU with float32 and retains
that CPU model for subsequent jobs.

## Current scope

The project supports timestamp inspection and direct word-level time stretching.
It does not synthesize replacement speech.
