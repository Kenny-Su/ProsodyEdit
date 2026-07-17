# Podcast Sentence Replacement Workflow

This workflow replaces selected sentences in a podcast with slowed CosyVoice generations while keeping the edit reproducible and minimizing audible splice artifacts.

## Overview

For a given podcast episode:

1. Convert the source audio to WAV and use WAV for all intermediate editing.
2. Use WhisperX to transcribe and obtain sentence-level timestamps.
3. Cut each sentence into an individual WAV file.
4. For selected target sentences, use CosyVoice to generate slowed replacement audio.
5. Trim only leading and ending silence from generated audio.
6. Match generated loudness to the surrounding original audio.
7. Splice replacements back into the episode with short crossfades.
8. Export MP3 only at final delivery, if needed.

## Audio Format

Use WAV as the editing format to avoid repeated lossy encoding. In the reference run, the working audio was:

- PCM `s16le`
- `44100 Hz`
- stereo

CosyVoice outputs may be mono and at a different sample rate. During splicing, convert generated clips to match the source WAV format.

## Transcription

Use WhisperX to generate sentence-level timestamps and word timings. The transcript should include, for each sentence:

- sentence text
- start time
- end time
- word-level timestamps, if available

The transcript is the timing source of truth for all later cuts and splice boundaries.

## Sentence Cutting

Cut each sentence from the source WAV using its WhisperX `start` and `end` timestamps. Name files consistently, for example:

```text
sentences/sentence_01.wav
sentences/sentence_02.wav
...
```

For a target sentence `N`, record:

```text
target_start = start time of sentence N
target_end   = end time of sentence N
previous_end = end time of sentence N - 1
next_start   = start time of sentence N + 1
```

Use `target_start` and `target_end` for the CosyVoice prompt clip. Use `previous_end` and `next_start` for the final splice region.

## CosyVoice Generation

For each target sentence:

- Use the original sentence WAV as the voice prompt.
- Use the transcript sentence text as the generation text.
- Use CosyVoice zero-shot inference.
- Set `speed=0.95`.
- Save the untrimmed generated audio.

Prompt text format used in the reference run:

```text
You are a helpful assistant.<|endofprompt|>{sentence_text}
```

Keep both:

```text
generated/sentence_NN_slowdown.wav
generated/sentence_NN_slowdown_trimmed.wav
```

## Silence Trimming

Trim only leading and ending silence from generated audio. Do not remove internal pauses.

Rubric:

- Silence threshold: below `-50 dBFS`
- Analysis window: `0.01 s` RMS windows
- Minimum sustained sound duration: `0.10 s`
- Minimum preserved edge silence unit: `0.05 s`
- Leading silence: trim from the beginning until sustained non-silence is found
- Ending silence: scan backward using the same sustained non-silence rule

Operational definition:

1. Convert generated audio to PCM `s16le` before measuring silence.
2. Split the signal into short RMS analysis windows of `0.01 s`.
3. A window is silent when its RMS level is below `-50 dBFS`.
4. A boundary is treated as speech onset/offset only when non-silent windows continue for at least `0.10 s`.
5. Isolated clicks, breaths, or low-energy artifacts shorter than `0.10 s` at the boundary are ignored for onset/offset detection.
6. Internal pauses are preserved; the algorithm only trims the leading and trailing edges.

Rationale:

The sustained-sound criterion avoids falsely preserving leading or trailing artifacts that briefly exceed the silence threshold but are not part of the generated sentence. This is important for reproducible editing: a single above-threshold sample or short noise burst should not define the splice boundary.

Reference run observed:

- Sentence 02: `4.040000 s` to `3.700208 s`
- Sentence 06: `6.100000 s` to `5.866458 s`

## Loudness Matching

Measure the original sentence and generated trimmed sentence using FFmpeg `volumedetect` mean volume.

Apply gain using:

```text
gain_db = original_sentence_mean_volume - generated_trimmed_mean_volume
```

Reference run:

- Sentence 02:
  - original mean: `-23.2 dB`
  - generated trimmed mean: `-22.3 dB`
  - gain applied: `-0.9 dB`
- Sentence 06:
  - original mean: `-16.9 dB`
  - generated trimmed mean: `-17.8 dB`
  - gain applied: `+0.9 dB`

## Boundary-Aware Gain Ramp

If the audio before and after the replacement have noticeably different loudness, a single gain value may match one side but not the other. In that case, apply a gain ramp: the generated sentence starts with one gain value and gradually moves to another.

Measure four regions:

- Previous context: last `2.0 s` before the replacement region
- Next context: first `2.0 s` after the replacement region
- Generated start: first `2.0 s` of the trimmed generated sentence
- Generated end: last `2.0 s` of the trimmed generated sentence

Compute:

```text
start_gain = previous_context_mean - generated_start_mean
end_gain   = next_context_mean - generated_end_mean
```

Clamp both gains to avoid unnatural swelling:

```text
gain = clamp(gain, -3 dB, +3 dB)
```

Then linearly interpolate gain across the generated sentence:

```text
gain(t) = start_gain + (end_gain - start_gain) * (t / generated_duration)
```

Reference run for one sentence:

- Previous `2.0 s` context: `-18.8 dB`
- Next `2.0 s` context: `-18.9 dB`
- Generated first `2.0 s`: `-21.0 dB`
- Generated last `2.0 s`: `-22.0 dB`
- Raw gains: `+2.2 dB` to `+3.1 dB`
- Applied ramp after clamping: `+2.2 dB` to `+3.0 dB`

## Crossfade Splicing

Hard cuts are audible because generated speech differs in texture, channel layout, noise floor, and loudness. Use short crossfades at every replacement boundary.

Rubric:

- Crossfade duration: `0.150 s`
- Fade curve: linear fade-out plus linear fade-in
- FFmpeg curve setting: `c1=tri:c2=tri`

Boundary rule:

```text
left_original_end  = previous_end + 0.150
right_original_start = next_start - 0.150
```

This keeps `150 ms` of original boundary audio as overlap for each crossfade.

For two replacements, the sequence is:

```text
original prefix
crossfade into generated sentence 1
crossfade back into original middle
crossfade into generated sentence 2
crossfade back into original suffix
```

Reference run boundaries:

- Sentence 02:
  - previous sentence end: `7.118 s`
  - next sentence start: `11.340 s`
  - left original kept to `7.268 s`
  - right original resumed from `11.190 s`
- Sentence 06:
  - previous sentence end: `26.455 s`
  - next sentence start: `32.396 s`
  - left original kept to `26.605 s`
  - right original resumed from `32.246 s`

## Output

Save the smoothed WAV as the main edited output. In the reference run:

```text
generated/original_with_sentence_02_and_06_slowdown_crossfade.wav
```

The reference output was:

- PCM `s16le`
- `44100 Hz`
- stereo
- `47.403696 s`

Optional preview clips around each splice are useful for listening checks.

## Final Delivery

If a compressed deliverable is needed, export from the final WAV once:

```text
final WAV -> final MP3
```

Avoid MP3 in the intermediate editing path.
