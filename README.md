# ProsodyEdit

ProsodyEdit is a local GUI for inspecting sentence- and word-level timestamps
produced by Qwen3-ASR and Qwen3-ForcedAligner. Upload a WAV, transcribe it,
inspect every timed unit, and click a tile to hear its exact aligned interval.

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

Create a dedicated environment for both the GUI server and Qwen inference.

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
.venv/bin/python run_gui.py
```

You may also run `python run_gui.py`; the launcher detects an interpreter
outside the project environment and restarts itself with `.venv/bin/python`.

Open [http://127.0.0.1:8765](http://127.0.0.1:8765), upload a short WAV, and
click **Transcribe**. This is also the recommended first-run smoke test before
processing a long recording.

The first launch creates the ignored `prosody_gui/config.json`. The server
loads both models lazily on the first transcription and reuses them for later
jobs. Defaults are:

- ASR: `Qwen/Qwen3-ASR-1.7B`
- Forced aligner: `Qwen/Qwen3-ForcedAligner-0.6B`
- Device: MPS with unsupported-operation CPU fallback
- Language: English; set `qwen_language` to an empty string for detection
- Generation limit: 2048 tokens

See `prosody_gui/config.example.json` for all settings. When MPS initialization
or inference fails, ProsodyEdit logs the error, releases the MPS model, and
retries the complete job on CPU with float32. CPU inference can be substantially
slower.

## Timestamp inspection

Qwen's forced aligner supplies text, start time, and end time, but no alignment
confidence. The GUI therefore reports timestamp coverage instead of inventing
a score. It displays timestamps to millisecond precision and creates exact WAV
previews for every sentence and timed unit.

The native Qwen `ASRTranscription` and `ForcedAlignResult` objects remain in
memory for the lifetime of the server. ProsodyEdit does not export or maintain
a transcript JSON file. Uploading a new WAV or stopping the server discards the
in-memory transcript.

## Word-level slowdown

After transcription, select any words with their checkboxes, choose a speed
from `0.50` to `0.99`, an optional `0–6 dB` volume boost, and optional pauses
of `0–500 ms` before and after each selected run. ProsodyEdit applies FFmpeg
`atempo`, `volume`, `adelay`, and `apad`, then concatenates the result in
timeline order. Consecutive selected words are merged from the first word's
start to the last word's end, including their internal pauses. Nonconsecutive
selections remain separate chunks, and all audio between those chunks is
unchanged.

Click **Add effect group** to save those settings, then select another set of
words and add a group with different values. A word may belong to only one
group. Review or remove groups from the list, then click **Create edited WAV**
to apply every group in one timeline-ordered FFmpeg export.
The edited episode becomes longer by the added duration of the slowed words.

Use the Edited player to review the result or click **Download WAV**. Creating
another edit replaces the previous temporary `edited.wav`; the browser receives
a cache-busted URL for the newest version.

The official Qwen wrapper automatically splits long audio to the forced
aligner's supported chunk length and restores absolute offsets when merging
the results. Test a short recording first, then verify a recording longer than
five minutes on the target machine before relying on a long batch.
