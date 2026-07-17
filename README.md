# ProsodyEdit

ProsodyEdit is a local GUI for transcribing audio with WhisperX and replacing
selected sentences with CosyVoice-generated speech.

## Clone

Clone with both source dependencies:

```bash
git clone --recurse-submodules <PROSODYEDIT_REPOSITORY_URL>
cd ProsodyEdit
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## Python environments

Use Python 3.10 for CosyVoice:

```bash
python3.10 -m venv CosyVoice/.venv
source CosyVoice/.venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<81"
python -m pip install -r CosyVoice/requirements.txt
deactivate
```

Create a separate WhisperX environment:

```bash
python3.10 -m venv whisperx/.venv
source whisperx/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./whisperx
deactivate
```

FFmpeg and FFprobe must also be available on `PATH`. On macOS with Homebrew:

```bash
brew install ffmpeg
```

## CosyVoice model

Model weights are intentionally not stored in Git. Download
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` into this exact directory:

```text
CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/
```

Using ModelScope from the CosyVoice environment:

```bash
source CosyVoice/.venv/bin/activate
python -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')"
deactivate
```

The directory should contain files such as `llm.pt`, `flow.pt`, `hift.pt`,
`speech_tokenizer_v3.onnx`, and `cosyvoice3.yaml`.

## Run

The GUI server itself only uses the Python standard library:

```bash
python3 run_gui.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

The first launch creates `prosody_gui/config.json`. This file is ignored
because it contains machine-specific paths. Portable defaults resolve
CosyVoice and WhisperX relative to the repository:

```text
CosyVoice/.venv/bin/python
CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
whisperx/.venv/bin/python
```

Input and output directories can be changed from the GUI. See
`prosody_gui/config.example.json` for optional tuning settings.
