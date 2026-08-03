from __future__ import annotations

import json
import gc
import math
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import AppConfig


LogFn = Callable[[str], None]
OUTPUT_DIR = Path.home() / "Downloads" / "ProsodyEdit-output"
TRANSCRIPTS: dict[str, Any] = {}
_QWEN_MODEL: Any = None
_QWEN_MODEL_KEY: tuple[str, str, str, int] | None = None
_MPS_DISABLED_REASON: str | None = None

# Must be set before the lazy torch import performed during transcription.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SENTENCE_END_RE = re.compile(r"[.!?。！？][\"'”’）】》]*$")
NO_LEADING_SPACE_RE = re.compile(r"^[,.;:!?%。，、；：！？）】》”’]")
NO_TRAILING_SPACE_RE = re.compile(r"[（【《“‘/]$")
CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")


@dataclass
class Word:
    index: int
    sentence_index: int
    sentence_word_index: int
    start: float
    end: float
    text: str
    score: float | None = None
    word_audio: str | None = None


@dataclass
class Sentence:
    index: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    sentence_audio: str | None = None


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._")
    return cleaned or f"episode_{uuid.uuid4().hex[:8]}"


def configure_output_dir(path: str | Path) -> Path:
    global OUTPUT_DIR
    output = Path(path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise NotADirectoryError(output)
    OUTPUT_DIR = output
    return output


def reset_workspace() -> None:
    TRANSCRIPTS.clear()
    if not OUTPUT_DIR.exists():
        return
    for path in OUTPUT_DIR.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(OUTPUT_DIR))


def resolve_media(relative: str) -> Path:
    path = (OUTPUT_DIR / relative).resolve()
    if path != OUTPUT_DIR and OUTPUT_DIR not in path.parents:
        raise ValueError("Media path is outside the configured output directory.")
    return path


def episode_dir(name: str) -> Path:
    return OUTPUT_DIR / safe_name(name)


def display_episode_dir(name: str) -> Path:
    return episode_dir(name)


def original_wav(ep_dir: Path) -> Path:
    return ep_dir / "original.wav"


def episode_metadata_json(ep_dir: Path) -> Path:
    return ep_dir / "episode.json"


def sentences_dir(ep_dir: Path) -> Path:
    return ep_dir / "sentences"


def words_dir(ep_dir: Path) -> Path:
    return ep_dir / "words"


def edited_wav(ep_dir: Path) -> Path:
    return ep_dir / "edited.wav"


def word_file_stem(index: int) -> str:
    return f"word_{index:04d}"


def run_command(
    args: list[str],
    log: LogFn,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    log("$ " + " ".join(args))
    run_env = os.environ.copy() if env is None else env
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        args,
        cwd=str(cwd) if cwd else None,
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip())
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(args)}")


def ffmpeg(config: AppConfig, args: list[str], log: LogFn) -> None:
    run_command([config.ffmpeg, "-hide_banner", "-y", *args], log)


def import_wav(
    source: Path,
    requested_name: str | None,
    log: LogFn,
    display_name: str | None = None,
) -> dict:
    if source.suffix.lower() != ".wav":
        raise ValueError("Input must be a .wav file.")
    if not source.exists():
        raise FileNotFoundError(source)
    name = safe_name(requested_name or source.stem)
    ep_dir = episode_dir(name)
    ep_dir.mkdir(parents=True, exist_ok=True)
    sentences_dir(ep_dir).mkdir(exist_ok=True)
    words_dir(ep_dir).mkdir(exist_ok=True)
    dest = original_wav(ep_dir)
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    visible_name = (display_name or requested_name or source.stem).strip() or name
    episode_metadata_json(ep_dir).write_text(
        json.dumps({"display_name": visible_name}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"Imported WAV into {rel(dest)}.")
    return get_episode(name)


def set_transcript(name: str, result: Any) -> None:
    TRANSCRIPTS[safe_name(name)] = result


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def timestamp_items(result: Any) -> list[Any]:
    timestamps = _field(result, "time_stamps")
    if timestamps is None:
        return []
    items = getattr(timestamps, "items", None)
    if items is not None and not callable(items):
        return list(items)
    if isinstance(timestamps, dict):
        raw = timestamps.get("items", [])
        return list(raw) if isinstance(raw, (list, tuple)) else []
    if isinstance(timestamps, (list, tuple)):
        if len(timestamps) == 1 and hasattr(timestamps[0], "items"):
            return list(timestamps[0].items)
        return list(timestamps)
    try:
        return list(timestamps)
    except TypeError:
        return []


def join_units(units: Iterable[str]) -> str:
    text = ""
    for raw in units:
        unit = raw.strip()
        if not unit:
            continue
        if not text:
            text = unit
        elif NO_LEADING_SPACE_RE.search(unit) or NO_TRAILING_SPACE_RE.search(text):
            text += unit
        elif CJK_RE.search(unit) or CJK_RE.search(text[-1:]):
            text += unit
        else:
            text += " " + unit
    return text


def _timed_units(result: Any) -> list[tuple[str, float, float]]:
    units: list[tuple[str, float, float]] = []
    for item in timestamp_items(result):
        text = str(_field(item, "text", "") or "").strip()
        try:
            start = float(_field(item, "start_time"))
            end = float(_field(item, "end_time"))
        except (TypeError, ValueError):
            continue
        if not text or not math.isfinite(start) or not math.isfinite(end):
            continue
        if start < 0 or end <= start:
            continue
        units.append((text, start, end))
    return units


def parse_sentences(ep_dir: Path) -> list[Sentence]:
    result = TRANSCRIPTS.get(ep_dir.name)
    if result is None:
        return []
    sentences: list[Sentence] = []
    word_index = 0
    grouped: list[list[tuple[str, float, float]]] = []
    pending: list[tuple[str, float, float]] = []
    for unit in _timed_units(result):
        pending.append(unit)
        if SENTENCE_END_RE.search(unit[0]):
            grouped.append(pending)
            pending = []
    if pending:
        grouped.append(pending)

    for idx, group in enumerate(grouped, 1):
        words: list[Word] = []
        for sentence_word_index, (text, start, end) in enumerate(group, 1):
            word_index += 1
            stem = word_file_stem(word_index)
            word_wav = words_dir(ep_dir) / f"{stem}.wav"
            words.append(
                Word(
                    index=word_index,
                    sentence_index=idx,
                    sentence_word_index=sentence_word_index,
                    start=start,
                    end=end,
                    text=text,
                    word_audio=rel(word_wav) if word_wav.exists() else None,
                )
            )
        sentence_wav = sentences_dir(ep_dir) / f"sentence_{idx:02d}.wav"
        sentences.append(
            Sentence(
                index=idx,
                start=group[0][1],
                end=group[-1][2],
                text=join_units(unit[0] for unit in group),
                words=words,
                sentence_audio=rel(sentence_wav) if sentence_wav.exists() else None,
            )
        )
    return sentences


def parse_words(ep_dir: Path) -> list[Word]:
    return [word for sentence in parse_sentences(ep_dir) for word in sentence.words]


def get_episode(name: str) -> dict:
    ep_dir = display_episode_dir(name)
    sentences = parse_sentences(ep_dir)
    display_name = ep_dir.name
    metadata_path = episode_metadata_json(ep_dir)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        display_name = str(metadata.get("display_name") or display_name)
    edited = edited_wav(ep_dir)
    return {
        "name": ep_dir.name,
        "display_name": display_name,
        "path": rel(ep_dir) if ep_dir.exists() else None,
        "original": rel(original_wav(ep_dir)) if original_wav(ep_dir).exists() else None,
        "edited": rel(edited) if edited.exists() else None,
        "edited_version": edited.stat().st_mtime_ns if edited.exists() else None,
        "transcript": ep_dir.name in TRANSCRIPTS,
        "sentences": [asdict(sentence) for sentence in sentences],
        "word_count": sum(len(sentence.words) for sentence in sentences),
    }


def _clear_qwen_model() -> None:
    global _QWEN_MODEL, _QWEN_MODEL_KEY
    _QWEN_MODEL = None
    _QWEN_MODEL_KEY = None
    gc.collect()
    try:
        import torch

        torch.mps.empty_cache()
    except Exception:
        pass


def _load_qwen_model(config: AppConfig, device: str, log: LogFn) -> Any:
    global _QWEN_MODEL, _QWEN_MODEL_KEY
    key = (config.qwen_asr_model, config.qwen_aligner_model, device, config.qwen_max_new_tokens)
    if _QWEN_MODEL is not None and _QWEN_MODEL_KEY == key:
        return _QWEN_MODEL
    _clear_qwen_model()
    import torch
    from qwen_asr import Qwen3ASRModel

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this PyTorch runtime")
    dtype = torch.float16 if device == "mps" else torch.float32
    log(f"Loading Qwen3-ASR and forced aligner on {device}.")
    _QWEN_MODEL = Qwen3ASRModel.from_pretrained(
        config.qwen_asr_model,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=config.qwen_max_new_tokens,
        forced_aligner=config.qwen_aligner_model,
        forced_aligner_kwargs={"dtype": dtype, "device_map": device},
    )
    _QWEN_MODEL_KEY = key
    return _QWEN_MODEL


def _transcribe_native(wav: Path, config: AppConfig, device: str, log: LogFn) -> Any:
    model = _load_qwen_model(config, device, log)
    return model.transcribe(
        audio=str(wav),
        language=config.qwen_language or None,
        return_time_stamps=True,
    )[0]


def transcribe_episode(name: str, config: AppConfig, log: LogFn) -> dict:
    global _MPS_DISABLED_REASON
    ep_dir = display_episode_dir(name)
    wav = original_wav(ep_dir)
    if not wav.exists():
        raise FileNotFoundError(wav)
    device = config.qwen_device
    if device == "mps" and _MPS_DISABLED_REASON:
        log(f"MPS was disabled after an earlier failure: {_MPS_DISABLED_REASON}")
        device = "cpu"
    try:
        result = _transcribe_native(wav, config, device, log)
    except (RuntimeError, NotImplementedError) as exc:
        if device != "mps":
            raise
        _MPS_DISABLED_REASON = str(exc)
        log(f"MPS inference failed: {exc}")
        log("Retrying the complete transcription on CPU with float32.")
        _clear_qwen_model()
        result = _transcribe_native(wav, config, "cpu", log)
    edited_wav(ep_dir).unlink(missing_ok=True)
    set_transcript(name, result)
    log(f"Transcript ready in memory ({len(timestamp_items(result))} timed units).")
    cut_transcript_audio(name, config, log)
    return get_episode(name)


def cut_transcript_audio(name: str, config: AppConfig, log: LogFn) -> dict:
    ep_dir = display_episode_dir(name)
    wav = original_wav(ep_dir)
    sentences_dir(ep_dir).mkdir(exist_ok=True)
    words_dir(ep_dir).mkdir(exist_ok=True)
    for sentence in parse_sentences(ep_dir):
        out = sentences_dir(ep_dir) / f"sentence_{sentence.index:02d}.wav"
        duration = max(0.01, sentence.end - sentence.start)
        ffmpeg(
            config,
            [
                "-ss",
                f"{sentence.start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(wav),
                "-c:a",
                "pcm_s16le",
                str(out),
            ],
            log,
        )
        for word in sentence.words:
            word_out = words_dir(ep_dir) / f"{word_file_stem(word.index)}.wav"
            word_duration = max(0.01, word.end - word.start)
            ffmpeg(
                config,
                [
                    "-ss",
                    f"{word.start:.3f}",
                    "-t",
                    f"{word_duration:.3f}",
                    "-i",
                    str(wav),
                    "-c:a",
                    "pcm_s16le",
                    str(word_out),
                ],
                log,
            )
    log("Sentence and word preview WAV files are ready.")
    return get_episode(name)


def slow_words(
    name: str,
    word_ids: Iterable[int],
    speed: float,
    config: AppConfig,
    log: LogFn,
    gain_db: float = 0.0,
    pause_before_ms: int = 0,
    pause_after_ms: int = 0,
) -> dict:
    return edit_word_groups(
        name,
        [
            {
                "word_ids": list(word_ids),
                "speed": speed,
                "gain_db": gain_db,
                "pause_before_ms": pause_before_ms,
                "pause_after_ms": pause_after_ms,
            }
        ],
        config,
        log,
    )


def edit_word_groups(
    name: str,
    edits: Iterable[dict[str, Any]],
    config: AppConfig,
    log: LogFn,
) -> dict:
    edit_list = list(edits)
    if not edit_list:
        raise ValueError("Add at least one word effect group.")
    ep_dir = display_episode_dir(name)
    wav = original_wav(ep_dir)
    if not wav.exists():
        raise FileNotFoundError(wav)
    available = {word.index: word for word in parse_words(ep_dir)}
    assigned: set[int] = set()
    chunks: list[dict[str, Any]] = []

    for edit_number, edit in enumerate(edit_list, 1):
        try:
            speed = float(edit.get("speed", 0.95))
            gain_db = float(edit.get("gain_db", 0.0))
            pause_before_ms = int(edit.get("pause_before_ms", 0))
            pause_after_ms = int(edit.get("pause_after_ms", 0))
            requested = sorted(set(int(value) for value in edit.get("word_ids", [])))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid settings in effect group {edit_number}.") from exc
        if not math.isfinite(speed) or not 0.5 <= speed < 1.0:
            raise ValueError("Speed must be at least 0.50 and less than 1.00.")
        if not math.isfinite(gain_db) or not 0.0 <= gain_db <= 6.0:
            raise ValueError("Volume boost must be between 0 and 6 dB.")
        if not 0 <= pause_before_ms <= 500 or not 0 <= pause_after_ms <= 500:
            raise ValueError("Pauses must be between 0 and 500 milliseconds.")
        if not requested:
            raise ValueError(f"Effect group {edit_number} has no selected words.")
        duplicates = assigned.intersection(requested)
        if duplicates:
            raise ValueError(f"Words cannot belong to multiple effect groups: {sorted(duplicates)}.")
        missing = [word_id for word_id in requested if word_id not in available]
        if missing:
            raise ValueError(f"Unknown word indexes: {', '.join(map(str, missing))}.")
        assigned.update(requested)
        selected_words = [available[word_id] for word_id in requested]
        runs: list[list[Word]] = []
        for word in selected_words:
            if runs and word.index == runs[-1][-1].index + 1:
                runs[-1].append(word)
            else:
                runs.append([word])
        for run in runs:
            chunks.append(
                {
                    "start": run[0].start,
                    "end": run[-1].end,
                    "speed": speed,
                    "gain_db": gain_db,
                    "pause_before_ms": pause_before_ms,
                    "pause_after_ms": pause_after_ms,
                    "word_count": len(run),
                    "edit_number": edit_number,
                }
            )

    chunks.sort(key=lambda chunk: (chunk["start"], chunk["end"]))
    for previous, current in zip(chunks, chunks[1:]):
        if current["start"] < previous["end"]:
            raise ValueError("Effect group intervals overlap.")

    filters: list[str] = []
    labels: list[str] = []
    cursor = 0.0

    def add_segment(expression: str) -> None:
        label = f"part{len(labels)}"
        filters.append(f"[0:a]{expression},asetpts=PTS-STARTPTS[{label}]")
        labels.append(f"[{label}]")

    for chunk in chunks:
        start, end = chunk["start"], chunk["end"]
        if start > cursor + 0.000001:
            add_segment(f"atrim=start={cursor:.6f}:end={start:.6f}")
        effects = [f"atrim=start={start:.6f}:end={end:.6f}", f"atempo={chunk['speed']:.6f}"]
        if chunk["gain_db"]:
            effects.append(f"volume={chunk['gain_db']:.3f}dB")
        if chunk["pause_before_ms"]:
            effects.append(f"adelay=delays={chunk['pause_before_ms']}:all=1")
        if chunk["pause_after_ms"]:
            effects.append(f"apad=pad_dur={chunk['pause_after_ms'] / 1000:.3f}")
        add_segment(",".join(effects))
        cursor = end
    add_segment(f"atrim=start={cursor:.6f}")
    output = edited_wav(ep_dir)
    filter_complex = ";".join(filters + [f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]"])
    log(f"Applying {len(edit_list)} effect group(s) to {len(assigned)} word(s) in {len(chunks)} chunk(s).")
    ffmpeg(
        config,
        [
            "-i",
            str(wav),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        log,
    )
    log(f"Edited WAV ready: {rel(output)}.")
    return get_episode(name)
