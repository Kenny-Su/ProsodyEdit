from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import wave
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from .config import ROOT, AppConfig


LogFn = Callable[[str], None]
OUTPUT_DIR = Path.home() / "Downloads" / "ProsodyEdit-output"


@dataclass
class Sentence:
    index: int
    start: float
    end: float
    text: str
    sentence_audio: str | None = None
    generated_audio: str | None = None
    trimmed_audio: str | None = None


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


def transcript_json(ep_dir: Path) -> Path:
    return ep_dir / "original.json"


def sentences_dir(ep_dir: Path) -> Path:
    return ep_dir / "sentences"


def generated_dir(ep_dir: Path) -> Path:
    return ep_dir / "generated"


def run_command(
    args: list[str],
    log: LogFn,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    log("$ " + " ".join(args))
    proc = subprocess.Popen(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
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


def ffprobe_duration(config: AppConfig, path: Path) -> float:
    out = subprocess.check_output(
        [
            config.ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def import_wav(source: Path, requested_name: str | None, log: LogFn) -> dict:
    if source.suffix.lower() != ".wav":
        raise ValueError("Input must be a .wav file.")
    if not source.exists():
        raise FileNotFoundError(source)
    name = safe_name(requested_name or source.stem)
    ep_dir = episode_dir(name)
    ep_dir.mkdir(parents=True, exist_ok=True)
    sentences_dir(ep_dir).mkdir(exist_ok=True)
    generated_dir(ep_dir).mkdir(exist_ok=True)
    dest = original_wav(ep_dir)
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    log(f"Imported WAV into {rel(dest)}.")
    return get_episode(name)


def parse_sentences(ep_dir: Path) -> list[Sentence]:
    path = transcript_json(ep_dir)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    sentences: list[Sentence] = []
    for idx, segment in enumerate(data.get("segments", []), 1):
        sentence_wav = sentences_dir(ep_dir) / f"sentence_{idx:02d}.wav"
        gen_wav = generated_dir(ep_dir) / f"sentence_{idx:02d}_slowdown.wav"
        trimmed_wav = generated_dir(ep_dir) / f"sentence_{idx:02d}_slowdown_trimmed.wav"
        sentences.append(
            Sentence(
                index=idx,
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=str(segment.get("text", "")).strip(),
                sentence_audio=rel(sentence_wav) if sentence_wav.exists() else None,
                generated_audio=rel(gen_wav) if gen_wav.exists() else None,
                trimmed_audio=rel(trimmed_wav) if trimmed_wav.exists() else None,
            )
        )
    return sentences


def get_episode(name: str) -> dict:
    ep_dir = display_episode_dir(name)
    final_files = sorted(
        generated_dir(ep_dir).glob("original_with_sentence_*_slowdown*.wav"),
        key=lambda path: path.stat().st_mtime,
    )
    return {
        "name": ep_dir.name,
        "path": rel(ep_dir) if ep_dir.exists() else None,
        "original": rel(original_wav(ep_dir)) if original_wav(ep_dir).exists() else None,
        "transcript": rel(transcript_json(ep_dir)) if transcript_json(ep_dir).exists() else None,
        "sentences": [asdict(s) for s in parse_sentences(ep_dir)],
        "final": rel(final_files[-1]) if final_files else None,
    }


def list_episodes() -> list[dict]:
    items: list[dict] = []
    if OUTPUT_DIR.exists():
        for path in sorted(OUTPUT_DIR.iterdir()):
            if path.is_dir() and (original_wav(path).exists() or transcript_json(path).exists()):
                items.append(get_episode(path.name))
    return items


def transcribe_episode(name: str, config: AppConfig, log: LogFn) -> dict:
    ep_dir = display_episode_dir(name)
    wav = original_wav(ep_dir)
    if not wav.exists():
        raise FileNotFoundError(wav)
    args = [
        config.whisperx_python,
        str(ROOT / "whisperx" / "whisperx" / "__main__.py"),
        str(wav),
        "--model",
        config.whisperx_model,
        "--device",
        config.whisperx_device,
        "--compute_type",
        config.whisperx_compute_type,
        "--batch_size",
        str(config.whisperx_batch_size),
        "--segment_resolution",
        "sentence",
        "--output_format",
        "json",
        "--output_dir",
        str(ep_dir),
        "--print_progress",
        "True",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "whisperx")
    run_command(args, log, cwd=ROOT, env=env)
    candidate = ep_dir / f"{wav.stem}.json"
    if candidate.exists() and candidate != transcript_json(ep_dir):
        candidate.replace(transcript_json(ep_dir))
    if not transcript_json(ep_dir).exists():
        raise RuntimeError("WhisperX did not produce original.json.")
    log(f"Transcript ready: {rel(transcript_json(ep_dir))}.")
    cut_sentences(name, config, log)
    return get_episode(name)


def cut_sentences(name: str, config: AppConfig, log: LogFn) -> dict:
    ep_dir = display_episode_dir(name)
    wav = original_wav(ep_dir)
    out_dir = sentences_dir(ep_dir)
    out_dir.mkdir(exist_ok=True)
    for sentence in parse_sentences(ep_dir):
        out = out_dir / f"sentence_{sentence.index:02d}.wav"
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
    log("Sentence WAV files are ready.")
    return get_episode(name)


def generate_selected(name: str, ids: Iterable[int], config: AppConfig, log: LogFn) -> dict:
    ep_dir = display_episode_dir(name)
    selected = {int(i) for i in ids}
    if not selected:
        raise ValueError("Select at least one sentence.")
    if not parse_sentences(ep_dir):
        raise RuntimeError("Transcript is missing. Run transcription first.")
    cut_sentences(name, config, log)
    for sentence in parse_sentences(ep_dir):
        if sentence.index not in selected:
            continue
        prompt = sentences_dir(ep_dir) / f"sentence_{sentence.index:02d}.wav"
        raw = generated_dir(ep_dir) / f"sentence_{sentence.index:02d}_slowdown.wav"
        args = [
            config.cosyvoice_python,
            str(Path(__file__).resolve().parent / "cosyvoice_generate.py"),
            "--model-dir",
            config.cosyvoice_model_dir,
            "--text",
            sentence.text,
            "--prompt-text",
            f"You are a helpful assistant.<|endofprompt|>{sentence.text}",
            "--prompt-wav",
            str(prompt),
            "--output",
            str(raw),
            "--speed",
            str(config.speed),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "CosyVoice")
        run_command(args, log, cwd=ROOT / "CosyVoice", env=env)
        trim_and_match_sentence(ep_dir, sentence.index, config, log)
    return get_episode(name)


def measure_mean_volume(path: Path, config: AppConfig, log: LogFn) -> float:
    proc = subprocess.run(
        [config.ffmpeg, "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if "mean_volume:" in line:
            match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", line)
            if match:
                value = float(match.group(1))
                log(f"{path.name} mean volume: {value:.1f} dB")
                return value
    raise RuntimeError(f"Could not measure mean volume for {path}")


def trim_and_match_sentence(ep_dir: Path, index: int, config: AppConfig, log: LogFn) -> Path:
    source_sentence = sentences_dir(ep_dir) / f"sentence_{index:02d}.wav"
    raw = generated_dir(ep_dir) / f"sentence_{index:02d}_slowdown.wav"
    pcm = generated_dir(ep_dir) / f"sentence_{index:02d}_slowdown_pcm.wav"
    trimmed = generated_dir(ep_dir) / f"sentence_{index:02d}_slowdown_trimmed.wav"
    matched = generated_dir(ep_dir) / f"sentence_{index:02d}_slowdown_matched.wav"
    if not source_sentence.exists():
        raise FileNotFoundError(
            f"Sentence audio is missing for sentence {index:02d}: {rel(source_sentence)}. "
            "Run Transcribe first so sentence clips are cut."
        )
    if not raw.exists():
        raise FileNotFoundError(
            f"Generated replacement is missing for sentence {index:02d}: {rel(raw)}. "
            "Run Generate Selected before splicing."
        )
    ffmpeg(config, ["-i", str(raw), "-c:a", "pcm_s16le", str(pcm)], log)
    trim_wav_edges(pcm, trimmed, config.silence_threshold_db, config.min_sound, config.min_silence, log)
    original_mean = measure_mean_volume(source_sentence, config, log)
    generated_mean = measure_mean_volume(trimmed, config, log)
    gain = max(-config.gain_clamp_db, min(config.gain_clamp_db, original_mean - generated_mean))
    log(f"Applying {gain:+.1f} dB gain to sentence {index:02d}.")
    ffmpeg(config, ["-i", str(trimmed), "-af", f"volume={gain:.3f}dB", "-c:a", "pcm_s16le", str(matched)], log)
    shutil.copy2(matched, trimmed)
    return trimmed


def trim_wav_edges(
    source: Path,
    dest: Path,
    threshold_db: float,
    min_sound: float,
    min_silence: float,
    log: LogFn,
) -> None:
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(params.nframes)
    if params.sampwidth != 2:
        raise RuntimeError(f"Expected 16-bit PCM WAV for trimming: {source}")
    frame_width = params.nchannels * params.sampwidth
    threshold = 32767 * math.pow(10.0, threshold_db / 20.0)
    total_frames = params.nframes
    window_frames = max(1, int(params.framerate * 0.01))
    min_sound_frames = max(1, int(params.framerate * min_sound))
    min_silence_frames = max(1, int(params.framerate * min_silence))

    def window_rms(start_frame: int, end_frame: int) -> float:
        start = start_frame * frame_width
        end = end_frame * frame_width
        chunk = frames[start:end]
        if not chunk:
            return 0.0
        sample_count = len(chunk) // params.sampwidth
        samples = struct.unpack("<" + "h" * sample_count, chunk)
        return math.sqrt(sum(sample * sample for sample in samples) / max(1, sample_count))

    def window_is_sound(start_frame: int, end_frame: int) -> bool:
        return window_rms(start_frame, end_frame) > threshold

    def has_sustained_sound(start_frame: int, step: int) -> bool:
        end_frame = min(total_frames, start_frame + min_sound_frames)
        if step < 0:
            end_frame = start_frame
            start_frame = max(0, end_frame - min_sound_frames)
        sound_frames = 0
        frame = start_frame
        while frame < end_frame:
            next_frame = min(end_frame, frame + window_frames)
            if window_is_sound(frame, next_frame):
                sound_frames += next_frame - frame
            frame = next_frame
        return sound_frames >= min_sound_frames

    first = 0
    while first < total_frames:
        end = min(total_frames, first + window_frames)
        if window_is_sound(first, end) and has_sustained_sound(first, 1):
            break
        first = end
    last = total_frames - 1
    while last >= first:
        start = max(first, last - window_frames + 1)
        if window_is_sound(start, last + 1) and has_sustained_sound(last + 1, -1):
            break
        last = start - 1

    while first > 0:
        prev = max(0, first - min_silence_frames)
        if window_rms(prev, first) <= threshold:
            break
        first = prev

    while last < total_frames - 1:
        next_last = min(total_frames - 1, last + min_silence_frames)
        if window_rms(last + 1, next_last + 1) <= threshold:
            break
        last = next_last

    if first >= total_frames or last < first:
        first, last = 0, total_frames - 1
    start_byte = first * frame_width
    end_byte = (last + 1) * frame_width
    with wave.open(str(dest), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames[start_byte:end_byte])
    original_duration = total_frames / params.framerate
    trimmed_duration = (last - first + 1) / params.framerate
    log(f"Trimmed {source.name}: {original_duration:.3f}s -> {trimmed_duration:.3f}s.")


def selected_output_name(ids: list[int]) -> str:
    joined = "_and_".join(f"{i:02d}" for i in ids)
    return f"original_with_sentence_{joined}_slowdown_crossfade.wav"


def splice_episode(name: str, ids: Iterable[int], config: AppConfig, log: LogFn) -> dict:
    ep_dir = display_episode_dir(name)
    selected = sorted({int(i) for i in ids})
    if not selected:
        raise ValueError("Select at least one sentence.")
    sentences = parse_sentences(ep_dir)
    by_id = {s.index: s for s in sentences}
    for idx in selected:
        if idx not in by_id:
            raise ValueError(f"Sentence {idx} is not in the transcript.")
        raw = generated_dir(ep_dir) / f"sentence_{idx:02d}_slowdown.wav"
        if not raw.exists():
            raise FileNotFoundError(
                f"Generated replacement is missing for sentence {idx:02d}: {rel(raw)}. "
                "Run Generate Selected first, then Splice Final WAV."
            )
        trim_and_match_sentence(ep_dir, idx, config, log)

    wav = original_wav(ep_dir)
    out_dir = generated_dir(ep_dir)
    out_dir.mkdir(exist_ok=True)
    final = out_dir / selected_output_name(selected)
    duration = ffprobe_duration(config, wav)
    xfade = max(0.0, float(config.crossfade))

    with tempfile.TemporaryDirectory(prefix="prosody_splice_") as temp_name:
        temp = Path(temp_name)
        clips: list[Path] = []
        cursor = 0.0
        for idx in selected:
            sentence = by_id[idx]
            prev_end = by_id[idx - 1].end if idx - 1 in by_id else sentence.start
            next_start = by_id[idx + 1].start if idx + 1 in by_id else sentence.end
            if idx == 1 and cursor == 0.0:
                left_end = cursor
            else:
                left_end = min(duration, max(cursor, prev_end + xfade))
            if idx == len(sentences):
                right_start = duration
            else:
                right_start = min(duration, max(0.0, next_start - xfade))
            if left_end > cursor + 0.005:
                clip = temp / f"clip_{len(clips):03d}_original.wav"
                extract_original_span(wav, cursor, left_end, clip, config, log)
                clips.append(clip)
            replacement = generated_dir(ep_dir) / f"sentence_{idx:02d}_slowdown_trimmed.wav"
            clips.append(replacement)
            cursor = max(cursor, right_start)
        if cursor < duration - 0.005:
            clip = temp / f"clip_{len(clips):03d}_original.wav"
            extract_original_span(wav, cursor, duration, clip, config, log)
            clips.append(clip)
        joined = join_with_crossfades(clips, final, xfade, config, log, temp)
        log(f"Final WAV ready: {rel(joined)}")
    return get_episode(name)


def extract_original_span(source: Path, start: float, end: float, out: Path, config: AppConfig, log: LogFn) -> None:
    duration = max(0.01, end - start)
    ffmpeg(
        config,
        ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source), "-c:a", "pcm_s16le", str(out)],
        log,
    )


def join_with_crossfades(
    clips: list[Path],
    final: Path,
    xfade: float,
    config: AppConfig,
    log: LogFn,
    temp: Path,
) -> Path:
    if not clips:
        raise RuntimeError("No clips to join.")
    if len(clips) == 1:
        shutil.copy2(clips[0], final)
        return final
    current = clips[0]
    for i, clip in enumerate(clips[1:], 1):
        out = final if i == len(clips) - 1 else temp / f"join_{i:03d}.wav"
        duration_a = ffprobe_duration(config, current)
        duration_b = ffprobe_duration(config, clip)
        effective = min(xfade, max(0.01, duration_a / 4), max(0.01, duration_b / 4))
        ffmpeg(
            config,
            [
                "-i",
                str(current),
                "-i",
                str(clip),
                "-filter_complex",
                f"[0:a][1:a]acrossfade=d={effective:.3f}:c1=tri:c2=tri[a]",
                "-map",
                "[a]",
                "-c:a",
                "pcm_s16le",
                str(out),
            ],
            log,
        )
        current = out
    return final


def run_all(name: str, ids: Iterable[int], config: AppConfig, log: LogFn) -> dict:
    if not transcript_json(display_episode_dir(name)).exists():
        transcribe_episode(name, config, log)
    else:
        cut_sentences(name, config, log)
    generate_selected(name, ids, config, log)
    return splice_episode(name, ids, config, log)
