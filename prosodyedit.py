"""Compare fixed audio editing with CosyVoice emphasis resynthesis."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import urllib.request
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("config.json")
MFA_MODEL = "english_us_arpa"
COSYVOICE_MODEL = "pretrained_models/Fun-CosyVoice3-0.5B"
EDIT_SPEED = 0.95
EDIT_GAIN_DB = 2
SEED = 1986

PLANNING_PROMPT = """Select words and short phrases to emphasize in an educational recording.

Input tokens use `[S<sentence_id>] <word_id>:<word>`. Select at most eight
sentences and at most two phrases per selected sentence. Each phrase must be
one to five consecutive words. Prefer key concepts, causal relationships,
contrasts, and conclusions. Distribute selections across the passage and avoid
whole clauses or sentences. Return only the selected word IDs."""


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def normalize(text):
    text = unicodedata.normalize("NFKD", text).casefold()
    return "".join(character for character in text if character.isalnum())


def script_words(text):
    words = []
    sentence = 0
    for token in text.split():
        if normalize(token):
            words.append({"text": token, "sentence": sentence})
        if re.search(r"[.!?][\"'’”)]*$", token):
            sentence += 1
    return words


def parse_textgrid(path):
    from praatio import textgrid

    grid = textgrid.openTextgrid(str(path), includeEmptyIntervals=False)
    tier_name = next(name for name in grid.tierNames if name.casefold().endswith("words"))
    return [
        (entry.label, float(entry.start), float(entry.end))
        for entry in grid.getTier(tier_name).entries
        if normalize(entry.label) not in {"", "sil", "sp", "spn", "eps"}
    ]


def map_alignment(tokens, intervals):
    words = []
    interval_index = 0
    for word_id, token in enumerate(tokens):
        expected = normalize(token["text"])
        first = interval_index
        spoken = ""
        while interval_index < len(intervals) and len(spoken) < len(expected):
            spoken += normalize(intervals[interval_index][0])
            interval_index += 1
        if spoken != expected:
            raise ValueError(f"MFA mismatch at {token['text']!r}: heard {spoken!r}")
        words.append(
            {
                "id": word_id,
                "text": token["text"],
                "start": intervals[first][1],
                "end": intervals[interval_index - 1][2],
                "sentence": token["sentence"],
            }
        )
    if interval_index != len(intervals):
        raise ValueError("MFA alignment contains words missing from script.txt")
    return words


def alignment(input_dir):
    path = input_dir / "alignment.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data["words"]

    audio = input_dir / "audio.wav"
    script = input_dir / "script.txt"
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        corpus = workspace / "corpus"
        output = workspace / "output"
        corpus.mkdir()
        output.mkdir()
        shutil.copyfile(audio, corpus / "recording.wav")
        shutil.copyfile(script, corpus / "recording.txt")
        subprocess.run(
            [
                "mfa", "align", str(corpus), MFA_MODEL, MFA_MODEL, str(output),
                "--g2p_model_path", MFA_MODEL, "--output_format", "long_textgrid",
                "--single_speaker", "--clean", "--overwrite",
                "--temporary_directory", str(workspace / "mfa"),
            ],
            check=True,
        )
        textgrid_path = next(output.rglob("recording.TextGrid"))
        words = map_alignment(script_words(script.read_text()), parse_textgrid(textgrid_path))
    write_json(path, {"words": words})
    return words


def planning_transcript(words):
    parts = []
    sentence = None
    for word in words:
        if word["sentence"] != sentence:
            sentence = word["sentence"]
            parts.append(f"\n[S{sentence}]")
        parts.append(f"{word['id']}:{word['text']}")
    return " ".join(parts).strip()


def emphasis_plan(input_dir, words, config):
    path = input_dir / "emphasis_plan.json"
    if path.exists():
        return json.loads(path.read_text())["word_ids"]

    schema = {
        "type": "object",
        "properties": {
            "word_ids": {"type": "array", "items": {"type": "integer", "minimum": 0}}
        },
        "required": ["word_ids"],
        "additionalProperties": False,
    }
    body = {
        "model": config["openai_model"],
        "messages": [
            {"role": "developer", "content": PLANNING_PROMPT},
            {"role": "user", "content": planning_transcript(words)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "emphasis_plan", "strict": True, "schema": schema},
        },
    }
    request = urllib.request.Request(
        config["openai_base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {config['openai_api_key']}",
            "Content-Type": "application/json",
            "User-Agent": "ProsodyEdit/1.0",
        },
    )
    with urllib.request.urlopen(request) as response:
        result = json.load(response)
    word_ids = json.loads(result["choices"][0]["message"]["content"])["word_ids"]
    write_json(path, {"word_ids": word_ids})
    return word_ids


def consecutive_spans(words, word_ids):
    spans = []
    for word_id in word_ids:
        word = words[word_id]
        if (
            spans
            and word_id == spans[-1]["word_ids"][-1] + 1
            and word["sentence"] == words[word_id - 1]["sentence"]
        ):
            spans[-1]["word_ids"].append(word_id)
            spans[-1]["end"] = word["end"]
        else:
            spans.append(
                {"word_ids": [word_id], "start": word["start"], "end": word["end"]}
            )
    return spans


def render_edit(input_dir, words, word_ids):
    source = input_dir / "audio.wav"
    output = input_dir / "audio.edited.wav"
    spans = consecutive_spans(words, word_ids)
    if not spans:
        shutil.copyfile(source, output)
        return output

    filters = []
    labels = []

    def add_part(expression):
        label = f"part{len(labels)}"
        filters.append(f"[0:a]{expression},asetpts=PTS-STARTPTS[{label}]")
        labels.append(label)

    cursor = 0
    for span in spans:
        if span["start"] > cursor:
            add_part(f"atrim=start={cursor}:end={span['start']}")
        add_part(
            f"atrim=start={span['start']}:end={span['end']},"
            f"atempo={EDIT_SPEED},volume={EDIT_GAIN_DB}dB"
        )
        cursor = span["end"]
    add_part(f"atrim=start={cursor}")
    filters.append(
        f"{''.join(f'[{label}]' for label in labels)}"
        f"concat=n={len(labels)}:v=0:a=1[out]"
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-filter_complex", ";".join(filters), "-map", "[out]", "-c:a", "pcm_s16le",
            str(output),
        ],
        check=True,
    )
    return output


def group_sentences(words):
    sentences = {}
    for word in words:
        sentences.setdefault(word["sentence"], []).append(word)
    return sentences


def marked_sentence(sentence, selected):
    tokens = []
    for index, word in enumerate(sentence):
        token = word["text"]
        if word["id"] in selected and (index == 0 or sentence[index - 1]["id"] not in selected):
            token = "<strong>" + token
        if word["id"] in selected and (
            index + 1 == len(sentence) or sentence[index + 1]["id"] not in selected
        ):
            token = re.sub(r"([.,!?;:]+)$", r"</strong>\1", token)
            if not token.endswith(tuple(".,!?;:")):
                token += "</strong>"
        tokens.append(token)
    return " ".join(tokens)


def selected_phrases(sentence, selected):
    phrases = []
    current = []
    for word in sentence:
        if word["id"] in selected:
            current.append(word["text"].strip(".,!?;:"))
        elif current:
            phrases.append(" ".join(current))
            current = []
    if current:
        phrases.append(" ".join(current))
    return phrases


def load_cosyvoice(root):
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/prosodyedit-numba")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path[:0] = [str(root), str(root / "third_party/Matcha-TTS")]
    import wetext

    # text_frontend=False makes the downloadable WeText normalizer unnecessary.
    wetext.Normalizer = lambda **_: None
    from cosyvoice.cli.cosyvoice import AutoModel

    return AutoModel(model_dir=str(root / COSYVOICE_MODEL))


def render_cosyvoice(input_dir, words, word_ids, root):
    source = input_dir / "audio.wav"
    output = input_dir / "audio.cosyvoice.wav"
    if not word_ids:
        shutil.copyfile(source, output)
        return output

    model = load_cosyvoice(root)
    import torch
    import torchaudio
    from cosyvoice.utils.common import set_all_random_seed
    sentences = group_sentences(words)
    selected = set(word_ids)
    sentence_ids = sorted({words[word_id]["sentence"] for word_id in word_ids})
    source_audio, sample_rate = torchaudio.load(str(source))

    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        replacements = []
        for sentence_id in sentence_ids:
            sentence = sentences[sentence_id]
            start, end = sentence[0]["start"], sentence[-1]["end"]
            reference = workspace / f"reference-{sentence_id}.wav"
            reference_audio = source_audio[:, round(start * sample_rate):round(end * sample_rate)]
            reference_audio = torchaudio.functional.resample(
                reference_audio, sample_rate, model.sample_rate
            )
            torchaudio.save(str(reference), reference_audio, model.sample_rate)
            phrases = selected_phrases(sentence, selected)
            instruction = (
                "You are a helpful assistant. Please strongly emphasize the phrases "
                + " and ".join(f'\"{phrase}\"' for phrase in phrases)
                + ", making them slower, louder, and more prominent than the surrounding words."
                "<|endofprompt|>"
            )
            set_all_random_seed(SEED)
            [result] = model.inference_instruct2(
                marked_sentence(sentence, selected), instruction, str(reference),
                stream=False, text_frontend=False,
            )
            generated_audio = torchaudio.functional.resample(
                result["tts_speech"], model.sample_rate, sample_rate
            )
            replacements.append(
                (round(start * sample_rate), round(end * sample_rate), generated_audio)
            )

        parts = []
        cursor = 0
        for start, end, generated_audio in replacements:
            parts.extend((source_audio[:, cursor:start], generated_audio))
            cursor = end
        parts.append(source_audio[:, cursor:])
        torchaudio.save(
            str(output), torch.cat(parts, dim=1), sample_rate,
            encoding="PCM_S", bits_per_sample=16,
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    config = json.loads(CONFIG_PATH.read_text())
    words = alignment(input_dir)
    word_ids = emphasis_plan(input_dir, words, config)
    print("Emphasis:", " ".join(words[word_id]["text"] for word_id in word_ids))
    print(render_edit(input_dir, words, word_ids))
    print(render_cosyvoice(input_dir, words, word_ids, Path(config["cosyvoice_root"]).expanduser()))


if __name__ == "__main__":
    main()
