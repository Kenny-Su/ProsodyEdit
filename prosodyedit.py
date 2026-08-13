"""ProsodyEdit — ask an LLM which words in a recording deserve emphasis, then
render those words slower and louder with FFmpeg.

    python3 prosodyedit.py path/to/input_dir

The input directory holds `audio.wav` and `timestamp.json`, a forced aligner's
flat list of {"text", "start_time", "end_time"} words in timeline order.
ProsodyEdit writes `audio.edited.wav` and `edit_log.json` beside them and never
touches the originals.
"""

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

SPEED = 0.8  # emphasized words are stretched to this fraction of their tempo
CONTRAST_DB = 5.0  # ...and stand this much above the rest of the recording

# The contrast is applied by ducking everything else rather than by boosting the
# emphasized words: source recordings are usually mastered near 0 dBFS, so a
# boost would hard-clip the very words it is meant to highlight.

CONFIG_PATH = Path(__file__).parent / "config.json"

SYSTEM_PROMPT = """You are directing prosody edits for an educational spoken-word \
recording. You are given a transcript as "<word_id>:<word text>" tokens in timeline \
order. The transcript comes from a forced aligner and carries no punctuation or \
capitalization, so infer sentence and phrase boundaries from the wording itself. \
Decide which words deserve emphasis to help listeners notice information that is \
important for understanding the passage. Prioritize key concepts, causal or logical \
relationships, important contrasts or qualifications, and conclusions or \
implications. Do not select words merely because they are names, numbers, or \
technical terms.

Use emphasis sparingly. Most words should be left alone; only mark words where it \
clearly helps direct attention to important information. Emphasis is binary: a word \
is either emphasized or not, so only mark the words that truly deserve it.

Consecutive ids are rendered as one continuous stretch of emphasized speech. \
Prefer short contiguous phrases when several adjacent words together express an \
important idea. Do not split a meaningful phrase merely to exclude articles, \
prepositions, or auxiliary verbs. Keep emphasized stretches short and avoid whole \
clauses or sentences.

Respond with strict JSON only, no prose, matching exactly:
{"word_ids": [int, ...]}

Return {"word_ids": []} if no emphasis is warranted."""


def select_emphasis(words):
    """Ask the model which word ids deserve emphasis, in timeline order."""
    config = json.loads(CONFIG_PATH.read_text())
    transcript = " ".join(f"{i}:{word['text']}" for i, word in enumerate(words))
    request = urllib.request.Request(
        config["openai_base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(
            {
                "model": config["openai_model"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                "response_format": {"type": "json_object"},
                # Qwen3 models think by default; the reasoning pass can outrun
                # the read timeout (and burns tokens) on a long transcript.
                "enable_thinking": False,
            }
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['openai_api_key']}",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        reply = json.load(response)["choices"][0]["message"]["content"]
    chosen = set(json.loads(reply)["word_ids"])
    return sorted(chosen & set(range(len(words))))  # ignore ids the model invents


def spans(words, chosen):
    """Merge runs of consecutive emphasized words into one span each."""
    merged = []
    for position, index in enumerate(chosen):
        word = words[index]
        if position and index == chosen[position - 1] + 1:
            merged[-1][1] = word["end_time"]
        else:
            merged.append([word["start_time"], word["end_time"]])
    return merged


def render(source, output, emphasis):
    """Slow each span, ducking the audio between spans to make them stand out."""
    filters, labels = [], []

    def part(expression):
        labels.append(f"[p{len(labels)}]")
        filters.append(f"[0:a]{expression},asetpts=PTS-STARTPTS{labels[-1]}")

    cursor = 0.0
    for start, end in emphasis:
        if start > cursor:
            part(f"atrim=start={cursor:.3f}:end={start:.3f},volume=-{CONTRAST_DB}dB")
        part(f"atrim=start={start:.3f}:end={end:.3f},atempo={SPEED}")
        cursor = end
    part(f"atrim=start={cursor:.3f},volume=-{CONTRAST_DB}dB")
    graph = ";".join(filters + [f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]"])

    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
        + ["-filter_complex", graph, "-map", "[out]", "-c:a", "pcm_s16le", str(output)],
        check=True,
    )


def main():
    input_dir = Path(sys.argv[1])
    words = json.loads((input_dir / "timestamp.json").read_text())

    chosen = select_emphasis(words)
    render(input_dir / "audio.wav", input_dir / "audio.edited.wav", spans(words, chosen))

    log = [{**word, "emphasized": i in set(chosen)} for i, word in enumerate(words)]
    (input_dir / "edit_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))

    print(f"Emphasized {len(chosen)} of {len(words)} words:")
    print(" ".join(words[i]["text"] for i in chosen))
    print(f"Wrote {input_dir / 'audio.edited.wav'} and {input_dir / 'edit_log.json'}")


if __name__ == "__main__":
    main()
