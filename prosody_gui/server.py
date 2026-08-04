from __future__ import annotations

import json
import logging
import mimetypes
import tempfile
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import STATIC_DIR, load_config
from .jobs import Job, JobManager
from . import pipeline


LOGGER = logging.getLogger("prosodyedit.server")
CONFIG = load_config()
WORKSPACE = tempfile.TemporaryDirectory(prefix="prosodyedit_workspace_")
pipeline.configure_output_dir(WORKSPACE.name)
JOBS = JobManager()
CURRENT_EPISODE: str | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "ProsodyGUI/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/":
            return self.send_file(STATIC_DIR / "index.html")
        if path.startswith("/static/"):
            return self.send_file(STATIC_DIR / path.removeprefix("/static/"))
        if path.startswith("/media/"):
            return self.send_media(path.removeprefix("/media/"))
        if path == "/api/current":
            return self.send_json({"episode": current_episode(), "config": public_config()})
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            job = JOBS.get(job_id)
            if not job:
                return self.send_error_json(HTTPStatus.NOT_FOUND, "Job not found.")
            return self.send_json(job_payload(job))
        return self.send_error_json(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/upload-wav":
                return self.upload_wav()
            payload = self.read_json()
            if JOBS.busy():
                raise ValueError("Wait for the current operation to finish.")
            if path == "/api/transcribe":
                name = require_name(payload)
                job = JOBS.start("transcribe", lambda j: pipeline.transcribe_episode(name, CONFIG, j.log))
                return self.send_json(job_payload(job))
            if path == "/api/slow-words":
                name = require_name(payload)
                raw_edits = payload.get("edits")
                if isinstance(raw_edits, list):
                    if not all(isinstance(edit, dict) for edit in raw_edits):
                        raise ValueError("Each effect group must be an object.")
                    job = JOBS.start(
                        "slow-words",
                        lambda j: pipeline.edit_word_groups(name, raw_edits, CONFIG, j.log),
                    )
                    return self.send_json(job_payload(job))
                raw_ids = payload.get("word_ids")
                if not isinstance(raw_ids, list):
                    raise ValueError("word_ids must be a list.")
                try:
                    word_ids = [int(value) for value in raw_ids]
                    speed = float(payload.get("speed", 0.95))
                    gain_db = float(payload.get("gain_db", 0.0))
                except (TypeError, ValueError) as exc:
                    raise ValueError("Invalid word selection or edit settings.") from exc
                job = JOBS.start(
                    "slow-words",
                    lambda j: pipeline.slow_words(
                        name,
                        word_ids,
                        speed,
                        CONFIG,
                        j.log,
                        gain_db,
                    ),
                )
                return self.send_json(job_payload(job))
            if path == "/api/ai-effects":
                name = require_name(payload)
                job = JOBS.start("ai-effects", lambda j: pipeline.ai_auto_edit(name, CONFIG, j.log))
                return self.send_json(job_payload(job))
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Not found.")
        except Exception as exc:  # noqa: BLE001 - API error response
            LOGGER.exception("Request failed: %s %s", self.command, path)
            return self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status)

    def send_file(self, path: Path) -> None:
        path = path.resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.exists() or not path.is_file():
            return self.send_error_json(HTTPStatus.NOT_FOUND, "File not found.")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_media(self, relative: str) -> None:
        rel = urllib.parse.unquote(relative)
        try:
            path = pipeline.resolve_media(rel)
        except ValueError:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Media not found.")
        if not path.exists() or not path.is_file():
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Media not found.")
        content_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 256):
                self.wfile.write(chunk)

    def upload_wav(self) -> None:
        if JOBS.busy():
            raise ValueError("Wait for the current operation to finish.")
        filename = decode_upload_filename(self.headers.get("X-Filename", ""))
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("The uploaded WAV is empty.")
        source_label = Path(filename).stem
        name = pipeline.safe_name(source_label)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="prosody_upload_", suffix=".wav", delete=False) as handle:
                temp_path = Path(handle.name)
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("The WAV upload ended before the full file was received.")
                    handle.write(chunk)
                    remaining -= len(chunk)
        except Exception:
            if temp_path:
                temp_path.unlink(missing_ok=True)
            raise

        def import_upload(job: Job) -> dict:
            global CURRENT_EPISODE
            assert temp_path is not None
            try:
                CURRENT_EPISODE = None
                pipeline.reset_workspace()
                result = pipeline.import_wav(
                    temp_path,
                    name,
                    job.log,
                    display_name=source_label,
                )
                CURRENT_EPISODE = result["name"]
                return result
            finally:
                temp_path.unlink(missing_ok=True)

        try:
            job = JOBS.start("upload-wav", import_upload)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        self.send_json(job_payload(job), HTTPStatus.ACCEPTED)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def require_name(payload: dict[str, Any]) -> str:
    name = str(payload.get("episode", "")).strip()
    if not name:
        raise ValueError("No uploaded audio is selected.")
    return name


def decode_upload_filename(value: str) -> str:
    filename = Path(urllib.parse.unquote(value).replace("\\", "/")).name
    if not filename or Path(filename).suffix.lower() != ".wav":
        raise ValueError("Choose a WAV file to upload.")
    return filename


def current_episode() -> dict[str, Any] | None:
    if not CURRENT_EPISODE:
        return None
    ep_dir = pipeline.episode_dir(CURRENT_EPISODE)
    if not ep_dir.exists():
        return None
    return pipeline.get_episode(CURRENT_EPISODE)


def job_payload(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "action": job.action,
        "status": job.status,
        "result": job.result,
        "error": job.error,
    }


def public_config() -> dict[str, Any]:
    return {
        "model": CONFIG.qwen_asr_model,
        "align_model": CONFIG.qwen_aligner_model,
        "language": CONFIG.qwen_language,
        "device": CONFIG.qwen_device,
    }


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ProsodyEdit GUI: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
