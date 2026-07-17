from __future__ import annotations

import json
import mimetypes
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import STATIC_DIR, load_config, save_config
from .jobs import Job, JobManager
from . import pipeline


CONFIG = load_config()
pipeline.configure_output_dir(CONFIG.output_dir)
JOBS = JobManager()


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
        if path == "/api/episodes":
            return self.send_json({"episodes": pipeline.list_episodes(), "config": public_config()})
        if path.startswith("/api/episode/"):
            name = urllib.parse.unquote(path.removeprefix("/api/episode/"))
            return self.send_json(pipeline.get_episode(name))
        if path.startswith("/api/jobs/") and path.endswith("/events"):
            job_id = path.split("/")[3]
            return self.send_events(job_id)
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
            payload = self.read_json()
            if path == "/api/directories":
                return self.update_directories(payload)
            if path == "/api/import-wav":
                source = resolve_input_path(payload.get("path", ""))
                name = payload.get("name") or source.stem
                job = JOBS.start("import-wav", lambda j: pipeline.import_wav(source, name, j.log))
                return self.send_json(job_payload(job))
            if path == "/api/transcribe":
                name = require_name(payload)
                job = JOBS.start("transcribe", lambda j: pipeline.transcribe_episode(name, CONFIG, j.log))
                return self.send_json(job_payload(job))
            if path == "/api/selection":
                return self.send_json({"selected": selected_ids(payload)})
            if path == "/api/generate":
                name = require_name(payload)
                ids = selected_ids(payload)
                job = JOBS.start("generate", lambda j: pipeline.generate_selected(name, ids, CONFIG, j.log))
                return self.send_json(job_payload(job))
            if path == "/api/splice":
                name = require_name(payload)
                ids = selected_ids(payload)
                job = JOBS.start("splice", lambda j: pipeline.splice_episode(name, ids, CONFIG, j.log))
                return self.send_json(job_payload(job))
            if path == "/api/run-all":
                name = require_name(payload)
                ids = selected_ids(payload)
                job = JOBS.start("run-all", lambda j: pipeline.run_all(name, ids, CONFIG, j.log))
                return self.send_json(job_payload(job))
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Not found.")
        except Exception as exc:  # noqa: BLE001 - API error response
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

    def update_directories(self, payload: dict[str, Any]) -> None:
        input_dir = Path(str(payload.get("input_dir", ""))).expanduser().resolve()
        output_dir = Path(str(payload.get("output_dir", ""))).expanduser().resolve()
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {output_dir}")
        CONFIG.input_dir = str(input_dir)
        CONFIG.output_dir = str(pipeline.configure_output_dir(output_dir))
        save_config(CONFIG)
        self.send_json({"config": public_config(), "episodes": pipeline.list_episodes()})

    def send_events(self, job_id: str) -> None:
        job = JOBS.get(job_id)
        if not job:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Job not found.")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for line in job.logs:
            self.write_event(line)
        while True:
            item = job.events.get()
            if item is None:
                self.write_event("[[DONE]]")
                break
            self.write_event(item)

    def write_event(self, line: str) -> None:
        self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
        self.wfile.flush()

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def require_name(payload: dict[str, Any]) -> str:
    name = str(payload.get("episode", "")).strip()
    if not name:
        raise ValueError("Missing episode name.")
    return name


def selected_ids(payload: dict[str, Any]) -> list[int]:
    ids = [int(item) for item in payload.get("sentence_ids", [])]
    if not ids:
        raise ValueError("Select at least one sentence.")
    return ids


def resolve_input_path(value: Any) -> Path:
    raw = Path(str(value)).expanduser()
    source = raw if raw.is_absolute() else Path(CONFIG.input_dir) / raw
    source = source.resolve()
    input_dir = Path(CONFIG.input_dir).resolve()
    if source != input_dir and input_dir not in source.parents:
        raise ValueError("Input WAV must be inside the configured input directory.")
    return source


def job_payload(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "action": job.action,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "logs": job.logs[-200:],
    }


def public_config() -> dict[str, Any]:
    return {
        "input_dir": CONFIG.input_dir,
        "output_dir": CONFIG.output_dir,
        "speed": CONFIG.speed,
        "crossfade": CONFIG.crossfade,
        "silence_threshold_db": CONFIG.silence_threshold_db,
        "whisperx_python": CONFIG.whisperx_python,
        "cosyvoice_python": CONFIG.cosyvoice_python,
        "cosyvoice_model_dir": CONFIG.cosyvoice_model_dir,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ProsodyEdit GUI: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
