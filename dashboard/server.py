from __future__ import annotations

import argparse
import json
import mimetypes
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .metrics import read_snapshot

ROOT = Path(__file__).resolve().parent
SAMPLE_METRICS = ROOT / "sample_metrics.json"


def _default_metrics() -> dict[str, object]:
    snapshot = read_snapshot(SAMPLE_METRICS)
    return snapshot or {
        "sync_method": "parameter_server",
        "compression": "none",
        "losses": [],
        "throughput": [],
        "communication_time": [],
        "worker_health": {},
        "restarts": 0,
    }


def _safe_static_path(request_path: str) -> Path | None:
    relative_path = request_path.lstrip("/") or "index.html"
    candidate = (ROOT / relative_path).resolve()
    if ROOT not in candidate.parents and candidate != ROOT:
        return None
    if candidate.is_dir():
        candidate = candidate / "index.html"
    return candidate


def _load_metrics_payload(metrics_path: Path | None) -> dict[str, object]:
    if metrics_path is not None:
        snapshot = read_snapshot(metrics_path)
        if snapshot is not None:
            return snapshot
    return _default_metrics()


def _content_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


@dataclass
class DashboardConfig:
    metrics_path: Path | None = None


def create_handler(config: DashboardConfig) -> type[BaseHTTPRequestHandler]:
    class DashboardRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # pragma: no cover - quiet server
            return

        def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", _content_type(path))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed_url = urlparse(self.path)
            if parsed_url.path == "/api/metrics":
                self._send_json(_load_metrics_payload(config.metrics_path))
                return
            if parsed_url.path == "/api/health":
                self._send_json({"status": "ok"})
                return

            static_path = _safe_static_path(parsed_url.path)
            if static_path is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if static_path.name == "index.html" and not static_path.exists():
                static_path = ROOT / "index.html"
            self._send_file(static_path)

    return DashboardRequestHandler


def create_server(host: str = "127.0.0.1", port: int = 8000, metrics_path: str | Path | None = None) -> ThreadingHTTPServer:
    config = DashboardConfig(metrics_path=Path(metrics_path) if metrics_path is not None else None)
    handler = create_handler(config)
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MiniDistributed dashboard server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--metrics-file", default=None, help="Path to a live metrics JSON file")
    args = parser.parse_args()

    server = create_server(host=args.host, port=args.port, metrics_path=args.metrics_file)
    print(f"MiniDistributed dashboard serving on http://{args.host}:{server.server_address[1]}")
    if args.metrics_file:
        print(f"Live metrics: {args.metrics_file}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
