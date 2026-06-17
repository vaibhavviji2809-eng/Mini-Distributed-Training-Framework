from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingMetrics:
    sync_method: str = "parameter_server"
    compression: str = "none"
    losses: list[float] = field(default_factory=list)
    throughput: list[float] = field(default_factory=list)
    communication_time: list[float] = field(default_factory=list)
    worker_health: dict[int, str] = field(default_factory=dict)
    restarts: int = 0

    def record_step(self, step_metrics: dict[str, Any]) -> None:
        if "loss" in step_metrics:
            self.losses.append(float(step_metrics["loss"]))
        if "samples_per_sec" in step_metrics:
            self.throughput.append(float(step_metrics["samples_per_sec"]))
        if "communication_time" in step_metrics:
            self.communication_time.append(float(step_metrics["communication_time"]))

    def record_worker_health(self, rank: int, status: str) -> None:
        self.worker_health[int(rank)] = status

    def record_restart(self) -> None:
        self.restarts += 1

    def summary(self) -> dict[str, Any]:
        return {
            "sync_method": self.sync_method,
            "compression": self.compression,
            "losses": self.losses,
            "throughput": self.throughput,
            "communication_time": self.communication_time,
            "worker_health": self.worker_health,
            "restarts": self.restarts,
        }

    def write_snapshot(self, path: str | Path) -> Path:
        snapshot_path = Path(path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.summary(), indent=2, sort_keys=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=snapshot_path.parent, suffix=".tmp", encoding="utf-8") as temp_file:
            temp_file.write(payload)
            temp_path = Path(temp_file.name)
        temp_path.replace(snapshot_path)
        return snapshot_path


def read_snapshot(path: str | Path) -> dict[str, Any] | None:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        return None
    with snapshot_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)
