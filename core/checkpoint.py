from __future__ import annotations

import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CheckpointManager:
    directory: str | Path

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, payload: Any) -> Path:
        path = self.directory / filename
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=self.directory, suffix=".tmp") as temp_file:
            pickle.dump(payload, temp_file)
            temp_path = Path(temp_file.name)
        temp_path.replace(path)
        return path

    def load(self, filename: str) -> Any:
        path = self.directory / filename
        with path.open("rb") as file_handle:
            return pickle.load(file_handle)

    def exists(self, filename: str) -> bool:
        return (self.directory / filename).exists()

    def save_training_state(
        self,
        *,
        model_state: Any,
        optimizer_state: Any,
        epoch: int,
        step: int | None = None,
        history: list[dict[str, Any]] | None = None,
        filename: str = "checkpoint.pkl",
        extras: dict[str, Any] | None = None,
    ) -> Path:
        payload = {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "epoch": epoch,
            "step": step if step is not None else epoch,
            "history": history or [],
            "extras": extras or {},
        }
        return self.save(filename, payload)

    def load_training_state(self, filename: str = "checkpoint.pkl") -> dict[str, Any]:
        try:
            payload = self.load(filename)
        except EOFError as exc:
            raise RuntimeError(f"Checkpoint {filename!r} is incomplete or corrupted") from exc
        if not isinstance(payload, dict):
            raise TypeError("Checkpoint payload must be a dictionary")
        return payload
