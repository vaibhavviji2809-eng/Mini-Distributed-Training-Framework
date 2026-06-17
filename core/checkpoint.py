from __future__ import annotations

import pickle
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
        with path.open("wb") as file_handle:
            pickle.dump(payload, file_handle)
        return path

    def load(self, filename: str) -> Any:
        path = self.directory / filename
        with path.open("rb") as file_handle:
            return pickle.load(file_handle)

    def save_training_state(
        self,
        *,
        model_state: Any,
        optimizer_state: Any,
        epoch: int,
        history: list[dict[str, Any]] | None = None,
        filename: str = "checkpoint.pkl",
        extras: dict[str, Any] | None = None,
    ) -> Path:
        payload = {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "epoch": epoch,
            "history": history or [],
            "extras": extras or {},
        }
        return self.save(filename, payload)

    def load_training_state(self, filename: str = "checkpoint.pkl") -> dict[str, Any]:
        payload = self.load(filename)
        if not isinstance(payload, dict):
            raise TypeError("Checkpoint payload must be a dictionary")
        return payload

