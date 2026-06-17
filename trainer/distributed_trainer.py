from __future__ import annotations

import copy
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.checkpoint import CheckpointManager
from ..core.communication import CommunicationHub
from ..core.parameter_server import ParameterServer
from ..core.worker import SGDOptimizer, run_worker_from_config


@dataclass
class DistributedTrainer:
    model: Any
    workers: int = 4
    learning_rate: float = 0.01
    batch_size: int = 32
    steps_per_worker: int = 32
    seed: int = 0
    inputs: np.ndarray | None = None
    targets: np.ndarray | None = None
    checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        self.model = copy.deepcopy(self.model)
        self.history: list[dict[str, Any]] = []

    def _resolve_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        if self.inputs is not None and self.targets is not None:
            return np.asarray(self.inputs), np.asarray(self.targets)
        if hasattr(self.model, "default_training_data"):
            sample_count = max(self.workers * self.batch_size * self.steps_per_worker, 256)
            return self.model.default_training_data(
                num_samples=sample_count,
                seed=self.seed,
                vocab_size=getattr(self.model, "vocab_size", 32),
                context_length=getattr(self.model, "context_length", 8),
            )
        raise ValueError("DistributedTrainer needs data or a model that provides default_training_data()")

    def _split_dataset(self, inputs: np.ndarray, targets: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        indices = np.array_split(np.arange(len(inputs)), self.workers)
        shards = []
        for shard_indices in indices:
            if len(shard_indices) == 0:
                shards.append((inputs, targets))
            else:
                shards.append((inputs[shard_indices], targets[shard_indices]))
        return shards

    def train(self) -> dict[str, Any]:
        ctx = mp.get_context("spawn")
        inputs, targets = self._resolve_dataset()
        shards = self._split_dataset(inputs, targets)

        hub = CommunicationHub(self.workers, ctx=ctx)
        metrics_queue = ctx.Queue()
        server = ParameterServer(
            model=copy.deepcopy(self.model),
            learning_rate=self.learning_rate,
            num_workers=self.workers,
            grad_queue=hub.to_server,
            worker_queues=hub.to_workers,
            metrics_queue=metrics_queue,
            max_steps=self.steps_per_worker,
        )

        processes: list[mp.Process] = []
        model_state = self.model.state_dict()
        model_config = self.model.config_dict() if hasattr(self.model, "config_dict") else {}
        for rank, (worker_inputs, worker_targets) in enumerate(shards):
            process = ctx.Process(
                target=run_worker_from_config,
                kwargs={
                    "model_class": self.model.__class__,
                    "model_config": model_config,
                    "model_state": model_state,
                    "rank": rank,
                    "data": worker_inputs,
                    "targets": worker_targets,
                    "optimizer": SGDOptimizer(learning_rate=self.learning_rate),
                    "batch_size": self.batch_size,
                    "steps": self.steps_per_worker,
                    "to_server": hub.to_server,
                    "inbox": hub.worker_queue(rank),
                    "metrics_queue": metrics_queue,
                    "seed": self.seed,
                },
                name=f"worker-{rank}",
            )
            process.start()
            processes.append(process)

        final_state = server.run()

        for process in processes:
            process.join()
            if process.exitcode not in (0, None):
                raise RuntimeError(f"{process.name} exited with code {process.exitcode}")

        self.model.load_state_dict(final_state)
        self.history = server.history

        if self.checkpoint_path is not None:
            checkpoint = CheckpointManager(self.checkpoint_path)
            checkpoint.save_training_state(
                model_state=self.model.state_dict(),
                optimizer_state={"learning_rate": self.learning_rate},
                epoch=self.steps_per_worker,
                history=self.history,
                filename="distributed_checkpoint.pkl",
            )

        return {
            "final_loss": self.history[-1]["loss"] if self.history else None,
            "steps": self.steps_per_worker,
            "workers": self.workers,
            "history": self.history,
        }

    fit = train
