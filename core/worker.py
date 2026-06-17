from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .communication import receive, send


@dataclass
class SGDOptimizer:
    learning_rate: float = 0.01


@dataclass
class Worker:
    rank: int
    model: Any
    data: np.ndarray
    targets: np.ndarray
    optimizer: SGDOptimizer
    batch_size: int
    steps: int
    to_server: Any
    inbox: Any
    metrics_queue: Any | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        self.model = copy.deepcopy(self.model)
        self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)
        if len(self.data) == 0:
            raise ValueError("Worker needs at least one training sample")

    def _next_batch_indices(self, rng: np.random.Generator) -> np.ndarray:
        replace = len(self.data) < self.batch_size
        return rng.choice(len(self.data), size=self.batch_size, replace=replace)

    def run(self) -> None:
        init_message = receive(self.inbox)
        if init_message.get("type") != "weights":
            raise RuntimeError("Worker expected initial weights from the parameter server")
        self.model.load_state_dict(init_message["state"])

        rng = np.random.default_rng(self.seed + self.rank)
        for step in range(self.steps):
            batch_indices = self._next_batch_indices(rng)
            batch_inputs = self.data[batch_indices]
            batch_targets = self.targets[batch_indices]
            loss, gradients = self.model.loss_and_gradients(batch_inputs, batch_targets)

            send(
                self.to_server,
                {
                    "type": "grad",
                    "rank": self.rank,
                    "step": step,
                    "loss": float(loss),
                    "gradients": gradients,
                    "samples": int(len(batch_inputs)),
                },
            )

            response = receive(self.inbox)
            message_type = response.get("type")
            if message_type == "stop":
                break
            if message_type != "weights":
                raise RuntimeError(f"Worker {self.rank} received unexpected message: {message_type!r}")
            self.model.load_state_dict(response["state"])

            if self.metrics_queue is not None:
                self.metrics_queue.put(
                    {
                        "rank": self.rank,
                        "step": step + 1,
                        "loss": float(loss),
                    }
                )

        send(self.to_server, {"type": "done", "rank": self.rank})


def run_worker(worker: Worker) -> None:
    try:
        worker.run()
    except Exception as exc:  # pragma: no cover - multiprocessing error path
        try:
            worker.to_server.put(
                {
                    "type": "error",
                    "rank": worker.rank,
                    "error": repr(exc),
                }
            )
        finally:
            raise


def run_worker_from_config(
    *,
    model_class: type,
    model_config: dict[str, Any],
    model_state: dict[str, np.ndarray],
    rank: int,
    data: np.ndarray,
    targets: np.ndarray,
    optimizer: SGDOptimizer,
    batch_size: int,
    steps: int,
    to_server: Any,
    inbox: Any,
    metrics_queue: Any | None = None,
    seed: int = 0,
) -> None:
    model = model_class(**model_config)
    model.load_state_dict(model_state)
    worker = Worker(
        rank=rank,
        model=model,
        data=data,
        targets=targets,
        optimizer=optimizer,
        batch_size=batch_size,
        steps=steps,
        to_server=to_server,
        inbox=inbox,
        metrics_queue=metrics_queue,
        seed=seed,
    )
    run_worker(worker)
