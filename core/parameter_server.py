from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .communication import all_reduce, broadcast, receive


@dataclass
class ParameterServer:
    model: Any
    learning_rate: float
    num_workers: int
    grad_queue: Any
    worker_queues: list[Any]
    metrics_queue: Any | None = None
    max_steps: int = 1
    history: list[dict[str, Any]] = field(default_factory=list)

    def run(self) -> dict[str, Any]:
        start_time = time.perf_counter()
        initial_state = self.model.state_dict()
        broadcast(self.worker_queues, {"type": "weights", "state": initial_state, "step": 0})

        for step in range(self.max_steps):
            receive_start = time.perf_counter()
            gradients = []
            losses = []
            samples_seen = 0
            for _ in range(self.num_workers):
                message = receive(self.grad_queue)
                if message.get("type") == "error":
                    raise RuntimeError(f"Worker {message.get('rank')} failed: {message.get('error')}")
                if message.get("type") != "grad":
                    continue
                gradients.append(message["gradients"])
                losses.append(float(message["loss"]))
                samples_seen += int(message.get("samples", 0))

            if not gradients:
                raise RuntimeError("ParameterServer did not receive any gradients")
            averaged_gradients = all_reduce(gradients, op="mean")
            self.model.apply_gradients(averaged_gradients, self.learning_rate)
            current_state = self.model.state_dict()
            broadcast(self.worker_queues, {"type": "weights", "state": current_state, "step": step + 1})

            elapsed = time.perf_counter() - start_time
            communication_time = time.perf_counter() - receive_start
            step_metrics = {
                "step": step + 1,
                "loss": sum(losses) / len(losses),
                "samples_per_sec": samples_seen / elapsed if elapsed else 0.0,
                "communication_time": communication_time,
            }
            self.history.append(step_metrics)
            if self.metrics_queue is not None:
                self.metrics_queue.put(step_metrics)

        final_state = self.model.state_dict()
        broadcast(self.worker_queues, {"type": "stop", "state": final_state, "step": self.max_steps})
        return final_state
