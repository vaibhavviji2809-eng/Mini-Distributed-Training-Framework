from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .communication import all_reduce, broadcast, decompress_payload, receive, ring_all_reduce


class WorkerFailure(RuntimeError):
    pass


@dataclass
class ParameterServer:
    model: Any
    learning_rate: float
    num_workers: int
    grad_queue: Any
    worker_queues: list[Any]
    metrics_queue: Any | None = None
    max_steps: int = 1
    start_step: int = 0
    sync_method: str = "parameter_server"
    compression: str = "none"
    receive_timeout: float = 2.0
    worker_processes: list[Any] | None = None
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None
    metrics_callback: Callable[[dict[str, Any]], None] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def _dead_worker(self) -> Any | None:
        if not self.worker_processes:
            return None
        for process in self.worker_processes:
            if process.exitcode not in (None, 0):
                return process
            if process.exitcode is None and not process.is_alive():
                return process
        return None

    def _collect_gradient(self) -> dict[str, Any]:
        while True:
            try:
                message = receive(self.grad_queue, timeout=self.receive_timeout)
            except TimeoutError as exc:
                failed_process = self._dead_worker()
                if failed_process is not None:
                    raise WorkerFailure(
                        f"{failed_process.name} exited with code {failed_process.exitcode}"
                    ) from exc
                continue
            if message.get("type") == "error":
                raise WorkerFailure(f"Worker {message.get('rank')} failed: {message.get('error')}")
            if message.get("type") == "grad":
                return message

    def run(self) -> dict[str, Any]:
        start_time = time.perf_counter()
        initial_state = self.model.state_dict()
        broadcast(self.worker_queues, {"type": "weights", "state": initial_state, "step": self.start_step})

        for step in range(self.max_steps):
            receive_start = time.perf_counter()
            gradients = []
            losses = []
            samples_seen = 0
            for _ in range(self.num_workers):
                message = self._collect_gradient()
                gradients.append(message["gradients"])
                losses.append(float(message["loss"]))
                samples_seen += int(message.get("samples", 0))

            if not gradients:
                raise RuntimeError("ParameterServer did not receive any gradients")
            compressed_gradients = decompress_payload(gradients, mode=self.compression)
            if self.sync_method == "ring":
                averaged_gradients = ring_all_reduce(compressed_gradients, op="mean")
            else:
                averaged_gradients = all_reduce(compressed_gradients, op="mean")
            self.model.apply_gradients(averaged_gradients, self.learning_rate)
            current_state = self.model.state_dict()
            completed_step = self.start_step + step + 1
            broadcast(self.worker_queues, {"type": "weights", "state": current_state, "step": completed_step})

            elapsed = time.perf_counter() - start_time
            communication_time = time.perf_counter() - receive_start
            step_metrics = {
                "step": completed_step,
                "loss": sum(losses) / len(losses),
                "samples_per_sec": samples_seen / elapsed if elapsed else 0.0,
                "communication_time": communication_time,
                "sync_method": self.sync_method,
                "compression": self.compression,
            }
            self.history.append(step_metrics)
            if self.metrics_queue is not None:
                self.metrics_queue.put(step_metrics)
            if self.metrics_callback is not None:
                self.metrics_callback(step_metrics)
            if self.checkpoint_callback is not None:
                self.checkpoint_callback(
                    {
                        "model_state": current_state,
                        "optimizer_state": {"learning_rate": self.learning_rate},
                        "epoch": completed_step,
                        "step": completed_step,
                        "history": list(self.history),
                        "extras": {
                            "sync_method": self.sync_method,
                            "compression": self.compression,
                        },
                    }
                )

        final_state = self.model.state_dict()
        completed_steps = self.start_step + self.max_steps
        broadcast(self.worker_queues, {"type": "stop", "state": final_state, "step": completed_steps})
        return {
            "model_state": final_state,
            "completed_steps": completed_steps,
            "history": self.history,
        }
