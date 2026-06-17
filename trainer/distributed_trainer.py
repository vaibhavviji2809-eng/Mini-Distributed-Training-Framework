from __future__ import annotations

import copy
import multiprocessing as mp
from queue import Empty
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.checkpoint import CheckpointManager
from ..core.communication import CommunicationHub
from ..core.parameter_server import ParameterServer, WorkerFailure
from ..core.worker import SGDOptimizer, run_worker_from_config
from ..dashboard.metrics import TrainingMetrics


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
    checkpoint_name: str = "distributed_checkpoint.pkl"
    checkpoint_interval: int = 1
    resume_from_checkpoint: bool = False
    sync_method: str = "parameter_server"
    compression: str = "none"
    fault_tolerant: bool = True
    max_restarts: int = 2
    receive_timeout: float = 2.0

    def __post_init__(self) -> None:
        self.model = copy.deepcopy(self.model)
        self.history: list[dict[str, Any]] = []
        self.metrics = TrainingMetrics(sync_method=self.sync_method, compression=self.compression)

    def _checkpoint_manager(self) -> CheckpointManager | None:
        if self.checkpoint_path is None:
            return None
        return CheckpointManager(self.checkpoint_path)

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

    def _drain_metrics_queue(self, metrics_queue: Any) -> None:
        while True:
            try:
                message = metrics_queue.get_nowait()
            except Empty:
                break
            if not isinstance(message, dict):
                continue
            if message.get("status") in {"started", "done"}:
                self.metrics.record_worker_health(int(message["rank"]), str(message["status"]))
                continue
            if "step" in message:
                self.metrics.record_step(message)

    def _cleanup_processes(self, processes: list[mp.Process]) -> None:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)

    def _spawn_workers(
        self,
        *,
        ctx: mp.context.BaseContext,
        hub: CommunicationHub,
        shards: list[tuple[np.ndarray, np.ndarray]],
        model_state: dict[str, np.ndarray],
        model_config: dict[str, Any],
        metrics_queue: Any,
        steps: int,
        start_step: int,
    ) -> list[mp.Process]:
        processes: list[mp.Process] = []
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
                    "steps": steps,
                    "to_server": hub.to_server,
                    "inbox": hub.worker_queue(rank),
                    "metrics_queue": metrics_queue,
                    "seed": self.seed,
                    "start_step": start_step,
                    "compression": self.compression,
                },
                name=f"worker-{rank}",
            )
            process.start()
            processes.append(process)
        return processes

    def _save_checkpoint(self, checkpoint_manager: CheckpointManager | None, payload: dict[str, Any]) -> None:
        if checkpoint_manager is None:
            return
        if self.checkpoint_interval > 0 and payload["step"] % self.checkpoint_interval != 0:
            return
        checkpoint_manager.save_training_state(
            model_state=payload["model_state"],
            optimizer_state=payload["optimizer_state"],
            epoch=int(payload["epoch"]),
            step=int(payload["step"]),
            history=list(payload["history"]),
            filename=self.checkpoint_name,
            extras=payload.get("extras", {}),
        )

    def train(self) -> dict[str, Any]:
        ctx = mp.get_context("spawn")
        inputs, targets = self._resolve_dataset()
        shards = self._split_dataset(inputs, targets)
        checkpoint_manager = self._checkpoint_manager()
        start_step = 0
        base_history: list[dict[str, Any]] = []

        if self.resume_from_checkpoint and checkpoint_manager is not None and checkpoint_manager.exists(self.checkpoint_name):
            checkpoint = checkpoint_manager.load_training_state(self.checkpoint_name)
            self.model.load_state_dict(checkpoint["model_state"])
            self.history = list(checkpoint.get("history", []))
            base_history = list(self.history)
            start_step = int(checkpoint.get("step", checkpoint.get("epoch", 0)))
            for entry in self.history:
                self.metrics.record_step(entry)

        if start_step >= self.steps_per_worker:
            return {
                "final_loss": self.history[-1]["loss"] if self.history else None,
                "steps": start_step,
                "workers": self.workers,
                "history": self.history,
                "metrics": self.metrics.summary(),
                "resumed": self.resume_from_checkpoint,
                "restarts": 0,
            }

        if checkpoint_manager is not None:
            self._save_checkpoint(
                checkpoint_manager,
                {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": {"learning_rate": self.learning_rate},
                    "epoch": start_step,
                    "step": start_step,
                    "history": list(self.history),
                    "extras": {
                        "sync_method": self.sync_method,
                        "compression": self.compression,
                    },
                },
            )

        restart_count = 0
        while start_step < self.steps_per_worker:
            remaining_steps = self.steps_per_worker - start_step
            hub = CommunicationHub(self.workers, ctx=ctx)
            metrics_queue = ctx.Queue()
            model_state = self.model.state_dict()
            model_config = self.model.config_dict() if hasattr(self.model, "config_dict") else {}
            processes = self._spawn_workers(
                ctx=ctx,
                hub=hub,
                shards=shards,
                model_state=model_state,
                model_config=model_config,
                metrics_queue=metrics_queue,
                steps=remaining_steps,
                start_step=start_step,
            )

            def checkpoint_callback(payload: dict[str, Any]) -> None:
                self._save_checkpoint(checkpoint_manager, payload)

            server = ParameterServer(
                model=copy.deepcopy(self.model),
                learning_rate=self.learning_rate,
                num_workers=self.workers,
                grad_queue=hub.to_server,
                worker_queues=hub.to_workers,
                metrics_queue=metrics_queue,
                max_steps=remaining_steps,
                start_step=start_step,
                sync_method=self.sync_method,
                compression=self.compression,
                receive_timeout=self.receive_timeout,
                worker_processes=processes,
                checkpoint_callback=checkpoint_callback if checkpoint_manager is not None else None,
            )

            try:
                result = server.run()
            except WorkerFailure:
                self._cleanup_processes(processes)
                restart_count += 1
                self.metrics.record_restart()
                if not self.fault_tolerant or restart_count > self.max_restarts:
                    raise
                if checkpoint_manager is None or not checkpoint_manager.exists(self.checkpoint_name):
                    raise
                checkpoint = checkpoint_manager.load_training_state(self.checkpoint_name)
                self.model.load_state_dict(checkpoint["model_state"])
                self.history = list(checkpoint.get("history", self.history))
                base_history = list(self.history)
                start_step = int(checkpoint.get("step", checkpoint.get("epoch", start_step)))
                continue

            self._cleanup_processes(processes)
            self._drain_metrics_queue(metrics_queue)
            self.model.load_state_dict(result["model_state"])
            self.history = base_history + list(result["history"])
            start_step = int(result["completed_steps"])
            break

        return {
            "final_loss": self.history[-1]["loss"] if self.history else None,
            "steps": start_step,
            "workers": self.workers,
            "history": self.history,
            "metrics": self.metrics.summary(),
            "resumed": self.resume_from_checkpoint or start_step > 0,
            "restarts": self.metrics.restarts,
        }

    fit = train
