from __future__ import annotations

from queue import Empty
from pathlib import Path

import numpy as np
import pytest

from MiniDistributed import DistributedTrainer, TinyGPT
from MiniDistributed.core.communication import compress_payload, decompress_payload, ring_all_reduce
from MiniDistributed.core.parameter_server import ParameterServer, WorkerFailure


def test_compression_round_trip():
    payload = {
        "weight": np.array([1.0, 2.5], dtype=np.float32),
        "bias": np.array([0.25], dtype=np.float32),
    }
    compressed = compress_payload(payload, mode="fp16")
    assert compressed["weight"].dtype == np.float16

    restored = decompress_payload(compressed, mode="fp16")
    assert restored["weight"].dtype == np.float32
    assert np.allclose(restored["weight"], payload["weight"], atol=1e-3)
    assert np.allclose(restored["bias"], payload["bias"], atol=1e-3)


def test_ring_all_reduce_mean():
    gradients = [
        {"weights": np.array([1.0, 2.0], dtype=np.float32)},
        {"weights": np.array([3.0, 6.0], dtype=np.float32)},
        {"weights": np.array([5.0, 10.0], dtype=np.float32)},
    ]
    averaged = ring_all_reduce(gradients, op="mean")
    assert np.allclose(averaged["weights"], np.array([3.0, 6.0], dtype=np.float32))


def test_checkpoint_resume_and_metrics(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    compact_model = TinyGPT(vocab_size=16, context_length=4, embedding_dim=8, hidden_dim=8)

    trainer = DistributedTrainer(
        model=compact_model,
        workers=1,
        batch_size=2,
        steps_per_worker=1,
        learning_rate=0.05,
        checkpoint_path=str(checkpoint_dir),
        sync_method="ring",
        compression="fp16",
    )
    result = trainer.train()
    assert result["steps"] == 1
    assert result["metrics"]["sync_method"] == "ring"
    assert result["metrics"]["compression"] == "fp16"
    assert Path(checkpoint_dir / "distributed_checkpoint.pkl").exists()

    resumed = DistributedTrainer(
        model=compact_model,
        workers=1,
        batch_size=2,
        steps_per_worker=2,
        learning_rate=0.05,
        checkpoint_path=str(checkpoint_dir),
        resume_from_checkpoint=True,
        sync_method="ring",
        compression="fp16",
    )
    resumed_result = resumed.train()
    assert resumed_result["resumed"] is True
    assert resumed_result["steps"] == 2
    assert len(resumed_result["history"]) == 2
    assert resumed_result["metrics"]["losses"]


def test_worker_failure_detection():
    class DeadProcess:
        name = "worker-0"
        exitcode = 1

        def is_alive(self) -> bool:
            return False

    class EmptyQueue:
        def get(self, timeout=None):
            raise Empty

    server = ParameterServer(
        model=TinyGPT(),
        learning_rate=0.05,
        num_workers=1,
        grad_queue=EmptyQueue(),
        worker_queues=[],
        receive_timeout=0.01,
        worker_processes=[DeadProcess()],
    )

    with pytest.raises(WorkerFailure):
        server._collect_gradient()
