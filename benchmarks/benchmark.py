from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MiniDistributed import TinyGPT
from MiniDistributed.core.communication import all_reduce
from MiniDistributed.trainer.trainer import Trainer


def run_single_worker() -> tuple[float, float, float]:
    model = TinyGPT()
    inputs, targets = model.default_training_data(num_samples=128, seed=0)
    trainer = Trainer(model=model, learning_rate=0.05, batch_size=16, epochs=1)
    start = time.perf_counter()
    history = trainer.train(inputs, targets)
    elapsed = time.perf_counter() - start
    samples_per_second = len(inputs) / elapsed if elapsed else 0.0
    return elapsed, history[-1]["loss"], samples_per_second


def run_distributed(workers: int) -> tuple[float, float, float]:
    master_model = TinyGPT()
    inputs, targets = master_model.default_training_data(num_samples=workers * 16, seed=1)
    input_shards = np.array_split(inputs, workers)
    target_shards = np.array_split(targets, workers)

    start = time.perf_counter()
    loss_values = []
    gradients_per_worker = []
    for shard_inputs, shard_targets in zip(input_shards, target_shards):
        worker_model = master_model.clone()
        loss, gradients = worker_model.loss_and_gradients(shard_inputs, shard_targets)
        loss_values.append(loss)
        gradients_per_worker.append(gradients)

    averaged_gradients = all_reduce(gradients_per_worker, op="mean")
    master_model.apply_gradients(averaged_gradients, 0.05)
    elapsed = time.perf_counter() - start
    total_samples = len(inputs)
    samples_per_second = total_samples / elapsed if elapsed else 0.0
    return elapsed, float(sum(loss_values) / len(loss_values)), samples_per_second


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MiniDistributed trainers.")
    parser.add_argument("--workers", nargs="*", type=int, default=[2, 4], help="Worker counts to benchmark.")
    args = parser.parse_args()

    print("MiniDistributed benchmark", flush=True)
    print(flush=True)
    single_time, single_loss, single_throughput = run_single_worker()
    print(f"single-worker | time={single_time:.3f}s | loss={single_loss:.4f} | samples/sec={single_throughput:.1f}", flush=True)

    for workers in args.workers:
        elapsed, loss, throughput = run_distributed(workers)
        print(f"{workers}-worker      | time={elapsed:.3f}s | loss={loss:.4f} | samples/sec={throughput:.1f}", flush=True)


if __name__ == "__main__":
    main()
