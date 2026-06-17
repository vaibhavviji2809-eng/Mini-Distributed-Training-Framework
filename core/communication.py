from __future__ import annotations

import multiprocessing as mp
from queue import Empty
from typing import Any, Iterable

import numpy as np


def send(queue: Any, message: Any) -> None:
    queue.put(message)


def receive(queue: Any, timeout: float | None = None) -> Any:
    try:
        return queue.get(timeout=timeout)
    except Empty as exc:
        raise TimeoutError("Timed out waiting for a message") from exc


def broadcast(queues: Iterable[Any], message: Any) -> None:
    for queue in queues:
        queue.put(message)


def _reduce_values(values: list[Any], op: str) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _reduce_values([value[key] for value in values], op) for key in first}
    if isinstance(first, np.ndarray):
        stacked = np.stack(values, axis=0)
        if op == "sum":
            return stacked.sum(axis=0)
        if op == "mean":
            return stacked.mean(axis=0)
        raise ValueError(f"Unsupported reduction op: {op}")
    if isinstance(first, tuple):
        return tuple(_reduce_values([value[index] for value in values], op) for index in range(len(first)))
    if isinstance(first, list):
        return [_reduce_values([value[index] for value in values], op) for index in range(len(first))]
    if isinstance(first, (int, float, np.number)):
        total = sum(values)
        return total / len(values) if op == "mean" else total
    raise TypeError(f"Unsupported value type for all_reduce: {type(first)!r}")


def all_reduce(values: list[Any], op: str = "mean") -> Any:
    if not values:
        raise ValueError("all_reduce needs at least one value")
    return _reduce_values(values, op)


class CommunicationHub:
    def __init__(self, num_workers: int, ctx: mp.context.BaseContext | None = None) -> None:
        self.ctx = ctx or mp.get_context("spawn")
        self.to_server = self.ctx.Queue()
        self.to_workers = [self.ctx.Queue() for _ in range(num_workers)]

    def worker_queue(self, rank: int) -> Any:
        return self.to_workers[rank]

