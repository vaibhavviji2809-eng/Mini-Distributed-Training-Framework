from __future__ import annotations

import multiprocessing as mp
import copy
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


def compress_payload(payload: Any, mode: str = "none") -> Any:
    if mode == "none":
        return payload
    if mode != "fp16":
        raise ValueError(f"Unsupported compression mode: {mode}")
    if isinstance(payload, dict):
        return {key: compress_payload(value, mode) for key, value in payload.items()}
    if isinstance(payload, list):
        return [compress_payload(value, mode) for value in payload]
    if isinstance(payload, tuple):
        return tuple(compress_payload(value, mode) for value in payload)
    if isinstance(payload, np.ndarray):
        if np.issubdtype(payload.dtype, np.floating):
            return payload.astype(np.float16)
        return payload
    if isinstance(payload, (float, np.floating)):
        return np.float16(payload)
    return payload


def decompress_payload(payload: Any, mode: str = "none") -> Any:
    if mode == "none":
        return payload
    if mode != "fp16":
        raise ValueError(f"Unsupported compression mode: {mode}")
    if isinstance(payload, dict):
        return {key: decompress_payload(value, mode) for key, value in payload.items()}
    if isinstance(payload, list):
        return [decompress_payload(value, mode) for value in payload]
    if isinstance(payload, tuple):
        return tuple(decompress_payload(value, mode) for value in payload)
    if isinstance(payload, np.ndarray):
        if np.issubdtype(payload.dtype, np.floating):
            return payload.astype(np.float32)
        return payload
    if isinstance(payload, (float, np.floating)):
        return np.float32(payload)
    return payload


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


def _scale_value(value: Any, scale: float) -> Any:
    if isinstance(value, dict):
        return {key: _scale_value(item, scale) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value * scale
    if isinstance(value, tuple):
        return tuple(_scale_value(item, scale) for item in value)
    if isinstance(value, list):
        return [_scale_value(item, scale) for item in value]
    if isinstance(value, (int, float, np.number)):
        return type(value)(value * scale)
    return copy.deepcopy(value)


def all_reduce(values: list[Any], op: str = "mean") -> Any:
    if not values:
        raise ValueError("all_reduce needs at least one value")
    return _reduce_values(values, op)


def ring_all_reduce(values: list[Any], op: str = "mean") -> Any:
    if not values:
        raise ValueError("ring_all_reduce needs at least one value")
    if len(values) == 1:
        return copy.deepcopy(values[0])
    running = copy.deepcopy(values[0])
    for value in values[1:]:
        running = _reduce_values([running, value], op="sum")
    if op == "sum":
        return running
    if op == "mean":
        return _scale_value(running, 1.0 / len(values))
    raise ValueError(f"Unsupported reduction op: {op}")


class CommunicationHub:
    def __init__(self, num_workers: int, ctx: mp.context.BaseContext | None = None) -> None:
        self.ctx = ctx or mp.get_context("spawn")
        self.to_server = self.ctx.Queue()
        self.to_workers = [self.ctx.Queue() for _ in range(num_workers)]

    def worker_queue(self, rank: int) -> Any:
        return self.to_workers[rank]
