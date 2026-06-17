from __future__ import annotations

import copy
from typing import Any

import numpy as np


def _iterate_minibatches(inputs: np.ndarray, targets: np.ndarray, batch_size: int):
    for start in range(0, len(inputs), batch_size):
        end = start + batch_size
        yield inputs[start:end], targets[start:end]


class Trainer:
    def __init__(self, model: Any, learning_rate: float = 0.01, batch_size: int = 32, epochs: int = 5) -> None:
        self.model = copy.deepcopy(model)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.history: list[dict[str, float]] = []

    def train(self, inputs: np.ndarray | None = None, targets: np.ndarray | None = None):
        if inputs is None or targets is None:
            if not hasattr(self.model, "default_training_data"):
                raise ValueError("A dataset is required when the model does not provide default_training_data()")
            inputs, targets = self.model.default_training_data(num_samples=1024, seed=0)

        inputs = np.asarray(inputs)
        targets = np.asarray(targets)

        for epoch in range(self.epochs):
            epoch_losses = []
            for batch_inputs, batch_targets in _iterate_minibatches(inputs, targets, self.batch_size):
                loss, gradients = self.model.loss_and_gradients(batch_inputs, batch_targets)
                self.model.apply_gradients(gradients, self.learning_rate)
                epoch_losses.append(float(loss))
            self.history.append({"epoch": float(epoch + 1), "loss": float(np.mean(epoch_losses))})
        return self.history

    fit = train

