from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MiniDistributed.trainer.trainer import Trainer


class LinearRegressionModel:
    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.weight = rng.normal(0.0, 0.5, size=(1, 1)).astype(np.float32)
        self.bias = np.zeros((1,), dtype=np.float32)

    def clone(self) -> "LinearRegressionModel":
        cloned = LinearRegressionModel()
        cloned.load_state_dict(self.state_dict())
        return cloned

    def state_dict(self):
        return {"weight": self.weight.copy(), "bias": self.bias.copy()}

    def load_state_dict(self, state) -> None:
        self.weight = np.asarray(state["weight"], dtype=np.float32).copy()
        self.bias = np.asarray(state["bias"], dtype=np.float32).copy()

    def loss_and_gradients(self, inputs, targets):
        inputs = np.asarray(inputs, dtype=np.float32).reshape(-1, 1)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        predictions = inputs @ self.weight + self.bias
        error = predictions - targets
        loss = float((error**2).mean())
        grad_weight = (2.0 / len(inputs)) * inputs.T @ error
        grad_bias = (2.0 / len(inputs)) * error.sum(axis=0)
        return loss, {"weight": grad_weight, "bias": grad_bias}

    def apply_gradients(self, gradients, learning_rate):
        self.weight -= learning_rate * gradients["weight"]
        self.bias -= learning_rate * gradients["bias"]


def make_dataset(num_samples: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(-2.0, 2.0, size=(num_samples, 1)).astype(np.float32)
    targets = 3.0 * inputs[:, 0] - 1.5 + rng.normal(0.0, 0.15, size=num_samples)
    return inputs, targets.astype(np.float32)


def main() -> None:
    inputs, targets = make_dataset()
    model = LinearRegressionModel()
    trainer = Trainer(model=model, learning_rate=0.05, batch_size=32, epochs=12)
    history = trainer.train(inputs, targets)
    print("final_loss=", history[-1]["loss"])
    print("weight=", trainer.model.weight.ravel()[0])
    print("bias=", trainer.model.bias.ravel()[0])


if __name__ == "__main__":
    main()
