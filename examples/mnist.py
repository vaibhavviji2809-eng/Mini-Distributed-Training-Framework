from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MiniDistributed.trainer.trainer import Trainer


class SyntheticDigitClassifier:
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 64, num_classes: int = 10, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        scale = 0.05
        self.w1 = rng.normal(0.0, scale, size=(input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = rng.normal(0.0, scale, size=(hidden_dim, num_classes)).astype(np.float32)
        self.b2 = np.zeros(num_classes, dtype=np.float32)

    def clone(self) -> "SyntheticDigitClassifier":
        cloned = SyntheticDigitClassifier()
        cloned.load_state_dict(self.state_dict())
        return cloned

    def state_dict(self):
        return {"w1": self.w1.copy(), "b1": self.b1.copy(), "w2": self.w2.copy(), "b2": self.b2.copy()}

    def load_state_dict(self, state) -> None:
        self.w1 = np.asarray(state["w1"], dtype=np.float32).copy()
        self.b1 = np.asarray(state["b1"], dtype=np.float32).copy()
        self.w2 = np.asarray(state["w2"], dtype=np.float32).copy()
        self.b2 = np.asarray(state["b2"], dtype=np.float32).copy()

    def loss_and_gradients(self, inputs, targets):
        inputs = np.asarray(inputs, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.int64)
        hidden_linear = inputs @ self.w1 + self.b1
        hidden = np.maximum(hidden_linear, 0.0)
        logits = hidden @ self.w2 + self.b2
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        batch_indices = np.arange(len(targets))
        loss = -np.log(probabilities[batch_indices, targets] + 1e-12).mean()

        grad_logits = probabilities.copy()
        grad_logits[batch_indices, targets] -= 1.0
        grad_logits /= len(targets)
        grad_w2 = hidden.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ self.w2.T
        grad_hidden[hidden_linear <= 0.0] = 0.0
        grad_w1 = inputs.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)
        gradients = {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}
        return float(loss), gradients

    def apply_gradients(self, gradients, learning_rate):
        self.w1 -= learning_rate * gradients["w1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.w2 -= learning_rate * gradients["w2"]
        self.b2 -= learning_rate * gradients["b2"]


def make_synthetic_mnist(num_samples: int = 2048, seed: int = 0):
    rng = np.random.default_rng(seed)
    inputs = rng.normal(0.0, 1.0, size=(num_samples, 28 * 28)).astype(np.float32)
    targets = rng.integers(0, 10, size=num_samples, dtype=np.int64)
    inputs[np.arange(num_samples), targets % inputs.shape[1]] += 2.0
    return inputs, targets


def main() -> None:
    inputs, targets = make_synthetic_mnist()
    model = SyntheticDigitClassifier()
    trainer = Trainer(model=model, learning_rate=0.02, batch_size=64, epochs=8)
    history = trainer.train(inputs, targets)
    print("final_loss=", history[-1]["loss"])


if __name__ == "__main__":
    main()
