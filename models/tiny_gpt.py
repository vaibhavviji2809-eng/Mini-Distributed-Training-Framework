from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


@dataclass
class TinyGPT:
    vocab_size: int = 32
    context_length: int = 8
    embedding_dim: int = 32
    hidden_dim: int = 64
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = 0.02
        self.token_embedding = rng.normal(0.0, scale, size=(self.vocab_size, self.embedding_dim)).astype(np.float32)
        self.position_embedding = rng.normal(0.0, scale, size=(self.context_length, self.embedding_dim)).astype(np.float32)
        self.projection_weight = rng.normal(0.0, scale, size=(self.embedding_dim, self.hidden_dim)).astype(np.float32)
        self.projection_bias = np.zeros(self.hidden_dim, dtype=np.float32)
        self.output_weight = rng.normal(0.0, scale, size=(self.hidden_dim, self.vocab_size)).astype(np.float32)
        self.output_bias = np.zeros(self.vocab_size, dtype=np.float32)

    @staticmethod
    def default_training_data(num_samples: int = 1024, seed: int = 0, vocab_size: int = 32, context_length: int = 8):
        rng = np.random.default_rng(seed)
        inputs = rng.integers(0, vocab_size, size=(num_samples, context_length), dtype=np.int64)
        target_signal = inputs[:, 0] * 3 + inputs[:, -1] * 5 + inputs.sum(axis=1)
        targets = np.mod(target_signal, vocab_size).astype(np.int64)
        return inputs, targets

    def clone(self) -> "TinyGPT":
        cloned = copy.deepcopy(self)
        cloned.load_state_dict(self.state_dict())
        return cloned

    def config_dict(self) -> dict[str, int]:
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
        }

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "token_embedding": self.token_embedding.copy(),
            "position_embedding": self.position_embedding.copy(),
            "projection_weight": self.projection_weight.copy(),
            "projection_bias": self.projection_bias.copy(),
            "output_weight": self.output_weight.copy(),
            "output_bias": self.output_bias.copy(),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.token_embedding = np.asarray(state["token_embedding"], dtype=np.float32).copy()
        self.position_embedding = np.asarray(state["position_embedding"], dtype=np.float32).copy()
        self.projection_weight = np.asarray(state["projection_weight"], dtype=np.float32).copy()
        self.projection_bias = np.asarray(state["projection_bias"], dtype=np.float32).copy()
        self.output_weight = np.asarray(state["output_weight"], dtype=np.float32).copy()
        self.output_bias = np.asarray(state["output_bias"], dtype=np.float32).copy()

    def _forward_internal(self, inputs: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        inputs = np.asarray(inputs, dtype=np.int64)
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        token_vectors = self.token_embedding[inputs]
        positional = self.position_embedding[None, :, :]
        combined = token_vectors + positional
        pooled = combined.mean(axis=1)
        pre_activation = pooled @ self.projection_weight + self.projection_bias
        hidden = np.tanh(pre_activation)
        logits = hidden @ self.output_weight + self.output_bias
        cache = {
            "inputs": inputs,
            "combined": combined,
            "pooled": pooled,
            "pre_activation": pre_activation,
            "hidden": hidden,
        }
        return logits, cache

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        logits, _ = self._forward_internal(inputs)
        return logits

    def loss_and_gradients(self, inputs: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        logits, cache = self._forward_internal(inputs)
        targets = np.asarray(targets, dtype=np.int64)
        if targets.ndim == 0:
            targets = targets[None]

        probabilities = _softmax(logits)
        batch_indices = np.arange(len(targets))
        loss = -np.log(probabilities[batch_indices, targets] + 1e-12).mean()

        grad_logits = probabilities.copy()
        grad_logits[batch_indices, targets] -= 1.0
        grad_logits /= len(targets)

        grad_output_weight = cache["hidden"].T @ grad_logits
        grad_output_bias = grad_logits.sum(axis=0)

        grad_hidden = grad_logits @ self.output_weight.T
        grad_pre_activation = grad_hidden * (1.0 - np.tanh(cache["pre_activation"]) ** 2)
        grad_projection_weight = cache["pooled"].T @ grad_pre_activation
        grad_projection_bias = grad_pre_activation.sum(axis=0)

        grad_pooled = grad_pre_activation @ self.projection_weight.T
        grad_combined = np.repeat(grad_pooled[:, None, :], self.context_length, axis=1) / self.context_length

        grad_token_embedding = np.zeros_like(self.token_embedding)
        grad_position_embedding = np.zeros_like(self.position_embedding)
        for position in range(self.context_length):
            np.add.at(grad_token_embedding, cache["inputs"][:, position], grad_combined[:, position, :])
            grad_position_embedding[position] += grad_combined[:, position, :].sum(axis=0)

        gradients = {
            "token_embedding": grad_token_embedding,
            "position_embedding": grad_position_embedding,
            "projection_weight": grad_projection_weight,
            "projection_bias": grad_projection_bias,
            "output_weight": grad_output_weight,
            "output_bias": grad_output_bias,
        }
        return float(loss), gradients

    def apply_gradients(self, gradients: dict[str, np.ndarray], learning_rate: float) -> None:
        self.token_embedding -= learning_rate * gradients["token_embedding"]
        self.position_embedding -= learning_rate * gradients["position_embedding"]
        self.projection_weight -= learning_rate * gradients["projection_weight"]
        self.projection_bias -= learning_rate * gradients["projection_bias"]
        self.output_weight -= learning_rate * gradients["output_weight"]
        self.output_bias -= learning_rate * gradients["output_bias"]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs).argmax(axis=1)

    def evaluate_accuracy(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        predictions = self.predict(inputs)
        return float((predictions == np.asarray(targets)).mean())
