import numpy as np

from MiniDistributed import DistributedTrainer, TinyGPT
from MiniDistributed.core.communication import all_reduce


def test_distributed_trainer_setup():
    model = TinyGPT()
    trainer = DistributedTrainer(model=model, workers=2, batch_size=8, steps_per_worker=2, learning_rate=0.05)
    inputs, targets = trainer._resolve_dataset()
    shards = trainer._split_dataset(inputs, targets)

    assert len(shards) == 2
    assert sum(len(shard_inputs) for shard_inputs, _ in shards) == len(inputs)
    assert all(len(shard_inputs) > 0 for shard_inputs, _ in shards)


def test_all_reduce_mean():
    gradients = [
        {"weights": np.array([1.0, 2.0], dtype=np.float32)},
        {"weights": np.array([3.0, 6.0], dtype=np.float32)},
    ]
    averaged = all_reduce(gradients, op="mean")
    assert np.allclose(averaged["weights"], np.array([2.0, 4.0], dtype=np.float32))

