# MiniDistributed

A small, readable distributed-training playground built around multiprocessing.

![Distributed architecture](docs/assets/distributed-architecture.svg)

## Install

```bash
pip install mini-distributed-training-framework
```

For local development:

```bash
git clone https://github.com/vaibhavviji2809-eng/Mini-Distributed-Training-Framework.git
cd Mini-Distributed-Training-Framework
pip install -e ".[dev]"
```

## Quick Start

```python
from MiniDistributed import DistributedTrainer, TinyGPT

model = TinyGPT()
trainer = DistributedTrainer(model=model, workers=4)
trainer.train()
```

## TinyGPT Example

```python
from MiniDistributed import DistributedTrainer, TinyGPT

model = TinyGPT()
trainer = DistributedTrainer(
    model=model,
    workers=4,
    batch_size=32,
    steps_per_worker=24,
    learning_rate=0.05,
)
result = trainer.train()
print(result["final_loss"])
```

## Single-Worker Example

```python
from MiniDistributed.examples.linear_regression import LinearRegressionModel, make_dataset
from MiniDistributed.trainer.trainer import Trainer

inputs, targets = make_dataset()
model = LinearRegressionModel()
trainer = Trainer(model=model, learning_rate=0.05, batch_size=32, epochs=12)
trainer.train(inputs, targets)
```

## What Is Included

- Queue-based communication layer
- Synchronous parameter server
- Worker processes with sharded data
- Checkpoint save/load helpers
- TinyGPT toy model for distributed demos
- Single-worker trainer and runnable examples
- GitHub Actions test workflow

## Project Layout

- `core/` process coordination, communication, checkpointing
- `trainer/` single-worker and distributed trainers
- `models/` TinyGPT toy model
- `examples/` runnable demos
- `docs/` design notes and roadmap
- `dashboard/` future observability UI notes

## Example Commands

```bash
python -m MiniDistributed.examples.linear_regression
python -m MiniDistributed.examples.mnist
python -m MiniDistributed.examples.tiny_gpt
python benchmarks/benchmark.py
```

The benchmark script compares a single-worker training pass with an in-process
distributed data-parallel step. Pass custom worker counts if you want to scale
the comparison:

```bash
python benchmarks/benchmark.py --workers 2 4 8
```

## Roadmap

- Ring all-reduce
- Gradient compression
- Checkpoint resume
- Dashboard metrics
- Fault tolerance
