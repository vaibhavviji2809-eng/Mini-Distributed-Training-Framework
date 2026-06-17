# MiniDistributed

A small, readable distributed-training playground built around multiprocessing.

## Quick Start

```python
from MiniDistributed import DistributedTrainer, TinyGPT

model = TinyGPT()
trainer = DistributedTrainer(model=model, workers=4)
trainer.train()
```

## What Is Included

- A queue-based communication layer
- A synchronous parameter server
- Worker processes with sharded data
- Checkpoint save/load helpers
- TinyGPT toy model for distributed demos
- Single-worker trainer and runnable examples

## Layout

- `core/` process coordination, communication, checkpointing
- `trainer/` single-worker and distributed trainers
- `models/` TinyGPT toy model
- `examples/` runnable demos
- `docs/` design notes and roadmap
- `dashboard/` future observability UI notes

