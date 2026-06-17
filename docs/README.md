# Design Notes

## Phase Roadmap

1. Multiprocessing workers
2. Worker abstraction with local shards
3. Queue-based communication
4. Parameter server
5. Synchronous distributed SGD
6. All-reduce averaging
7. Distributed trainer wrapper
8. Data parallel benchmarks
9. Checkpointing
10. Dashboard
11. TinyGPT distributed training

## Current Implementation

- Uses `multiprocessing` with spawn-safe worker entrypoints
- Synchronizes gradients on every step
- Averages gradients on the parameter server
- Ships checkpoints with `pickle`

## Future Extensions

- Ring all-reduce
- Gradient compression
- Fault tolerance
- Process health monitoring

