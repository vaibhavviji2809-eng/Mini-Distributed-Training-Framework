# Dashboard

The dashboard works in two modes:

- Open `dashboard/index.html` directly for the static viewer
- Run `mini-distributed-dashboard --metrics-file runs/live_metrics.json` for live polling

When the live server is running, the page automatically polls `/api/metrics`
and updates the charts, worker health, and restart counter.

The trainer can write to the same file with:

```python
trainer = DistributedTrainer(
    model=TinyGPT(),
    workers=4,
    dashboard_metrics_path="runs/live_metrics.json",
)
```

You can also load:

- The sample metrics file at `dashboard/sample_metrics.json`
- The JSON output from `trainer.train()["metrics"]`

The dashboard shows:

- Worker state
- Loss curves
- Throughput
- Gradient synchronization latency
- Communication overhead

Structured metrics come from `dashboard/metrics.py`, including:

- Step loss
- Throughput
- Communication time
- Worker health
- Restart counts
