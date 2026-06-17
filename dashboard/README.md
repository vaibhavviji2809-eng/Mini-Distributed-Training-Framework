# Dashboard

Open `dashboard/index.html` directly in a browser to inspect the training UI.

You can load either:

- The sample metrics file at `dashboard/sample_metrics.json`
- The JSON output from `trainer.train()["metrics"]`

The dashboard shows:

- Worker state
- Loss curves
- Throughput
- Gradient synchronization latency
- Communication overhead

The trainer emits structured metrics through `dashboard/metrics.py`, including:

- Step loss
- Throughput
- Communication time
- Worker health
- Restart counts

Suggested next step: connect the dashboard to a live training run or host it
through a tiny local web server.
