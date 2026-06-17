# Dashboard

This folder is reserved for a future live dashboard.

Potential views:

- Worker state
- Loss curves
- Throughput
- Gradient synchronization latency
- Communication overhead

The trainer now emits structured metrics through `dashboard/metrics.py`, including:

- Step loss
- Throughput
- Communication time
- Worker health
- Restart counts

Suggested next step: expose these metrics through a small web app or a desktop UI.
