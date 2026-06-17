from __future__ import annotations

import json
import threading
from urllib.request import urlopen

from MiniDistributed import DistributedTrainer, TinyGPT
from MiniDistributed.dashboard.metrics import TrainingMetrics
from MiniDistributed.dashboard.server import create_server


def test_dashboard_server_serves_live_metrics(tmp_path):
    metrics_path = tmp_path / "live_metrics.json"
    metrics = TrainingMetrics(sync_method="ring", compression="fp16")
    metrics.record_step(
        {
            "loss": 2.75,
            "samples_per_sec": 1234.5,
            "communication_time": 0.18,
        }
    )
    metrics.record_worker_health(0, "started")
    metrics.write_snapshot(metrics_path)

    server = create_server(port=0, metrics_path=metrics_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/api/metrics") as response:
            payload = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert payload["sync_method"] == "ring"
    assert payload["compression"] == "fp16"
    assert payload["losses"] == [2.75]
    assert payload["worker_health"]["0"] == "started"


def test_trainer_writes_live_dashboard_metrics(tmp_path):
    dashboard_path = tmp_path / "live_metrics.json"
    model = TinyGPT(vocab_size=16, context_length=4, embedding_dim=8, hidden_dim=8)

    trainer = DistributedTrainer(
        model=model,
        workers=1,
        batch_size=2,
        steps_per_worker=1,
        learning_rate=0.05,
        dashboard_metrics_path=str(dashboard_path),
    )

    result = trainer.train()
    payload = json.loads(dashboard_path.read_text(encoding="utf-8"))

    assert result["steps"] == 1
    assert payload["losses"]
    assert payload["throughput"]
    assert payload["worker_health"]["0"] == "done"
