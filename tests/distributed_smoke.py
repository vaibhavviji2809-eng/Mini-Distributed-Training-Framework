from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MiniDistributed import DistributedTrainer, TinyGPT


def main() -> None:
    model = TinyGPT()
    trainer = DistributedTrainer(model=model, workers=2, batch_size=8, steps_per_worker=2, learning_rate=0.05)
    result = trainer.train()
    print(f"final_loss= {result['final_loss']}")
    print(f"workers= {result['workers']}")
    print(f"steps= {result['steps']}")


if __name__ == "__main__":
    main()

