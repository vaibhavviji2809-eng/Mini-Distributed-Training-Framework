from MiniDistributed.core.checkpoint import CheckpointManager


def test_checkpoint_round_trip(tmp_path):
    manager = CheckpointManager(tmp_path)
    manager.save_training_state(
        model_state={"weight": [1, 2, 3]},
        optimizer_state={"learning_rate": 0.01},
        epoch=7,
        history=[{"step": 1, "loss": 3.14}],
        filename="state.pkl",
    )

    payload = manager.load_training_state("state.pkl")
    assert payload["epoch"] == 7
    assert payload["optimizer_state"]["learning_rate"] == 0.01
    assert payload["history"][0]["loss"] == 3.14
