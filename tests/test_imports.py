from MiniDistributed import DistributedTrainer, TinyGPT, Trainer


def test_package_imports():
    assert TinyGPT is not None
    assert Trainer is not None
    assert DistributedTrainer is not None

