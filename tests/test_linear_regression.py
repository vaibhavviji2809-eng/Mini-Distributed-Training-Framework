from MiniDistributed.examples.linear_regression import LinearRegressionModel, make_dataset
from MiniDistributed.trainer.trainer import Trainer


def test_linear_regression_converges():
    inputs, targets = make_dataset(num_samples=256, seed=1)
    model = LinearRegressionModel(seed=1)
    trainer = Trainer(model=model, learning_rate=0.05, batch_size=32, epochs=12)
    history = trainer.train(inputs, targets)
    assert history[-1]["loss"] < history[0]["loss"]
    assert abs(float(trainer.model.weight.ravel()[0]) - 3.0) < 0.2
    assert abs(float(trainer.model.bias.ravel()[0]) + 1.5) < 0.2

