"""Coverage tests for lexnlp.ml.gensim_utils."""

from types import SimpleNamespace

import pytest

from lexnlp.ml.gensim_utils import DummyGensimKeyedVectors, TrainingCallback


def _fake_model() -> SimpleNamespace:
    return SimpleNamespace(epochs=5, total_train_time=1.5, vector_size=10, window=5, min_count=1, dm=1)


def test_dummy_vectors_stores_vector_size():
    assert DummyGensimKeyedVectors(50).vector_size == 50


def test_dummy_vectors_missing_attribute_raises():
    with pytest.raises(AttributeError, match="This is a DummyDV!"):
        DummyGensimKeyedVectors(50).most_similar("anything")


def test_callback_init_starts_at_zero():
    callback = TrainingCallback()
    assert callback.completed_epochs == 0
    assert callback.epoch == 0


def test_callback_on_epoch_begin_increments_and_prints(capsys):
    callback = TrainingCallback()
    callback.on_epoch_begin(_fake_model())
    assert callback.epoch == 1
    assert capsys.readouterr().out == "Started epoch 1 / 5\n"


def test_callback_on_epoch_end_counts_completion(capsys):
    callback = TrainingCallback()
    callback.on_epoch_begin(_fake_model())
    callback.on_epoch_end(_fake_model())
    assert callback.completed_epochs == 1
    out = capsys.readouterr().out
    assert "[Epoch 1 | total_train_time: 1.5]" in out


def test_callback_on_train_begin_prints_config(capsys):
    TrainingCallback().on_train_begin(_fake_model())
    out = capsys.readouterr().out
    assert "Started training..." in out
    assert "Gensim version:" in out
    assert "model.vector_size=10" in out


def test_callback_on_train_end_prints_finished(capsys):
    TrainingCallback().on_train_end(_fake_model())
    assert capsys.readouterr().out == "Ended training.\n"
