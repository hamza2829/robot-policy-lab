import json

import numpy as np
import pytest
import torch

from robot_policy_lab.data import write_dataset
from robot_policy_lab.evaluate import evaluate, wilson_interval
from robot_policy_lab.policy import Policy, load_policy, masked_loss, train


def test_policy_shape_and_padding_loss():
    policy = Policy(4)
    assert policy(torch.zeros(3, 12)).shape == (3, 4, 2)
    assert policy.act(np.zeros(12)).shape == (4, 2)
    predicted = torch.tensor([[[1.0, 1.0], [99.0, 99.0]]])
    assert masked_loss(predicted, torch.zeros_like(predicted), torch.tensor([[1, 0]])) == 1


def test_training_reduces_loss_and_reload(dataset, tmp_path):
    output = tmp_path / "policy.pt"
    result = train(dataset, output, chunk_size=3, epochs=12)
    assert result["final_loss"] < result["initial_loss"] * 0.6
    assert result["validation_loss"] < result["initial_loss"]
    assert not set(result["training_seeds"]) & set(result["validation_seeds"])
    model, metadata = load_policy(output)
    assert metadata == result
    assert json.loads(output.with_suffix(".json").read_text()) == result
    assert np.isfinite(model.act(np.zeros(12))).all()


def test_training_argument_errors(dataset, episodes, tmp_path):
    with pytest.raises(ValueError, match="positive"):
        train(dataset, tmp_path / "bad.pt", epochs=0)
    single = write_dataset([episodes[0]], tmp_path / "single")
    with pytest.raises(ValueError, match="two episodes"):
        train(single, tmp_path / "bad.pt")


def test_evaluation_smoke_determinism_and_raw_results(dataset, tmp_path):
    one, chunk = tmp_path / "one.pt", tmp_path / "chunk.pt"
    train(dataset, one, 1, 2)
    train(dataset, chunk, 4, 2)
    output = tmp_path / "evaluation.json"
    report = evaluate(one, chunk, episodes=3, output=output)
    assert report == evaluate(one, chunk, episodes=3)
    assert json.loads(output.read_text()) == report
    assert set(report["policies"]) == {"expert", "random", "one_step", "chunked"}
    assert report["policies"]["expert"]["successes"] == 3
    for item in report["policies"].values():
        assert len(item["outcomes"]) == 3
        assert item["success_rate"] == item["successes"] / 3
        assert sum(item["failure_counts"].values()) + item["successes"] == 3


def test_evaluation_protocol_errors(dataset, tmp_path):
    checkpoint = tmp_path / "one.pt"
    train(dataset, checkpoint, 1, 1)
    with pytest.raises(ValueError, match="overlap"):
        evaluate(checkpoint, episodes=1, start_seed=0)
    with pytest.raises(ValueError, match="at least two"):
        evaluate(chunked=checkpoint, episodes=1)
    chunk = tmp_path / "chunk.pt"
    train(dataset, chunk, 4, 1)
    with pytest.raises(ValueError, match="exactly one"):
        evaluate(one_step=chunk, episodes=1)
    checkpoint_data = torch.load(chunk, weights_only=True)
    checkpoint_data["metadata"]["dataset_version"] = "different"
    torch.save(checkpoint_data, chunk)
    with pytest.raises(ValueError, match="same dataset"):
        evaluate(checkpoint, chunk, episodes=1)
    with pytest.raises(ValueError, match="positive"):
        evaluate(episodes=0)
    assert wilson_interval(0, 100)[0] == pytest.approx(0)
    assert wilson_interval(100, 100)[1] == pytest.approx(1)
