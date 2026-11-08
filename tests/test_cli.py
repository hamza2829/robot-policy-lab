import json

import pytest

from robot_policy_lab.cli import main


def test_cli_pipeline(tmp_path, capsys):
    assert main(["generate-data", "--output", str(tmp_path / "data"), "--episodes", "3"]) == 0
    dataset = capsys.readouterr().out.strip()
    assert main(["validate-data", dataset]) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 3
    one, chunk = tmp_path / "one.pt", tmp_path / "chunk.pt"
    for size, path in [(1, one), (4, chunk)]:
        assert main(["train", dataset, "--output", str(path), "--chunk-size", str(size),
                     "--epochs", "1"]) == 0
    assert main(["evaluate", "--one-step", str(one), "--chunked", str(chunk),
                 "--episodes", "2", "--output", str(tmp_path / "results.json")]) == 0
    assert (tmp_path / "results.json").exists()


def test_cli_clear_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["validate-data", str(tmp_path / "missing")])
    assert exc.value.code == 2
    assert "error: cannot read dataset" in capsys.readouterr().err
