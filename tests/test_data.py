import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from robot_policy_lab.data import (
    SCHEMA,
    EpisodeDataset,
    InvalidEpisode,
    collect_episode,
    episode_hash,
    generate_data,
    sha256,
    validate_dataset,
    validate_episode,
    write_dataset,
)
from robot_policy_lab.sim import PushEnv


def mutate(table, change):
    rows = table.to_pylist()
    change(rows)
    return pa.Table.from_pylist(rows, schema=SCHEMA)


def test_good_schema(episodes):
    assert validate_episode(episodes[0])[-1]["success"]
    assert episodes[0].schema.equals(SCHEMA)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda rows: rows[0].update(seed=-1), "nonnegative"),
        (lambda rows: rows[1].update(seed=999), "mixed"),
        (lambda rows: rows[1].update(timestep=8), "contiguous"),
        (lambda rows: rows[0].update(action=[2, 0]), "range"),
        (lambda rows: rows[0].update(action=[float("nan"), 0]), "non-finite"),
        (lambda rows: rows[0].update(action=[None, 0]), "non-finite"),
        (lambda rows: rows[0].update(observation=None), "null"),
        (lambda rows: rows[0].update(success=True), "terminal"),
        (lambda rows: rows[-1].update(success=False), "terminal"),
        (lambda rows: rows[0]["observation"].__setitem__(6, 0.6), "block relative"),
        (lambda rows: rows[0]["observation"].__setitem__(8, 0.6), "target relative"),
        (lambda rows: rows[0]["observation"].__setitem__(10, 3), "range"),
        (lambda rows: rows[1]["observation"].__setitem__(10, 0.5), "continuity"),
    ],
)
def test_broken_values(episodes, change, reason):
    with pytest.raises(InvalidEpisode, match=reason):
        validate_episode(mutate(episodes[0], change))


def test_wrong_schema_and_empty(episodes):
    with pytest.raises(InvalidEpisode, match="schema"):
        validate_episode(episodes[0].drop(["action"]))
    with pytest.raises(InvalidEpisode, match="empty"):
        validate_episode(pa.Table.from_pylist([], schema=SCHEMA))


def test_false_terminal_distance(episodes):
    rows = episodes[0].to_pylist()
    final = rows[-1]["next_observation"]
    final[4] += 0.1
    final[8] += 0.1
    # Change the target consistently for the complete episode.
    for row in rows:
        for key in ("observation", "next_observation"):
            if row[key] is not final:
                row[key][4] += 0.1
                row[key][8] += 0.1
    with pytest.raises(InvalidEpisode, match="target distance"):
        validate_episode(pa.Table.from_pylist(rows, schema=SCHEMA))


def test_changing_target(episodes):
    def change(rows):
        rows[0]["next_observation"][4] += 0.01
        rows[0]["next_observation"][8] += 0.01
    with pytest.raises(InvalidEpisode, match="target changed"):
        validate_episode(mutate(episodes[0], change))


def test_hash_stability_and_changes(tmp_path, episodes):
    table = episodes[0]
    file = tmp_path / "copy.parquet"
    pq.write_table(table, file, compression=None)
    assert episode_hash(table) == episode_hash(pq.read_table(file))
    changed = mutate(table, lambda rows: rows[0].update(action=[0.123, 0.456]))
    assert episode_hash(table) != episode_hash(changed)
    first = write_dataset(episodes, tmp_path / "first")
    second = write_dataset(list(reversed(episodes)), tmp_path / "second")
    assert first.name == second.name
    assert write_dataset(episodes, tmp_path / "first") == first
    assert len(validate_dataset(first)) == len(episodes)


def test_write_rejects_empty_and_duplicates(tmp_path, episodes):
    with pytest.raises(InvalidEpisode, match="no accepted"):
        write_dataset([], tmp_path)
    with pytest.raises(InvalidEpisode, match="duplicate"):
        write_dataset([episodes[0], episodes[0]], tmp_path)


def test_loader_shapes_masks_and_boundaries(dataset, episodes):
    data = EpisodeDataset(dataset, chunk_size=4, seeds={0})
    observation, action, mask = data[0]
    assert observation.shape == (12,)
    assert action.shape == (4, 2)
    assert mask.shape == (4,)
    assert observation.dtype == action.dtype == mask.dtype == torch.float32
    assert len(data) == episodes[0].num_rows
    np.testing.assert_allclose(action, episodes[0]["action"].to_pylist()[:4])
    _, final_actions, final_mask = data[-1]
    assert final_mask.tolist() == [1, 0, 0, 0]
    assert not final_actions[1:].any()
    with pytest.raises(ValueError, match="positive"):
        EpisodeDataset(dataset, 0)
    with pytest.raises(InvalidEpisode, match="no episodes"):
        EpisodeDataset(dataset, seeds={999})


@pytest.mark.parametrize("kind", ["file_hash", "content_hash", "rows", "version", "schema",
                                  "unsafe", "missing", "duplicate", "extra", "order", "seed"])
def test_manifest_corruption(dataset, kind):
    manifest_file = dataset / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    if kind == "extra":
        (dataset / "extra.parquet").write_bytes(b"junk")
    elif kind == "order":
        manifest["episodes"].reverse()
    elif kind == "duplicate":
        manifest["episodes"].append(manifest["episodes"][0])
    elif kind == "schema":
        manifest["schema_version"] = 999
    elif kind == "version":
        manifest["version"] = "bad"
    elif kind in ("unsafe", "missing"):
        manifest["episodes"][0]["file"] = "../bad.parquet" if kind == "unsafe" else "bad.parquet"
    elif kind == "seed":
        manifest["episodes"][0]["seed"] = 999
    else:
        manifest["episodes"][0][kind] = 999 if kind == "rows" else "bad"
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(InvalidEpisode):
        validate_dataset(dataset)


def test_tampered_parquet(dataset):
    manifest_file = dataset / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    file = dataset / manifest["episodes"][0]["file"]
    file.write_bytes(b"broken")
    manifest["episodes"][0]["file_hash"] = sha256(file.read_bytes())
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(InvalidEpisode, match="cannot read"):
        validate_dataset(dataset)


def test_generation_and_rejection(tmp_path, monkeypatch):
    path = generate_data(tmp_path, episodes=3, start_seed=50)
    assert len(validate_dataset(path)) == 3
    with pytest.raises(ValueError, match="positive"):
        generate_data(tmp_path, episodes=0)
    with pytest.raises(InvalidEpisode, match="terminal"):
        validate_episode(collect_episode(1, PushEnv(horizon=1)))
    monkeypatch.setattr("robot_policy_lab.data.collect_episode",
                        lambda seed, env: collect_episode(seed, PushEnv(horizon=1)))
    with pytest.raises(InvalidEpisode, match="no accepted"):
        generate_data(tmp_path / "bad", episodes=1)
