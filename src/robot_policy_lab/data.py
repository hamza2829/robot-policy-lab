"""Versioned Parquet episodes with semantic hashes and strict validation."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from robot_policy_lab.sim import ACTION_DIM, OBS_DIM, PushEnv, expert_action

SCHEMA_VERSION = 1
SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.int64(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("timestep", pa.int32(), nullable=False),
        pa.field("observation", pa.list_(pa.float32(), OBS_DIM), nullable=False),
        pa.field("action", pa.list_(pa.float32(), ACTION_DIM), nullable=False),
        pa.field("next_observation", pa.list_(pa.float32(), OBS_DIM), nullable=False),
        pa.field("success", pa.bool_(), nullable=False),
        pa.field("terminated", pa.bool_(), nullable=False),
    ]
)


class InvalidEpisode(ValueError):
    """A human-readable reason why an episode cannot enter the dataset."""


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_episode(table: pa.Table) -> list[dict]:
    if not table.schema.equals(SCHEMA, check_metadata=False):
        raise InvalidEpisode("schema mismatch: expected fixed schema version 1")
    if table.num_rows == 0:
        raise InvalidEpisode("empty episode")
    if any(column.null_count for column in table.columns):
        raise InvalidEpisode("null values are forbidden")
    rows = table.to_pylist()
    seed, episode_id = rows[0]["seed"], rows[0]["episode_id"]
    if seed < 0 or episode_id < 0:
        raise InvalidEpisode("seed and episode_id must be nonnegative")
    for index, row in enumerate(rows):
        if row["seed"] != seed or row["episode_id"] != episode_id:
            raise InvalidEpisode(f"row {index}: mixed episode IDs or seeds")
        if row["timestep"] != index:
            raise InvalidEpisode(f"row {index}: timesteps must be contiguous from zero")
        for key in ("observation", "next_observation", "action"):
            values = np.asarray(row[key], dtype=float)
            if not np.isfinite(values).all():
                raise InvalidEpisode(f"row {index}: {key} contains non-finite or null values")
            limit = np.ones(ACTION_DIM) if key == "action" else np.array([0.7] * 10 + [2, 2])
            if np.any(np.abs(values) > limit + 1e-6):
                raise InvalidEpisode(f"row {index}: {key} outside allowed range")
            if key != "action":
                if not np.allclose(values[6:8], values[2:4] - values[:2], atol=1e-6):
                    raise InvalidEpisode(f"row {index}: inconsistent block relative position")
                if not np.allclose(values[8:10], values[4:6] - values[2:4], atol=1e-6):
                    raise InvalidEpisode(f"row {index}: inconsistent target relative position")
                if not np.allclose(values[4:6], rows[0]["observation"][4:6], atol=1e-6):
                    raise InvalidEpisode(f"row {index}: target changed within episode")
        if index and not np.allclose(
            row["observation"], rows[index - 1]["next_observation"], atol=1e-6
        ):
            raise InvalidEpisode(f"row {index}: broken transition continuity")
        final = index == len(rows) - 1
        if row["terminated"] != final or row["success"] != final:
            raise InvalidEpisode(f"row {index}: require one successful terminal row at episode end")
    if np.linalg.norm(np.asarray(rows[-1]["next_observation"])[8:10]) >= 0.025:
        raise InvalidEpisode("terminal success is inconsistent with target distance")
    return rows


def episode_hash(table: pa.Table) -> str:
    payload = {"schema_version": SCHEMA_VERSION, "rows": validate_episode(table)}
    return sha256(canonical_json(payload))


def collect_episode(seed: int, env: PushEnv | None = None) -> pa.Table:
    env = env or PushEnv()
    obs = env.reset(seed)
    rows = []
    done = False
    while not done:
        action = expert_action(obs)
        next_obs, success, done = env.step(action)
        rows.append(
            dict(
                episode_id=seed,
                seed=seed,
                timestep=len(rows),
                observation=obs.tolist(),
                action=action.tolist(),
                next_observation=next_obs.tolist(),
                success=success,
                terminated=done,
            )
        )
        obs = next_obs
    return pa.Table.from_pylist(rows, schema=SCHEMA)


def write_dataset(tables: list[pa.Table], root: Path, rejected: list[dict] | None = None) -> Path:
    if not tables:
        raise InvalidEpisode("dataset contains no accepted episodes")
    ordered = sorted(tables, key=lambda table: table["seed"][0].as_py())
    hashes = [episode_hash(table) for table in ordered]
    seeds = [table["seed"][0].as_py() for table in ordered]
    ids = [table["episode_id"][0].as_py() for table in ordered]
    if len(set(seeds)) != len(seeds) or len(set(ids)) != len(ids):
        raise InvalidEpisode("duplicate episode seed or ID")
    version = sha256(canonical_json({"schema_version": SCHEMA_VERSION, "episodes": hashes}))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / version
    if destination.exists():
        validate_dataset(destination)
        return destination
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    try:
        entries = []
        for table, digest, seed in zip(ordered, hashes, seeds, strict=True):
            name = f"episode-{seed:08d}.parquet"
            pq.write_table(table, staging / name, compression="zstd")
            entries.append(
                dict(file=name, seed=seed, rows=table.num_rows, content_hash=digest,
                     file_hash=sha256((staging / name).read_bytes()))
            )
        manifest = dict(
            schema_version=SCHEMA_VERSION,
            version=version,
            episodes=entries,
            rejected=rejected or [],
            generator="scripted-expert-v1",
            model_hash=sha256((Path(__file__).parent / "models/push.xml").read_bytes()),
        )
        (staging / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def generate_data(root: Path, episodes: int = 128, start_seed: int = 0) -> Path:
    if episodes < 1 or start_seed < 0:
        raise ValueError("episodes must be positive and start_seed nonnegative")
    accepted, rejected = [], []
    env = PushEnv()
    for seed in range(start_seed, start_seed + episodes):
        table = collect_episode(seed, env)
        try:
            validate_episode(table)
        except InvalidEpisode as exc:
            rejected.append(dict(seed=seed, reason=str(exc)))
        else:
            accepted.append(table)
    return write_dataset(accepted, root, rejected)


def validate_dataset(path: Path) -> list[pa.Table]:
    path = Path(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["schema_version"] != SCHEMA_VERSION or not manifest["episodes"]:
            raise InvalidEpisode("unsupported manifest schema or empty dataset")
        tables, hashes, seeds, ids, filenames = [], [], set(), set(), set()
        for entry in manifest["episodes"]:
            name = entry["file"]
            if Path(name).name != name or not name.endswith(".parquet"):
                raise InvalidEpisode("manifest contains unsafe episode path")
            file = path / name
            if sha256(file.read_bytes()) != entry["file_hash"]:
                raise InvalidEpisode(f"{name}: file hash mismatch")
            table = pq.read_table(file)
            digest = episode_hash(table)
            seed = table["seed"][0].as_py()
            episode_id = table["episode_id"][0].as_py()
            if digest != entry["content_hash"] or table.num_rows != entry["rows"]:
                raise InvalidEpisode(f"{name}: content hash or row count mismatch")
            if seed != entry["seed"] or seed in seeds or episode_id in ids or name in filenames:
                raise InvalidEpisode(f"{name}: duplicate or inconsistent episode identity")
            tables.append(table)
            hashes.append(digest)
            seeds.add(seed)
            ids.add(episode_id)
            filenames.add(name)
        if list(seeds) and [entry["seed"] for entry in manifest["episodes"]] != sorted(seeds):
            raise InvalidEpisode("manifest episodes must be sorted by seed")
        if {file.name for file in path.glob("*.parquet")} != filenames:
            raise InvalidEpisode("untracked Parquet files")
        version = sha256(canonical_json({"schema_version": SCHEMA_VERSION, "episodes": hashes}))
        if manifest["version"] != version or path.name != version:
            raise InvalidEpisode("dataset version hash mismatch")
        return tables
    except InvalidEpisode:
        raise
    except (OSError, ValueError, KeyError, TypeError, pa.ArrowException) as exc:
        raise InvalidEpisode(f"cannot read dataset: {exc}") from exc


class EpisodeDataset(Dataset):
    """Return (observation [12], future actions [K, 2], valid mask [K]).

    Windows never cross episode boundaries. Missing terminal actions are masked.
    """

    def __init__(self, path: Path, chunk_size: int = 1, seeds: set[int] | None = None):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        observations, actions, masks = [], [], []
        for table in validate_dataset(path):
            if seeds is not None and table["seed"][0].as_py() not in seeds:
                continue
            rows = table.to_pylist()
            for index, row in enumerate(rows):
                chunk = np.zeros((chunk_size, ACTION_DIM), dtype=np.float32)
                mask = np.zeros(chunk_size, dtype=np.float32)
                future = rows[index:index + chunk_size]
                chunk[:len(future)] = [step["action"] for step in future]
                mask[:len(future)] = 1
                observations.append(row["observation"])
                actions.append(chunk)
                masks.append(mask)
        if not observations:
            raise InvalidEpisode("no episodes selected by loader")
        self.observations = torch.tensor(np.asarray(observations), dtype=torch.float32)
        self.actions = torch.tensor(np.asarray(actions), dtype=torch.float32)
        self.masks = torch.tensor(np.asarray(masks), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.observations[index], self.actions[index], self.masks[index]
