"""CPU behavior cloning with one-step and open-loop action chunks."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from robot_policy_lab.data import EpisodeDataset, validate_dataset
from robot_policy_lab.sim import ACTION_DIM, OBS_DIM


class Policy(nn.Module):
    def __init__(self, chunk_size: int = 1):
        super().__init__()
        self.chunk_size = chunk_size
        self.register_buffer("mean", torch.zeros(OBS_DIM))
        self.register_buffer("scale", torch.ones(OBS_DIM))
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, ACTION_DIM * chunk_size), nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = (observation - self.mean) / self.scale
        return self.net(normalized).reshape(-1, self.chunk_size, ACTION_DIM)

    @torch.inference_mode()
    def act(self, observation: np.ndarray) -> np.ndarray:
        return self(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))[0].numpy()


def masked_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((predicted - target) ** 2) * mask.unsqueeze(-1)).sum() / (mask.sum() * ACTION_DIM)


def train(
    dataset: Path, output: Path, chunk_size: int = 4, epochs: int = 50, seed: int = 42
) -> dict:
    if epochs < 1 or chunk_size < 1:
        raise ValueError("epochs and chunk_size must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    tables = validate_dataset(dataset)
    seeds = sorted(table["seed"][0].as_py() for table in tables)
    if len(seeds) < 2:
        raise ValueError("training requires at least two episodes for a held-out split")
    split_rng = np.random.default_rng(seed)
    shuffled = split_rng.permutation(seeds).tolist()
    validation_seeds = set(shuffled[:max(1, len(shuffled) // 5)])
    training_seeds = set(seeds) - validation_seeds
    training = EpisodeDataset(dataset, chunk_size, training_seeds)
    validation = EpisodeDataset(dataset, chunk_size, validation_seeds)
    loader = DataLoader(
        training, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    model = Policy(chunk_size)
    model.mean.copy_(training.observations.mean(0))
    model.scale.copy_(training.observations.std(0).clamp(min=0.01))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    def score(data: EpisodeDataset) -> float:
        with torch.inference_mode():
            return float(masked_loss(model(data.observations), data.actions, data.masks))

    initial = score(training)
    history = []
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        for observations, targets, mask in loader:
            loss = masked_loss(model(observations), targets, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        validation_loss = score(validation)
        history.append(dict(epoch=epoch + 1, train_loss=score(training), val_loss=validation_loss))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        chunk_size=chunk_size, epochs=epochs, seed=seed, dataset_version=Path(dataset).name,
        training_seeds=sorted(training_seeds), validation_seeds=sorted(validation_seeds),
        initial_loss=initial, final_loss=score(training), validation_loss=score(validation),
        training_rows=len(training), validation_rows=len(validation), history=history,
    )
    torch.save(dict(state_dict=model.state_dict(), chunk_size=chunk_size, metadata=metadata), output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_policy(path: Path) -> tuple[Policy, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Policy(checkpoint["chunk_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["metadata"]
