"""Paired seeded evaluations, raw episode outcomes, and failure analysis."""

import json
import math
from pathlib import Path

import numpy as np
import torch

from robot_policy_lab.policy import load_policy
from robot_policy_lab.sim import PushEnv, expert_action


def wilson_interval(successes: int, total: int) -> list[float]:
    z = 1.96
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def evaluate(
    one_step: Path | None = None, chunked: Path | None = None,
    episodes: int = 128, start_seed: int = 10000, output: Path | None = None,
) -> dict:
    if episodes < 1 or start_seed < 0:
        raise ValueError("episodes must be positive and start_seed nonnegative")
    torch.set_num_threads(1)
    models, metadata = {}, {}
    for name, path in (("one_step", one_step), ("chunked", chunked)):
        if path is not None:
            models[name], metadata[name] = load_policy(path)
    if "one_step" in models and models["one_step"].chunk_size != 1:
        raise ValueError("one_step checkpoint must predict exactly one action")
    if "chunked" in models and models["chunked"].chunk_size < 2:
        raise ValueError("chunked checkpoint must predict at least two actions")
    evaluation_seeds = set(range(start_seed, start_seed + episodes))
    for details in metadata.values():
        used = set(details["training_seeds"]) | set(details["validation_seeds"])
        if used & evaluation_seeds:
            raise ValueError("evaluation seeds overlap training or validation episodes")
    if len({item["dataset_version"] for item in metadata.values()}) > 1:
        raise ValueError("comparison checkpoints must use the same dataset version")
    env = PushEnv()
    results = {}
    for name in ["random", "expert", *models]:
        outcomes = []
        for seed in sorted(evaluation_seeds):
            observation = env.reset(seed)
            initial_distance = env.distance
            random = np.random.default_rng(seed + 1000000)
            queue = []
            done = success = False
            while not done:
                if name == "random":
                    action = random.uniform(-1, 1, 2).astype(np.float32)
                elif name == "expert":
                    action = expert_action(observation)
                else:
                    if not queue:
                        queue = list(models[name].act(observation))
                    action = queue.pop(0)
                observation, success, done = env.step(action)
            outcomes.append(dict(
                seed=seed, success=success, steps=env.steps, initial_distance=initial_distance,
                final_distance=env.distance,
                final_block_position=observation[2:4].tolist(),
                target_position=observation[4:6].tolist(),
                failure_reason=None if success else (
                    "moved_away" if env.distance > initial_distance
                    else "timeout_outside_stable_goal"
                ),
            ))
        successes = sum(item["success"] for item in outcomes)
        failures = [item for item in outcomes if not item["success"]]
        results[name] = dict(
            successes=successes, episodes=episodes, success_rate=successes / episodes,
            wilson_95=wilson_interval(successes, episodes),
            mean_final_distance=float(np.mean([item["final_distance"] for item in outcomes])),
            failure_counts={
                reason: sum(item["failure_reason"] == reason for item in failures)
                for reason in sorted({item["failure_reason"] for item in failures})
            },
            outcomes=outcomes,
        )
    report = dict(
        start_seed=start_seed, episodes=episodes, policies=results,
        checkpoints={name: {key: value for key, value in item.items() if key != "history"}
                     for name, item in metadata.items()},
        protocol="shared held-out seeds; execute all K actions before observing again; CPU",
    )
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report
