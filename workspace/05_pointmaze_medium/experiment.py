"""Compare trajectory-based shaping on the MuJoCo PointMaze Medium task."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from numba import njit

ENVIRONMENT = "PointMaze_Medium-v3"
# Cells use zero-based (row, column) coordinates. Training samples from
# these starts, while evaluation visits every start once per checkpoint.
GOAL = (1, 6)
STARTS = ((2, 5), (2, 4), (3, 4), (4, 4), (3, 2), (4, 1), (6, 1), (6, 5))
EVALUATION_STARTS = STARTS
METHODS = ("sarsa", "return_all", "fixed_mix", "return_gate")
# MuJoCo still simulates continuous positions and velocities. Only forces are discretized.
FORCES = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
], dtype=float)
# Give diagonal and cardinal actions the same force magnitude.
FORCES[5:] /= np.sqrt(2)
TILINGS = 8
TILE_SLOTS = 2048
FEATURES = TILINGS * TILE_SLOTS


@dataclass(frozen=True)
class Config:
    steps: int
    cap: int = 250
    checkpoint: int = 10_000
    gamma: float = 0.99
    lam: float = 0.9
    alpha: float = 0.1 / TILINGS
    epsilon: float = 0.15
    warmup: float = 0.25
    scale: float = 0.1
    publish_every: int = 10


@njit(cache=True)
def encode(observation):
    """Tile-code position and velocity using eight offset grids."""
    # Finer position tiles distinguish nearby corridors. Coarser velocity
    # tiles let states with similar momentum share features.
    widths = (0.4, 0.4, 1.0, 1.0)
    strides = (1, 3, 5, 7)
    tiles = np.empty(TILINGS, dtype=np.int64)
    for layer in range(TILINGS):
        key = 0
        for dimension in range(4):
            offset = (layer * strides[dimension] % TILINGS) / TILINGS
            coordinate = int(np.floor(observation[dimension] / widths[dimension] + offset))
            # Hash the four tile coordinates into a fixed-size layer.
            key = (key * 131 + coordinate) % TILE_SLOTS
        tiles[layer] = layer * TILE_SLOTS + key
    return tiles


@njit(cache=True)
def action_values(weights, tiles):
    values = np.zeros(len(FORCES))
    # Each action has its own block of tile weights in the flat array.
    for action in range(len(FORCES)):
        for tile in tiles:
            values[action] += weights[action * FEATURES + tile]
    return values


@njit(cache=True)
def true_online_update(weights, traces, active, seen, size, indices,
                       alpha, decay, error, correction):
    """Apply the Dutch trace update only to features touched this episode."""
    # The Dutch trace coefficient uses the overlap with the old trace,
    # so measure it before applying the gamma * lambda decay.
    overlap = 0.0
    for index in indices:
        overlap += traces[index]
    for entry in range(size):
        traces[active[entry]] *= decay
    increment = 1.0 - alpha * decay * overlap
    for index in indices:
        if not seen[index]:
            active[size] = index
            size += 1
            seen[index] = True
        traces[index] += increment
    # Apply the TD error and the true-online correction through the trace.
    # Then subtract the correction on the current active features only.
    for entry in range(size):
        index = active[entry]
        weights[index] += alpha * error * traces[index] + alpha * correction * traces[index]
    for index in indices:
        weights[index] -= alpha * correction
    return size


class Memory:
    """Weighted last-visit returns from distinct successful trajectories."""

    def __init__(self, size):
        self.weight = np.zeros(size)
        self.total = np.zeros(size)
        self.count = np.zeros(size, dtype=int)

    def add(self, bins, gamma):
        # A repeated bin keeps its last visit, so one route contributes at
        # most one return and one evidence count per position.
        last_visits = {position_bin: time for time, position_bin in enumerate(bins)}
        quality = 1.0 / len(bins)
        # Evaporate both parts of the weighted mean. Keep the trajectory
        # counts cumulative because the gate uses them as evidence.
        self.weight *= 0.99
        self.total *= 0.99
        for position_bin, time in last_visits.items():
            # The only nonzero reward is the final +1, so the remaining
            # return is just that reward discounted back to the visit.
            remaining_return = gamma ** (len(bins) - 1 - time)
            self.weight[position_bin] += quality
            self.total[position_bin] += quality * remaining_return
            self.count[position_bin] += 1

    def potential(self, scale):
        # Unseen states use zero, the sparse environment's return on failure.
        return scale * np.divide(
            self.total, self.weight, out=np.zeros_like(self.total), where=self.weight > 0
        )


def published_potential(method, full_memory, online_memory, anchor, scale):
    if method == "sarsa":
        return np.zeros_like(anchor)
    # This baseline accumulates warm-up and later successes in one memory.
    if method == "return_all":
        return full_memory.potential(scale)
    if method == "fixed_mix":
        online_weight = 0.5
    else:
        # The gate starts at zero online weight in each bin and approaches
        # 0.5 as distinct post-warm-up successes visit that bin.
        online_weight = 0.5 * online_memory.count / (online_memory.count + 5)
    return (1 - online_weight) * anchor + online_weight * online_memory.potential(scale)


def make_environment(config, **render_options):
    gym.register_envs(gymnasium_robotics)
    environment = gym.make(
        ENVIRONMENT, continuing_task=False, reward_type="sparse",
        max_episode_steps=config.cap, **render_options,
    )
    # PointMaze v3 exposes this on MazeEnv rather than as a constructor option.
    environment.unwrapped.position_noise_range = 0.0
    return environment


def reset(environment, start, seed=None):
    observation, _ = environment.reset(
        seed=seed, options={"reset_cell": np.array(start), "goal_cell": np.array(GOAL)}
    )
    return observation["observation"]


@njit(cache=True)
def position_bin(observation):
    """Use four potential bins per maze cell in each axis, independently of velocity."""
    x = min(31, max(0, int(np.floor((observation[0] + 4) * 4))))
    y = min(31, max(0, int(np.floor((observation[1] + 4) * 4))))
    return y * 32 + x


def evaluate(environment, weights):
    # Use a separate environment and greedy actions with the current weights.
    # Each entry is the number of steps to the goal, or None for a timeout.
    lengths = []
    for start in EVALUATION_STARTS:
        observation = reset(environment, start, seed=72_001)
        for step in range(1, environment.spec.max_episode_steps + 1):
            action = int(np.argmax(action_values(weights, encode(observation))))
            result, reward, terminated, truncated, _ = environment.step(FORCES[action])
            observation = result["observation"]
            if terminated or truncated:
                lengths.append(step if reward == 1 else None)
                break
    successes = [length for length in lengths if length is not None]
    return {
        "success_rate": len(successes) / len(EVALUATION_STARTS),
        "steps_on_success": float(np.mean(successes)) if successes else None,
        "steps_by_start": lengths,
    }


def train(environment, evaluation, method, seed, config, *, weights_out=None):
    weights = np.zeros(len(FORCES) * FEATURES)
    traces = np.zeros_like(weights)
    active = np.empty(weights.size, dtype=np.int64)
    seen = np.zeros(weights.size, dtype=np.bool_)
    # Give each method the same initial random streams for this training seed.
    start_rng = np.random.default_rng(seed)
    action_rng = np.random.default_rng(seed + 137)
    # Start without shaping. The anchor is the frozen warm-up potential.
    # Only successes after the anchor is set enter online_memory.
    full_memory, online_memory = Memory(1024), Memory(1024)
    potential = np.zeros(1024)
    anchor = None
    warmup_hash = hashlib.sha256()
    warmup_steps = warmup_successes = online_successes = episodes_since_update = 0
    step = active_size = 0
    first_success = None
    max_boundary_error = 0.0
    records = [{"step": 0, **evaluate(evaluation, weights)}]

    def choose(tiles):
        values = action_values(weights, tiles)
        if action_rng.random() < config.epsilon:
            action = int(action_rng.integers(len(FORCES)))
        else:
            # Randomize ties so an untrained sparse-reward policy can explore.
            action = int(action_rng.choice(np.flatnonzero(values == values.max())))
        return action, values[action]

    while step < config.steps:
        start = STARTS[int(start_rng.integers(len(STARTS)))]
        observation = reset(environment, start, seed=seed if step == 0 else None)
        # The learned weights are retained, while traces belong to one episode.
        touched = active[:active_size]
        traces[touched] = 0
        seen[touched] = False
        active_size = 0
        old_value = 0.0
        tiles = encode(observation)
        action, _ = choose(tiles)
        current_bin = position_bin(observation)
        bins = []
        discounted_shaping = 0.0
        initial_potential = potential[current_bin]
        discount = 1.0

        while step < config.steps:
            bins.append(current_bin)
            result, reward, terminated, truncated, _ = environment.step(FORCES[action])
            if anchor is None:
                warmup_hash.update(np.asarray(observation, dtype="<f8").tobytes())
                warmup_hash.update(bytes([action]))
            observation = result["observation"]
            next_bin = position_bin(observation)
            done = terminated or truncated
            # Success and timeout both end the capped task. Zero the next
            # potential and the bootstrap value at either boundary.
            if done:
                next_value = next_potential = 0.0
            else:
                next_tiles = encode(observation)
                next_action, next_value = choose(next_tiles)
                next_potential = potential[next_bin]
            indices = action * FEATURES + tiles
            value = weights[indices].sum()
            # Potential-based shaping supplies a discount-corrected difference.
            # Compute the SARSA target using the action chosen at the next state.
            shaping = config.gamma * next_potential - potential[current_bin]
            error = reward + shaping + config.gamma * next_value - value
            active_size = true_online_update(
                weights, traces, active, seen, active_size, indices, config.alpha,
                config.gamma * config.lam, error, value - old_value,
            )
            discounted_shaping += discount * shaping
            discount *= config.gamma
            step += 1
            if step % config.checkpoint == 0 or step == config.steps:
                records.append({"step": step, **evaluate(evaluation, weights)})
            if done:
                # With one potential per episode and a zero endpoint, the
                # discounted shaping terms should sum to -initial_potential.
                max_boundary_error = max(max_boundary_error, abs(discounted_shaping + initial_potential))
                break
            # Cache the next action value from before the weight update
            # for the true-online correction at the next step.
            tiles, action, old_value, current_bin = next_tiles, next_action, next_value, next_bin

        if not done:
            break  # Budget ended inside an episode. Only complete trajectories enter memory.
        if reward == 1:
            first_success = first_success or step
            full_memory.add(bins, config.gamma)
            if anchor is None:
                warmup_successes += 1
            else:
                online_memory.add(bins, config.gamma)
                online_successes += 1
        # End warm-up at a complete episode. Include that episode in the
        # anchor and begin shaping on the following episode.
        if anchor is None and step >= config.warmup * config.steps:
            warmup_steps = step
            anchor = full_memory.potential(config.scale)
            potential = np.zeros_like(anchor) if method == "sarsa" else anchor.copy()
        elif anchor is not None:
            episodes_since_update += 1
            # Publish at episode boundaries so a route uses one potential
            # throughout. Count both successful and timed-out episodes.
            if episodes_since_update == config.publish_every:
                potential = published_potential(method, full_memory, online_memory, anchor, config.scale)
                episodes_since_update = 0

    assert np.isfinite(weights).all()
    assert max_boundary_error < 1e-9
    if weights_out is not None:
        # Capture the final policy for playback without adding weights to the JSON results.
        np.copyto(weights_out, weights)
    return {
        "method": method, "seed": seed, "records": records,
        "warmup_steps": warmup_steps, "warmup_successes": warmup_successes,
        "warmup_hash": warmup_hash.hexdigest(), "online_successes": online_successes,
        "first_training_success": first_success, "max_boundary_error": max_boundary_error,
    }


def run_seed(job):
    seed, config, output = job
    environment, evaluation = make_environment(config), make_environment(config)
    try:
        rows = [train(environment, evaluation, method, seed, config) for method in METHODS]
        # Check the warm-up observations and actions match across all arms
        # before treating the four results as a paired comparison.
        assert len({row["warmup_hash"] for row in rows}) == 1, seed
        (output / f"seed_{seed}.json").write_text(json.dumps(rows, indent=2) + "\n")
    finally:
        environment.close()
        evaluation.close()
    return seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=81_000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1000 or args.seeds < 1 or args.workers < 1 or args.seed_start < 0:
        parser.error("Use at least 1000 steps and positive seed and worker counts.")
    args.out.mkdir(parents=True, exist_ok=False)
    config = Config(steps=args.steps)
    manifest = {
        "environment": ENVIRONMENT, "config": asdict(config), "methods": METHODS,
        "package_versions": {name: version(name) for name in (
            "gymnasium", "gymnasium-robotics", "mujoco", "numpy", "numba"
        )},
        "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "force_vectors": FORCES.tolist(), "position_noise_range": 0.0,
        "goal_cell": GOAL, "training_start_cells": STARTS, "evaluation_start_cells": EVALUATION_STARTS,
        "seeds": list(range(args.seed_start, args.seed_start + args.seeds)),
        "status": "running",
    }
    path = args.out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = [(seed, config, args.out) for seed in manifest["seeds"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for seed in pool.map(run_seed, jobs):
            print(f"Finished seed {seed}", flush=True)
    manifest["status"] = "complete"
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
