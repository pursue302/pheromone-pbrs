"""Train a common AntMaze warm start and compare four sparse-reward SAC variants."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch

from agent import Replay, SAC

ENVIRONMENT = "AntMaze_UMaze-v5"
GOAL = (1, 1)
STARTS = {"near": GOAL, "middle": (1, 2), "far": (3, 1)}
NEAR_OFFSET = (-0.65, 0.0)
START_PROBABILITIES = (0.60, 0.25, 0.15)
METHODS = ("sac", "return_all", "fixed_mix", "return_gate")
# Fixed layout with quarter-cell position bins and a small border for clipping.
BINS = 20
OBSERVATION_SIZE = 31  # 27 ant features, two position and two goal coordinates.


@dataclass(frozen=True)
class Config:
    warmup_steps: int = 120_000
    comparison_steps: int = 100_000
    checkpoint: int = 20_000
    evaluation_episodes: int = 6
    episode_limit: int = 300
    train_every: int = 4
    comparison_train_every: int = 8
    batch_size: int = 128
    random_steps: int = 2000
    gamma: float = 0.99
    scale: float = 0.1
    publish_every: int = 10


def make_environment(config):
    gym.register_envs(gymnasium_robotics)
    env = gym.make(
        ENVIRONMENT, continuing_task=False, reward_type="sparse",
        include_cfrc_ext_in_observation=False, max_episode_steps=config.episode_limit,
    )
    # The U-maze cells are four units wide. Keep modest perturbations of both
    # start and goal within their cells, with evaluation seeds held fixed.
    env.unwrapped.position_noise_range = 0.10
    return env


def reset(env, start, seed=None):
    observation, _ = env.reset(seed=seed, options={
        "goal_cell": np.asarray(GOAL), "reset_cell": np.asarray(start),
    })
    if start == GOAL:
        # The training and evaluation near tier starts inside the goal cell,
        # outside its success radius. Preserve all other Ant state variables.
        ant = env.unwrapped.ant_env
        position = observation["desired_goal"] + NEAR_OFFSET
        qpos = ant.data.qpos.copy()
        qpos[:2] = position
        ant.set_state(qpos, ant.data.qvel.copy())
        observation["achieved_goal"] = position
    return observation


def encode(observation):
    body = observation["observation"].astype(np.float32)
    body[-14:] = np.clip(body[-14:] / 5, -5, 5)
    return np.concatenate((
        body, observation["achieved_goal"].astype(np.float32) / 6,
        observation["desired_goal"].astype(np.float32) / 6,
    ))


def bin_position(point):
    x = min(BINS - 1, max(0, int(np.floor(point[0] + 10))))
    y = min(BINS - 1, max(0, int(np.floor(point[1] + 10))))
    return y * BINS + x


def route_distance(point, goal):
    """Approximate the remaining U-route through two fixed corridor portals."""
    upper = (4, 2)
    lower = (4, -2)
    if point[1] < -2:
        return float(np.linalg.norm(point - lower) + 4 + np.linalg.norm(goal - upper))
    if point[1] < 2:
        return float(np.linalg.norm(point - upper) + np.linalg.norm(goal - upper))
    return float(np.linalg.norm(point - goal))


class Memory:
    """Weighted last-visit returns from actual successful episodes."""

    def __init__(self):
        self.weight = np.zeros(BINS * BINS)
        self.total = np.zeros_like(self.weight)
        self.count = np.zeros_like(self.weight)

    def deposit(self, positions, gamma):
        weight = 1 / len(positions)
        self.weight *= 0.99
        self.total *= 0.99
        for position, last in {b: t for t, b in enumerate(positions)}.items():
            self.weight[position] += weight
            self.total[position] += weight * gamma ** (len(positions) - 1 - last)
            self.count[position] += 1

    def potential(self, scale):
        values = np.divide(self.total, self.weight, out=np.zeros_like(self.weight),
                           where=self.weight > 0)
        return (scale * values).astype(np.float32)


def publish(method, full, recent, anchor, scale):
    if method == "sac":
        return np.zeros_like(anchor)
    if method == "return_all":
        return full.potential(scale)
    weight = 0.5 if method == "fixed_mix" else 0.5 * recent.count / (recent.count + 5)
    return ((1 - weight) * anchor + weight * recent.potential(scale)).astype(np.float32)


def evaluation(env, agent, config, *, seed_base=12_000_000):
    """Measure the frozen actor under the original sparse task, by start tier."""
    by_start = {}
    for tier_index, (tier, start) in enumerate(STARTS.items()):
        lengths = []
        for episode in range(config.evaluation_episodes):
            obs = reset(env, start, seed=seed_base + tier_index * 1000 + episode)
            for step in range(1, config.episode_limit + 1):
                obs, reward, terminated, truncated, _ = env.step(agent.act(encode(obs), greedy=True))
                if terminated or truncated:
                    lengths.append(step if reward == 1 else None)
                    break
        by_start[tier] = lengths
    flat = [length for lengths in by_start.values() for length in lengths]
    return {
        "success_rate": sum(length is not None for length in flat) / len(flat),
        "discounted_return": sum(config.gamma ** (length - 1) for length in flat
                                 if length is not None) / len(flat),
        "steps_by_start": by_start,
    }


def train_phase(env, test_env, agent, replay, memory, config, seed, method, *, checkpoint_hook=None):
    """Use a single dense prefix, then fork into four matched sparse-reward arms."""
    warmup = method == "warmup"
    steps = config.warmup_steps if warmup else config.comparison_steps
    train_every = config.train_every if warmup else config.comparison_train_every
    prefix = 0 if warmup else config.warmup_steps
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    starts = list(STARTS.values())
    online = Memory()
    anchor = memory.potential(config.scale)
    potential = publish(method, memory, online, anchor, config.scale) if not warmup else anchor
    # Give all three trajectory-shaping arms the same initial warm-up potential.
    if method in ("return_all", "fixed_mix", "return_gate"):
        potential = anchor.copy()
    records = [{"step": prefix, **evaluation(test_env, agent, config)}]
    episode_successes = completed = 0
    boundary_error = 0.0
    printed_at = time.monotonic()
    step = 0
    while step < steps:
        start = starts[int(rng.choice(len(starts), p=START_PROBABILITIES))]
        obs = reset(env, start, seed=int(rng.integers(2**31)))
        state = encode(obs)
        position = bin_position(obs["achieved_goal"])
        distance = route_distance(obs["achieved_goal"], obs["desired_goal"]) if warmup else 0.0
        bins = []
        initial_potential = potential[position]
        discount, shaped_total = 1.0, 0.0
        while step < steps:
            bins.append(position)
            action = (rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
                      if warmup and step < config.random_steps else agent.act(state))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = encode(obs)
            next_position = bin_position(obs["achieved_goal"])
            next_distance = route_distance(obs["achieved_goal"], obs["desired_goal"]) if warmup else 0.0
            replay.add(state, action, next_state, done, reward, distance - next_distance,
                       position, next_position)
            state, position, distance = next_state, next_position, next_distance
            if not warmup:
                next_potential = 0.0 if done else potential[next_position]
                shaped_total += discount * (config.gamma * next_potential - potential[bins[-1]])
                discount *= config.gamma
            step += 1
            if step % train_every == 0 and replay.size >= config.batch_size:
                batch = replay.sample(rng, config.batch_size)
                if warmup:
                    # The common pretraining policy gets a route progress hint.
                    rewards = 3 * batch["reward"] + batch["progress"] - 0.015
                else:
                    # Recompute both ends with the current published potential.
                    # This avoids stale rewards in an off-policy replay buffer.
                    mask = 1 - batch["done"][:, 0]
                    rewards = (batch["reward"] + config.gamma * mask * potential[batch["next_position"]]
                               - potential[batch["position"]])
                agent.update(batch, rewards)
            if step % config.checkpoint == 0 or step == steps:
                record = {"step": prefix + step, **evaluation(test_env, agent, config)}
                records.append(record)
                if checkpoint_hook is not None:
                    checkpoint_hook(record, episode_successes)
                elapsed = time.monotonic() - printed_at
                print(f"{seed} {method} {prefix + step}: success {record['success_rate']:.1%}"
                      f" ({elapsed:.1f}s since last report)", flush=True)
                printed_at = time.monotonic()
            if done:
                break
        if not done:
            break
        completed += 1
        if reward == 1:
            episode_successes += 1
            memory.deposit(bins, config.gamma)
            if not warmup:
                online.deposit(bins, config.gamma)
        if not warmup:
            boundary_error = max(boundary_error, abs(shaped_total + initial_potential))
            if completed % config.publish_every == 0:
                potential = publish(method, memory, online, anchor, config.scale)
    assert boundary_error < 1e-6
    return {
        "method": method, "records": records, "training_successes": episode_successes,
        "completed_episodes": completed, "max_telescoping_error": float(boundary_error),
    }


def write_record(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_warmup(job):
    seed, config, output, checkpoint = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    with make_environment(config) as env, make_environment(config) as test_env:
        agent = SAC(OBSERVATION_SIZE, env.action_space.shape[0], config.gamma)
        replay = Replay(config.warmup_steps + config.comparison_steps,
                        OBSERVATION_SIZE, env.action_space.shape[0])
        memory = Memory()
        selected = None

        def consider_checkpoint(record, successes):
            nonlocal selected
            if successes < 5:
                return
            # Checkpoint selection uses a different reset bank from reporting.
            score = evaluation(test_env, agent, config, seed_base=11_000_000)
            key = (score["success_rate"], score["discounted_return"])
            if selected is not None and key <= tuple(selected["selection_score"]):
                return
            selected = {
                "training_step": record["step"], "training_successes": successes,
                "selection_score": list(key), "evaluation": record,
            }
            if checkpoint is not None:
                # The private checkpoint is generated and consumed within this run.
                with checkpoint.open("wb") as handle:
                    pickle.dump((agent, replay, memory), handle, protocol=pickle.HIGHEST_PROTOCOL)

        warmup = train_phase(env, test_env, agent, replay, memory, config, seed, "warmup",
                             checkpoint_hook=consider_checkpoint)
    record = {"seed": seed, "warmup": warmup, "selected_checkpoint": selected, "methods": {}}
    write_record(output / f"seed_{seed}.json", record)
    return record


def run_comparison(job):
    seed, method, config, checkpoint = job
    torch.set_num_threads(1)
    with checkpoint.open("rb") as handle:
        agent, replay, memory = pickle.load(handle)
    with make_environment(config) as env, make_environment(config) as test_env:
        result = train_phase(env, test_env, agent, replay, memory,
                             config, seed + 4_000_000, method)
    return seed, method, result


def training_ready(records):
    """Require a useful common checkpoint before running comparison arms."""
    selected = [row["selected_checkpoint"] for row in records.values()]
    mean_success = np.mean([row["selection_score"][0] if row else 0 for row in selected])
    min_trajectories = min(row["training_successes"] if row else 0 for row in selected)
    return float(mean_success), min_trajectories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="Run only the shared warm-up.")
    parser.add_argument("--warmup-steps", type=int, default=120_000)
    parser.add_argument("--comparison-steps", type=int, default=100_000)
    parser.add_argument("--checkpoint", type=int, default=20_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--comparison-train-every", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=92_000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    bounds = (args.warmup_steps, args.comparison_steps, args.checkpoint,
              args.train_every, args.comparison_train_every, args.seeds, args.workers)
    if min(bounds) <= 0 or args.seed_start < 0:
        parser.error("Use positive budgets and worker counts, and a nonnegative seed.")
    args.out.mkdir(parents=True, exist_ok=False)
    config = Config(warmup_steps=args.warmup_steps, comparison_steps=args.comparison_steps,
                    checkpoint=args.checkpoint, train_every=args.train_every,
                    comparison_train_every=args.comparison_train_every)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    manifest = {
        "status": "running", "pilot": args.pilot, "environment": ENVIRONMENT,
        "config": asdict(config), "seeds": seeds, "goal_cell": GOAL,
        "start_cells": STARTS, "near_goal_offset": NEAR_OFFSET,
        "start_probabilities": START_PROBABILITIES,
        "methods": METHODS,
        "comparison_readiness": {"mean_selection_success": 0.25, "min_successful_trajectories": 5},
        "selection_reset_seed_base": 11_000_000, "evaluation_reset_seed_base": 12_000_000,
        "versions": {name: version(name) for name in ("gymnasium", "gymnasium-robotics", "mujoco", "torch")},
    }
    path = args.out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    with TemporaryDirectory(dir=args.out) as staging:
        checkpoints = {seed: Path(staging) / f"seed_{seed}.pkl" for seed in seeds}
        warmup_jobs = [(seed, config, args.out, None if args.pilot else checkpoints[seed])
                       for seed in seeds]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            records = {row["seed"]: row for row in pool.map(run_warmup, warmup_jobs)}
            mean_success, min_trajectories = training_ready(records)
            ready = mean_success >= 0.25 and min_trajectories >= 5
            manifest["common_checkpoint"] = {
                "mean_selection_success": mean_success,
                "min_successful_trajectories": min_trajectories,
                "ready": ready,
            }
            write_record(path, manifest)
            print(f"Common checkpoint: {mean_success:.1%} selection success, "
                  f"at least {min_trajectories} successes per seed, ready={ready}", flush=True)
            if not args.pilot and not ready:
                manifest["status"] = "warmup_below_threshold"
                write_record(path, manifest)
                raise RuntimeError("Common checkpoint did not meet the comparison thresholds.")
            if not args.pilot:
                futures = [
                    pool.submit(run_comparison, (seed, method, config, checkpoints[seed]))
                    for seed in seeds for method in METHODS
                ]
                for future in as_completed(futures):
                    seed, method, result = future.result()
                    record = records[seed]
                    expected = dict(record["selected_checkpoint"]["evaluation"], step=config.warmup_steps)
                    assert result["records"][0] == expected
                    record["methods"][method] = result
                    write_record(args.out / f"seed_{seed}.json", record)
                    print(f"Completed seed {seed}, {method}", flush=True)
    manifest["status"] = "complete"
    write_record(path, manifest)


if __name__ == "__main__":
    main()
