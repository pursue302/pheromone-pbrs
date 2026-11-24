"""Controlled modifications of Bromuri (2019)'s released PA-SARSA code.

The original class supplies tile coding and global pheromone deposition. The
stepping loop below makes the audited fixes explicit and adds sampling feedback.
No intrinsic bonus is added to the TD reward or to pheromone trajectory quality.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import csv
from dataclasses import dataclass, asdict, replace
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from numba import njit
from author_port.PAApproximatedSarsaLambdaAgent import PAApproximatedSarsaLambdaAgent

HERE = Path(__file__).resolve().parent
METHODS = ('sarsa', 'sarsa_count', 'sarsa_feedback', 'fixed', 'fixed_count',
           'fixed_feedback', 'dynamic', 'dynamic_count', 'dynamic_feedback',
           'dynamic_random')
SCENARIOS = ('mountain_car', 'open', 'u_detour', 'four_rooms', 'slippery_rooms')
METRICS = ('original_return', 'original_discounted_return', 'success_rate', 'steps')


@dataclass
class Config:
    grid_steps: int = 30000
    mc_steps: int = 60000
    seeds: int = 20
    seed_start: int = 1000
    checkpoint: int = 5000
    epsilon: float = 0.025
    random_epsilon: float = 0.1
    gain: float = 1.0
    rho: float = 0.99
    freeze_fraction: float = 0.25
    evaluation_episodes: int = 20
    grid_cap: int = 200
    mc_cap: int = 1000


@njit(cache=True)
def feature_indices(obs, offsets, lo, hi, tiles):
    actions, dimensions, layers = offsets.shape
    memory = tiles ** dimensions
    result = np.empty((actions, layers + 1), np.int64)
    for a in range(actions):
        for layer in range(layers):
            index = 0
            for d in range(dimensions):
                normalized = (obs[d] - lo[d]) / (hi[d] - lo[d])
                value = int((normalized + offsets[a, d, layer]) * tiles)
                value = min(tiles - 1, max(0, value))
                index = index * tiles + value
            result[a, layer] = index + layer * memory
        result[a, layers] = layers * memory
    return result


@njit(cache=True)
def values(weights, indices):
    out = np.zeros(indices.shape[0])
    for a in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            out[a] += weights[a, indices[a, j]]
    return out


@njit(cache=True)
def true_online_update(w, traces, indices, action, alpha, gamma, lam, delta, delta_q):
    """Literal equations of upstream PA TrueOnline lines 283--293."""
    total = 0.0
    for index in indices:
        total += traces[action, index]
    dutch = 1.0 - alpha * gamma * lam * total
    for index in indices:
        traces[action, index] += dutch
    for a in range(w.shape[0]):
        for j in range(w.shape[1]):
            w[a, j] += alpha * (delta + delta_q) * traces[a, j]
    for index in indices:
        w[action, index] -= alpha * delta_q
    traces *= gamma * lam


@njit(cache=True)
def local_pheromone_update(tau, past, present, future, aold, action, ap, rho, layers):
    """Literal upstream PA local update, including sequential overlapping writes."""
    pv = 0.0
    fv = 0.0
    for index in past:
        pv += tau[aold, index]
    for index in future:
        fv += tau[ap, index]
    delta = (fv - pv) / (2.0 * layers)
    if delta > 0:
        for index in past:
            tau[aold, index] = rho * tau[aold, index] + (1.0 - rho) * delta
        for index in present:
            tau[action, index] = rho * tau[action, index] + (1.0 - rho) * delta
    else:
        for index in present:
            tau[action, index] = rho * tau[action, index] - (1.0 - rho) * delta
        for index in future:
            tau[ap, index] = rho * tau[ap, index] - (1.0 - rho) * delta
    for index in present:
        tau[action, index] *= rho


def concentration(pheromone_values):
    """Zero for uniform scores; tends to one when an action dominates."""
    v = np.maximum(pheromone_values, 0.0)
    return float((v.max() - v.min()) / (v.max() + 1e-12))


def exploration_bonus(counts, pheromone_values, kind, gain):
    if kind == 'none':
        return np.zeros_like(counts, dtype=float)
    gate = concentration(pheromone_values) if kind == 'feedback' else 1.0
    return gain * gate / np.sqrt(1.0 + counts)


def scenario_layout(key):
    size = 11
    board = np.full((size, size), '.', dtype='<U1')
    start, goal = (10, 0), (0, 10)
    if key == 'u_detour':
        board[4, 3:8] = '#'
        board[5:9, 3] = '#'
        board[5:9, 7] = '#'
        start, goal = (6, 5), (2, 5)
    elif key in ('four_rooms', 'slippery_rooms'):
        board[5, :] = '#'
        board[:, 5] = '#'
        board[5, (2, 8)] = '.'
        board[(2, 8), 5] = '.'
        start, goal = (9, 1), (1, 9)
    board[start] = 'S'
    board[goal] = 'G'
    return [''.join(row) for row in board]


def course_grid_class():
    path = HERE.parent / 'cs4620j_nb1/cs4620j_nb1/cs4620j/envs.py'
    spec = importlib.util.spec_from_file_location('feedback_course_env', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GridWorld


class Space:
    def __init__(self, n):
        self.n = n


class GridEnv:
    def __init__(self, key, seed, cap):
        self.key, self.cap = key, cap
        self.layout = scenario_layout(key)
        self.base = course_grid_class()(self.layout, slip=0.15 if key == 'slippery_rooms' else 0,
                                       step_reward=-1.0, goal_reward=0.0, continuing=False)
        self.rng = np.random.default_rng(seed)
        self.actions = Space(4)
        self.low, self.high = np.array([0., 0.]), np.array([10., 10.])
        self.gamma, self.learning_rate = 0.99, 0.1
        self.n_count = self.base.n_states
        self._max_episode_steps = cap
        self.observations = np.array([self.base.s_to_rc(s) for s in range(self.n_count)], float)
        self.active = np.array([s for s in range(self.n_count)
                                if s not in self.base.wall_states | self.base.terminal_states])
        self.transitions = self.base.P
        self.shortest = self._shortest()

    def _shortest(self):
        distances = {self.base.start_state: 0}
        queue = [self.base.start_state]
        for s in queue:
            for a in range(4):
                for prob, sp, _, terminal in self.transitions[s][a]:
                    if prob > 0 and sp not in distances:
                        distances[sp] = distances[s] + 1
                        queue.append(sp)
                        if terminal:
                            return distances[sp]
        raise RuntimeError('unreachable goal')

    def reset(self, position=None):
        self.s, self.t = self.base.start_state, 0
        self.last_success = False
        return self.observations[self.s]

    def step(self, action):
        ts = self.transitions[self.s][action]
        u, cumulative = self.rng.random(), 0.0
        for p, sp, reward, terminated in ts:
            cumulative += p
            if u < cumulative:
                break
        self.s, self.t = int(sp), self.t + 1
        self.last_success = bool(terminated)
        return self.observations[self.s], float(reward), bool(terminated or self.t >= self.cap), {}

    def count_bin(self, obs):
        return int(obs[0]) * 11 + int(obs[1])


class MountainCar:
    def __init__(self, key, seed, cap):
        from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
        self.base = MountainCarEnv()
        self.base.reset(seed=seed)
        self.key, self.cap = key, cap
        self.actions = self.base.action_space
        self.low = np.array([-1.2, -0.07])
        self.high = np.array([0.6, 0.07])
        self.gamma, self.learning_rate = 1.0, 0.5
        self.n_count = 30 * 30
        self._max_episode_steps = cap
        self.shortest = None

    def reset(self, position=None):
        obs, _ = self.base.reset()
        if position is not None:
            self.base.state = (float(position), 0.0)
            obs = np.array(self.base.state, dtype=np.float32)
        self.t, self.last_success = 0, False
        return obs

    def step(self, action):
        obs, reward, terminated, _, info = self.base.step(int(action))
        self.t += 1
        self.last_success = bool(terminated)
        return obs, reward, bool(terminated or self.t >= self.cap), info

    def count_bin(self, obs):
        b = np.clip(((obs - self.low) / (self.high - self.low) * 30).astype(int), 0, 29)
        return int(b[0] * 30 + b[1])


def make_env(key, seed, cfg):
    return MountainCar(key, seed, cfg.mc_cap) if key == 'mountain_car' else GridEnv(key, seed, cfg.grid_cap)


class FastCoder:
    """Same indices as the pinned upstream tiling, with a faster projection."""
    def __init__(self, features, action):
        self.features, self.action = features, action
        self.size = features.size

    def __call__(self, obs):
        return self.features(obs)[self.action]


class Features:
    def __init__(self, agent, env):
        coders = agent.tile_coder
        self.offsets = np.array([coders[a].tilings[0].offset for a in range(env.actions.n)])
        tiling = coders[0].tilings[0]
        self.lo = tiling.state_range[0].reshape(-1)
        self.hi = tiling.state_range[1].reshape(-1)
        self.tiles = int(tiling.ntiles.reshape(-1)[0])
        self.size = coders[0].size
        self.grid_cache = None
        if isinstance(env, GridEnv):
            self.grid_cache = np.array([feature_indices(obs, self.offsets, self.lo, self.hi, self.tiles)
                                        for obs in env.observations])

    def __call__(self, obs):
        if self.grid_cache is not None:
            return self.grid_cache[int(obs[0]) * 11 + int(obs[1])]
        return feature_indices(obs, self.offsets, self.lo, self.hi, self.tiles)


def create_agent(env, seed, cfg):
    config = {'Strategy': 'TrueOnline', 'heuristic_dynamic': True, 'model_based': False,
              'model': None, 'decrease_exploration': False, 'eps': cfg.epsilon,
              'learning_rate': env.learning_rate, 'lambda': 0.9, 'discount': env.gamma,
              'n_iter': env.cap, 'rho': cfg.rho, 'tile_seed': seed,
              'Pheromone_strategy': 'hard'}
    with contextlib.redirect_stdout(io.StringIO()):
        agent = PAApproximatedSarsaLambdaAgent(env.low, env.high, env.actions, [10, 10], [10], my_config=config)
    return agent


def evaluation(agent, features, key, seed, cfg):
    env = make_env(key, seed + 918273, cfg)
    returns, discounted, successes, lengths = [], [], [], []
    n = cfg.evaluation_episodes if key in ('mountain_car', 'slippery_rooms') else 1
    positions = np.linspace(-0.6, -0.4, n)
    for ep in range(n):
        obs = env.reset(positions[ep] if key == 'mountain_car' else None)
        total, disc, factor = 0.0, 0.0, 1.0
        for t in range(env.cap):
            a = int(np.argmax(values(agent.tile_code_weights, features(obs))))
            obs, r, done, _ = env.step(a)
            total += r
            disc += factor * r
            factor *= env.gamma
            if done:
                break
        returns.append(total)
        discounted.append(disc)
        successes.append(env.last_success)
        lengths.append(t + 1)
    return dict(original_return=float(np.mean(returns)),
                original_discounted_return=float(np.mean(discounted)),
                success_rate=float(np.mean(successes)), steps=float(np.mean(lengths)))


def run_one(key, method, seed, cfg):
    started = time.perf_counter()
    env = make_env(key, seed + 451, cfg)
    rng = np.random.default_rng(seed + 2027)
    agent = create_agent(env, seed, cfg)
    features = Features(agent, env)
    agent.tile_coder = {a: FastCoder(features, a) for a in range(env.actions.n)}
    weights, traces = agent.tile_code_weights, agent.tile_code_trace_weights
    counts = np.zeros((env.n_count, env.actions.n), dtype=np.int64)
    shape_on = method.startswith(('fixed', 'dynamic'))
    fixed = method.startswith('fixed')
    kind = 'feedback' if method.endswith('feedback') else 'count' if method.endswith('count') else 'none'
    budget = cfg.mc_steps if key == 'mountain_car' else cfg.grid_steps
    freeze_at = int(cfg.freeze_fraction * budget)
    frozen_copy = None
    records, episodes = [], []
    first_goal = None
    best_length = None
    concentration_sum = bonus_sum = drift_sum = 0.0
    maximum_telescoping_error = 0.0
    step = 0

    def decision(obs):
        indices = features(obs)
        q = values(weights, indices)
        pher = values(agent.pheromone_trace, indices)
        bonus = exploration_bonus(counts[env.count_bin(obs)], pher, kind, cfg.gain)
        eps = cfg.random_epsilon if method == 'dynamic_random' else cfg.epsilon
        action = int(rng.integers(env.actions.n)) if rng.random() < eps else int(np.argmax(q + bonus))
        return action, indices, q, pher, bonus

    def checkpoint():
        measured = evaluation(agent, features, key, seed, cfg)
        observed = counts.sum(axis=1) > 0
        total_sa = len(env.active) * env.actions.n if isinstance(env, GridEnv) else None
        records.append(dict(step=step, **measured, visited_bins=int(observed.sum()),
                       visited_state_actions=int(np.count_nonzero(counts)),
                       state_action_coverage=float(np.count_nonzero(counts[env.active]) / total_sa)
                           if total_sa else None,
                       mean_concentration=concentration_sum / max(step, 1),
                       mean_action_bonus=bonus_sum / max(step, 1),
                       cumulative_local_potential_drift=drift_sum))

    checkpoint()
    while step < budget:
        obs = env.reset()
        traces.fill(0.0)  # Common fix: upstream TrueOnline carries traces across episodes.
        old_q = 0.0
        action, indices, q, pher, bonus = decision(obs)
        old_potential = pher[action] / agent.num_tiling
        initial_potential = old_potential
        trajectory = {}
        past_indices = indices[action].copy()
        past_action = action
        total_return = total_shaping = 0.0
        factor = 1.0
        for t in range(env.cap):
            concentration_sum += concentration(pher)
            bonus_sum += bonus[action]
            counts[env.count_bin(obs), action] += 1
            trajectory[(float(obs[0]), float(obs[1]), action)] = {
                'state': obs.copy(), 'action': action, 'time': t}
            next_obs, reward, done, _ = env.step(action)
            ap, next_indices, next_q, next_pher, next_bonus = decision(next_obs)
            potential = 0.0 if done else next_pher[ap] / agent.num_tiling
            shaping = env.gamma * potential - old_potential
            future_value = 0.0 if done else next_q[ap]
            present_value = q[action]
            delta = reward + (shaping if shape_on else 0.0) + env.gamma * future_value - present_value
            true_online_update(weights, traces, indices[action], action, agent.alpha,
                               env.gamma, agent.lmbd, delta, present_value - old_q)
            step += 1
            updates_allowed = not fixed or step <= freeze_at
            if agent.heuristic_activate and updates_allowed:
                before = next_pher[ap]
                local_pheromone_update(agent.pheromone_trace, past_indices, indices[action],
                                       next_indices[ap], past_action, action, ap, cfg.rho, agent.num_tiling)
                after = values(agent.pheromone_trace, next_indices)[ap]
                drift_sum += abs(after - before) / agent.num_tiling
            total_return += reward
            total_shaping += factor * shaping
            factor *= env.gamma
            old_q = future_value
            old_potential = potential  # Cache arrival-time value across pheromone updates.
            past_indices = indices[action].copy()
            past_action = action
            obs, action = next_obs, ap
            indices, q, pher, bonus = next_indices, next_q, next_pher, next_bonus
            # Recompute action values after the weight update, retaining chosen SARSA action.
            q = values(weights, indices)
            pher = values(agent.pheromone_trace, indices)
            if fixed and step == freeze_at:
                frozen_copy = agent.pheromone_trace.copy()
            if done:
                success = env.last_success
                if success:
                    first_goal = step if first_goal is None else first_goal
                    best_length = min(t + 1, best_length if best_length is not None else t + 1)
                    agent.heuristic_activate = True
                if updates_allowed and agent.heuristic_activate:
                    agent.handle_trajectory(result=success, t=max(t, 1), trajectory=trajectory)
                    agent.reinforce_heuristic_model()  # Author's released global updater.
                # Global update can occur exactly at freeze boundary, so snapshot after it.
                if fixed and step == freeze_at:
                    frozen_copy = agent.pheromone_trace.copy()
                maximum_telescoping_error = max(maximum_telescoping_error,
                                                abs(total_shaping + initial_potential))
                episodes.append(dict(step=step, length=t + 1, original_return=total_return,
                                     success=bool(success)))
            if step % cfg.checkpoint == 0 or step == budget:
                checkpoint()
            if done or step >= budget:
                break
    frozen_error = float(np.max(np.abs(agent.pheromone_trace - frozen_copy))) if fixed else None
    if fixed:
        assert frozen_error == 0.0, frozen_error
    assert np.all(np.isfinite(weights)) and np.all(np.isfinite(agent.pheromone_trace))
    assert maximum_telescoping_error < 1e-7
    result = dict(scenario=key, method=method, seed=seed, records=records,
                  first_goal_step=first_goal, best_training_length=best_length,
                  completed_episodes=len(episodes), training_successes=sum(e['success'] for e in episodes),
                  frozen_pheromone_change=frozen_error, max_telescoping_error=maximum_telescoping_error,
                  wall_seconds=time.perf_counter() - started, final=records[-1])
    # Small terminal snapshots support independent follow-up inspection.
    return result, weights, agent.pheromone_trace, counts


def source_hashes():
    sources = [HERE / 'experiment.py', HERE / 'build_port.py']
    sources += sorted((HERE / 'author_port').glob('*.py'))
    return {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}


def write_task(out, key, method, seed, cfg):
    result, weights, tau, counts = run_one(key, method, seed, cfg)
    stem = f'{key}__{method}__{seed}'
    path = Path(out)
    (path / 'runs' / (stem + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(path / 'snapshots' / (stem + '.npz'), weights=weights, pheromone=tau, counts=counts)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results')
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--grid-steps', type=int, default=30000)
    parser.add_argument('--mc-steps', type=int, default=60000)
    parser.add_argument('--checkpoint', type=int, default=5000)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--gain', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--methods', nargs='+', default=list(METHODS), choices=METHODS)
    parser.add_argument('--scenarios', nargs='+', default=list(SCENARIOS), choices=SCENARIOS)
    args = parser.parse_args()
    cfg = Config(grid_steps=args.grid_steps, mc_steps=args.mc_steps, seeds=args.seeds,
                 seed_start=args.seed_start, checkpoint=args.checkpoint,
                 evaluation_episodes=args.eval_episodes, gain=args.gain)
    out = HERE / args.out
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    (out / 'snapshots').mkdir(exist_ok=True)
    manifest = dict(config=asdict(cfg), methods=args.methods, scenarios=args.scenarios,
                    python=platform.python_version(), numpy=np.__version__, source_hashes=source_hashes(),
                    author_repository='https://github.com/SB-BISS/RLACOSarsaLambda',
                    author_commit='95786b39b9ee8c5e004e71ccfa790304cbb3f684',
                    tilecoding_commit='3fd90d31539e92d229996f48c4227ef5ea4663f6',
                    status='running')
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        for field in ('config', 'methods', 'scenarios', 'source_hashes'):
            if previous[field] != manifest[field]:
                raise RuntimeError(f'Resume mismatch: {field}; choose a new output folder')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    jobs = [(key, method, seed) for key in args.scenarios for seed in range(args.seed_start, args.seed_start + args.seeds)
            for method in args.methods]
    existing = [j for j in jobs if (out/'runs'/f'{j[0]}__{j[1]}__{j[2]}.json').exists()]
    jobs = [j for j in jobs if j not in existing]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(write_task, out, key, method, seed, cfg): (key, method, seed)
                   for key, method, seed in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            if i % 10 == 0 or i == len(jobs):
                print(json.dumps(dict(finished=i + len(existing), total=len(jobs) + len(existing),
                      elapsed=round(time.perf_counter()-start, 1), scenario=r['scenario'], method=r['method'],
                      seed=r['seed'], final_return=round(r['final']['original_return'], 2))), flush=True)
    manifest.update(status='complete', elapsed_seconds=time.perf_counter()-start,
                    unique_training_steps=sum((cfg.mc_steps if key=='mountain_car' else cfg.grid_steps)
                        * len(args.methods)*cfg.seeds for key in args.scenarios))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status':'complete', 'out':str(out), 'elapsed':manifest['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
