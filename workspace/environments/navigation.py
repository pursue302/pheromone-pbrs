import numpy as np
from gridworld import GridWorld


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
    return GridWorld


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


