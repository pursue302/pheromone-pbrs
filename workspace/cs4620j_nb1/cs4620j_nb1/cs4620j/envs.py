"""Tabular environments used across the four notebooks.

All environments here follow a small Gymnasium-compatible interface:

    s, info = env.reset(seed=...)
    s2, reward, terminated, truncated, info = env.step(action)

Tabular environments additionally expose an explicit model::

    env.n_states, env.n_actions
    env.P[s][a] -> list of (probability, next_state, reward, terminated)

which is the same structure Gymnasium's toy-text environments expose as
``env.unwrapped.P``. Notebook 1 turns this dictionary into dense arrays.
"""

from __future__ import annotations

import numpy as np

# Action encoding shared by every grid environment in this file.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ["up", "right", "down", "left"]
ACTION_DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


def get_P_dict(env) -> dict:
    """Return the ``P`` transition dictionary of a tabular environment.

    Works for both the course environments in this module and Gymnasium
    toy-text environments (``FrozenLake-v1``, ``CliffWalking-v0``), which keep
    it on the unwrapped instance.
    """
    if hasattr(env, "P"):
        return env.P
    return env.unwrapped.P


class TabularEnv:
    """Base class: sampling ``step`` is derived from the explicit model ``P``."""

    n_states: int
    n_actions: int
    P: dict
    start_state: int

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng()
        self.s = self._start_state()
        return self.s, {}

    def _start_state(self) -> int:
        return self.start_state

    def step(self, action: int):
        transitions = self.P[self.s][int(action)]
        probs = [t[0] for t in transitions]
        idx = self._rng.choice(len(transitions), p=probs)
        _, next_state, reward, terminated = transitions[idx]
        self.s = next_state
        return next_state, float(reward), bool(terminated), False, {}


# --------------------------------------------------------------------------
# GridWorld (Notebook 1)
# --------------------------------------------------------------------------

# Used by Notebook 1 experiment E2 (risk vs slip). Six rows give the optimal
# path room to back away from the cliff by more than one cell as slip grows.
CLIFF_LAYOUT = [
    "..........",
    "..........",
    "..........",
    "..........",
    "S........G",
    ".CCCCCCCC.",
]

# Used by Notebook 1 experiment E1 (discounting). A small reward sits next to
# the start and a large one is eight steps away, so gamma decides which is
# worth going for -- the discount changes the optimal policy, not just the
# scale of the values.
TWO_GOAL_LAYOUT = [
    ".........",
    ".........",
    "gS......G",
    ".........",
    ".........",
]

OPEN_LAYOUT = [
    "........",
    "..##....",
    "..##....",
    "S......G",
]


class GridWorld(TabularEnv):
    """Configurable grid world with an explicit transition model.

    The layout is a list of equal-length strings::

        '.'  ordinary cell        'S'  start
        '#'  wall (unreachable)   'G'  goal        (terminal)
                                  'g'  small goal  (terminal)
                                  'C'  cliff       (terminal)

    Dynamics: the intended action is taken with probability ``1 - slip``; the
    remaining probability is split evenly between the two perpendicular
    actions. Moving into a wall or off the grid leaves the agent in place.

    Parameters
    ----------
    layout : list of str
        Grid specification. Defaults to an 4x8 cliff world.
    slip : float
        Probability of *not* moving in the intended direction.
    continuing : bool
        If ``True``, entering a goal or cliff cell pays its reward and sends the
        agent back to the start instead of terminating the episode.
    step_reward, goal_reward, small_goal_reward, cliff_reward : float
        Reward received on entering an ordinary / goal / small-goal / cliff
        cell.
    """

    def __init__(
        self,
        layout=None,
        slip: float = 0.0,
        step_reward: float = -1.0,
        goal_reward: float = 0.0,
        small_goal_reward: float = 5.0,
        cliff_reward: float = -100.0,
        continuing: bool = False,
    ):
        self.layout = [row for row in (layout or CLIFF_LAYOUT)]
        self.n_rows, self.n_cols = len(self.layout), len(self.layout[0])
        self.slip = float(slip)
        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)
        self.small_goal_reward = float(small_goal_reward)
        self.cliff_reward = float(cliff_reward)
        self.continuing = bool(continuing)
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4
        self.start_state = self._find("S")
        self.terminal_states = {
            self._rc_to_s(r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self.layout[r][c] in "GgC"
        }
        self.wall_states = {
            self._rc_to_s(r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self.layout[r][c] == "#"
        }
        self.P = self._build_P()

    # -- geometry ----------------------------------------------------------
    def _rc_to_s(self, r: int, c: int) -> int:
        return r * self.n_cols + c

    def s_to_rc(self, s: int) -> tuple[int, int]:
        return divmod(int(s), self.n_cols)

    def _find(self, ch: str) -> int:
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.layout[r][c] == ch:
                    return self._rc_to_s(r, c)
        raise ValueError(f"layout contains no '{ch}' cell")

    def _move(self, r: int, c: int, action: int) -> tuple[int, int]:
        dr, dc = ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
            return r, c
        if self.layout[nr][nc] == "#":
            return r, c
        return nr, nc

    # -- model -------------------------------------------------------------
    def _outcome(self, r: int, c: int, action: int):
        nr, nc = self._move(r, c, action)
        cell = self.layout[nr][nc]
        if self.continuing and cell in "GgC":
            # Continuing formulation: collect the reward and teleport back to
            # the start instead of terminating. No absorbing states means the
            # policy transition matrix is row-stochastic, so the Bellman
            # operator contracts at exactly gamma -- which is what makes the
            # textbook rate observable in Notebook 1 experiment E3.
            reward = {"G": self.goal_reward, "g": self.small_goal_reward,
                      "C": self.cliff_reward}[cell]
            return self.start_state, reward, False
        if cell == "G":
            return self._rc_to_s(nr, nc), self.goal_reward, True
        if cell == "g":
            return self._rc_to_s(nr, nc), self.small_goal_reward, True
        if cell == "C":
            return self._rc_to_s(nr, nc), self.cliff_reward, True
        return self._rc_to_s(nr, nc), self.step_reward, False

    def _build_P(self) -> dict:
        P = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        for s in range(self.n_states):
            r, c = self.s_to_rc(s)
            for a in range(self.n_actions):
                if (s in self.terminal_states and not self.continuing) or s in self.wall_states:
                    # Absorbing with zero reward: V(terminal) = 0 falls out of
                    # the Bellman equation without any special-casing.
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                if self.slip <= 0.0:
                    outcomes = [(1.0, a)]
                else:
                    perpendicular = [(a + 1) % 4, (a - 1) % 4]
                    outcomes = [(1.0 - self.slip, a)]
                    outcomes += [(self.slip / 2.0, p) for p in perpendicular]
                merged: dict = {}
                for prob, act in outcomes:
                    key = self._outcome(r, c, act)
                    merged[key] = merged.get(key, 0.0) + prob
                P[s][a] = [(p, s2, rew, done) for (s2, rew, done), p in merged.items()]
        return P

    def render_policy(self, policy: np.ndarray) -> str:
        """ASCII rendering of a deterministic policy (for quick eyeballing)."""
        arrows = "^>v<"
        rows = []
        for r in range(self.n_rows):
            row = ""
            for c in range(self.n_cols):
                s = self._rc_to_s(r, c)
                cell = self.layout[r][c]
                row += cell if cell in "#GgC" else arrows[int(policy[s])]
            rows.append(row)
        return "\n".join(rows)


# --------------------------------------------------------------------------
# Random walk (Notebook 2, Sutton & Barto Example 6.2)
# --------------------------------------------------------------------------


class RandomWalk(TabularEnv):
    """The 5-state random walk. States are 0..n-1 plus two terminals.

    Every step moves left or right with probability 1/2 regardless of the
    action, so any policy is the same policy. Reward is 0 everywhere except on
    entering the right terminal, which gives +1. The true values are therefore
    ``i / (n + 1)`` for the i-th non-terminal state, which the notebook uses as
    ground truth for the RMS error curves.
    """

    def __init__(self, n: int = 5):
        self.n = int(n)
        self.n_states = self.n + 2  # index 0 and n+1 are terminal
        self.n_actions = 2
        self.start_state = (self.n + 1) // 2
        self.terminal_states = {0, self.n + 1}
        self.true_values = np.array(
            [0.0] + [(i + 1) / (self.n + 1) for i in range(self.n)] + [0.0]
        )
        self.P = self._build_P()

    def _build_P(self) -> dict:
        P = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if s in self.terminal_states:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                outcomes = []
                for step in (-1, +1):
                    s2 = s + step
                    reward = 1.0 if s2 == self.n + 1 else 0.0
                    outcomes.append((0.5, s2, reward, s2 in self.terminal_states))
                P[s][a] = outcomes
        return P

    def non_terminal_states(self) -> np.ndarray:
        return np.arange(1, self.n + 1)


# --------------------------------------------------------------------------
# Windy gridworld (Notebook 2, Sutton & Barto Example 6.5)
# --------------------------------------------------------------------------


class WindyGridworld:
    """7x10 windy gridworld with deterministic upward wind per column."""

    WIND = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)

    def __init__(self, n_rows: int = 7, n_cols: int = 10):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.n_states = n_rows * n_cols
        self.n_actions = 4
        self.start_rc = (3, 0)
        self.goal_rc = (3, 7)
        self.start_state = self._rc_to_s(*self.start_rc)
        self.goal_state = self._rc_to_s(*self.goal_rc)
        self.terminal_states = {self.goal_state}
        self._rng = np.random.default_rng()

    def _rc_to_s(self, r: int, c: int) -> int:
        return r * self.n_cols + c

    def s_to_rc(self, s: int) -> tuple[int, int]:
        return divmod(int(s), self.n_cols)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.s = self.start_state
        return self.s, {}

    def step(self, action: int):
        r, c = self.s_to_rc(self.s)
        dr, dc = ACTION_DELTAS[int(action)]
        nr = r + dr - self.WIND[c]
        nc = c + dc
        nr = int(np.clip(nr, 0, self.n_rows - 1))
        nc = int(np.clip(nc, 0, self.n_cols - 1))
        self.s = self._rc_to_s(nr, nc)
        terminated = self.s == self.goal_state
        return self.s, -1.0, terminated, False, {}


# --------------------------------------------------------------------------
# Dyna mazes (Notebook 4, Sutton & Barto Examples 8.2 and 8.3)
# --------------------------------------------------------------------------


class ChangingMaze:
    """6x9 maze whose wall layout changes once, at ``change_step``.

    ``BlockingMaze`` and ``ShortcutMaze`` below are the two instances used in
    Sutton & Barto. The environment counts *total interaction steps* (not
    episodes) and swaps its barrier the first time the count passes
    ``change_step``, which is what makes the plots in section E comparable to
    the book's figures.
    """

    N_ROWS, N_COLS = 6, 9
    START_RC = (5, 3)
    GOAL_RC = (0, 8)

    def __init__(self, barrier_before, barrier_after, change_step: int):
        self.barrier_before = set(barrier_before)
        self.barrier_after = set(barrier_after)
        self.change_step = int(change_step)
        self.n_states = self.N_ROWS * self.N_COLS
        self.n_actions = 4
        self.start_state = self._rc_to_s(*self.START_RC)
        self.goal_state = self._rc_to_s(*self.GOAL_RC)
        self.total_steps = 0
        self._rng = np.random.default_rng()

    def _rc_to_s(self, r: int, c: int) -> int:
        return r * self.N_COLS + c

    def s_to_rc(self, s: int) -> tuple[int, int]:
        return divmod(int(s), self.N_COLS)

    @property
    def barrier(self) -> set:
        return (
            self.barrier_before
            if self.total_steps < self.change_step
            else self.barrier_after
        )

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.s = self.start_state
        return self.s, {}

    def reset_counter(self) -> None:
        """Reset the global step counter (call once before a full run)."""
        self.total_steps = 0

    def step(self, action: int):
        r, c = self.s_to_rc(self.s)
        dr, dc = ACTION_DELTAS[int(action)]
        nr = int(np.clip(r + dr, 0, self.N_ROWS - 1))
        nc = int(np.clip(c + dc, 0, self.N_COLS - 1))
        if (nr, nc) in self.barrier:
            nr, nc = r, c
        self.s = self._rc_to_s(nr, nc)
        self.total_steps += 1
        terminated = self.s == self.goal_state
        return self.s, (1.0 if terminated else 0.0), terminated, False, {}


def BlockingMaze(change_step: int = 1000) -> ChangingMaze:
    """Short path on the right closes; a longer path on the left opens."""
    before = [(3, c) for c in range(0, 8)]
    after = [(3, c) for c in range(1, 9)]
    return ChangingMaze(before, after, change_step)


def ShortcutMaze(change_step: int = 3000) -> ChangingMaze:
    """A shorter path opens on the right while the old path stays available."""
    before = [(3, c) for c in range(1, 9)]
    after = [(3, c) for c in range(1, 8)]
    return ChangingMaze(before, after, change_step)


# --------------------------------------------------------------------------
# Bandits (Notebook 4, Sutton & Barto Chapter 2)
# --------------------------------------------------------------------------


class BanditTestbed:
    """The 10-armed testbed: q*(a) ~ N(0, 1), reward ~ N(q*(a), 1)."""

    def __init__(self, n_arms: int = 10, seed: int | None = None):
        self.n_arms = int(n_arms)
        self.rng = np.random.default_rng(seed)
        self.q_star = self.rng.normal(0.0, 1.0, size=self.n_arms)
        self.optimal_action = int(np.argmax(self.q_star))

    def pull(self, arm: int) -> float:
        return float(self.rng.normal(self.q_star[int(arm)], 1.0))

    @property
    def optimal_value(self) -> float:
        return float(self.q_star[self.optimal_action])


# --------------------------------------------------------------------------
# Matrix games (Notebook 4, multi-agent)
# --------------------------------------------------------------------------

MATRIX_GAMES = {
    # payoff[i, j] = (reward to player 0, reward to player 1)
    "matching_pennies": np.array(
        [[[+1.0, -1.0], [-1.0, +1.0]], [[-1.0, +1.0], [+1.0, -1.0]]]
    ),
    "prisoners_dilemma": np.array(
        [[[-1.0, -1.0], [-3.0, 0.0]], [[0.0, -3.0], [-2.0, -2.0]]]
    ),
}

ACTION_LABELS = {
    "matching_pennies": ("heads", "tails"),
    "prisoners_dilemma": ("cooperate", "defect"),
}


class MatrixGame:
    """Stateless two-player, two-action repeated game.

    ``step`` takes one action per player and returns one reward per player.
    There is no state, which is exactly the point: any non-stationarity the
    learners experience comes purely from each other.
    """

    def __init__(self, name: str = "matching_pennies"):
        if name not in MATRIX_GAMES:
            raise ValueError(f"unknown game {name!r}; choose from {list(MATRIX_GAMES)}")
        self.name = name
        self.payoff = MATRIX_GAMES[name]
        self.action_labels = ACTION_LABELS[name]
        self.n_players = 2
        self.n_actions = 2

    def step(self, actions) -> np.ndarray:
        a0, a1 = int(actions[0]), int(actions[1])
        return self.payoff[a0, a1].copy()


# --------------------------------------------------------------------------
# Blackjack (Notebook 2)
# --------------------------------------------------------------------------

BLACKJACK_N_STATES = 32 * 11 * 2


def blackjack_index(obs) -> int:
    """Map a Gymnasium Blackjack observation to a flat integer state.

    ``obs`` is ``(player_sum, dealer_showing, usable_ace)``. The flat index is
    ``(player_sum * 11 + dealer) * 2 + usable_ace``, so ``BLACKJACK_N_STATES``
    covers every reachable combination with room to spare.
    """
    player, dealer, ace = int(obs[0]), int(obs[1]), int(bool(obs[2]))
    return (player * 11 + dealer) * 2 + ace


class BlackjackTabular:
    """Gymnasium ``Blackjack-v1`` with integer states instead of tuples.

    Actions: 0 = stick, 1 = hit. Episodes are short (rarely more than three
    steps) and undiscounted, which is why Notebook 2 uses gamma = 1 here.
    """

    n_actions = 2
    n_states = BLACKJACK_N_STATES

    def __init__(self, natural: bool = False):
        import gymnasium as gym

        self.env = gym.make("Blackjack-v1", natural=natural, sab=False)

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        return blackjack_index(obs), info

    def step(self, action: int):
        obs, reward, term, trunc, info = self.env.step(int(action))
        return blackjack_index(obs), float(reward), bool(term), bool(trunc), info

    @staticmethod
    def stick_on_20_policy() -> np.ndarray:
        """The fixed policy evaluated in Notebook 2: stick on 20 or 21, else hit."""
        policy = np.ones(BLACKJACK_N_STATES, dtype=int)  # 1 = hit
        for player in range(32):
            for dealer in range(11):
                for ace in range(2):
                    if player >= 20:
                        policy[(player * 11 + dealer) * 2 + ace] = 0  # stick
        return policy

    @staticmethod
    def value_grid(V: np.ndarray, usable_ace: bool) -> np.ndarray:
        """Extract the ``(10, 10)`` surface for player sums 12-21, dealer 1-10."""
        ace = int(bool(usable_ace))
        grid = np.zeros((10, 10))
        for i, player in enumerate(range(12, 22)):
            for j, dealer in enumerate(range(1, 11)):
                grid[i, j] = V[(player * 11 + dealer) * 2 + ace]
        return grid


def make_cliffwalking():
    """Gymnasium ``CliffWalking-v1`` with the geometry attributes plotting needs.

    Note the version: ``CliffWalking-v0`` is deprecated in Gymnasium >= 1.3 and
    raises ``DeprecatedEnv``. The 4x12 layout, the -100 cliff and the reset-to-
    start-on-falling behaviour are the ones from Sutton & Barto Example 6.6.
    """
    import gymnasium as gym

    env = gym.make("CliffWalking-v1")
    u = env.unwrapped
    u.n_states = int(u.observation_space.n)
    u.n_actions = int(u.action_space.n)
    u.n_rows, u.n_cols = u.shape
    u.start_state = 36
    u.goal_state = 47
    u.terminal_states = {47}
    u.wall_states = set()
    u.layout = ["." * 12, "." * 12, "." * 12, "S" + "C" * 10 + "G"]
    u.s_to_rc = lambda s: divmod(int(s), u.n_cols)
    return env
