"""Checks against the released author implementation and reward accounting."""
import unittest
from dataclasses import replace
import numpy as np

from experiment import (Config, create_agent, make_env, Features, values,
                        true_online_update, local_pheromone_update,
                        concentration, exploration_bonus, run_one)


class ReferenceChecks(unittest.TestCase):
    def test_fast_projection_matches_pinned_upstream(self):
        cfg = Config()
        for key in ('mountain_car', 'four_rooms'):
            env = make_env(key, 91, cfg)
            agent = create_agent(env, 13, cfg)
            fast = Features(agent, env)
            rng = np.random.default_rng(5)
            observations = ([env.low, env.high] + list(rng.uniform(env.low, env.high, (100, 2))))
            if key != 'mountain_car':
                observations = list(env.observations)
            for obs in observations:
                for a in range(env.actions.n):
                    np.testing.assert_array_equal(fast(obs)[a], agent.tile_coder[a](obs))

    def test_full_scripted_episode_matches_author_weights_traces_and_local_pheromone(self):
        cfg = Config()
        template = make_env('mountain_car', 1, cfg)
        author = create_agent(template, 9, cfg)
        rng = np.random.default_rng(10)
        observations = rng.uniform(template.low, template.high, (26, 2))
        actions = rng.integers(0, 3, 26)
        rewards = rng.normal(-1, 0.1, 25)
        author.tile_code_weights[:] = rng.normal(0, .01, author.tile_code_weights.shape)
        author.pheromone_trace[:] = rng.uniform(.05, .2, author.pheromone_trace.shape)
        author.tile_code_trace_weights[:] = rng.uniform(0, .01, author.tile_code_trace_weights.shape)
        w = author.tile_code_weights.copy()
        traces = author.tile_code_trace_weights.copy()
        tau = author.pheromone_trace.copy()
        author.heuristic_activate = True
        author.reinforce_heuristic_model = lambda: None
        author.handle_trajectory = lambda **kwargs: None
        author.success_invariant = lambda **kwargs: False
        action_iterator = iter(actions)
        author.act = lambda obs: int(next(action_iterator))

        class Script:
            _max_episode_steps = 26
            def reset(self):
                self.i = 0
                return observations[0].copy()
            def step(self, a):
                assert a == actions[self.i]
                reward = rewards[self.i]
                self.i += 1
                return observations[self.i].copy(), reward, self.i == 25, {}

        old_potential, old_q = 0.0, 0.0
        past = author.tile_coder[actions[0]](observations[0])
        aold = actions[0]
        for t in range(25):
            a, ap = actions[t:t+2]
            present = author.tile_coder[a](observations[t])
            future = author.tile_coder[ap](observations[t+1])
            potential = tau[ap, future].sum()
            shaping = (author.gamma * potential - old_potential) / author.num_tiling
            q = w[a, present].sum()
            qp = w[ap, future].sum() if t < 24 else 0.0
            delta = rewards[t] + shaping + author.gamma * qp - q
            true_online_update(w, traces, present, a, author.alpha, author.gamma,
                               author.lmbd, delta, q - old_q)
            local_pheromone_update(tau, past, present, future, aold, a, ap,
                                   cfg.rho, author.num_tiling)
            old_q, old_potential = qp, potential
            past, aold = present, a
        author.learn(Script())
        np.testing.assert_allclose(w, author.tile_code_weights, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(traces, author.tile_code_trace_weights, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(tau, author.pheromone_trace, atol=1e-12, rtol=1e-12)

    def test_feedback_detects_action_imbalance_not_just_state_count(self):
        counts = np.array([100, 0, 100, 100])
        uniform = np.ones(4)
        focused = np.array([10., 1., 1., 1.])
        np.testing.assert_allclose(exploration_bonus(counts, uniform, 'feedback', 1.), 0.)
        bonus = exploration_bonus(counts, focused, 'feedback', 1.)
        self.assertGreater(bonus[1], 10 * bonus[0])
        self.assertGreater(concentration(focused), .89)

    def test_zero_gain_is_identical_and_bonus_never_changes_reward_accounting(self):
        cfg = Config(grid_steps=1200, mc_steps=1200, checkpoint=600, gain=0,
                     evaluation_episodes=2)
        plain = run_one('u_detour', 'dynamic', 41, cfg)
        feedback = run_one('u_detour', 'dynamic_feedback', 41, cfg)
        for first, second in zip(plain[1:], feedback[1:]):
            np.testing.assert_array_equal(first, second)
        self.assertEqual(plain[0]['records'], feedback[0]['records'])
        self.assertEqual(plain[0]['final']['step'], cfg.grid_steps)

    def test_fixed_control_really_freezes_and_pbrs_terms_telescope(self):
        cfg = Config(grid_steps=3500, checkpoint=700, evaluation_episodes=2)
        result, _, _, _ = run_one('open', 'fixed_count', 6, cfg)
        self.assertEqual(result['frozen_pheromone_change'], 0.)
        self.assertLess(result['max_telescoping_error'], 1e-8)
        self.assertGreater(result['completed_episodes'], 0)


if __name__ == '__main__':
    unittest.main()
