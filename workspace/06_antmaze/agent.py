"""A small CPU implementation of Soft Actor-Critic for continuous forces."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def network(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, outputs))


class SAC:
    def __init__(self, observation_size, action_size, gamma=0.99):
        self.actor = network(observation_size, 2 * action_size)
        self.critics = nn.ModuleList([network(observation_size + action_size, 1) for _ in range(2)])
        self.targets = nn.ModuleList([network(observation_size + action_size, 1) for _ in range(2)])
        self.targets.load_state_dict(self.critics.state_dict())
        self.targets.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=3e-4)
        self.log_temperature = torch.tensor(-2.3, requires_grad=True)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=3e-4)
        self.target_entropy = -float(action_size)
        self.gamma = gamma

    def policy(self, observations):
        mean, log_sd = self.actor(observations).chunk(2, dim=-1)
        log_sd = log_sd.clamp(-5, 1)
        noise = torch.randn_like(mean)
        raw_action = mean + log_sd.exp() * noise
        action = raw_action.tanh()
        # Include the tanh Jacobian in the log density of the bounded action.
        normal_log_prob = -0.5 * (noise.square() + np.log(2 * np.pi)) - log_sd
        log_jacobian = 2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))
        return action, (normal_log_prob - log_jacobian).sum(-1, keepdim=True)

    @torch.no_grad()
    def act(self, observation, greedy=False):
        inputs = torch.from_numpy(observation)
        if greedy:
            return self.actor(inputs).chunk(2, dim=-1)[0].tanh().numpy()
        return self.policy(inputs)[0].numpy()

    def update(self, batch, rewards):
        states, actions, next_states, done = (
            torch.from_numpy(batch[key]) for key in ("state", "action", "next_state", "done")
        )
        rewards = torch.from_numpy(rewards).reshape(-1, 1)
        temperature = self.log_temperature.detach().exp()
        with torch.no_grad():
            next_action, log_prob = self.policy(next_states)
            inputs = torch.cat((next_states, next_action), dim=-1)
            next_q = torch.minimum(self.targets[0](inputs), self.targets[1](inputs))
            target = rewards + self.gamma * (1 - done) * (next_q - temperature * log_prob)
        inputs = torch.cat((states, actions), dim=-1)
        critic_loss = sum(F.mse_loss(critic(inputs), target) for critic in self.critics)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critics.requires_grad_(False)
        action, log_prob = self.policy(states)
        inputs = torch.cat((states, action), dim=-1)
        q = torch.minimum(self.critics[0](inputs), self.critics[1](inputs))
        self.actor_optimizer.zero_grad(set_to_none=True)
        (temperature * log_prob - q).mean().backward()
        self.actor_optimizer.step()
        self.critics.requires_grad_(True)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        (-self.log_temperature * (log_prob.detach() + self.target_entropy)).mean().backward()
        self.temperature_optimizer.step()
        with torch.no_grad():
            self.log_temperature.clamp_(-7, 1)
            for target_param, critic_param in zip(self.targets.parameters(), self.critics.parameters()):
                target_param.lerp_(critic_param, 0.005)


class Replay:
    """Store original rewards and position bins so rewards can be rebuilt on sampling."""

    def __init__(self, capacity, observations, actions):
        self.arrays = {
            "state": np.empty((capacity, observations), dtype=np.float32),
            "action": np.empty((capacity, actions), dtype=np.float32),
            "next_state": np.empty((capacity, observations), dtype=np.float32),
            "done": np.empty((capacity, 1), dtype=np.float32),
            "reward": np.empty(capacity, dtype=np.float32),
            "progress": np.empty(capacity, dtype=np.float32),
            "position": np.empty(capacity, dtype=np.int32),
            "next_position": np.empty(capacity, dtype=np.int32),
        }
        self.capacity = capacity
        self.index = self.size = 0

    def add(self, state, action, next_state, done, reward, progress, position, next_position):
        values = (state, action, next_state, done, reward, progress, position, next_position)
        for array, value in zip(self.arrays.values(), values):
            array[self.index] = value
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng, count):
        indices = rng.integers(self.size, size=count)
        return {key: array[indices] for key, array in self.arrays.items()}
