import random
from collections import deque

from Policies import Policy
from RL_Environment.RLEnvironments import RLEnv


class Trajectory:
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.memory = list()
        self.t = -1

    def __getitem__(self, time: int) -> float:
        return self.memory[time - self.t - 1]

    def __setitem__(self, time, value) -> None:
        assert time == self.t + 1, "You can only add the next time step to the trajectory"

        if self.t < self.memory_size:
            self.memory.append(value)
        else:
            self.memory.pop(0)
            self.memory.append(value)

        self.t += 1


class PolicyTester:

    @staticmethod
    def test(episodes: int, max_steps: int, policy: Policy, env: RLEnv, gamma: float, verbose: bool = False) -> float:
        tot_reward = 0
        for episode in range(episodes):
            done = False
            step = 0
            state = env.reset()
            episode_reward = 0
            while not done and step < max_steps:
                if verbose:
                    env.render()
                action = policy.get(state)
                new_state, reward, done, info = env.step(action)
                episode_reward += gamma ** step * reward
                if verbose:
                    print(f"Reward: {reward}")
                state = new_state
                step += 1
            if verbose:
                print(f"Finished in {step} steps, total reward = {episode_reward}")
            tot_reward += episode_reward
        return tot_reward / episodes


class MemoryReplayer:

    def __init__(self, size) -> None:
        self.size = size
        self.memory = deque(maxlen=size)

    def add(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, sample_size):
        if len(self.memory) < sample_size:
            return []
        return random.sample(self.memory, sample_size)
