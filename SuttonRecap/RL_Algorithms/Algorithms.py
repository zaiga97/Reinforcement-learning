import random

import torch.optim
from torch import nn

from Policies import EpsilonGreedyPolicy, GreedyPolicy, TabularValueFunction, ValueFunction
from Policies.Policy import DifferentiablePolicy
from Policies.ValueFunctions import NNValueFunction

from .Utilities import Trajectory, MemoryReplayer
from .Parameters import Parameter

from RL_Environment import RLEnv


class RLAlgorithm:
    def train(self, *args) -> None:
        pass


class TD0(RLAlgorithm):
    def __init__(self, update_rule: str, alpha: Parameter, gamma: float, epsilon: Parameter,
                 env: RLEnv,
                 value_function: ValueFunction = None,
                 memory_replayer: MemoryReplayer = None,
                 memory_steps: int = 100) -> None:

        match update_rule:
            case "SARSA":
                self.boot_strap = self.sarsa_bootstrap
            case "Expected SARSA":
                self.boot_strap = self.expected_sarsa_bootstrap
            case "Q learning":
                self.boot_strap = self.q_learning_bootstrap
            case _:
                raise NotImplementedError
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env
        self.trained_episodes = 0
        if value_function is None:
            self.q = TabularValueFunction(env.get_possible_actions)
        else:
            self.q = value_function

        self.memory_replayer = memory_replayer
        self.memory_steps = memory_steps
        self.greedy_policy = GreedyPolicy(self.q)
        self.epsilon_greedy_policy = EpsilonGreedyPolicy(self.greedy_policy)

    def sarsa_bootstrap(self, next_state, epsilon) -> float:
        next_action = self.epsilon_greedy_policy.get(next_state, epsilon)
        return self.q.get(next_state, next_action)

    def q_learning_bootstrap(self, next_state, _) -> float:
        next_state_values = self.q.get_all(next_state)
        return max(list(next_state_values.values()))

    def expected_sarsa_bootstrap(self, next_state, epsilon) -> float:
        next_actions = self.epsilon_greedy_policy.get_prob(next_state, epsilon)
        next_value = 0
        for action in list(next_actions.keys()):
            next_value += next_actions[action] * self.q.get(next_state, action)
        return next_value

    def train(self, episodes, max_steps):
        for episode in range(self.trained_episodes, self.trained_episodes + episodes):
            # Initialize S
            state = self.env.reset()
            done = False
            step = 0

            while not done and step < max_steps:
                # Select an action
                action = self.epsilon_greedy_policy.get(state, self.epsilon(episode))
                # Take action
                next_state, reward, done, info = self.env.step(action)
                # Choose next move
                if done:
                    next_state = None
                self.__update_value_function(state, action, next_state, reward, episode)
                if self.memory_replayer is not None:
                    self.memory_replayer.add(state, action, next_state, reward)
                    memories = self.memory_replayer.sample(self.memory_steps)
                    for s, a, ns, r in memories:
                        self.__update_value_function(s, a, ns, r, episode)
                # Update state
                state = next_state
                step += 1
        self.trained_episodes += episodes

    def __update_value_function(self, state, action, next_state, reward, episode):
        if next_state is None:
            next_value = 0
        else:
            next_value = self.boot_strap(next_state, self.epsilon(episode))
        current_value = self.q.get(state, action)
        new_value = current_value + self.alpha(episode) * (reward + (self.gamma * next_value) - current_value)
        self.q.put(state, action, new_value)


class TDNSarsa:
    def __init__(self, alpha: Parameter, gamma: float, epsilon: Parameter, env: RLEnv, n: int = 3):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.env = env
        self.trained_episodes = 0
        self.qq = TabularValueFunction(env.get_possible_actions)
        self.q = TabularValueFunction(env.get_possible_actions)
        self.greedy_policy = GreedyPolicy(self.q)
        self.epsilon_greedy_policy = EpsilonGreedyPolicy(self.greedy_policy)

    def train(self, episodes, max_steps):
        for episode in range(self.trained_episodes, self.trained_episodes + episodes):
            # Initialize variables
            once = True
            t = 0
            T = max_steps
            # Initialize trajectories
            s = Trajectory(self.n)
            a = Trajectory(self.n)
            r = Trajectory(self.n)

            s[0] = self.env.reset()
            a[0] = self.epsilon_greedy_policy.get(s[0], self.epsilon(episode))
            r[0] = 0
            while t < T + self.n:
                # Take action
                s[t + 1], r[t + 1], done, info = self.env.step(a[t])
                # Choose next move
                if done and once:
                    T = t + 1
                    once = False

                a[t + 1] = self.epsilon_greedy_policy.get(s[t + 1], self.epsilon(episode))

                # Update state value of n step ago
                tao = t - self.n + 1
                if tao >= 0:
                    G = 0
                    for k in range(min(tao + self.n, T), tao, -1):
                        G += (self.gamma ** (k - tao)) * r[k]
                    if tao + self.n < T:
                        G += (self.gamma ** self.n) * self.q.get(s[tao + self.n], a[tao + self.n])
                    old_value = self.q.get(s[tao], a[tao])
                    new_value = old_value + self.alpha(episode) * (G - old_value)
                    self.q.put(s[tao], a[tao], new_value)
                t += 1
        self.trained_episodes += episodes


class GeneralizedTDN:
    def __init__(self, alpha: Parameter, gamma: float, epsilon: Parameter, env: RLEnv, n: int = 3, sigma: float = 0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.sigma = sigma
        self.n = n
        self.env = env
        self.trained_episodes = 0
        self.q = TabularValueFunction(env.get_possible_actions)
        self.greedy_policy = GreedyPolicy(self.q)
        self.epsilon_greedy_policy = EpsilonGreedyPolicy(self.greedy_policy)

    def train(self, episodes, max_steps):
        for episode in range(self.trained_episodes, self.trained_episodes + episodes):
            # Initialize variables
            once = True
            t = 0
            T = max_steps
            # Initialize trajectories
            s = Trajectory(self.n)
            a = Trajectory(self.n)
            d = Trajectory(self.n)
            q = Trajectory(self.n)
            pi = Trajectory(self.n)
            p = Trajectory(self.n)

            s[0] = self.env.reset()
            a[0] = self.epsilon_greedy_policy.get(s[0], self.epsilon(episode))
            q[0] = self.q.get(s[0], a[0])
            pi[0] = self.greedy_policy.get_prob(s[0])[a[0]]
            p[0] = pi[0] / self.epsilon_greedy_policy.get_prob(s[0], self.epsilon(episode))[a[0]]

            while t < T + self.n:
                # Take action
                s[t + 1], reward, done, info = self.env.step(a[t])
                # Choose next move
                a[t + 1] = self.epsilon_greedy_policy.get(s[t + 1], self.epsilon(episode))
                q[t + 1] = self.q.get(s[t + 1], a[t + 1])
                pi[t + 1] = self.greedy_policy.get_prob(s[t + 1])[a[t + 1]]
                p[t + 1] = pi[t + 1] / self.epsilon_greedy_policy.get_prob(s[t + 1], self.epsilon(episode))[a[t + 1]]

                action_prob = self.greedy_policy.get_prob(s[t + 1])
                V = 0
                for action in list(action_prob.keys()):
                    V += action_prob[action] * self.q.get(s[t + 1], action)

                if done:
                    if once:
                        T = t + 1
                        once = False
                    d[t] = reward - self.q.get(s[t], a[t])
                else:
                    d[t] = reward + self.gamma * (((self.sigma * q[t + 1]) + (1 - self.sigma) * V) - q[t])

                # Update state value of n step ago
                tao = t - self.n + 1
                if tao >= 0:
                    ro = 1
                    E = 1
                    G = q[tao]
                    for k in range(tao, min(tao + self.n, T)):
                        G = G + E * d[k]
                        E = self.gamma * E * ((1 - self.sigma) * pi[k + 1] + self.sigma)
                        ro = ro * ((1 - self.sigma) + (self.sigma * p[k]))
                    old_value = q[tao]
                    new_value = old_value + self.alpha(episode) * ro * (G - old_value)
                    self.q.put(s[tao], a[tao], new_value)
                t += 1
        self.trained_episodes += episodes


class ReinforceWithBaseline(RLAlgorithm):
    def __init__(self, diff_policy: DifferentiablePolicy,
                 diff_baseline: nn.Module,
                 alpha_policy: float,
                 alpha_baseline: float,
                 env: RLEnv,
                 gamma: float) -> None:
        self.gamma = gamma
        self.env = env
        self.diff_policy = diff_policy
        self.diff_policy_nn = diff_policy.policy_nn
        self.diff_baseline = diff_baseline
        self.policy_optimizer = torch.optim.Adam(self.diff_policy_nn.parameters(), lr=alpha_policy)
        self.baseline_optimizer = torch.optim.Adam(diff_baseline.parameters(), lr=alpha_baseline)

    def train(self, episodes: int, max_steps: int) -> None:
        for episode in range(episodes):
            t = 0
            s = Trajectory(max_steps)
            a = Trajectory(max_steps)
            r = Trajectory(max_steps)

            s[0] = self.env.reset()
            r[0] = None
            # Generate an episode
            while t < max_steps:
                a[t] = self.diff_policy.get(s[t])
                next_state, r[t + 1], done, info = self.env.step(a[t])
                t += 1
                if done or t == max_steps:
                    break
                s[t] = next_state

            T = t
            G = 0
            g_t = [0 for _ in range(T)]
            for t in range(T - 1, -1, -1):
                G = self.gamma * G + r[t + 1]
                g_t[t] = G

            s_t = torch.FloatTensor(s.memory)
            a_t = torch.LongTensor(a.memory).view(-1, 1)
            g_t = torch.FloatTensor(g_t).view(-1, 1)

            b_t = self.diff_baseline(s_t)
            with torch.no_grad():
                adv_t = g_t - b_t
            action_prob_t = self.diff_policy_nn(s_t).gather(1, a_t)
            # Update the policy network

            policy_loss = torch.mean(-torch.log(action_prob_t) * adv_t)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.diff_policy_nn.parameters(), 10)
            self.policy_optimizer.step()

            # Update the baseline network
            loss_fn = nn.MSELoss()
            baseline_loss = loss_fn(b_t, g_t)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.diff_baseline.parameters(), 10)
            self.baseline_optimizer.step()


class DoubleQ(RLAlgorithm):
    def __init__(self, alpha: Parameter, gamma: float, epsilon: Parameter,
                 env: RLEnv,
                 value_function1: NNValueFunction,
                 value_function2: NNValueFunction,
                 episodes_between_swaps: int = 100,
                 memory_replayer: MemoryReplayer = None,
                 memory_steps: int = 100) -> None:

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env
        self.trained_episodes = 0
        self.episodes_between_swaps = episodes_between_swaps

        self.q = (value_function1, value_function2)
        self.memory_replayer = memory_replayer
        self.memory_steps = memory_steps
        self.greedy_policy = (GreedyPolicy(value_function1), GreedyPolicy(value_function2))
        self.epsilon_greedy_policy = (
            EpsilonGreedyPolicy(self.greedy_policy[0]), EpsilonGreedyPolicy(self.greedy_policy[1]))

    def train(self, episodes, max_steps):
        for episode in range(self.trained_episodes, self.trained_episodes + episodes):
            # Initialize S
            state = self.env.reset()
            done = False
            step = 0

            while not done and step < max_steps:
                # Select an action
                action = self.epsilon_greedy_policy[0].get(state, self.epsilon(episode))
                # Take action
                next_state, reward, done, info = self.env.step(action)
                # Choose next move
                if done:
                    next_state = None
                self.__update_value_function(state, action, next_state, reward, episode)
                if self.memory_replayer is not None:
                    self.memory_replayer.add(state, action, next_state, reward)
                    memories = self.memory_replayer.sample(self.memory_steps)
                    for s, a, ns, r in memories:
                        self.__update_value_function(s, a, ns, r, episode)
                # Update state
                state = next_state
                step += 1
            self.trained_episodes += 1
            if self.trained_episodes % self.episodes_between_swaps:
                self.__update_target()

    def __update_target(self):
        self.q[1].qnn.load_state_dict(self.q[0].qnn.state_dict())

    def __update_value_function(self, state, action, next_state, reward, episode):
        if next_state is None:
            next_value = 0
        else:
            next_value = max(list(self.q[1].get_all(next_state).values()))
        current_value = self.q[0].get(state, action)
        new_value = current_value + self.alpha(episode) * (reward + (self.gamma * next_value) - current_value)
        self.q[0].put(state, action, new_value)
