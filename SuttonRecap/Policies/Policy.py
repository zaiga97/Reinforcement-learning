import random

import numpy as np
import torch
from torch import nn

from Policies.ValueFunctions import ValueFunction


class Policy:
    def get(self, *args):
        pass

    def get_prob(self, *args) -> dict:
        pass


class GreedyPolicy(Policy):
    def __init__(self, q: ValueFunction) -> None:
        self.q = q

    def get(self, state):
        actions_values = self.q.get_all(state)
        max_value = max(actions_values.values())
        best_actions = [k for (k, v) in actions_values.items() if v == max_value]
        return random.choice(best_actions)

    def get_prob(self, state) -> dict:
        actions_values = self.q.get_all(state)
        actions = list(actions_values.keys())
        max_value = max(actions_values.values())
        best_actions = [k for (k, v) in actions_values.items() if v == max_value]

        return {action: (0 if action not in best_actions else 1 / len(best_actions)) for action in actions}


class EpsilonGreedyPolicy(Policy):
    def __init__(self, base_policy: Policy) -> None:
        self.base_policy = base_policy

    def get(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(list(self.base_policy.get_prob(state).keys()))
        else:
            return self.base_policy.get(state)

    def get_prob(self, state, epsilon) -> dict:
        base_action_prob = self.base_policy.get_prob(state)
        actions = list(base_action_prob.keys())
        return {action: ((1 - epsilon) * base_action_prob[action] + epsilon / len(base_action_prob)) for action in
                actions}


class DifferentiablePolicy(Policy):
    def __init__(self, policy_nn: nn.Module) -> None:
        self.policy_nn = policy_nn

    def get(self, state):
        with torch.no_grad():
            tensor_state = torch.FloatTensor(state)
            action_probs = self.policy_nn(tensor_state)
            action_probs = action_probs.detach().numpy()
            action_size = action_probs.data.shape[0]
            action = np.random.choice(np.arange(action_size), p=action_probs)
        return action

    def get_prob(self, state) -> dict:
        with torch.no_grad():
            tensor_state = torch.FloatTensor(state)
            action_probs = self.policy_nn(tensor_state)
            action_probs = action_probs.detach().numpy()
        return action_probs
