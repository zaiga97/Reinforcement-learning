import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from Policies.Policy import DifferentiablePolicy
from RL_Algorithms import PolicyTester
from RL_Algorithms.Algorithms import ReinforceWithBaseline
from RL_Algorithms.nn_models import PolicyNN, LinearNN, SmallPolicyNN, SmallNN
from RL_Environment import RLFrozenLake

if __name__ == '__main__':

    def init_weight(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)

    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)
    onehot_env = RLFrozenLake(env, onehot_encoding=True)
    xy_env = RLFrozenLake(env, xy_encoding=True)

    onehot_policy_nn = PolicyNN(64, 4)
    onehot_baseline_nn = LinearNN(64, 1)
    xy_policy_nn = SmallPolicyNN(16, 4)
    xy_baseline_nn = SmallNN(16, 1)

    onehot_baseline_nn.apply(init_weight)
    xy_baseline_nn.apply(init_weight)
    onehot_policy_nn.apply(init_weight)
    xy_policy_nn.apply(init_weight)

    onehot_diff_policy = DifferentiablePolicy(onehot_policy_nn)
    xy_diff_policy = DifferentiablePolicy(xy_policy_nn)

    alpha_policy = 0.0005
    alpha_baseline = 0.0005
    gamma = 0.99

    onehot_pg_learning = ReinforceWithBaseline(onehot_diff_policy, onehot_baseline_nn, alpha_policy,
                                               alpha_baseline, onehot_env, gamma)
    xy_pg_learning = ReinforceWithBaseline(xy_diff_policy, xy_baseline_nn, alpha_policy, alpha_baseline,
                                           xy_env, gamma)

    episodes = 20000
    episodes_between_tests = 100
    test_size = 200
    max_step = 100
    onehot_perf = []
    xy_perf = []
    ep = []

    for episode in range(episodes):
        if episode % episodes_between_tests == 0:
            ep.append(episode)
            onehot_perf.append(PolicyTester.test(test_size, max_step, onehot_diff_policy, onehot_env, gamma))
            xy_perf.append(PolicyTester.test(test_size, max_step, xy_diff_policy, xy_env, gamma))
            print(f"Episode: {episode}  xy_perf = {xy_perf[-1]}, onehot_perf:{onehot_perf[-1]}")
            print(f"{xy_diff_policy.get_prob(xy_env.state_coord(0))}")
            print(f"{xy_baseline_nn(torch.Tensor(xy_env.state_coord(0)))}")
        onehot_pg_learning.train(1, max_step)
        xy_pg_learning.train(1, max_step)

    plt.plot(ep, onehot_perf, label="onehot")
    plt.plot(ep, xy_perf, label="xy")
    plt.legend()
    plt.show()

    np.savetxt("PolicyGrad_frozen_lake.csv",
               [p for p in zip(ep, onehot_perf, xy_perf)],
               delimiter=',')
