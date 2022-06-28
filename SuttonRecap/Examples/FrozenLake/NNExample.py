import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from Policies.ValueFunctions import NNValueFunction
from RL_Algorithms import TD0, ExpDecayParameter, ConstantParameter
from RL_Algorithms import PolicyTester
from RL_Algorithms.Utilities import MemoryReplayer
from RL_Algorithms.nn_models import LinearNN, SmallNN, init_weight
from RL_Environment import RLFrozenLake

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)
    onehot_env = RLFrozenLake(env, onehot_encoding=True)
    xy_env = RLFrozenLake(env, xy_encoding=True)

    learning_rate = 0.0005
    batch_size = 128

    onehot_nn = LinearNN(64, 4)
    xy_nn = SmallNN(16, 4)
    onehot_nn.apply(init_weight)
    xy_nn.apply(init_weight)

    onehot_optimizer = torch.optim.Adam(params=onehot_nn.parameters(), lr=learning_rate)
    xy_optimizer = torch.optim.Adam(params=xy_nn.parameters(), lr=learning_rate)
    onehot_value_function = NNValueFunction(onehot_nn, optimizer=onehot_optimizer, batch_size=batch_size)
    xy_value_function = NNValueFunction(xy_nn, optimizer=xy_optimizer, batch_size=batch_size)
    onehot_memory_replayer = MemoryReplayer(5000)
    xy_memory_replayer = MemoryReplayer(5000)

    alpha_const = ConstantParameter(1)
    epsilon = ExpDecayParameter(1, 1000)
    gamma = 0.99

    onehot_qnn_learning = TD0(update_rule="Q learning", alpha=alpha_const, gamma=gamma, epsilon=epsilon, env=onehot_env,
                              value_function=onehot_value_function, memory_replayer=onehot_memory_replayer,
                              memory_steps=4)
    xy_qnn_learning = TD0(update_rule="Q learning", alpha=alpha_const, gamma=gamma, epsilon=epsilon, env=xy_env,
                          value_function=xy_value_function, memory_replayer=xy_memory_replayer, memory_steps=4)

    # Testing the algorithm
    episodes = 10000
    episodes_between_tests = 100
    test_size = 200
    max_step = 100
    onehot_perf = []
    xy_perf = []
    ep = []

    for episode in range(episodes):
        if episode % episodes_between_tests == 0:
            ep.append(episode)
            onehot_perf.append(PolicyTester.test(test_size, max_step, onehot_qnn_learning.greedy_policy, onehot_env, gamma))
            xy_perf.append(PolicyTester.test(test_size, max_step, xy_qnn_learning.greedy_policy, xy_env, gamma))
            print(f"Episode:{episode} qnn_performance = {onehot_perf[-1]}")

        onehot_qnn_learning.train(1, max_step)
        xy_qnn_learning.train(1, max_step)

    plt.plot(ep, onehot_perf, label="Onehot")
    plt.plot(ep, xy_perf, label="XY")
    plt.legend()
    plt.show()

    torch.save(xy_nn.state_dict(), "results/NN_model")
    np.savetxt('DQN_frozen_lake.csv', [p for p in zip(ep, onehot_perf, xy_perf)], delimiter=',')
