import gym
import numpy as np
from matplotlib import pyplot as plt

from RL_Algorithms import TD0, PolicyTester, ExpDecayParameter
from RL_Algorithms.Utilities import MemoryReplayer
from RL_Environment import RLEnv

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)
    env = RLEnv(env)
    memory_replayer = MemoryReplayer(1000)

    alpha = ExpDecayParameter(0.5, 2000)
    epsilon = ExpDecayParameter(1, 2000)
    gamma = 0.99

    sarsa = TD0(update_rule="SARSA", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env)
    exp_sarsa = TD0(update_rule="Expected SARSA", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env)
    q_learning = TD0(update_rule="Q learning", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env)
    q_learning_with_memory = TD0(update_rule="Q learning", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env,
                                 memory_replayer=memory_replayer, memory_steps=4)

    # Testing the algorithms
    episodes = 10000
    episodes_between_tests = 100
    test_size = 200
    max_step = 100
    sarsa_perf = []
    exp_sarsa_perf = []
    q_learning_perf = []
    q_learning_with_memory_perf = []
    ep = []

    for episode in range(episodes):
        if episode % episodes_between_tests == 0:
            ep.append(episode)
            sarsa_perf.append(PolicyTester.test(test_size, max_step, sarsa.greedy_policy, env, gamma))
            exp_sarsa_perf.append(PolicyTester.test(test_size, max_step, exp_sarsa.greedy_policy, env, gamma))
            q_learning_perf.append(PolicyTester.test(test_size, max_step, q_learning.greedy_policy, env, gamma))
            q_learning_with_memory_perf.append(
                PolicyTester.test(test_size, max_step, q_learning_with_memory.greedy_policy, env, gamma))
            print(f"Episode: {episode}, q_performance: {q_learning_perf[-1]}")

        sarsa.train(1, max_step)
        exp_sarsa.train(1, max_step)
        q_learning.train(1, max_step)
        q_learning_with_memory.train(1, max_step)

    plt.plot(ep, sarsa_perf, label="SARSA")
    plt.plot(ep, exp_sarsa_perf, label="Expected SARSA")
    plt.plot(ep, q_learning_perf, label="Q learning")
    plt.plot(ep, q_learning_with_memory_perf, label="Q learning with memory")
    plt.legend()
    plt.show()

    np.savetxt("TD0_frozen_lake.csv",
               [p for p in zip(ep, sarsa_perf, exp_sarsa_perf, q_learning_perf, q_learning_with_memory_perf)],
               delimiter=',')
