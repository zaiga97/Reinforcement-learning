import numpy as np
from matplotlib import pyplot as plt

from RL_Algorithms import TD0, PolicyTester, ExpDecayParameter, GeneralizedTDN, TDNSarsa
from RL_Algorithms.Utilities import MemoryReplayer
from RL_Environment.RLEnvironments import SmallYahtzee

if __name__ == '__main__':
    n_dices = 5
    n_faces = 6
    env = SmallYahtzee(n_dices=n_dices, n_faces=n_faces)
    memory_replayer = MemoryReplayer(1000)

    alpha = ExpDecayParameter(0.5, 2000, 0.05)
    epsilon = ExpDecayParameter(1, 2000, 0.1)
    gamma = 0.99

    sarsa = TD0(update_rule="SARSA", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env)
    exp_sarsa = TD0(update_rule="Expected SARSA", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env)
    q_learning = TD0(update_rule="Q learning", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env)
    q_learning_with_memory = TD0(update_rule="Q learning", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env,
                                 memory_replayer=memory_replayer, memory_steps=4)

    tdn = GeneralizedTDN(alpha=alpha, gamma=gamma, epsilon=epsilon, env=env, n=3, sigma=0.3)
    tdn_sarsa = TDNSarsa(alpha=alpha, gamma=gamma, epsilon=epsilon, env=env, n=3)

    # Testing the algorithms
    episodes = 20000
    episodes_between_tests = 100
    test_size = 100
    max_step = 40
    sarsa_perf = []
    exp_sarsa_perf = []
    q_learning_perf = []
    q_learning_with_memory_perf = []
    tdn_perf = []
    tdn_sarsa_perf = []
    ep = []

    for episode in range(episodes):
        if episode % episodes_between_tests == 0:
            ep.append(episode)
            sarsa_perf.append(PolicyTester.test(test_size, max_step, sarsa.greedy_policy, env, gamma=1))
            exp_sarsa_perf.append(PolicyTester.test(test_size, max_step, exp_sarsa.greedy_policy, env, gamma=1))
            q_learning_perf.append(PolicyTester.test(test_size, max_step, q_learning.greedy_policy, env, gamma=1))
            q_learning_with_memory_perf.append(
                PolicyTester.test(test_size, max_step, q_learning.greedy_policy, env, gamma=1))
            tdn_perf.append(PolicyTester.test(test_size, max_step, tdn.greedy_policy, env, gamma=1))
            tdn_sarsa_perf.append(PolicyTester.test(test_size, max_step, tdn_sarsa.greedy_policy, env, gamma=1))
            print(f"Episode: {episode}")

        sarsa.train(1, max_step)
        exp_sarsa.train(1, max_step)
        q_learning.train(1, max_step)
        q_learning_with_memory.train(1, max_step)
        tdn.train(1, max_step)
        tdn_sarsa.train(1, max_step)

    plt.plot(ep, sarsa_perf, label="SARSA")
    plt.plot(ep, exp_sarsa_perf, label="Expected SARSA")
    plt.plot(ep, q_learning_perf, label="Q learning")
    plt.plot(ep, tdn_perf, label="tdn")
    plt.plot(ep, tdn_sarsa_perf, label="tdn_sarsa")
    plt.plot(ep, q_learning_with_memory_perf, label="Q learning with memory")
    plt.legend()
    plt.show()

    np.savetxt("Tabular_yahtzee.csv",
               [p for p in zip(ep, sarsa_perf, exp_sarsa_perf, q_learning_perf, q_learning_with_memory_perf, tdn_perf,
                               tdn_sarsa_perf)],
               delimiter=',')
