from matplotlib import pyplot as plt

from RL_Algorithms import TD0, PolicyTester, ExpDecayParameter, GeneralizedTDN, TDNSarsa
from RL_Algorithms.Utilities import MemoryReplayer
from RL_Environment.RLEnvironments import SmallYahtzee

if __name__ == '__main__':
    n_dices = 5
    n_faces = 6
    env = SmallYahtzee(n_dices=n_dices, n_faces=n_faces)
    tabular_memory_replayer = MemoryReplayer(10000)
    memory_replayer = MemoryReplayer(10000)

    alpha = ExpDecayParameter(0.5, 4000)
    epsilon = ExpDecayParameter(1, 2000, 0.1)
    gamma = 0.9

    sarsa = TD0(update_rule="SARSA", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env)
    exp_sarsa = TD0(update_rule="Expected SARSA", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env)
    q_learning = TD0(update_rule="Q learning", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env)
    q_learning_with_memory = TD0(update_rule="Q learning", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env,
                                 memory_replayer=tabular_memory_replayer, memory_steps=5)

    tdn = GeneralizedTDN(alpha=alpha, gamma=gamma, epsilon=epsilon, env=env, n=3, sigma=0.3)
    tdn_sarsa = TDNSarsa(alpha=alpha, gamma=gamma, epsilon=epsilon, env=env, n=3)

    obs_size = n_dices + n_faces + 2
    # value_nn = SmallNN(in_size=obs_size, out_size=env.n_actions)
    # optimizer = torch.optim.Adam(params=value_nn.parameters(), lr=0.005)
    # value_function = NNValueFunction(value_nn, optimizer=optimizer, batch_size=1024)
    # #
    # qnn_learning = TD0(update_rule="Q learning", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env,
    #                    value_function=value_function, memory_replayer=memory_replayer, memory_steps=4)
    #
    # policy_nn = PolicyNN(in_size=obs_size, out_size=env.n_actions)
    # baseline_nn = SmallNN(in_size=obs_size, out_size=1)
    # diff_policy = DifferentiablePolicy(policy_nn)
    # alpha_policy = 0.0001
    # alpha_baseline = 0.0002
    # pg_learning = ReinforceWithBaseline(diff_policy, baseline_nn, alpha_policy, alpha_baseline, env, gamma)

    # Testing the algorithms
    episodes = 10000
    episodes_between_tests = 200
    test_size = 10
    max_step = 40
    sarsa_perf = []
    exp_sarsa_perf = []
    q_learning_perf = []
    q_learning_with_memory_perf = []
    tdn_perf = []
    tdn_sarsa_perf = []
    qnn_learning_perf = []
    pg_learning_perf = []

    for episode in range(episodes):
        if episode % episodes_between_tests == 0:
            sarsa_perf.append(PolicyTester.test(test_size, max_step, sarsa.greedy_policy, env, gamma=1))
            exp_sarsa_perf.append(PolicyTester.test(test_size, max_step, exp_sarsa.greedy_policy, env, gamma=1))
            q_learning_perf.append(PolicyTester.test(test_size, max_step, q_learning.greedy_policy, env, gamma=1))
            q_learning_with_memory_perf.append(
                PolicyTester.test(test_size, max_step, q_learning.greedy_policy, env, gamma=1))
            tdn_perf.append(PolicyTester.test(test_size, max_step, tdn.greedy_policy, env, gamma=1))
            tdn_sarsa_perf.append(PolicyTester.test(test_size, max_step, tdn_sarsa.greedy_policy, env, gamma=1))
            # qnn_learning_perf.append(PolicyTester.test(test_size, max_step, qnn_learning.greedy_policy, env, gamma))
            # pg_learning_perf.append(PolicyTester.test(test_size, max_step, pg_learning.diff_policy, env, gamma))
            print(f"Episode: {episode}")

        sarsa.train(1, max_step)
        exp_sarsa.train(1, max_step)
        q_learning.train(1, max_step)
        q_learning_with_memory.train(1, max_step)
        tdn.train(1, max_step)
        tdn_sarsa.train(1, max_step)
        # qnn_learning.train(1, max_step)
        # pg_learning.train(1, max_step)

    plt.plot(range(len(sarsa_perf)), sarsa_perf, label="SARSA")
    plt.plot(range(len(exp_sarsa_perf)), exp_sarsa_perf, label="Expected SARSA")
    plt.plot(range(len(q_learning_perf)), q_learning_perf, label="Q learning")
    plt.plot(range(len(tdn_perf)), tdn_perf, label="tdn")
    plt.plot(range(len(tdn_sarsa_perf)), tdn_sarsa_perf, label="tdn_sarsa")
    plt.plot(range(len(q_learning_with_memory_perf)), q_learning_with_memory_perf, label="Q learning with memory")
    plt.plot(range(len(qnn_learning_perf)), qnn_learning_perf, label="Deep Qnn")
    # plt.plot(range(len(pg_learning_perf)), pg_learning_perf, label="Policy Gradient")
    plt.legend()
    plt.show()

