import gym
import numpy as np
from matplotlib import pyplot as plt

from RL_Algorithms import TD0, GeneralizedTDN, ExpDecayParameter, TDNSarsa
from RL_Algorithms import PolicyTester
from RL_Environment import RLEnv

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)
    env = RLEnv(env)

    alpha = ExpDecayParameter(0.5, 2000)
    epsilon = ExpDecayParameter(1, 2000)
    gamma = 0.99

    tdn = GeneralizedTDN(alpha=alpha, gamma=gamma, epsilon=epsilon, env=env, n=4, sigma=0.5)
    tdn_sarsa = TDNSarsa(alpha=alpha, gamma=gamma, epsilon=epsilon, env=env, n=4)

    # Testing the algorithms

    episodes = 20000
    episodes_between_tests = 100
    test_size = 200
    max_step = 100

    tdn_perf = []
    tdn_sarsa_perf = []
    ep = []

    for episode in range(episodes):
        if episode % episodes_between_tests == 0:
            ep.append(episode)
            tdn_perf.append(PolicyTester.test(test_size, max_step, tdn.greedy_policy, env, gamma))
            tdn_sarsa_perf.append(PolicyTester.test(test_size, max_step, tdn_sarsa.greedy_policy, env, gamma))
            print(f"Episode: {episode}, tdn_sarsa: {tdn_sarsa_perf[-1]}")
        tdn.train(1, max_step)
        tdn_sarsa.train(1, max_step)

    plt.plot(ep, tdn_perf, label="tdn")
    plt.plot(ep, tdn_sarsa_perf, label="tdn_sarsa")
    plt.legend()
    plt.show()

    np.savetxt("TDN_frozen_lake.csv",
               [p for p in zip(ep, tdn_perf, tdn_sarsa_perf)],
               delimiter=',')
