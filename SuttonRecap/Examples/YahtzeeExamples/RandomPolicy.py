import numpy as np
from matplotlib import pyplot as plt

from Policies import GreedyPolicy, TabularValueFunction
from RL_Algorithms import PolicyTester, ConstantParameter
from RL_Environment.RLEnvironments import SmallYahtzee

if __name__ == '__main__':
    n_dices = 5
    n_faces = 6
    env = SmallYahtzee(n_dices=n_dices, n_faces=n_faces)

    epsilon = ConstantParameter(1)
    gamma = 0.99

    random_policy = GreedyPolicy(TabularValueFunction(env.get_possible_actions))

    test_size = 10000
    max_step = 40

    random_perf = PolicyTester.test(test_size, max_step, random_policy, env, gamma=1)
    print(f"{random_perf}")