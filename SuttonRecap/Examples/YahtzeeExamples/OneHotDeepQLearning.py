import numpy as np
import torch
from matplotlib import pyplot as plt

from Policies.ValueFunctions import NNValueFunction
from RL_Algorithms import ExpDecayParameter, TD0, ConstantParameter
from RL_Algorithms.Utilities import MemoryReplayer, PolicyTester
from RL_Environment.RLEnvironments import SmallYahtzee
from RL_Algorithms.nn_models import SmallNN

if __name__ == "__main__":
    n_dices = 5
    n_faces = 6
    env = SmallYahtzee(n_dices=n_dices, n_faces=n_faces, onehot_encoding=True, punishment=True)
    test_env = SmallYahtzee(n_dices=n_dices, n_faces=n_faces, onehot_encoding=True, punishment=False)
    memory_replayer = MemoryReplayer(1000)

    alpha = ConstantParameter(1)
    learning_rate = 0.00025
    epsilon = ExpDecayParameter(1, 2000, 0.1)
    gamma = 0.99
    batch_size = 32

    def init_weight(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
            torch.nn.init.normal_(m.bias, mean=0.0, std=0.1)

    obs_size = n_faces + 3 + n_dices * n_faces + n_faces
    value_nn = SmallNN(in_size=obs_size, out_size=env.n_actions, hidden_size=(256, 128)).apply(init_weight)
    optimizer = torch.optim.RMSprop(params=value_nn.parameters(), lr=learning_rate)
    value_function = NNValueFunction(value_nn, optimizer=optimizer, batch_size=batch_size)
    #
    qnn_learning = TD0(update_rule="Q learning", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env,
                       value_function=value_function, memory_replayer=memory_replayer, memory_steps=4)

    episodes = 10000
    episodes_between_tests = 100
    test_size = 20
    max_step = 40

    perf = []
    ep = []

    for episode in range(episodes):
        if episode % episodes_between_tests == 0:
            ep.append(episode)
            perf.append(PolicyTester.test(test_size, max_step, qnn_learning.greedy_policy, test_env, gamma=1))
            print(f"Episode: {episode}: {perf[-1]}")
        qnn_learning.train(1, max_step)

    plt.plot(ep, perf, label="Deep Qnn")
    plt.legend()
    plt.show()

    torch.save(value_nn.state_dict(), "results/NN_model")

    PolicyTester.test(3, max_step, qnn_learning.greedy_policy, env, gamma=1, verbose=True)

    np.savetxt("Onehot_DQN_yahtzee.csv",
               [p for p in zip(ep, perf)],
               delimiter=',')
