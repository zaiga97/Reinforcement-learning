import torch
from matplotlib import pyplot as plt

from Policies.ValueFunctions import NNValueFunction
from RL_Algorithms import ExpDecayParameter, TD0
from RL_Algorithms.Utilities import MemoryReplayer, PolicyTester
from RL_Environment.RLEnvironments import SmallYahtzee
from RL_Algorithms.nn_models import SmallNN

if __name__ == "__main__":
    n_dices = 5
    n_faces = 6
    env = SmallYahtzee(n_dices=n_dices, n_faces=n_faces)
    memory_replayer = MemoryReplayer(200)

    alpha = ExpDecayParameter(0.5, 50000)
    epsilon = ExpDecayParameter(1, 2000, 0.1)
    gamma = 0.99

    def init_weight(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=1)
            torch.nn.init.normal_(m.bias, mean=0.0, std=1)

    obs_size = n_dices + n_faces + 2
    value_nn = SmallNN(in_size=obs_size, out_size=env.n_actions).apply(init_weight)
    optimizer = torch.optim.RMSprop(params=value_nn.parameters(), lr=0.00025)
    value_function = NNValueFunction(value_nn, optimizer=optimizer, batch_size=32)
    #
    qnn_learning = TD0(update_rule="Q learning", alpha=alpha, gamma=gamma, epsilon=epsilon, env=env,
                       value_function=value_function, memory_replayer=memory_replayer, memory_steps=4)

    episodes = 5000
    episodes_between_tests = 100
    test_size = 10
    max_step = 50

    qnn_learning_perf = []

    for episode in range(episodes):
        if episode % episodes_between_tests == 0:
            qnn_learning_perf.append(PolicyTester.test(test_size, max_step, qnn_learning.greedy_policy, env, gamma))
            print(f"Episode: {episode}")
        qnn_learning.train(1, max_step)

    plt.plot(range(len(qnn_learning_perf)), qnn_learning_perf, label="Deep Qnn")
    plt.legend()
    plt.show()

    torch.save(value_nn.state_dict(), "NN_model")
    for e in range(3):
        PolicyTester.test(test_size, max_step, qnn_learning.greedy_policy, test_env, gamma, verbose=True)
