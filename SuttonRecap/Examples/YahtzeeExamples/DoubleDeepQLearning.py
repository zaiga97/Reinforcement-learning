import torch
from matplotlib import pyplot as plt

from Policies.ValueFunctions import NNValueFunction
from RL_Algorithms import ExpDecayParameter
from RL_Algorithms.Algorithms import DoubleQ
from RL_Algorithms.Utilities import MemoryReplayer, PolicyTester
from RL_Environment.RLEnvironments import SmallYahtzee
from RL_Algorithms.nn_models import SmallNN

if __name__ == "__main__":
    n_dices = 5
    n_faces = 6
    env = SmallYahtzee(n_dices=n_dices, n_faces=n_faces, onehot_encoding=True)
    test_env = SmallYahtzee(n_dices=n_dices, n_faces=n_faces, onehot_encoding=True, punishment=False)
    memory_replayer = MemoryReplayer(1000)

    alpha = ExpDecayParameter(1, 50000)
    epsilon = ExpDecayParameter(1, 2000, 0.2)
    gamma = 0.9
    learning_rate = 0.00025
    batch_size = 32


    def init_weight(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
            torch.nn.init.normal_(m.bias, mean=0.0, std=0.1)


    obs_size = n_faces + 3 + n_dices * n_faces + n_faces
    value_nn1 = SmallNN(in_size=obs_size, out_size=env.n_actions).apply(init_weight)
    value_nn2 = SmallNN(in_size=obs_size, out_size=env.n_actions)
    value_nn2.load_state_dict(value_nn1.state_dict())
    optimizer = torch.optim.RMSprop(params=value_nn1.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.RMSprop(params=value_nn2.parameters(), lr=learning_rate)
    value_function1 = NNValueFunction(value_nn1, optimizer=optimizer, batch_size=batch_size)
    value_function2 = NNValueFunction(value_nn2, optimizer=optimizer2, batch_size=batch_size)

    double_q_learning = DoubleQ(alpha=alpha, gamma=gamma, epsilon=epsilon, env=env,
                                value_function1=value_function1, value_function2=value_function2,
                                memory_replayer=memory_replayer, memory_steps=4, episodes_between_swaps=100)

    episodes = 5000
    episodes_between_tests = 100
    test_size = 10
    max_step = 50

    perf = []

    for episode in range(episodes):
        if episode % episodes_between_tests == 0:
            perf.append(PolicyTester.test(test_size, max_step, double_q_learning.greedy_policy[0], test_env, gamma=1))
            print(f"Episode: {episode}: {perf[-1]}")
        double_q_learning.train(1, max_step)

    plt.plot(range(len(perf)), perf, label="Double Deep Q learning")
    plt.legend()
    plt.show()

    torch.save(value_nn1.state_dict(), "DoubleNN_model")
