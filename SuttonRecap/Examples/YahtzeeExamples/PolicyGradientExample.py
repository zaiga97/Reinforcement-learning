import numpy as np
import torch
from matplotlib import pyplot as plt

from Policies.Policy import DifferentiablePolicy

from RL_Algorithms import ExpDecayParameter
from RL_Algorithms.Algorithms import ReinforceWithBaseline
from RL_Algorithms.Utilities import MemoryReplayer, PolicyTester
from RL_Environment.RLEnvironments import SmallYahtzee
from RL_Algorithms.nn_models import SmallNN, SmallPolicyNN

if __name__ == "__main__":
    n_dices = 5
    n_faces = 6
    env = SmallYahtzee(n_dices=n_dices, n_faces=n_faces, onehot_encoding=True, punishment=True, force_roll=True)
    test_env = SmallYahtzee(n_dices=n_dices, n_faces=n_faces, onehot_encoding=True, punishment=False)

    gamma = 0.99

    def init_weight(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
            torch.nn.init.normal_(m.bias, mean=0.0, std=0.1)

    obs_size = n_faces + 3 + n_dices * n_faces + n_faces
    policy_nn = SmallPolicyNN(in_size=obs_size, out_size=env.n_actions, hidden_size=(128, 64))
    baseline_nn = SmallNN(in_size=obs_size, out_size=1, hidden_size=(128, 64))
    diff_policy = DifferentiablePolicy(policy_nn)
    alpha_policy = 0.0005
    alpha_baseline = 0.0005
    pg_learning = ReinforceWithBaseline(diff_policy, baseline_nn, alpha_policy, alpha_baseline, env, gamma)

    episodes = 20000
    episodes_between_tests = 100
    test_size = 20
    max_step = 40

    perf = []
    ep = []

    for episode in range(episodes):
        if episode % episodes_between_tests == 0:
            ep.append(episode)
            perf.append(PolicyTester.test(test_size, max_step, pg_learning.diff_policy, test_env, gamma=1))
            print(f"Episode: {episode}: {perf[-1]}")
        pg_learning.train(1, max_step)

    plt.plot(ep, perf, label="Reinforce")
    plt.legend()
    plt.show()

    torch.save(policy_nn.state_dict(), "results/Policy_NN_model")

    PolicyTester.test(3, max_step, pg_learning.diff_policy, test_env, gamma=1, verbose=True)

    np.savetxt("policy_grad_yahtzee.csv",
               [p for p in zip(ep, perf)],
               delimiter=',')
