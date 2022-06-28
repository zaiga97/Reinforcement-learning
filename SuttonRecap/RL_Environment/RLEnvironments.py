import copy
import random
from typing import List

import gym
import numpy as np
from gym.spaces import Discrete
from gym.vector.utils import spaces


class RLEnv:
    def __init__(self, base_env: gym.Env):
        super(RLEnv, self).__init__()
        self.base_env = base_env

    def get_possible_actions(self, state):

        if type(self.base_env.action_space) == Discrete:
            return list(range(self.base_env.action_space.n))
        else:
            raise NotImplementedError

    def step(self, action):
        return self.base_env.step(action)

    def reset(self):
        return self.base_env.reset()


class RLFrozenLake(RLEnv):
    def __init__(self, base_env: gym.Env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True),
                 onehot_encoding: bool = False, xy_encoding: bool = False):
        super(RLFrozenLake, self).__init__(base_env)
        self.onehot_encoding = onehot_encoding
        self.xy_encoding = xy_encoding

    def state_coord(self, state):
        if self.onehot_encoding:
            s = [0 for _ in range(64)]
            s[state] = 1
            return s
        elif self.xy_encoding:
            x = state % 8
            y = state // 8

            s = [0 for i in range(16)]
            s[x] = 1
            s[8 + y] = 1
            return s
        else:
            return state

    def step(self, action):
        state, reward, done, info = super(RLFrozenLake, self).step(action)
        return self.state_coord(state), reward, done, info

    def reset(self):
        return self.state_coord(super(RLFrozenLake, self).reset())


class TicTacToe(RLEnv):
    def __init__(self):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.player = 1
        self.done = False

    def get_state(self):
        return copy.deepcopy(self.board)

    def reset(self):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.player = 1
        self.done = False

    def get_possible_actions(self, state: tuple = (0, 0, 0, 0, 0, 0, 0, 0, 0)):
        return [i for i in range(9) if state[i] == 0]

    def __is_full(self):
        return not self.board.__contains__(0)

    def __row_win(self):
        for r in range(0, 9, 3):
            if self.board[r] == self.board[r + 1] == self.board[r + 2] != 0:
                return True, self.board[r]
        return False, 0

    def __col_win(self):
        for c in range(3):
            if self.board[c] == self.board[3 + c] == self.board[6 + c] != 0:
                return self.board[c]
        return False, 0

    def __diag_win(self):
        if self.board[0] == self.board[4] == self.board[8] != 0:
            return True, self.board[4]
        elif self.board[2] == self.board[4] == self.board[6] != 0:
            return True, self.board[4]
        return False, 0

    def step(self, action):
        # assert the move is a possible move (not necessarily legal)
        assert action in self.get_possible_actions()

        # If you try to cheat by doing a move that is not legal you will be penalized
        if action not in self.get_possible_actions(tuple(self.board)):
            return (), -10, True

        self.board[action] = self.player
        done, reward = self.is_final


class SmallYahtzee(RLEnv):

    def __init__(self, n_faces: int = 6, n_dices: int = 5, onehot_encoding: bool = False,
                 punishment: bool = True, force_roll: bool = False) -> None:
        self.dices = [random.randint(1, n_faces) for _ in range(n_dices)]
        self.scores = [-1 for _ in range(n_faces)]
        self.onehot_encoding = onehot_encoding
        self.punishment = punishment
        self.force_roll = force_roll

        self.round = 0
        self.subround = 0

        self.n_faces = n_faces
        self.n_dices = n_dices
        self.n_actions = (2 ** n_dices) + n_faces

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=np.array([0, 0]), high=np.array([n_faces, 2]), dtype=np.uint8),
            spaces.Box(low=np.array([1 for _ in range(n_dices)]), high=np.array([n_faces for _ in range(n_dices)]),
                       dtype=np.uint8),
            spaces.Box(low=np.array([-1 for _ in range(n_faces)]),
                       high=np.array([n_dices * i for i in range(1, n_faces + 1)]),
                       dtype=np.int16)
        ))

    def __get_observation(self):
        if self.onehot_encoding:
            _round = [0 if r != self.round else 1 for r in range(self.n_faces)]
            subround = [0 if sr != self.subround else 1 for sr in range(3)]
            dices = []
            for dice in range(self.n_dices):
                for val in range(1, self.n_faces + 1):
                    if self.dices[dice] == val:
                        dices.append(1)
                    else:
                        dices.append(0)
            return *_round, *subround, *dices, *self.scores
        else:
            return self.round, self.subround, *self.dices, *self.scores

    def __roll(self, to_reroll: List[bool] = None):
        if to_reroll is None:
            self.dices = [random.randint(1, self.n_faces) for _ in range(self.n_dices)]
        else:
            for i in range(self.n_dices):
                if to_reroll[i]:
                    self.dices[i] = random.randint(1, self.n_faces)

    def action_to_reroll_indexes(self, action: int):
        return [True if ((action // (2 ** i)) % 2 == 0) else False for i in range(self.n_dices)]

    def __score(self, action: int):
        category = action - (2 ** self.n_dices)
        good_dices = self.dices.count(category + 1)
        self.scores[category] = good_dices * (category + 1)
        self.__roll()
        return self.scores[category]

    def render(self):
        print(f"Round:{self.round},{self.subround}")
        print(f"Dices = {self.dices}")
        print(f"Scores = {self.scores}")

    def __is_legal(self, action: int):
        category = action - (2 ** self.n_dices)
        if category < 0 and self.subround < 2:
            return True
        elif category >= 0 and self.scores[category] == -1:
            if self.force_roll:
                if self.subround == 2:
                    return True
                return False
            else:
                return True
        else:
            return False

    def step(self, action: int):
        assert action in self.action_space
        done = False
        if self.__is_legal(action):
            if action < self.n_actions - self.n_faces:  # Reroll of some dices
                to_reroll = self.action_to_reroll_indexes(action)
                self.__roll(to_reroll)
                self.subround += 1
                reward = 0
            else:
                reward = self.__score(action)
                self.subround = 0
                self.round += 1
                if self.round == self.n_faces:
                    done = True
        elif self.punishment:
            reward = -1
        else:
            reward = 0

        return self.__get_observation(), reward, done, None

    def reset(self):
        self.__roll()
        self.scores = [-1 for _ in range(self.n_faces)]
        self.round = 0
        self.subround = 0
        return self.__get_observation()

    def get_possible_actions(self, state):
        actions = list(range(self.n_actions))
        possible_actions = [action for action in actions if self.__is_legal(action)]
        if len(possible_actions) == 0:
            possible_actions.append(0)
        return possible_actions
