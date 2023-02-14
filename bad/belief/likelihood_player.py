import sys
import os

from likelihood_hand_card import LikelihoodHandCard

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)

from action_network import ActionNetwork
from bad.encoding.observation import Observation
from hanabi_learning_environment.rl_env import HanabiEnv


class LikelihoodPlayer(list):
    def __init__(self, constants, idx_ply: int, observation: Observation,
                  action_network: ActionNetwork, last_act, old_likelihood, hint_matrix):
        self.idx_ply = idx_ply
        super().__init__(self.__init(constants, idx_ply, observation, 
                                     action_network, last_act, old_likelihood, hint_matrix))

    def __init(self, constants, idx_ply, observation, 
               action_network, last_act, old_likelihood, hint_matrix) -> list:
        likelihood_player = [LikelihoodHandCard(idx_ply, idx_card, constants, observation, 
                              action_network, last_act, old_likelihood, hint_matrix) 
                              for idx_card in range(constants.num_cards) ]
        return likelihood_player
