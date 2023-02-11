import sys
import os
import getopt

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)

from ftpubvec import RemaingCards
from hint_matrix_hand_card import HintMatrixHandCard
from hanabi_learning_environment.rl_env import HanabiEnv


class HintMatrixPlayer(dict):
    def __init__(self, observations: HanabiEnv, 
                       rem_cards: RemaingCards):
        self.observations = observations
        super().__init__(self.init())

    def init(self, observations: HanabiEnv, rem_cards: RemaingCards) -> dict:
        hint_matrix_player = [HintMatrixHandCard(observations, rem_cards)]
        return hint_matrix_player

    def update(self) -> None:
        """Update HintMatrixPlayer based on the rem_cards and hint_matrix_hand_card"""
        [hint_matrix_hand_card.update() for hint_matrix_hand_card in self.hint_matrix_player]

    

    

