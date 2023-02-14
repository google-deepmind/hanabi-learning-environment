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


class HintMatrixPlayer(list):
    def __init__(self, constants, idx_ply: int, rem_cards: RemaingCards):
        self.idx_ply = idx_ply
        super().__init__(self.__init(constants, rem_cards))

    def __init(self, constants, rem_cards: RemaingCards) -> list:
        hint_matrix_player = [HintMatrixHandCard(constants, idx_card, rem_cards)
                              for idx_card in range(constants.num_ply)]
        return hint_matrix_player



    

    

