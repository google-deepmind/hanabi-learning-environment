import sys
import os
import getopt

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)

from ftpubvec import RemaingCards
from hint_matrix_hand_card import HintMatrixHandCard

class PublicBelfHandCard(dict):
    def __init__(self, observation, rem_cards: RemaingCards, hint_matrix_hand_card: HintMatrixHandCard):
        self.observation = observation
        self.rem_cards = rem_cards
        self.hint_matrix_hand_card = hint_matrix_hand_card
        super().__init__(self.init())

    def init(self) -> dict:
        public_belief_hand_card = {'B': [3, 2, 2, 2, 1],
                                   'G': [3, 2, 2, 2, 1],
                                   'R': [3, 2, 2, 2, 1],
                                   'W': [3, 2, 2, 2, 1],
                                   'Y': [3, 2, 2, 2, 1]}
        return public_belief_hand_card

    def update(self) -> None:
        """Update PublicBelfHandCard based on the rem_cards and hint_matrix_hand_card"""
         

