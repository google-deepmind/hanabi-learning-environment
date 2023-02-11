import build_observation as build_observation
from ftpubvec import RemaingCards
from hanabi_learning_environment.rl_env import HanabiEnv
import sys
import os
import getopt

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)


class HintMatrixHandCard(dict):
    def __init__(self, observation: HanabiEnv, rem_cards: RemaingCards):
        self.observation = observation
        self.rem_cards = rem_cards
        self.max_rank = 4
        super().__init__(self.init())

    def init(self) -> dict:
        hint_matrix_hand_card = {'B': [1, 1, 1, 1, 1],
                                 'G': [1, 1, 1, 1, 1],
                                 'R': [1, 1, 1, 1, 1],
                                 'W': [1, 1, 1, 1, 1],
                                 'Y': [1, 1, 1, 1, 1]}
        return hint_matrix_hand_card

    def update(self) -> None:
        """Updates the HintMatrixHandCard based on the rem_cards"""
        for rem_card_per_color in self.rem_cards:
            for rank in range(self.max_rank + 1):
                rem_card_per_card = rem_card_per_color[rank]
                
                # Wenn es keine Karten mehr gibt kann diese auch 
                # nicht in der Hand sein 
                if rem_card_per_card == 0:
                    self[rem_card_per_card][rank] = 0

def main():
    observeration: HanabiEnv = build_observation.get_observation()
    rem_cards = RemaingCards(observeration)
    hint_matrix_hand_card = HintMatrixHandCard(observeration, rem_cards)
    print()


if __name__ == "__main__":
    main()
