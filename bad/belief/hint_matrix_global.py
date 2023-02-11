import numpy as np

import sys
import os
import getopt

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)

from hanabi_learning_environment.rl_env import HanabiEnv
from hint_matrix_hand_card import HintMatrixHandCard
from hint_matrix_player import HintMatrixPlayer
from ftpubvec import RemaingCards
from build_observation import get_observation


class HintMatrix(list):
    def __init__(self, hanabi_env: HanabiEnv, rem_cards: RemaingCards):
        self.num_ply = 2
        super().__init__(hanabi_env, rem_cards)


    def init(self, hanabi_env: HanabiEnv, rem_cards: RemaingCards) -> list:
        players_hands = [HintMatrixPlayer(hanabi_env['player_observations'][idx_ply], 
                         rem_cards) for idx_ply in range(self.num_ply)] 

        return players_hands
    
    def update(self) -> None:
        """Update based on rem_cards"""
        [self[idx_ply].update() for idx_ply in range(self.num_ply)]
        
    
    def convert_players_hand_to_one_hot(self, index_player: int):
        number_hand_cards = 5
        values = [list(self.players_hands[index_player][idx_hand_card].values()) \
                  for idx_hand_card in range(number_hand_cards)] 
        
        # Combine the list in list 
        values = sum(values, [])

        np_values = np.array(values)
        return np_values



def main():
    hanabi_env = get_observation()
    rem_cards = RemaingCards(hanabi_env)
    hint_matrix = HintMatrix(hanabi_env, rem_cards)
    print()
if __name__ == "__main__":
    main()