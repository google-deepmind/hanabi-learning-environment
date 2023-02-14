
import sys
import os
import getopt


from ftpubvec import RemaingCards
from hint_matrix_hand_card import HintMatrixHandCard
from build_hanabi_env import get_hanabi_env


currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)

from hanabi_learning_environment.rl_env import HanabiEnv
from constants import Constants

class PublicBelfHandCard(dict):
    def __init__(self, constants, idx_ply, idx_card, rem_cards, 
                 hint_matrix_hand_card, likelihood_hand_card):
        self.idx_card = idx_card
        self.idx_ply = idx_ply
        super().__init__(self.__init(constants, rem_cards, 
                         hint_matrix_hand_card, likelihood_hand_card))

    def __init(self, constants, rem_cards, 
               hint_matrix_hand_card, likelihood_hand_card) -> None:
        """Update PublicBelfHandCard based on the rem_cards and hint_matrix_hand_card"""

        public_belief_hand_card = {}
      
        for color in constants.colors:
            public_belief_color = [rem_cards[color][rank] * hint_matrix_hand_card[color][rank] * 
                                   likelihood_hand_card[color][rank]
                                   for rank in  range(constants.num_ranks + 1)]
            public_belief_hand_card.update({color: public_belief_color})
        
        # Normalize the public_belief_hand_card
        public_belief_hand_card = self.normalize_pub_belief_hand_card(
                                    constants, public_belief_hand_card)

        return public_belief_hand_card
    
    def normalize_pub_belief_hand_card(self, constants, public_belief_hand_card: dict) -> dict:
        """Normalize the public_belief_hand_card"""
        sum = 0
        for color in public_belief_hand_card:
            sum += sum(public_belief_hand_card[color])
        
        for color in public_belief_hand_card:
            public_belief_hand_card[color] = [public_belief_hand_card[color][rank] / sum 
                                              for rank in range(constants.num_ranks + 1)]

        return public_belief_hand_card


        

def main():
    hanabi_env: HanabiEnv = get_hanabi_env()
    print()


if __name__ == "__main__":
    main()
