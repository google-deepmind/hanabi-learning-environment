import sys
import os

from public_belief_player import PublicBeliefPlayer
from hint_matrix_global import HintMatrix
from ftpubvec import RemaingCards
from build_hanabi_env import get_hanabi_env

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)


from hanabi_learning_environment.rl_env import HanabiEnv
from constants import Constants


class PublicBelief(list):
    
    def __init__(self, hanabi_env: HanabiEnv, constants: Constants):
        observation = hanabi_env['player_observations'][0]
        # Init rem_cards,hint_matrix and hanabi_env due to dubug purpose 
        self.rem_cards = RemaingCards(hanabi_env)
        self.hint_matrix = HintMatrix(constants, self.rem_cards)
        self.hanabi_env = hanabi_env
        
        #self.num_colors_left = {'B':  10, 'G': 10,'R': 10,'W': 10,'Y': 10}        
        #self.num_ranks_left = {0:  15, 1: 10, 2: 10, 3: 10, 4: 5}
       
        super().__init__(self.init(hanabi_env, constants))
        
    def init(self, hanabi_env: HanabiEnv, constants: Constants):
        pub_belf = [PublicBeliefPlayer(constants, idx_ply, 
                    self.rem_cards, self.hint_matrix[idx_ply]) 
                    for idx_ply in range(constants.num_ply)]
        return pub_belf 

'''
    def get_private_belief_hand_card(self, agent_idx, card_idx):
        """Return private hand card belief
        This Belief takes into accounts the private knowledge and
        exclude all cards that are not possible based on private knowledge"""
        
        number_colors_left_private, number_ranks_left_private = self.get_privat_num_colors_ranks_left()

        hint_matrix_hand_card = self.hint_matrix[agent_idx][card_idx]
        private_hint_matrix_card = copy.deepcopy(hint_matrix_hand_card)

        for color in self.colors:
                for rank in range(self.max_rank + 1):
                    if hint_matrix_hand_card[color][rank] == 1:
                        card = {'color' : color, 'rank': rank}
                        
                        if (self.poss_card_in_hand(card, number_colors_left_private, 
                                                   number_ranks_left_private) is False):
                            private_hint_matrix_card[color][rank] = 0 
        
        return private_hint_matrix_card
            
    def get_privat_num_colors_ranks_left(self, idx_agent):
        """Return private num colors """
        number_colors_left_private = {
                                       'B': self.num_colors_left['B'],
                                       'G': self.num_colors_left['G'],
                                       'R': self.num_colors_left['R'],
                                       'W': self.num_colors_left['W'],
                                       'Y': self.num_colors_left['Y'],
                                     }
        
        number_ranks_left_private = {
                                      0: self.num_ranks_left[0],
                                      1: self.num_ranks_left[1],
                                      2: self.num_ranks_left[2],
                                      3: self.num_ranks_left[3],
                                      4: self.num_ranks_left[4]
                                    }
        
        for agent_idx, agent_hand in enumerate (self.observation[idx_agent]['observed_hands']):
            if agent_idx == 0:
                continue

            for card_idx, card in enumerate (agent_hand):
                card_in_cardknowledge = self.observation['card_knowledge'][agent_idx][card_idx]
                card_cardknowledge_color = card_in_cardknowledge['color']
                card_cardknowledge_rank = card_in_cardknowledge['rank']

                if (card_cardknowledge_color == None):
                    number_colors_left_private[card['color']] -= 1
                
                if (card_cardknowledge_rank == None):
                    number_ranks_left_private[card['rank']] -= 1
        
        return number_colors_left_private, number_ranks_left_private 

    def poss_card_in_hand(self, card, number_colors_left_private, number_ranks_left_private):
            card_color = card['color']
            card_rank = card['rank']

            if (number_ranks_left_private[card_rank] == 0 or
                number_colors_left_private[card_color] == 0):
                return False 

            else:
                return True 
'''

if __name__ == "__main__":
    hanabi_env = get_hanabi_env()
    pub_belf = PublicBelief(hanabi_env)
    print()        
            
