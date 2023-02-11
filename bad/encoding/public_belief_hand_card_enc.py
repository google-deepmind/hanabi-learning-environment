import sys
import os
import tensorflow as tf
import numpy as np 

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)

from belief.public_belief_hand_card import PublicBeliefHandCard
from card_probabilitiy import CardProbabilitiy

class PublicBeliefHandCardEnc():
    def __init__(self, public_belf_hand_card: PublicBeliefHandCard):
        
        colors = ['B', 'G', 'R', 'W', 'Y']
        max_rank = 5
        
        self.pro_hand_card = np.empty(0,int)
        
        for color in colors:
            for rank in range(max_rank):
                num_rem_cards = public_belf_hand_card[color][rank]
                self.pro_hand_card = np.append(self.pro_hand_card, 
                                               CardProbabilitiy[color][num_rem_cards])      

