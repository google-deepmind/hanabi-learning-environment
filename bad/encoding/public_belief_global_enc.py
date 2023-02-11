import sys
import os
import getopt
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from belief.public_belief_hand_card import PublicBeliefHandCard
from public_belief_player_enc import PublicBeliefPlayerEnc

class PublicBeliefGlobalEnc():
    """Inherit from list"""
    def __init__(self, public_belf):
        self.public_belf_player1 = PublicBeliefPlayerEnc(public_belf[0])
        self.public_belf_player2 = PublicBeliefPlayerEnc(public_belf[1])
    
    def to_one_hot(self):
        """Convert Public Belief to one-hot encoding"""
        result = np.concatenate(( 
            self.public_belf_player1.hand_card1,
            self.public_belf_player1.hand_card2,
            self.public_belf_player1.hand_card3,
            self.public_belf_player1.hand_card4,
            self.public_belf_player1.hand_card5,
            self.public_belf_player2.hand_card1,
            self.public_belf_player2.hand_card2,
            self.public_belf_player2.hand_card3,
            self.public_belf_player2.hand_card4,
            self.public_belf_player2.hand_card5
             ))
        return result