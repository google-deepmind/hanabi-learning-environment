import os
import sys 
import numpy as np
import tensorflow as tf
currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from hanabi_learning_environment.rl_env import HanabiEnv

class RemaingCards(dict):
    """Es handels sich um FtPubVec aus der Arbeit von Foster
       Beinhaltet die Information der remaing Cards in Form eines Dict """
    def __init__(self, hanabi_env: HanabiEnv) -> None:
        self.hanabi_env = hanabi_env
        super().__init__(self.init())

    def init(self):
        rep_color = [3,2,2,2,1]

        rem_cards = {'B': rep_color.copy(),
                     'G': rep_color.copy(),
                     'R': rep_color.copy(),
                     'W': rep_color.copy(),
                     'Y': rep_color.copy(),}
        
        return rem_cards
    
    def update(self, hanabi_env: HanabiEnv):
        """Based on the Public Information we calculate rem_cards
        For each card we find in public information (card_knowledge,
        discard_pile and firework) we reduce max number by one"""
        self.hanabi_env = hanabi_env
        self.rem_cards = self.init()

        self.update_based_on_card_knowledge()
        self.update_based_on_discard_pile()
        self.update_based_on_firework()

    def update_based_on_firework(self):
        """ Update rem_cards based on Firework"""

        # Jede Karte die im Firework liegt kann nicht 
        # mehr auf der Hand eines Spieles sein  
        firework = self.hanabi_env['player_observations'][0]['fireworks']
        for color, max_rank in firework.items():
            for rank in range(max_rank):
                self.rem_cards[color][rank] -= 1
    
    def update_based_on_card_knowledge(self):
        """Update rem_cards based on card_knowledge """
        
        card_knowledge = self.hanabi_env['player_observations'][0]['card_knowledge']
        
        # Prüfe alle Karten in card_knowledge
        for player_card_knowledge in card_knowledge:
            for card in player_card_knowledge:

                # Wenn eine Karte vollständig bekannt dann reduziere mc
                # Diese Karte kann ja nicht mehr einer anderen Hand sein
                if (card['rank'] is not None and  
                    card['color'] is not None):

                    self.rem_cards[card['color']][card['rank']] -= 1

    def update_based_on_discard_pile(self): 
        """Update mc based on discard_pile """

        # Jede Karte die im Discard Pile ist kann nicht mehr in der Hand 
        # eines anderen Spieler sein 
        discard_pile = self.hanabi_env['player_observations'][0]['discard_pile']
        for card in discard_pile:
            self.rem_cards[card['color']][card['rank']] -= 1   


    
