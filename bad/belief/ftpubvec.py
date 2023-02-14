import os
import sys 
import numpy as np
import tensorflow as tf
currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)

from hanabi_learning_environment.rl_env import HanabiEnv

class RemaingCards(dict):
    """Es handels sich um FtPubVec aus der Arbeit von Foster
       Beinhaltet die Information der remaing Cards in Form eines Dict """
    def __init__(self, hanabi_env: HanabiEnv, constants) -> None:
        super().__init__(self.__init(hanabi_env, constants))

    def init_start_conditions(self, constants)-> dict:
        

        rem_cards = {'B': constants.num_cards_per_ranks.copy(),
                     'G': constants.num_cards_per_ranks.copy(),
                     'R': constants.num_cards_per_ranks.copy(),
                     'W': constants.num_cards_per_ranks.copy(),
                     'Y': constants.num_cards_per_ranks.copy(),}
        
        return rem_cards
    
    def __init(self, hanabi_env: HanabiEnv, constants)-> dict:
        """Based on the Public Information we calculate rem_cards
        For each card we find in public information (card_knowledge,
        discard_pile and firework) we reduce max number by one"""
        rem_cards = self.init_start_conditions(constants)

        self.update_based_on_card_knowledge(hanabi_env, rem_cards)
        self.update_based_on_discard_pile(hanabi_env, rem_cards)
        self.update_based_on_firework(hanabi_env, rem_cards)

        return rem_cards

    def update_based_on_firework(self, hanabi_env: HanabiEnv, rem_cards)-> dict:
        """ Update rem_cards based on Firework"""

        # Jede Karte die im Firework liegt kann nicht 
        # mehr auf der Hand eines Spieles sein  
        firework = hanabi_env['player_observations'][0]['fireworks']
        for color, max_rank in firework.items():
            for rank in range(max_rank):
                rem_cards[color][rank] -= 1
        
        return rem_cards
    
    def update_based_on_card_knowledge(self, hanabi_env: HanabiEnv, rem_cards)-> dict:
        """Update rem_cards based on card_knowledge """
        
        card_knowledge = hanabi_env['player_observations'][0]['card_knowledge']
        
        # Prüfe alle Karten in card_knowledge
        for player_card_knowledge in card_knowledge:
            for card in player_card_knowledge:

                # Wenn eine Karte vollständig bekannt dann reduziere mc
                # Diese Karte kann ja nicht mehr einer anderen Hand sein
                if (card['rank'] is not None and  
                    card['color'] is not None):

                    rem_cards[card['color']][card['rank']] -= 1
        
        return rem_cards

    def update_based_on_discard_pile(self, hanabi_env: HanabiEnv, rem_cards)-> dict: 
        """Update mc based on discard_pile """

        # Jede Karte die im Discard Pile ist kann nicht mehr in der Hand 
        # eines anderen Spieler sein 
        discard_pile = hanabi_env['player_observations'][0]['discard_pile']
        for card in discard_pile:
            rem_cards[card['color']][card['rank']] -= 1   

        return rem_cards
    
