# pylint: disable=missing-module-docstring, wrong-import-position, import-error too-few-public-methods too-many-arguments
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.card import Card

class Hands():
    '''hand'''
    def __init__(self, observation: dict) -> None:
        # observation['current_player']
        curr_player_id:int = observation['current_player']

        self.own_cards = []
        own_cards = observation['player_observations'][curr_player_id]['card_knowledge'][0]
        for card in own_cards:
            self.own_cards.append(Card(card['color'], card['rank']))

        self.other_cards = []
        other_cards = observation['player_observations'][curr_player_id]['card_knowledge'][1]
        for card in other_cards:
            self.other_cards.append(Card(card['color'], card['rank']))
