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
        current_player:int = observation['current_player']
        print(f'current player id {current_player}')

        self.own_cards = []
        own_hand = observation['player_observations'][current_player]['observed_hands'][0]
        for card in own_hand:
            self.own_cards.append(Card(card['color'], card['rank']))

        self.other_cards = []
        other_hands = observation['player_observations'][current_player]['observed_hands'][1:4]
        for other_hand in other_hands:
            for card in other_hand:
                self.other_cards.append(Card(card['color'], card['rank']))
