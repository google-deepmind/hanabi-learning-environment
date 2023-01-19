# pylint: disable=missing-module-docstring, wrong-import-position, import-error too-few-public-methods too-many-arguments
import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.card import Card

class Hands():
    '''hand'''
    def __init__(self, observation: dict) -> None:
        current_player:int = observation['current_player']
        self.own_cards = np.empty(0, int)
        own_hand = observation['player_observations'][current_player]['observed_hands'][0]
        for card in own_hand:
            if int(card['rank']) == -1:
                own_card = Card('None', 0)
            else:
                own_card = Card(card['color'], card['rank'])

            self.own_cards = np.append(self.own_cards, own_card.color)
            self.own_cards = np.append(self.own_cards, own_card.rank)

        self.own_cards = self.adjust_card_length(self.own_cards)

        self.other_cards = np.empty(0, int)
        other_hands = observation['player_observations'][current_player]['observed_hands'][1:4]
        for other_hand in other_hands:
            for card in other_hand:
                other_card = Card(card['color'], card['rank'])
                self.other_cards = np.append(self.other_cards, other_card.color)
                self.other_cards = np.append(self.other_cards, other_card.rank)

        self.other_cards = self.adjust_card_length(self.other_cards)

    def adjust_card_length(self, cards: np.array) -> np.array:
        '''adjust card length'''
        fill_with_zeros: int = 60 - len(cards)
        if fill_with_zeros > 0:
            cards = np.append(cards, np.zeros((fill_with_zeros, ), dtype=int))
        return cards
