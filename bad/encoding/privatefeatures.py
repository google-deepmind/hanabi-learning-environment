# pylint: disable=missing-module-docstring, wrong-import-position, import-error, too-few-public-methods too-many-arguments
import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.hand import Hand

class PrivateFeatures:
    '''private features'''
    def __init__(self, observation: dict) -> None:
        '''init'''
        self.observation = observation
        current_player_id = observation['current_player']
        self.other_player_hand = self.convert_other_player_hand()
        # other players hand
    def convert_other_player_hand(self) -> Hand:
        return Hand(self.observation)
