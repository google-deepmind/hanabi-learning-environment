# pylint: disable=missing-module-docstring, wrong-import-position, import-error, too-few-public-methods too-many-arguments, disable=too-many-instance-attributes
import sys
import os
import numpy as np
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.fireworkrank import FireworkRank
from bad.encoding.legal_actions import LegalActions

class PublicFeatures:
    '''public features'''
    def __init__(self, observation: dict) -> None:
        '''init'''
        self.observation = observation
        self.curr_player = observation['current_player']

        self.life_tokens_left = self.convert_life_tokens()
        self.hint_tokens_left = self.convert_information_tokens()
        self.current_player = self.convert_current_player()
        self.last_action = self.convert_last_action()
        self.firework = FireworkRank(observation)
        self.legal_actions = LegalActions(observation)

    def convert_life_tokens(self) -> np.ndarray:
        ''' convert life tokens '''
        life_tokens = self.observation['player_observations'][self.curr_player]['life_tokens']
        return tf.keras.utils.to_categorical(life_tokens, num_classes=4, dtype=int)

    def convert_information_tokens(self) -> np.ndarray:
        ''' convert information tokens '''
        in_to = self.observation['player_observations'][self.curr_player]['information_tokens']
        return tf.keras.utils.to_categorical(in_to, num_classes=9, dtype=int)

    def convert_current_player(self) -> np.ndarray:
        '''converts current player'''
        return tf.keras.utils.to_categorical(self.curr_player, num_classes=2, dtype=int)

    def convert_last_action(self) -> np.ndarray:
        '''convert last action'''
        max_action: int = self.observation['max_action']
        last_action: int = self.observation['last_action']

        return tf.keras.utils.to_categorical(last_action, num_classes=max_action, dtype=int)
