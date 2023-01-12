# pylint: disable=missing-module-docstring, wrong-import-position, import-error, too-few-public-methods too-many-arguments
import sys
import os
import numpy as np
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.fireworkrank import FireworkRank

class PublicFeatures:
    '''public features'''
    def __init__(self, observation: dict) -> None:
        '''init'''
        self.observation = observation
        self.current_player = observation['current_player']

        self.life_tokens_left = self.convert_life_tokens()
        self.hint_tokens_left = self.convert_information_tokens()
        self.current_player = self.convert_current_player()

        self.firework = FireworkRank(observation)

    def convert_life_tokens(self) -> np.ndarray:
        ''' convert life tokens '''
        life_tokens = self.observation['player_observations'][self.current_player]['life_tokens']
        return tf.keras.utils.to_categorical(life_tokens, num_classes=4, dtype=int)

    def convert_information_tokens(self) -> np.ndarray:
        ''' convert information tokens '''
        in_to = self.observation['player_observations'][self.current_player]['information_tokens']
        return tf.keras.utils.to_categorical(in_to, num_classes=9, dtype=int)

    def convert_current_player(self):
        '''converts current player'''
        return tf.keras.utils.to_categorical(self.current_player, num_classes=2, dtype=int)
