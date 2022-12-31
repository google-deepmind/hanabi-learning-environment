# pylint: disable=missing-module-docstring, wrong-import-position, import-error
import string
import sys
import os
import numpy as np
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.observation import Observation

class ObservationConverter:
    ''' converting hanabi observation to vector observation '''
    def __init__(self, observation: dict) -> None:
        self.observation = observation

    def convert(self) -> Observation:
        ''' one hot encoding '''

        life_tokens_left = self.convert_life_tokens()
        information_tokens_left = self.convert_information_tokens()
        firework_rank_red = self.convert_firework_rank_color('R')
        firework_rank_yellow = self.convert_firework_rank_color('Y')
        firework_rank_green = self.convert_firework_rank_color('G')
        firework_rank_white = self.convert_firework_rank_color('W')
        firework_rank_blue = self.convert_firework_rank_color('B')

        return Observation(life_tokens_left, information_tokens_left, \
        firework_rank_red, firework_rank_yellow, firework_rank_green, \
        firework_rank_white, firework_rank_blue)

    def convert_life_tokens(self) -> np.ndarray:
        ''' convert life tokens '''
        life_tokens: int = self.observation['life_tokens']
        return tf.keras.utils.to_categorical(life_tokens, num_classes=4, dtype=int)

    def convert_information_tokens(self) -> np.ndarray:
        ''' convert information tokens '''
        information_tokens: int = self.observation['information_tokens']
        return tf.keras.utils.to_categorical(information_tokens, num_classes=9, dtype=int)

    def convert_firework_rank_color(self, color: string) -> np.ndarray:
        ''' convert firework rank color '''
        rank = self.observation['fireworks'][color]
        return tf.keras.utils.to_categorical(rank, num_classes=6, dtype=int)
