# pylint: disable=missing-module-docstring too-few-public-methods too-many-arguments
import string
import numpy as np
import tensorflow as tf


class FireworkRank:
    ''' firework rank '''
    def __init__(self, observation: dict) -> None:
        '''init'''
        self.observation = observation

        self.red = self.convert_firework_rank_color('R')
        self.yellow = self.convert_firework_rank_color('Y')
        self.green = self.convert_firework_rank_color('G')
        self.white  = self.convert_firework_rank_color('W')
        self.blue = self.convert_firework_rank_color('B')

    def convert_firework_rank_color(self, color: string) -> np.ndarray:
        ''' convert firework rank color '''
        rank = self.observation['fireworks'][color]
        return tf.keras.utils.to_categorical(rank, num_classes=6, dtype=int)
