# pylint: disable=missing-module-docstring, wrong-import-position, import-error
import sys
import os
import numpy as np

from bad.encoding.fireworkrank import FireworkRank

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)


class PublicFeatures:
    '''public features'''
    def __init__(self,life_tokens_left : np.ndarray, hint_tokens_left: np.ndarray,red:np.ndarray,\
        yellow:np.ndarray, green:np.ndarray, white:np.ndarray, blue:np.ndarray) -> None:
        '''init'''
        self.life_tokens_left : np.ndarray = life_tokens_left
        self.hint_tokens_left : np.ndarray = hint_tokens_left
        self.firework = FireworkRank(red, yellow, green, white, blue)
