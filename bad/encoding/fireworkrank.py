# pylint: disable=missing-module-docstring, wrong-import-position, import-error too-few-public-methods too-many-arguments
import string
import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)


from bad.encoding.ranktointconverter import RankToIntConverter

class FireworkRank():
    ''' firework rank '''
    def __init__(self, observation: dict) -> None:
        '''init'''
        self.observation = observation
        self.current_player = observation['current_player']

        self.red = self.convert_firework_rank_color('R')
        self.yellow = self.convert_firework_rank_color('Y')
        self.green = self.convert_firework_rank_color('G')
        self.white  = self.convert_firework_rank_color('W')
        self.blue = self.convert_firework_rank_color('B')

    def convert_firework_rank_color(self, color: string) -> np.ndarray:
        ''' convert firework rank color '''
        rank = self.observation['player_observations'][self.current_player]['fireworks'][color]
        rank_converter = RankToIntConverter()
        return rank_converter.convert(rank)
