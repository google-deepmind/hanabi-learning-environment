# pylint: disable=missing-module-docstring
import numpy as np


class FireworkRank:
    ''' firework rank '''
    def __init__(self, red:np.ndarray, yellow:np.ndarray, \
         green:np.ndarray, white:np.ndarray, blue:np.ndarray) -> None:
        self.red = red
        self.yellow = yellow
        self.green = green
        self.white = white
        self.blue = blue
