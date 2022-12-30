import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.fireworkrank import FireworkRank

class PublicFeatures:
    
    def __init__(self,lifeTokensLeft : np.ndarray, hintTokensLeft: np.ndarray, red:np.ndarray, yellow:np.ndarray, green:np.ndarray, white:np.ndarray, blue:np.ndarray) -> None:
        self.LifeTokensLeft : np.ndarray = lifeTokensLeft
        self.HintTokensLeft : np.ndarray = hintTokensLeft
        self.Firework = FireworkRank(red, yellow, green, white, blue)