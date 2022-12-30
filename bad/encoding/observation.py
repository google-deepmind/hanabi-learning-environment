import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.publicfeatures import PublicFeatures

class Observation:

    def __init__(self, lifeTokensLeft : np.ndarray, hintTokensLeft: np.ndarray, red:np.ndarray, yellow:np.ndarray, green:np.ndarray, white:np.ndarray, blue:np.ndarray):
        self.PublicFeatures = PublicFeatures(lifeTokensLeft, hintTokensLeft, red, yellow, green, white, blue)