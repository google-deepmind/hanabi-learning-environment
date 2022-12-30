import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)


class ObservationVector:

    def __init__(self, lifeTokensLeft : np.ndarray, hintTokensLeft: np.ndarray):
        self.lifeTokensLeft : np.ndarray = lifeTokensLeft
        self.hintTokensLeft : np.ndarray = hintTokensLeft        