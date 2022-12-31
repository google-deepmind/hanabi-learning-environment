# pylint: disable=missing-module-docstring, wrong-import-position, import-error too-few-public-methods too-many-arguments
import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.publicfeatures import PublicFeatures

class Observation:
    '''observation'''
    def __init__(self, life_tokens_left : np.ndarray, hint_tokens_left: np.ndarray,\
        red:np.ndarray, yellow:np.ndarray, green:np.ndarray, white:np.ndarray,\
        blue:np.ndarray):
        self.public_features = PublicFeatures(life_tokens_left, hint_tokens_left, \
            red, yellow, green, white, blue)
