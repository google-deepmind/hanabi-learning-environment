# pylint: disable=missing-module-docstring, wrong-import-position, import-error too-few-public-methods too-many-arguments
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.publicfeatures import PublicFeatures

class Observation:
    '''observation'''
    def __init__(self, observation: dict):
        self.public_features = PublicFeatures(observation)
