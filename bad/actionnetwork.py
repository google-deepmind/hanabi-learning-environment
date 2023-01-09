# pylint: disable=missing-module-docstring, wrong-import-position, import-error
import sys
import os
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.observation import Observation

class ActionNetwork():
    ''' action network '''

    def __init__(self) -> None:
        self.model = None

    def build(self) -> None:
        '''build'''
        print('building network')
        self.model = tf.keras.Sequential()

    def train_observation(self, observation: Observation) -> int:
        '''input observation, output action'''
        print(f'traning network with firework blue: {observation.public_features.firework.blue}')
