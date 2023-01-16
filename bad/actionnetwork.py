# pylint: disable=missing-module-docstring, wrong-import-position, import-error
import sys
import os
import numpy as np
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.observation import Observation

class ActionNetwork():
    ''' action network '''

    def __init__(self) -> None:
        self.model = None

    def build(self, observation: Observation) -> None:
        '''build'''
        shape = observation.to_array().shape
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=shape),
            tf.keras.layers.Dense(2, activation="relu", name="layer1"),
            tf.keras.layers.Dense(3, activation="relu", name="layer2")
        ])
        #with tf.device('/cpu:0'):
        # self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def print_summary(self):
        '''print summary'''
        self.model.summary()

    def train_observation(self, observation: Observation) -> int:
        '''input observation, output action'''
        print(f'traning network with firework blue: {observation.public_features.firework.blue}')
        network_input = observation.to_array()
        result = self.model(network_input.astype(np.float32))
        return result
