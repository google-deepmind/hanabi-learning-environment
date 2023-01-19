# pylint: disable=missing-module-docstring, wrong-import-position, import-error
import sys
import os
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.observation import Observation
from bad.bayesianaction import BayesianAction

class ActionNetwork():
    ''' action network '''

    def __init__(self) -> None:
        self.model = None

    def build(self, observation: Observation, max_action: int) -> None:
        '''build'''
        shape = observation.to_array().shape
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=shape, name="input"),
            tf.keras.layers.Dense(384, activation="relu", name="layer1"),
            tf.keras.layers.Dense(384, activation="relu", name="layer2"),
            tf.keras.layers.Dense(max_action, activation='softmax', name='Output_Layer')
        ])

    def print_summary(self):
        '''print summary'''
        self.model.summary()

    def get_action(self, observation: Observation) -> BayesianAction:
        '''get action'''
        network_input = observation.to_array()
        reshaped = tf.reshape(network_input, [1, network_input.shape[0]])

        result = self.model(reshaped)
        return BayesianAction(result.numpy()[0])
    def train_observation(self, observation: Observation) -> BayesianAction:
        '''input observation, output action'''
        return self.get_action(observation)
