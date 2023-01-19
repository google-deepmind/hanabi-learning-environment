# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, unnecessary-pass, too-few-public-methods
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.actionnetwork import ActionNetwork
from bad.encoding.observation import Observation
from bad.bayesianaction import BayesianAction

class Policy:
    '''policy'''
    def __init__(self, network: ActionNetwork) -> None:
        '''init'''
        self.network = network
    def get_action(self, observation: Observation) -> BayesianAction:
        '''get action'''
        return self.network.get_action(observation)
