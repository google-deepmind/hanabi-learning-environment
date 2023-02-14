# pylint: disable=missing-module-docstring, wrong-import-position, too-few-public-methods
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.action_network import ActionNetwork
from bad.encoding.observation import Observation
from bad.bayesian_action import BayesianAction
from bad.encoding.public_belief_global_enc import PublicBeliefGlobalEnc

class Policy:
    '''policy'''
    def __init__(self, network: ActionNetwork) -> None:
        '''init'''
        self.network = network
    def get_action(self, observation: Observation, public_belief: PublicBeliefGlobalEnc) -> BayesianAction:
        '''get action'''
        return self.network.get_action(observation, public_belief)
