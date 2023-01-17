# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable, no-method-argument
import sys
import os
from bad.bad_agent import BadAgent

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from hanabi_learning_environment import pyhanabi, rl_env
from bad.actionnetwork import ActionNetwork
from bad.encoding.observationconverter import ObservationConverter
from bad.policy import Policy

class RunEpisode:
    '''runs an episode'''
    def __init__(self, network: ActionNetwork) -> None:
        '''init'''
        self.policy = Policy(network)
        self.agents = [BadAgent(self.policy), BadAgent(self.policy)]

    def run_episode() -> None:
        '''run'''
        pass
