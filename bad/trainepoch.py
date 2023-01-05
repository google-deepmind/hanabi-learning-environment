# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from hanabi_learning_environment import pyhanabi, rl_env
from bad.encoding.observation import Observation
from bad.actionnetwork import ActionNetwork

class TrainEpoch:
    '''train epoch'''
    def run(self, network: ActionNetwork) -> None:
        '''train within an environment'''
        players:int = 2
        hanabi_environment = rl_env.make('Hanabi-Full', players, \
            pyhanabi.AgentObservationType.CARD_KNOWLEDGE.SEER)

        hanabi_observation = self.environment.reset()
        vector_observation: Observation = Observation(hanabi_observation)
        move = network.train()

        self.env.step(move)
