# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods
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
    def __init__(self) -> None:
        pass
    def train(self) -> None:
        '''train within an environment'''
        network: ActionNetwork = ActionNetwork()
        network.build()

        players:int = 2
        hanabi_environment = rl_env.make('Hanabi-Full', players, \
            pyhanabi.AgentObservationType.CARD_KNOWLEDGE.SEER)

        print(f'created environment with score: {hanabi_environment.state.score} ') #pylint

        hanabi_observation = self.environment.reset()
        encoded_observation: Observation = Observation(hanabi_observation)
        network.train_observation(encoded_observation)

        # self.env.step(move)
