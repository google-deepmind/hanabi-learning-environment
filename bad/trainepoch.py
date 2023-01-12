# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods
import random
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
        players:int = 2
        self.hanabi_environment = rl_env.make('Hanabi-Full', players, \
            pyhanabi.AgentObservationType.CARD_KNOWLEDGE.SEER)
    def train(self) -> None:
        '''train within an environment'''
        network: ActionNetwork = ActionNetwork()
        network.build()

        hanabi_observation = self.hanabi_environment.reset()

        done = False
        while not done:

            legal_actions = self.hanabi_environment.state.legal_moves()
            action = random.choice(legal_actions)
            action_int: int = self.hanabi_environment.game.get_move_uid(action)

            encoded_observation: Observation = Observation(hanabi_observation)
            network.train_observation(encoded_observation)
            _, _, done, _ = self.hanabi_environment.step(action_int)

        print(f'finish: {self.hanabi_environment.state}')
        print(f'score: {self.hanabi_environment.state.score}')
