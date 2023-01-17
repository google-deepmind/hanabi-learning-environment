# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable
import sys
import os

import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from hanabi_learning_environment import pyhanabi, rl_env
from bad.actionnetwork import ActionNetwork
from bad.encoding.observationconverter import ObservationConverter

class TrainEpoch:
    '''train epoch'''
    def __init__(self) -> None:
        players:int = 2
        self.hanabi_environment = rl_env.make('Hanabi-Full', players, \
            pyhanabi.AgentObservationType.CARD_KNOWLEDGE.SEER)

    def set_extra_observation(self, hanabi_observation:dict, last_action: int, max_action:int, \
         legal_moves: list) -> None:
        '''set extra observation'''
        hanabi_observation['last_action'] = last_action
        hanabi_observation['max_action'] = max_action

        legal_moves_int = np.empty(0, int)
        for legal_move in legal_moves:
            legal_moves_int = np.append(legal_moves_int, \
                self.hanabi_environment.game.get_move_uid(legal_move))

        hanabi_observation['legal_actions'] = legal_moves_int

    def train(self, batch_size: int) -> ActionNetwork:
        '''train within an environment'''
        hanabi_observation = self.hanabi_environment.reset()
        # one more move because of no-action move on the beginning
        #fake an action that does not exists
        max_moves: int = self.hanabi_environment.game.max_moves() + 1
        max_actions = max_moves + 1 # 0 index based

        self.set_extra_observation(hanabi_observation, max_moves, max_actions, \
             self.hanabi_environment.state.legal_moves())

        observation_converter: ObservationConverter = ObservationConverter()
        observation = observation_converter.convert(hanabi_observation)

        network: ActionNetwork = ActionNetwork()
        network.build(observation, max_actions)

        done = False
        while not done:

            for batch in range(batch_size):
                # legal_actions = self.hanabi_environment.state.legal_moves()
                bad = network.train_observation(observation)
                next_action = bad.random_action()
                next_move = self.hanabi_environment.game.get_move(next_action)

                if not self.hanabi_environment.state.move_is_legal(next_move):
                    print('illegal move')

                observation_after_step, _, done, _ = self.hanabi_environment.step(next_action)

                self.set_extra_observation(observation_after_step, next_action, max_actions, \
                    self.hanabi_environment.state.legal_moves())

                # backpropagation
                observation = observation_converter.convert(observation_after_step)

        print(f'finish: {self.hanabi_environment.state}')
        print(f'score: {self.hanabi_environment.state.score}')

        return network
