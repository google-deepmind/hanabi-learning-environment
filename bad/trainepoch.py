# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable
import sys
import os

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
    def train(self, batch_size: int) -> None:
        '''train within an environment'''
        hanabi_observation = self.hanabi_environment.reset()
        # one more move because of no-action move on the beginning
        max_moves: int = self.hanabi_environment.game.max_moves() + 1
        max_actions = max_moves + 1 # 0 index based
        hanabi_observation['last_action'] = max_moves #fake an action that does not exists
        hanabi_observation['max_action'] = max_actions
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
                observation_after_step, _, done, _ = self.hanabi_environment.step(next_action)
                observation_after_step['last_move'] = next_action
                observation_after_step['max_action'] = max_actions
                # backpropagation
                observation = observation_converter.convert(observation_after_step)

        print(f'finish: {self.hanabi_environment.state}')
        print(f'score: {self.hanabi_environment.state.score}')
