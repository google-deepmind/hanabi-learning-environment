# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from hanabi_learning_environment import pyhanabi, rl_env
from bad.actionnetwork import ActionNetwork
from bad.encoding.observationconverter import ObservationConverter
from bad.set_extra_observation import SetExtraObservation

class TrainEpoch:
    '''train epoch'''
    def __init__(self) -> None:
        players:int = 2
        self.hanabi_environment = rl_env.make('Hanabi-Full', players, \
            pyhanabi.AgentObservationType.CARD_KNOWLEDGE.SEER)

    def train(self, batch_size: int) -> ActionNetwork:
        '''train within an environment'''
        hanabi_observation = self.hanabi_environment.reset()
        # one more move because of no-action move on the beginning
        #fake an action that does not exists
        max_moves: int = self.hanabi_environment.game.max_moves() + 1
        max_actions = max_moves + 1 # 0 index based

        seo = SetExtraObservation()
        seo.set_extra_observation(hanabi_observation, max_moves, max_actions, \
             self.hanabi_environment.state.legal_moves_int())

        observation_converter: ObservationConverter = ObservationConverter()
        observation = observation_converter.convert(hanabi_observation)

        network: ActionNetwork = ActionNetwork()
        network.build(observation, max_actions)

        done = False
        while not done:

            for batch in range(batch_size):
                # legal_actions = self.hanabi_environment.state.legal_moves()
                bad = network.train_observation(observation)
                next_action = bad.decode_action(self.hanabi_environment.state.legal_moves_int())
                next_move = self.hanabi_environment.game.get_move(next_action)

                observation_after_step, _, done, _ = self.hanabi_environment.step(next_action)

                seo.set_extra_observation(observation_after_step, next_action, max_actions, \
                    self.hanabi_environment.state.legal_moves_int())

                observation = observation_converter.convert(observation_after_step)

        return network
