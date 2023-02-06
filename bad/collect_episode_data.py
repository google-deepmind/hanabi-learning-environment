# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from hanabi_learning_environment import rl_env
from bad.action_network import ActionNetwork
from bad.encoding.observationconverter import ObservationConverter
from bad.set_extra_observation import SetExtraObservation
from bad.buffer import Buffer
from bad.collect_episode_data_result import CollectEpisodeDataResult

class CollectEpisodeData:
    '''train episode'''
    def __init__(self, hanabi_observation:dict, hanabi_environment: rl_env.HanabiEnv) -> None:
        self.hanabi_observation = hanabi_observation
        self.hanabi_environment = hanabi_environment

    def collect(self) \
         -> CollectEpisodeDataResult:
        '''train within an environment'''

        copied_state = self.hanabi_environment.state.copy()

        buffer = Buffer()

        self.hanabi_environment.state = copied_state.copy()
        # one more move because of no-action move on the beginning
        #fake an action that does not exists
        max_moves: int = self.hanabi_environment.game.max_moves() + 1
        max_actions = max_moves + 1 # 0 index based

        seo = SetExtraObservation()
        seo.set_extra_observation(self.hanabi_observation, max_moves, max_actions, \
            self.hanabi_environment.state.legal_moves_int())

        observation_converter: ObservationConverter = ObservationConverter()
        observation = observation_converter.convert(self.hanabi_observation)

        network: ActionNetwork = ActionNetwork()
        network.build(observation, max_actions)

        done = False
        while not done:

            # legal_actions = self.hanabi_environment.state.legal_moves()
            bad = network.train_observation(observation)
            next_action = bad.decode_action(self.hanabi_environment.state.legal_moves_int())
            # next_move = self.hanabi_environment.game.get_move(next_action)

            observation_after_step, reward, done, _ = self.hanabi_environment.step(next_action)

            buffer.add(self.hanabi_observation, next_action, reward)

            seo.set_extra_observation(observation_after_step, next_action, max_actions, \
                self.hanabi_environment.state.legal_moves_int())

            observation = observation_converter.convert(observation_after_step)
            self.hanabi_observation = observation_after_step

        return CollectEpisodeDataResult(network, buffer)
