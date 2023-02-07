# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods

import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.collect_episode_data import CollectEpisodeData
from bad.collect_episode_data_result import CollectEpisodeDataResult
from bad.collect_episodes_data_results import CollectEpisodesDataResults
from bad.constants import Constants
from hanabi_learning_environment import pyhanabi, rl_env
from bad.action_network import ActionNetwork
from bad.encoding.observationconverter import ObservationConverter
from bad.set_extra_observation import SetExtraObservation
from bad.reward_to_go_calculation import RewardToGoCalculation
from bad.train_batch_result import TrainBatchResult

class TrainBatch:
    '''train batch'''
    def __init__(self, network: ActionNetwork) -> None:
        '''init'''
        self.network = network

    def collect_data(self, batch_size:int, players: int) -> CollectEpisodesDataResults:
        '''collect data'''
        collect_episodes_result = CollectEpisodesDataResults()

        while len(collect_episodes_result.results) < batch_size:

            constants = Constants()
            hanabi_environment = rl_env.make(constants.environment_name, players, \
            pyhanabi.AgentObservationType.SEER)
            hanabi_observation = hanabi_environment.reset()
            max_moves: int = hanabi_environment.game.max_moves() + 1
            max_actions = max_moves + 1 # 0 index based

            seo = SetExtraObservation()
            seo.set_extra_observation(hanabi_observation, max_moves, max_actions, \
                hanabi_environment.state.legal_moves_int())
            observation_converter: ObservationConverter = ObservationConverter()
            self.network.build(observation_converter.convert(hanabi_observation), max_actions)

            ce_data = CollectEpisodeData(hanabi_observation, hanabi_environment, self.network)
            episode_data_result: CollectEpisodeDataResult = \
                ce_data.collect() # hier werden die Daten gesammelt

            collect_episodes_result.add(episode_data_result)

        return collect_episodes_result

    def run(self, batch_size: int, discount: float) -> TrainBatchResult:
        '''init'''
        print('train')
        players:int = 2
        collected_data = self.collect_data(batch_size, players)

        reward_to_go_calculation = RewardToGoCalculation(discount)
        reward_to_go_calculation.run(collected_data)

        return TrainBatchResult()
