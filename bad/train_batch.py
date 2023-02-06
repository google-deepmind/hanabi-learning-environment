# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-many-function-args, ungrouped-imports

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


class TrainBatch:
    '''train batch'''
    def __init__(self) -> None:
        pass

    def run(self, batch_size: int) -> None:
        '''init'''
        print('train')

        players:int = 2

        collect_episodes_result = CollectEpisodesDataResults()

        while len(collect_episodes_result.results) < batch_size:

            constants = Constants()
            hanabi_environment = rl_env.make(constants.environment_name, players, \
            pyhanabi.AgentObservationType.CARD_KNOWLEDGE.SEER)
            hanabi_observation = hanabi_environment.reset()

            collect_episode_data = CollectEpisodeData()
            episode_data_result: CollectEpisodeDataResult = \
                 collect_episode_data.collect(hanabi_observation, hanabi_environment)

            collect_episodes_result.add(episode_data_result)
