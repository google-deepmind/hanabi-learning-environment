# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods, line-too-long

import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.collect_episodes_data_results import CollectEpisodesDataResults
from bad.rewards_to_go_calculation_result import RewardsToGoCalculationResult
from bad.rewards_to_go_episode_calculation_result import RewardsToGoEpisodeCalculationResult


class RewardToGoCalculation:
    ''''calculate reward to go'''
    def __init__(self, gamma: float) -> None:
        self.gamma = gamma

    def run(self,collected_data_results: CollectEpisodesDataResults) -> RewardsToGoCalculationResult:
        '''run'''

        result = RewardsToGoCalculationResult()

        for collected_data_result in collected_data_results.results:
            ep_result = RewardsToGoEpisodeCalculationResult()
            result.append(ep_result)

            for index in range(len(collected_data_result.buffer.rewards)):
                current_reward = float(np.sum(collected_data_result.buffer.rewards[index:]))
                ep_result.append(current_reward * np.power(self.gamma, index + 1))

        return result
