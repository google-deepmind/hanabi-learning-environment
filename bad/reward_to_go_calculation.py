# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods, line-too-long

import sys
import os
import numpy as np


currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.collect_episodes_data_results import CollectEpisodesDataResults
from bad.rewards_to_go_episode_calculation_result import RewardsToGoEpisodeCalculationResult
from bad.rewards_to_go_calculation_result import RewardsToGoCalculationResult

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
            buffer = collected_data_result.buffer

            for index in range(len(buffer.rewards)):
                current_reward = float(np.sum(buffer.rewards[index:]))
                discounted_reward = current_reward * np.power(self.gamma, index + 1)

                categorical = buffer.actions[index].categorical
                sampled_action = buffer.actions[index].sampled_action
                observation = buffer.observation[index]
                # loss calculation
                current_loss = -(discounted_reward * float(categorical.log_prob(sampled_action).numpy()))

                ep_result.append(discounted_reward, current_loss, observation)

        return result
