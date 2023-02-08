# pylint: disable=missing-module-docstring, wrong-import-position, too-few-public-methods
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.rewards_to_go_episode_calculation_result import RewardsToGoEpisodeCalculationResult

class RewardsToGoCalculationResult:
    '''RewardToGoCalculationResult'''
    def __init__(self) -> None:
        self.results: list[RewardsToGoEpisodeCalculationResult] = []

    def append(self, result: RewardsToGoEpisodeCalculationResult):
        '''append'''
        self.results.append(result)
