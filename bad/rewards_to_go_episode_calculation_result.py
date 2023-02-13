# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods

import statistics

import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.observation import Observation


class RewardsToGoEpisodeCalculationResult:
    '''RewardsToGoEpisodeCalculationResult'''
    def __init__(self) -> None:
        self.rewards: list[float] = []
        self.losses: list[float] = []
        self.observation: list[Observation] = []

    def append(self, reward: float, loss: float, observation: Observation) -> None:
        '''add'''
        self.rewards.append(reward)
        self.losses.append(loss)
        self.observation.append(observation)

    def mean_loss(self) -> float:
        '''mean losses'''
        return statistics.mean(self.losses)
