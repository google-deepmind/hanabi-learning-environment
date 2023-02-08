# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods

import statistics


class RewardsToGoEpisodeCalculationResult:
    '''RewardsToGoEpisodeCalculationResult'''
    def __init__(self) -> None:
        self.rewards: list[float] = []
        self.losses: list[float] = []

    def append(self, reward: float, loss: float) -> None:
        '''add'''
        self.rewards.append(reward)
        self.losses.append(loss)

    def mean_loss(self) -> float:
        '''mean losses'''
        return statistics.mean(self.losses)
