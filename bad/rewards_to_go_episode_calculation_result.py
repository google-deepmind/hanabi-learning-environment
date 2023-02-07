# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods

class RewardsToGoEpisodeCalculationResult:
    '''RewardsToGoEpisodeCalculationResult'''
    def __init__(self) -> None:
        self.result: list[float] = []

    def append(self, reward: float) -> None:
        '''add'''
        self.result.append(reward)
