# pylint: disable=missing-module-docstring, wrong-import-position, too-few-public-methods

import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.collect_episode_data_result import CollectEpisodeDataResult


class CollectEpisodesDataResults:
    '''collect episode data results'''
    def __init__(self) -> None:
        self.results: list[CollectEpisodeDataResult] = []

    def add(self, result: CollectEpisodeDataResult) -> None:
        '''add'''
        self.results.append(result)
