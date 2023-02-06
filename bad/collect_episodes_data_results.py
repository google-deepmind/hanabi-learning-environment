# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable

import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.collect_episode_data_result import CollectEpisodeDataResult
from bad.action_network import ActionNetwork

class CollectEpisodesDataResults:
    '''collect episode data results'''
    def __init__(self, network: ActionNetwork) -> None:
        self.network = network
        self.results = []

    def add(self, result: CollectEpisodeDataResult) -> None:
        '''add'''
        self.results.append(result)
