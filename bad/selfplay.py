# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-many-function-args, ungrouped-imports
import random
import sys
import os
import numpy as np
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.run_eposiode import RunEpisode
from bad.print_episode_selfplay import PrintEpisodeSelfPlay
from bad.print_total_selfplay import PrintTotalSelfPlay
from bad.action_network import ActionNetwork

class SelfPlay:
    '''self play'''
    def __init__(self, network: ActionNetwork) -> None:
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.network = network

    def run(self, episodes: int) -> None:
        '''self play'''
        print('self play')

        total_reward = 0
        max_reward = 0
        perfect_games = 0

        for episode in range(episodes):

            run_episode = RunEpisode(self.network)
            episode_result = run_episode.run(episode)

            if episode_result.reward > max_reward:
                max_reward = episode_result.reward
            total_reward += episode_result.reward

            if episode_result.reward == 25:
                perfect_games = perfect_games+1

            print_selfplay = PrintEpisodeSelfPlay(episode_result)
            print_selfplay.print()

        print('')
        print_total_selfplay = PrintTotalSelfPlay(episodes, total_reward, max_reward, perfect_games)
        print_total_selfplay.print()
