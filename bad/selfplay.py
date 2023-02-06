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
from bad.collect_episode_data import CollectEpisodeData
from bad.print_episode_selfplay import PrintEpisodeSelfPlay
from bad.print_total_selfplay import PrintTotalSelfPlay
from bad.collect_episode_data_result import CollectEpisodeDataResult
from bad.collect_episodes_data_results import CollectEpisodesDataResults
from hanabi_learning_environment import pyhanabi, rl_env
from bad.constants import Constants

class Runner:
    '''runner'''
    def __init__(self) -> None:
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.network = None

    def train_batch(self, batch_size:int) -> None:
        '''train'''
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


    def self_play(self, episodes: int) -> None:
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
