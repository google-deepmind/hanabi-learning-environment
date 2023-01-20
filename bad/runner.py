# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-many-function-args
import random
import sys
import os
import numpy as np
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.runeposiode import RunEpisode
from bad.trainepoch import TrainEpoch
from bad.print_episode_selfplay import PrintEpisodeSelfPlay
from bad.print_total_selfplay import PrintTotalSelfPlay

class Runner:
    '''runner'''
    def __init__(self) -> None:
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.network = None

    def train(self, episodes: int, batch_size:int) -> None:
        '''train'''
        print('train')
        for episode in range(episodes):
            print(f"begin training for {episode}")
            train_epoch = TrainEpoch()
            self.network = train_epoch.train(batch_size)

    def self_play(self, episodes: int) -> None:
        '''self play'''
        print('self play')

        total_reward = 0
        max_reward = 0

        for episode in range(episodes):
            # observations = self.environment.reset()
            run_episode = RunEpisode(self.network)
            episode_result = run_episode.run(episode)

            if episode_result.reward > max_reward:
                max_reward = episode_result.reward
            total_reward += episode_result.reward

            print_selfplay = PrintEpisodeSelfPlay(episode_result)
            print_selfplay.print()

        print('')
        print_total_selfplay = PrintTotalSelfPlay(episodes, total_reward, max_reward)
        print_total_selfplay.print()
