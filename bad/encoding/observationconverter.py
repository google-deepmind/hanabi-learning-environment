import string
import sys
import os
import numpy as np
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.observation import Observation

class ObservationConverter:

    def __init__(self, observation: dict) -> None:
        self.observation = observation

    '''
    one hot encoding
    '''
    def Convert(self) -> Observation:

        lifeTokensLeft = self.GetLifeTokens()
        informationTokensLeft = self.GetInformationTokens()
        fireworkrankred = self.GetFireWorkRankColor('R')
        fireworkrankyellow = self.GetFireWorkRankColor('Y')
        fireworkrankgreen = self.GetFireWorkRankColor('G')
        fireworkrankwhite = self.GetFireWorkRankColor('W')
        fireworkrankblue = self.GetFireWorkRankColor('B')

        return Observation(lifeTokensLeft, informationTokensLeft, fireworkrankred, fireworkrankyellow, fireworkrankgreen, fireworkrankwhite, fireworkrankblue)

    def GetLifeTokens(self) -> np.ndarray:
        lifeTokens: int = self.observation['life_tokens']
        return tf.keras.utils.to_categorical(lifeTokens, num_classes=4, dtype=int)

    def GetInformationTokens(self) -> np.ndarray:
        informationTokens: int = self.observation['information_tokens']
        return tf.keras.utils.to_categorical(informationTokens, num_classes=9, dtype=int)

    def GetFireWorkRankColor(self, color: string) -> np.ndarray:
        rank = self.observation['fireworks'][color];
        return tf.keras.utils.to_categorical(rank, num_classes=6, dtype=int)
