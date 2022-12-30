import sys
import os
import numpy as np
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.gamevector import GameVector

class ObservationToVectorConverter:

    '''
    one hot encoding
    '''
    def Convert(self, observation: dict) -> GameVector:

        lifeTokens: int = observation['life_tokens']
        lifeTokensLeft: np.ndarray = tf.keras.utils.to_categorical(lifeTokens, num_classes=4, dtype=int)

        informationTokens: int = observation['information_tokens']
        informationTokensLeft: np.ndarray = tf.keras.utils.to_categorical(informationTokens, num_classes=9, dtype=int)
        return GameVector(lifeTokensLeft, informationTokensLeft)