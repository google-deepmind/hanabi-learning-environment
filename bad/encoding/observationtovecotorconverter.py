import sys
import os
import numpy as np
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.vectors.observationvector import ObservationVector

class ObservationToVectorConverter:

    '''
    one hot encoding
    '''
    def Convert(self, observation: dict) -> ObservationVector:

        lifeTokens: int = observation['life_tokens']
        lifeTokensLeft: np.ndarray = tf.keras.utils.to_categorical(lifeTokens, num_classes=4, dtype=int)

        informationTokens: int = observation['information_tokens']
        informationTokensLeft: np.ndarray = tf.keras.utils.to_categorical(informationTokens, num_classes=9, dtype=int)
        
        
        return ObservationVector(lifeTokensLeft, informationTokensLeft)