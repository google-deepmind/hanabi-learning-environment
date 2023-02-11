import numpy as np
import tensorflow as tf


class NumRemCardsToIntConverter:
    ''' rank converter'''
    def convert(self, rem_cards:int) -> np.ndarray:
        '''convert'''
        return tf.keras.utils.to_categorical(rem_cards, num_classes=6, dtype=int)