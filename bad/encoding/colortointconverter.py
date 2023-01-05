# pylint: disable=missing-module-docstring, wrong-import-position, import-error too-few-public-methods too-many-arguments

import string
import numpy as np
import tensorflow as tf


class ColorToIntConverter:
    '''color to int converter'''

    def __init__(self) -> None:
        '''init'''
        self.colors = {'R': 0, 'Y': 1, 'G': 2, 'W': 3, 'B': 4}

    def convert(self, col: string) -> np.ndarray:
        '''convert'''
        return tf.keras.utils.to_categorical(self.convert_color_to_int(col), num_classes=6, \
            dtype=int)

    def convert_color_to_int(self, col: string) -> int:
        '''convert color to int'''
        return self.colors[col]
