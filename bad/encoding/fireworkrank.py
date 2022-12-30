import numpy as np


class FireworkRank:
    def __init__(self, red:np.ndarray, yellow:np.ndarray, green:np.ndarray, white:np.ndarray, blue:np.ndarray) -> None:
        self.Red = red
        self.Yellow = yellow
        self.Green = green
        self.White = white
        self.Blue = blue