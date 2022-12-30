import sys
import os

from keras.models import Sequential

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.actionnetworkbase import ActionNetworkBase

class ActionNetwork(ActionNetworkBase):
    
    def build(self) -> None:
        model = Sequential()
