import sys
import os

from abc import ABC, abstractmethod

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.observation import Observation

class ActionNetworkBase(ABC):

    @abstractmethod
    def Execute(self, observation: Observation):
        pass

