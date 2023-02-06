# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.action_network import ActionNetwork
from bad.buffer import Buffer


class CollectEpisodeDataResult:
    def __init__(self, action_network: ActionNetwork, buffer: Buffer) -> None:
        self.action_network = action_network
        self.buffer = buffer
