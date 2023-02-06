# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.buffer import Buffer


class CollectEpisodeDataResult:
    '''collect episode data result'''
    def __init__(self, buffer: Buffer) -> None:
        ''''init'''
        self.buffer = buffer
