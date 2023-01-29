# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from hanabi_learning_environment.pyhanabi import HanabiState


class EpochState():
    '''epoch state'''
    def __init__(self, parent, action: int, hanabi_state: HanabiState) -> None:
        self.parent:EpochState = parent
        self.action:int = action
        self.hanabi_state = hanabi_state
        self.childs = []
        if self.parent is not None:
            self.parent.register_childs(self)

    def register_childs(self, child):
        '''register childs'''
        self.childs.append(child)
