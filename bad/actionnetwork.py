# pylint: disable=missing-module-docstring, wrong-import-position, import-error
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

class ActionNetwork():
    ''' action network '''
    def build(self) -> None:
        '''build'''
        print('building network')

    def train(self) -> int:
        '''train'''
        print('traning network')
