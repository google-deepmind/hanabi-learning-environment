import unittest
import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.observationconverter import ObservationConverter

class ObservationConverterEncodingTest(unittest.TestCase):

    def test_converterhaseverything(self):
        
        observation = {}
        observation['life_tokens'] = 3
        observation['information_tokens'] = 8

        converter = ObservationConverter()
        gamevector = converter.Convert(observation)
        
        expectedLifeTokensLeft = np.zeros(shape=4, dtype=int)
        expectedLifeTokensLeft[3] = 1

        expectedHintTokensLeft = np.zeros(shape=9, dtype=int)
        expectedHintTokensLeft[8] = 1
                
        self.assertTrue(np.array_equal(gamevector.lifeTokensLeft, expectedLifeTokensLeft))
        self.assertTrue(np.array_equal(gamevector.hintTokensLeft, expectedHintTokensLeft))

    def test_converterhasnothing(self):
        
        observation = {}
        observation['life_tokens'] = 0
        observation['information_tokens'] = 0

        converter = ObservationConverter()
        gamevector = converter.Convert(observation)
        
        expectedLifeTokensLeft = np.zeros(shape=4, dtype=int)
        expectedLifeTokensLeft[0] = 1

        expectedHintTokensLeft = np.zeros(shape=9, dtype=int)
        expectedHintTokensLeft[0] = 1
                
        self.assertTrue(np.array_equal(gamevector.lifeTokensLeft, expectedLifeTokensLeft))
        self.assertTrue(np.array_equal(gamevector.hintTokensLeft, expectedHintTokensLeft))

if __name__ == '__main__':
    unittest.main()