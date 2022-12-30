import unittest
import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.observationconverter import ObservationConverter
from bad.encoding.observation import Observation

class ObservationConverterEncodingTest(unittest.TestCase):

    def test_converterhaseverything(self):
        
        observation = {}
        observation['life_tokens'] = 3
        observation['information_tokens'] = 8
        
        observation['fireworks'] = {}
        observation['fireworks']['B'] = 5
        observation['fireworks']['G'] = 5
        observation['fireworks']['R'] = 5
        observation['fireworks']['W'] = 5
        observation['fireworks']['Y'] = 5
        
        expectedLifeTokensLeft = np.zeros(shape=4, dtype=int)
        expectedLifeTokensLeft[3] = 1

        expectedHintTokensLeft = np.zeros(shape=9, dtype=int)
        expectedHintTokensLeft[8] = 1

        expectedfireworkrankred = np.zeros(shape=6, dtype=int)
        expectedfireworkrankred[5] = 1

        expectedfireworkrankgreen = np.zeros(shape=6, dtype=int)
        expectedfireworkrankgreen[5] = 1

        expectedfireworkrankblue = np.zeros(shape=6, dtype=int)
        expectedfireworkrankblue[5] = 1

        expectedfireworkrankwhite = np.zeros(shape=6, dtype=int)
        expectedfireworkrankwhite[5] = 1

        expectedfireworkrankyellow = np.zeros(shape=6, dtype=int)
        expectedfireworkrankyellow[5] = 1

        converter = ObservationConverter(observation)
        gamevector: Observation = converter.Convert()

        self.assertTrue(np.array_equal(gamevector.PublicFeatures.LifeTokensLeft, expectedLifeTokensLeft))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.HintTokensLeft, expectedHintTokensLeft))

        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.Red, expectedfireworkrankred))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.Green, expectedfireworkrankgreen))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.Blue, expectedfireworkrankblue))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.White, expectedfireworkrankwhite))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.Yellow, expectedfireworkrankyellow))

    def test_converterhasnothing(self):
        
        observation = {}
        observation['life_tokens'] = 0
        observation['information_tokens'] = 0

        observation['fireworks'] = {}
        observation['fireworks']['B'] = 0
        observation['fireworks']['G'] = 0
        observation['fireworks']['R'] = 0
        observation['fireworks']['W'] = 0
        observation['fireworks']['Y'] = 0

        expectedLifeTokensLeft = np.zeros(shape=4, dtype=int)
        expectedLifeTokensLeft[0] = 1

        expectedHintTokensLeft = np.zeros(shape=9, dtype=int)
        expectedHintTokensLeft[0] = 1

        expectedfireworkrankred = np.zeros(shape=6, dtype=int)
        expectedfireworkrankred[0] = 1

        expectedfireworkrankgreen = np.zeros(shape=6, dtype=int)
        expectedfireworkrankgreen[0] = 1

        expectedfireworkrankblue = np.zeros(shape=6, dtype=int)
        expectedfireworkrankblue[0] = 1

        expectedfireworkrankwhite = np.zeros(shape=6, dtype=int)
        expectedfireworkrankwhite[0] = 1

        expectedfireworkrankyellow = np.zeros(shape=6, dtype=int)
        expectedfireworkrankyellow[0] = 1

        converter = ObservationConverter(observation)
        gamevector = converter.Convert()
        
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.LifeTokensLeft, expectedLifeTokensLeft))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.HintTokensLeft, expectedHintTokensLeft))

        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.Red, expectedfireworkrankred))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.Green, expectedfireworkrankgreen))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.Blue, expectedfireworkrankblue))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.White, expectedfireworkrankwhite))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.Yellow, expectedfireworkrankyellow))

    def test_differentfireworks(self):
        
        observation = {}
        observation['life_tokens'] = 3
        observation['information_tokens'] = 8
        
        observation['fireworks'] = {}
        observation['fireworks']['B'] = 1
        observation['fireworks']['G'] = 2
        observation['fireworks']['R'] = 3
        observation['fireworks']['W'] = 4
        observation['fireworks']['Y'] = 5
        
        expectedfireworkrankblue = np.zeros(shape=6, dtype=int)
        expectedfireworkrankblue[1] = 1

        expectedfireworkrankgreen = np.zeros(shape=6, dtype=int)
        expectedfireworkrankgreen[2] = 1

        expectedfireworkrankred = np.zeros(shape=6, dtype=int)
        expectedfireworkrankred[3] = 1

        expectedfireworkrankwhite = np.zeros(shape=6, dtype=int)
        expectedfireworkrankwhite[4] = 1

        expectedfireworkrankyellow = np.zeros(shape=6, dtype=int)
        expectedfireworkrankyellow[5] = 1

        converter = ObservationConverter(observation)
        gamevector: Observation = converter.Convert()

        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.Red, expectedfireworkrankred))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.Green, expectedfireworkrankgreen))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.Blue, expectedfireworkrankblue))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.White, expectedfireworkrankwhite))
        self.assertTrue(np.array_equal(gamevector.PublicFeatures.Firework.Yellow, expectedfireworkrankyellow))

if __name__ == '__main__':
    unittest.main()