# pylint: disable=missing-module-docstring, wrong-import-position, import-error too-many-locals too-many-statements
import unittest
import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.card import Card


class TestCards(unittest.TestCase):
    '''test cards'''
    def test_card_encoding(self):
        '''test card encoding'''

        red_card_one: Card = Card('R', 1)
        red_card_two: Card = Card('R', 2)
        red_card_three: Card = Card('R', 3)
        red_card_four: Card = Card('R', 4)
        red_card_five: Card = Card('R', 5)

        yellow_card_one: Card = Card('Y', 1)
        yellow_card_two: Card = Card('Y', 2)
        yellow_card_three: Card = Card('Y', 3)
        yellow_card_four: Card = Card('Y', 4)
        yellow_card_five: Card = Card('Y', 5)

        green_card_one: Card = Card('G', 1)
        green_card_two: Card = Card('G', 2)
        green_card_three: Card = Card('G', 3)
        green_card_four: Card = Card('G', 4)
        green_card_five: Card = Card('G', 5)

        white_card_one: Card = Card('W', 1)
        white_card_two: Card = Card('W', 2)
        white_card_three: Card = Card('W', 3)
        white_card_four: Card = Card('W', 4)
        white_card_five: Card = Card('W', 5)

        blue_card_one: Card = Card('B', 1)
        blue_card_two: Card = Card('B', 2)
        blue_card_three: Card = Card('B', 3)
        blue_card_four: Card = Card('B', 4)
        blue_card_five: Card = Card('B', 5)

        expected_red_card_one_rank = np.zeros(shape=6, dtype=int)
        expected_red_card_one_rank[1] = 1
        expected_red_card_two_rank = np.zeros(shape=6, dtype=int)
        expected_red_card_two_rank[2] = 1
        expected_red_card_three_rank = np.zeros(shape=6, dtype=int)
        expected_red_card_three_rank[3] = 1
        expected_red_card_four_rank = np.zeros(shape=6, dtype=int)
        expected_red_card_four_rank[4] = 1
        expected_red_card_five_rank = np.zeros(shape=6, dtype=int)
        expected_red_card_five_rank[5] = 1

        expected_yellow_card_one_rank = np.zeros(shape=6, dtype=int)
        expected_yellow_card_one_rank[1] = 1
        expected_yellow_card_two_rank = np.zeros(shape=6, dtype=int)
        expected_yellow_card_two_rank[2] = 1
        expected_yellow_card_three_rank = np.zeros(shape=6, dtype=int)
        expected_yellow_card_three_rank[3] = 1
        expected_yellow_card_four_rank = np.zeros(shape=6, dtype=int)
        expected_yellow_card_four_rank[4] = 1
        expected_yellow_card_five_rank = np.zeros(shape=6, dtype=int)
        expected_yellow_card_five_rank[5] = 1

        expected_green_card_one_rank = np.zeros(shape=6, dtype=int)
        expected_green_card_one_rank[1] = 1
        expected_green_card_two_rank = np.zeros(shape=6, dtype=int)
        expected_green_card_two_rank[2] = 1
        expected_green_card_three_rank = np.zeros(shape=6, dtype=int)
        expected_green_card_three_rank[3] = 1
        expected_green_card_four_rank = np.zeros(shape=6, dtype=int)
        expected_green_card_four_rank[4] = 1
        expected_green_card_five_rank = np.zeros(shape=6, dtype=int)
        expected_green_card_five_rank[5] = 1

        expected_white_card_one_rank = np.zeros(shape=6, dtype=int)
        expected_white_card_one_rank[1] = 1
        expected_white_card_two_rank = np.zeros(shape=6, dtype=int)
        expected_white_card_two_rank[2] = 1
        expected_white_card_three_rank = np.zeros(shape=6, dtype=int)
        expected_white_card_three_rank[3] = 1
        expected_white_card_four_rank = np.zeros(shape=6, dtype=int)
        expected_white_card_four_rank[4] = 1
        expected_white_card_five_rank = np.zeros(shape=6, dtype=int)
        expected_white_card_five_rank[5] = 1

        expected_blue_card_blue_rank = np.zeros(shape=6, dtype=int)
        expected_blue_card_blue_rank[1] = 1
        expected_blue_card_two_rank = np.zeros(shape=6, dtype=int)
        expected_blue_card_two_rank[2] = 1
        expected_blue_card_three_rank = np.zeros(shape=6, dtype=int)
        expected_blue_card_three_rank[3] = 1
        expected_blue_card_four_rank = np.zeros(shape=6, dtype=int)
        expected_blue_card_four_rank[4] = 1
        expected_blue_card_five_rank = np.zeros(shape=6, dtype=int)
        expected_blue_card_five_rank[5] = 1

        self.assertTrue(np.array_equal(red_card_one.rank,   expected_red_card_one_rank))
        self.assertTrue(np.array_equal(red_card_two.rank,   expected_red_card_two_rank))
        self.assertTrue(np.array_equal(red_card_three.rank, expected_red_card_three_rank))
        self.assertTrue(np.array_equal(red_card_four.rank,  expected_red_card_four_rank))
        self.assertTrue(np.array_equal(red_card_five.rank,  expected_red_card_five_rank))

        self.assertTrue(np.array_equal(yellow_card_one.rank,   expected_yellow_card_one_rank))
        self.assertTrue(np.array_equal(yellow_card_two.rank,   expected_yellow_card_two_rank))
        self.assertTrue(np.array_equal(yellow_card_three.rank, expected_yellow_card_three_rank))
        self.assertTrue(np.array_equal(yellow_card_four.rank,  expected_yellow_card_four_rank))
        self.assertTrue(np.array_equal(yellow_card_five.rank,  expected_yellow_card_five_rank))

        self.assertTrue(np.array_equal(green_card_one.rank,   expected_green_card_one_rank))
        self.assertTrue(np.array_equal(green_card_two.rank,   expected_green_card_two_rank))
        self.assertTrue(np.array_equal(green_card_three.rank, expected_green_card_three_rank))
        self.assertTrue(np.array_equal(green_card_four.rank,  expected_green_card_four_rank))
        self.assertTrue(np.array_equal(green_card_five.rank,  expected_green_card_five_rank))

        self.assertTrue(np.array_equal(white_card_one.rank,   expected_white_card_one_rank))
        self.assertTrue(np.array_equal(white_card_two.rank,   expected_white_card_two_rank))
        self.assertTrue(np.array_equal(white_card_three.rank, expected_white_card_three_rank))
        self.assertTrue(np.array_equal(white_card_four.rank,  expected_white_card_four_rank))
        self.assertTrue(np.array_equal(white_card_five.rank,  expected_white_card_five_rank))

        self.assertTrue(np.array_equal(blue_card_one.rank,   expected_blue_card_blue_rank))
        self.assertTrue(np.array_equal(blue_card_two.rank,   expected_blue_card_two_rank))
        self.assertTrue(np.array_equal(blue_card_three.rank, expected_blue_card_three_rank))
        self.assertTrue(np.array_equal(blue_card_four.rank,  expected_blue_card_four_rank))
        self.assertTrue(np.array_equal(blue_card_five.rank,  expected_blue_card_five_rank))
