from hanabi_learning_environment.rl_env import HanabiEnv
from action_network import ActionNetwork
from bad.encoding.observation import Observation
import sys
import os

from likelihood_global import Likelihood
from hint_matrix_global import HintMatrix

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)


class LikelihoodHandCard(dict):
    def __init__(self, idx_ply: int, idx_card: int, constants, observation: Observation,
                 act_network: ActionNetwork, last_act, old_likelihood, hint_matrix):
        self.idx_ply = idx_ply
        self.idx_card = idx_card
        super().__init__(self.init(constants, observation,
                                   act_network, last_act, old_likelihood, hint_matrix))

    def init(self, constants, observation: Observation, act_network: ActionNetwork,
             last_act=None, old_likelihood: Likelihood = None,
             hint_matrix: HintMatrix = None) -> dict:
        """Initialize / Update the likelihood based on the action, observation and action network
           By initializing the likelihood, the first time last_act,old_likelihood and last_act are None

        Args:
            constants (HanabiEnv): Hanabi environment 
            observation (Observation): Part of Input from Network
            act_network (ActionNetwork): Network that predicts the action
            last_act (_type_, None): the last action taken by the agent
            old_likelihood (Likelihood, None): Likelikhood from the previous step
            old_public_belief (PublicBelief, None): Public belief from the previous step
            hint_matrix (HintMatrix, None): Hint matrix from the previous step
        Returns:
            dict: _description_
        """

        if (old_likelihood is None 
            and last_act is None
            and observation is None):
            '''Initialize the likelihood for the first time'''
            return self.first_init_likelihood(constants)

        # Update the likelihood based on the old likelihood, action, input from network and action network
        return self.update_likelihood(constants, observation, act_network, last_act, old_likelihood, hint_matrix)

    def first_init_likelihood(self, constants) -> dict:
        """Initialize the likelihood for the first time"""

        likelihood_hand_card = {}

        list_one = [1 for _ in range(constants.num_ranks + 1)]
        for color in constants.colors:
            likelihood_hand_card.update({color: list_one.copy()})

        return likelihood_hand_card

    def update_likelihood(self, constants, observation: Observation, act_network: ActionNetwork,
                          last_act, old_likelihood: Likelihood, hint_matrix) -> dict:
        """Update the likelihood based on the old likelihood, action, input from network and action network"""

        # Get all possible hand_card combinations
        hand_card_combinations = self.hand_card_combinations(constants)

        # Get the possiblity for hand_card_combinations
        hand_card_combinations_possibility = self.hand_card_combinations_pos(
            act_network_output, last_act)

        # Get action from network based on hand_card_combinations and observation (input from network)
        act_network_output = self.output_action_network(
            hand_card_combinations, observation, act_network)

        # Update the hand_card_combinations_possibility
        # based on the action from network
        # (Eliminate the hand_card_combinations that are not possible)
        hand_card_combinations_possibility = self.update_hand_card_combinations_pos(
            hand_card_combinations_possibility, act_network_output, last_act, hint_matrix)

        # Normalize the hand_card_combinations_possibility
        hand_card_combinations_possibility = self.normalize_hand_card_possibility(
            hand_card_combinations_possibility)

        # Get the possiblity for each possible hand_card
        hand_card_possibility = self.hand_card_possibility(
            hand_card_combinations_possibility)

        # Update the likelihood based on the hand_card_combinations_possibility
        likelihood = self.calculate_new_likelihood(
            old_likelihood, hand_card_possibility)

        return likelihood

    def hand_card_combinations(self, constants) -> list:
        """Returns all possible hand_card combinations"""

        # Get all possible cards
        card_combinations = self.card_combinations(constants)

        # Build all possible hand_card combinations
        hand_card_combinations = []
        for card1 in card_combinations:
            for card2 in card_combinations:
                for card3 in card_combinations:
                    hand_card_combinations.append([card1, card2, card3])

        return hand_card_combinations

    def card_combinations(self, constants) -> list:
        """Return all possible cards"""
        card_combinations = [{'color': color, 'rank': rank} for color in constants.colors
                             for rank in range(constants.max_rank + 1)]

        return card_combinations

    def output_action_network(self, hand_card_combinations, observation: Observation, 
        act_network: ActionNetwork) -> list:
        """Returns the action from network based on hand_card_combinations and observation (input from network)"""
        pass

    def hand_card_combinations_pos(self, hand_card_combinations, public_belief) -> list:
        """Returns the possiblity for hand_card_combinations"""

        hand_card_combinations_possibility = []
        for card_combination in hand_card_combinations:
            card1_possiblity = self.card_possibility(
                card_combination[0], public_belief)
            card2_possiblity = self.card_possibility(
                card_combination[1], public_belief)
            card3_possiblity = self.card_possibility(
                card_combination[2], public_belief)
            hand_card_combinations_possibility.append(
                card1_possiblity * card2_possiblity * card3_possiblity)

        return hand_card_combinations_possibility

    def card_possibility(self, card, public_belief):
        """Returns the possibility for a card based on the public belief"""
        card_color = card['color']
        card_rank = card['rank']
        card_possibility = public_belief[card_color][card_rank]

        return card_possibility

    def update_hand_card_combinations_pos(self, hand_card_combinations_possibility,
                                          act_network_output, last_act, hint_matrix):
        """Update the hand_card_combinations_possibility based on the action from network
           (Eliminate the hand_card_combinations that are not possible)
        """

        # Eliminate the hand_card_combinations that are not possible
        # based on the action from network
        hand_card_combinations_possibility = self.update_hand_card_combinations_pos_based_on_act_net_output(
            hand_card_combinations_possibility, act_network_output, last_act)

        # Eliminate the hand_card_combinations that are not possible based on hint_matrix
        hand_card_combinations_possibility = self.update_hand_card_combinations_pos_based_on_hint_matrix(
            hand_card_combinations_possibility, hint_matrix)

        return hand_card_combinations_possibility

    def update_hand_card_combinations_pos_based_on_act_net_output(self, hand_card_combinations_possibility,
                                                                  act_network_output, last_act):
        """Eliminate the hand_card_combinations that are not possible based on the action from network"""
        for idx, action in enumerate(act_network_output):
            if action != last_act:
                hand_card_combinations_possibility[idx] = 0

        return hand_card_combinations_possibility

    def update_hand_card_combinations_pos_based_on_hint_matrix(self, hand_card_combinations_possibility,
                                                               hint_matrix):
        """Eliminate the hand_card_combinations that are not possible based on hint_matrix"""

        for idx_hand_card_combination, hand_card_combination in hand_card_combinations_possibility:
            for idx_card, card in enumerate(hand_card_combination):
                card_rank = card['rank']
                card_color = card['color']
                if hint_matrix[self.idx_ply][idx_card][card_color][card_rank] == 0:
                    hand_card_combinations_possibility[idx_hand_card_combination] = 0

        return hand_card_combinations_possibility

    def normalize_hand_card_possibility(self, hand_card_combinations_possibility):
        """Normalize the hand_card_combinations_possibility"""

        sum = 0
        for possibility in hand_card_combinations_possibility:
            sum += possibility

        for i, possibility in enumerate(hand_card_combinations_possibility):
            hand_card_combinations_possibility[i] = possibility / sum

        return hand_card_combinations_possibility

    def hand_card_possibility(self, hand_card_combinations_possibility, hand_cards_combination)-> dict:
        """Get the possiblity for each possible hand_card"""
        all_possible_cards = self.card_combinations()

        # Init dict for hand_card_possibility
        hand_card_possibility = {}
        for card in all_possible_cards:
            hand_card_possibility.update({card: 0})

        # Get for each card the possibility based on the hand_card_combinations_possibility   
        for idx, hand_card_combination_pos in enumerate(hand_card_combinations_possibility):
            
            # The card possibilty is based on the card_idx
            card = hand_cards_combination[idx][self.idx_card]

            hand_card_possibility[card] += hand_card_combination_pos

        return hand_card_possibility

    def calculate_new_likelihood(self, old_likelihood: dict, hand_card_combinations_possibility):
        """Update the likelihood based on the hand_card_combinations_possibility and the old likelihood"""
        new_likelihood = {}
        for color, color_likelihood in old_likelihood.items():
            new_color_likelihood = [hand_card_combinations_possibility[color][rank] * color_likelihood[rank]
                                    for rank in range(len(color_likelihood))]
            new_likelihood.update({color: new_color_likelihood})

        return new_likelihood
