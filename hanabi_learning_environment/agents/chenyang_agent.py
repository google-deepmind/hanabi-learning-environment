# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Chenyang Agent."""

from hanabi_learning_environment.rl_env import Agent
from hanabi_learning_environment.pyhanabi import HanabiCard
from hanabi_learning_environment.pyhanabi import HanabiCardKnowledge


class ChenyangAgent(Agent):
    """Agent that applies a simple heuristic."""
    colors = {'G', 'W', 'Y', 'B', 'R'}
    ranks = {1, 2, 3, 4, 5}

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        self.config = config
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get('information_tokens', 8)

    @staticmethod
    def playable_card(card, observation, fireworks):
        possible_card = ChenyangAgent.get_possible(card, observation)
        all_playable_card = ChenyangAgent.all_playable_cards(fireworks)
        return not [False for card in possible_card if card not in all_playable_card]

    @staticmethod
    def discardble_card(card, observation, fireworks):
        possible_card = ChenyangAgent.get_possible(card, observation)
        all_discardble_card = ChenyangAgent.all_discardble_cards(fireworks)
        return not [False for card in possible_card if card not in all_discardble_card]

    @staticmethod
    def get_possible(card_index, observation):
        hints = observation['card_knowledge'][0]
        possible_colors = []
        possible_ranks = []
        if hints[card_index]['color'] is None:
            [possible_colors.append(color) for color in ChenyangAgent.colors]
        else:
            possible_colors.append(hints[list.index(card_index)]['color'])
        if hints[card_index]['rank'] is None:
            [possible_ranks.append(rank) for rank in ChenyangAgent.ranks]
        else:
            possible_ranks.append(hints[list.index(card_index)]['rank'])
        possible_cards = [HanabiCard(color, rank) for color in possible_colors for rank in possible_ranks]
        return possible_cards

    @staticmethod
    def all_playable_cards(fireworks):
        all_playable_cards = []
        for card in fireworks:
            all_playable_cards.append(card)
        return all_playable_cards

    @staticmethod
    def all_discardble_cards(fireworks):
        all_discardble_cards = []
        for card in fireworks:
            [all_discardble_cards.append(HanabiCard(card['color'], rank)) for rank in range(card['rank'])]
        return all_discardble_cards

    def act(self, observation):
        """Act based on an observation."""
        if observation['current_player_offset'] != 0:
            return None

        fireworks = observation['fireworks']
        my_hand = observation['observed_hands'][0]
        you_hand = observation['observed_hands'][1]

        for card_index, card in enumerate(my_hand):
            if ChenyangAgent.playable_card(card_index, observation, fireworks):
                return {'action_type': 'PLAY', 'card_index': card_index}

        if observation['information tokens'] < self.max_information_tokens:
            for card_index, card in enumerate(my_hand):
                if ChenyangAgent.discardble_card(card_index, observation, fireworks):
                    return {'action_type': 'DISCARD', 'card_index': card_index}

        you_hints = observation['card_knowledge'][1]

        # Check if it's possible to hint a card to your colleagues.

        if observation['information_tokens'] > 0:
            # Check if there are any playable cards in the hands of the colleagues.
            for card_index, card in enumerate(you_hand):
                if ChenyangAgent.playable_card(card_index, observation, fireworks) and you_hints[card_index][
                    'rank'] is None:
                    return {
                        'action_type': 'REVEAL_RANK',
                        'rank': card['rank'],
                        'target_offset': 1
                    }
                if ChenyangAgent.playable_card(card, observation, fireworks) and you_hints[card_index]['color'] is None:
                    return {
                        'action_type': 'REVEAL_COLOR',
                        'color': card['color'],
                        'target_offset': 1
                    }

        # If no card is hintable then discard or play.
        if observation['information_tokens'] < self.max_information_tokens:
            for card_index, hint in enumerate(observation['card_knowledge'][0]):
                if hint['color'] is None and hint['rank'] is None:
                    return {'action_type': 'DISCARD', 'card_index': card_index}
        else:
            for card_index, hint in enumerate(observation['card_knowledge'][1]):
                if hint['rank'] is None:
                    return {'action_type': 'REVEAL_RANK',
                            'rank': observation['observed_hands'][1][card_index]['rank'],
                            'target_offset': 1}
