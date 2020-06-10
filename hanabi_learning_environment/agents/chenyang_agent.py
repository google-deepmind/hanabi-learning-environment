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
    def playable_card(card_knowledge, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        possible_card = ChenyangAgent.get_possible(card_knowledge)
        all_playable_card = ChenyangAgent.all_playable_cards(fireworks)
        return set(possible_card).issubset(set(all_playable_card))

    def get_possible(self, card_knowledge):
        possible_colors = []
        possible_ranks = []
        for color in ChenyangAgent.colors:
            if card_knowledge.color_plausible(color):
                possible_colors.append(color)
        for rank in ChenyangAgent.ranks:
            if card_knowledge.rank_plausible(rank):
                possible_ranks.append(rank)
        possible_cards = [HanabiCard(color, rank) for color in possible_colors for rank in possible_ranks]
        return possible_cards

    def all_playable_cards(self, fireworks):
        all_playable_cards = []
        for card in fireworks:
            all_playable_cards.append(card)
        return all_playable_cards

    def all_discardble_cards(self, fireworks):
        all_discardble_cards = []
        for card in fireworks:
            [all_discardble_cards.append(HanabiCard(card['color'], rank)) for rank in range(card['rank'])]
        return all_discardble_cards

    def act(self, observation):
        """Act based on an observation."""

        if observation['current_player_offset'] != 0:
            return None

        # Check if there are any pending hints and play the card corresponding to
        # the hint.
        for card_index, hint in enumerate(observation['card_knowledge'][0]):
            if hint['color'] is not None:
                if observation['life_tokens'] > 1:
                    return {'action_type': 'PLAY', 'card_index': card_index}

        # Check if it's possible to hint a card to your colleagues.
        fireworks = observation['fireworks']
        if observation['information_tokens'] > 0:
            # Check if there are any playable cards in the hands of the colleagues.
            for player_offset in range(1, observation['num_players']):
                player_hand = observation['observed_hands'][player_offset]
                player_hints = observation['card_knowledge'][player_offset]
                # Check if the card in the hand of the colleagues is playable.
                for card, hint in zip(player_hand, player_hints):
                    if ChenyangAgent.playable_card(card,
                                                   fireworks) and hint['color'] is None:
                        return {
                            'action_type': 'REVEAL_COLOR',
                            'color': card['color'],
                            'target_offset': player_offset
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
