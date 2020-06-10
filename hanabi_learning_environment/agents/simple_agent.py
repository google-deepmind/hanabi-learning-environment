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
"""Simple Agent."""

from hanabi_learning_environment.rl_env import Agent


class SimpleAgent(Agent):
    """Agent that applies a simple heuristic."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        self.config = config
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get('information_tokens', 8)

    @staticmethod
    def playable_card(card, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        return card['rank'] == fireworks[card['color']]

    @staticmethod
    def my_playable_card(card_index, observation, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        return observation['card_knowledge'][0][card_index]['rank'] == fireworks[
            observation['card_knowledge'][0][card_index]['color']]

    @staticmethod
    def discardble_card(card, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        return card['rank'] < fireworks[card['color']]

    @staticmethod
    def my_discardble_card(card_index, observation, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        return observation['card_knowledge'][0][card_index]['rank'] < fireworks[
            observation['card_knowledge'][0][card_index]['color']]

    def act(self, observation):
        """Act based on an observation."""
        # colors={'G','W','Y','B','R'}
        # ranks={1,2,3,4,5}

        if observation['current_player_offset'] != 0:
            return None

        my_cards_list = list(observation['observed_hands'][0])
        my_hints_list = list(observation['card_knowledge'][0])

        for card_index in range(len(my_cards_list)):
            if my_hints_list[card_index]['color'] is not None and my_hints_list[card_index]['rank'] is not None:
                if SimpleAgent.my_playable_card(card_index, observation, observation['fireworks']):
                    return {'action_type': 'PLAY', 'card_index': card_index}

        for card_index in range(len(my_cards_list)):
            if my_hints_list[card_index]['color'] is not None and my_hints_list[card_index]['rank'] is not None:
                if SimpleAgent.my_discardble_card(card_index, observation, observation['fireworks']):
                    return {'action_type': 'DISCARD', 'card_index': card_index}
        # Check if there are any pending hints and play the card corresponding to
        # the hint.
        # for card_index, hint in enumerate(observation['card_knowledge'][0]):
        #     if hint['color'] is not None or hint['rank'] is not None:
        #         if observation['life_tokens'] > 1:
        #             return {'action_type': 'PLAY', 'card_index': card_index}

        # Check if it's possible to hint a card to your colleagues.
        fireworks = observation['fireworks']
        if observation['information_tokens'] > 0:
            # Check if there are any playable cards in the hands of the colleagues.
            for player_offset in range(1, observation['num_players']):
                player_hand = observation['observed_hands'][player_offset]
                player_hints = observation['card_knowledge'][player_offset]
                # Check if the card in the hand of the colleagues is playable.
                for card, hint in zip(player_hand, player_hints):
                    if SimpleAgent.playable_card(card,
                                                 fireworks) and hint['color'] is None:
                        return {
                            'action_type': 'REVEAL_COLOR',
                            'color': card['color'],
                            'target_offset': player_offset
                        }
                    if SimpleAgent.playable_card(card,
                                                 fireworks) and hint['rank'] is None:
                        return {
                            'action_type': 'REVEAL_RANK',
                            'rank': card['rank'],
                            'target_offset': player_offset
                        }
                    if SimpleAgent.discardble_card(card,
                                                   fireworks) and hint['color'] is None:
                        return {
                            'action_type': 'REVEAL_COLOR',
                            'color': card['color'],
                            'target_offset': player_offset
                        }
                    if SimpleAgent.discardble_card(card,
                                                   fireworks) and hint['rank'] is None:
                        return {
                            'action_type': 'REVEAL_RANK',
                            'rank': card['rank'],
                            'target_offset': player_offset
                        }

        # If no card is hintable then discard or play.
        if observation['information_tokens'] < 0.5 * self.max_information_tokens:
            for card_index, hint in enumerate(observation['card_knowledge'][0]):
                if hint['color'] is None and hint['rank'] is None:
                    move = {'action_type': 'DISCARD', 'card_index': card_index}
                    if move in observation['legal_moves']:
                        return move
                if hint['color'] is None or hint['rank'] is None:
                    move = {'action_type': 'DISCARD', 'card_index': card_index}
                    if move in observation['legal_moves']:
                        return move
        else:
            for card_index, hint in enumerate(observation['card_knowledge'][1]):
                if hint['rank'] is None:
                    return {'action_type': 'REVEAL_RANK',
                            'rank': observation['observed_hands'][1][card_index]['rank'],
                            'target_offset': 1}
                if hint['color'] is None:
                    return {'action_type': 'REVEAL_COLOR',
                            'color': observation['observed_hands'][1][card_index]['color'],
                            'target_offset': 1}
