# Developed by Lorenzo Mambretti
# June 2019
"""A simple heuristic-based agent"""

from rl_env import Agent

global colors
colors = ['Y', 'B', 'W', 'R', 'G']


class SecondAgent(Agent):
    """Agent that applies a simple heuristic."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        self.config = config
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get('information_tokens', 8)

    def playable_card(self, card, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        if card['color'] == None and card['rank'] != None:
            for color in colors:
                if fireworks[color] == card['rank']:
                    continue
                else:
                    return False

            return True
        elif card['color'] == None or card['rank'] == None:
            return False
        else:
            return card['rank'] == fireworks[card['color']]

    def act(self, observation):
        """Act based on an observation."""
        if observation['current_player_offset'] != 0:
            return None

        fireworks = observation['fireworks']

        # Check if there are any pending hints and play the card corresponding to
        # the hint.
        for card_index, hint in enumerate(observation['card_knowledge'][0]):
            if self.playable_card(hint, fireworks):
                return {'action_type': 'PLAY', 'card_index': card_index}

        # Check if it's possible to hint a card to your colleagues.
        if observation['information_tokens'] > 0:
            # Check if there are any playable cards in the hands of the opponents.
            for player_offset in range(1, observation['num_players']):
                player_hand = observation['observed_hands'][player_offset]
                player_hints = observation['card_knowledge'][player_offset]
                # Check if the card in the hand of the opponent is playable.
                for card, hint in zip(player_hand, player_hints):
                    if self.playable_card(card,
                                          fireworks) and hint['color'] is None:
                        return {
                            'action_type': 'REVEAL_COLOR',
                            'color': card['color'],
                            'target_offset': player_offset
                        }
                    elif self.playable_card(card, fireworks) and hint['rank'] is None:
                        return {
                            'action_type': 'REVEAL_RANK',
                            'rank': card['rank'],
                            'target_offset': player_offset
                        }

        # If no card is hintable then discard or play.
        if observation['information_tokens'] < self.max_information_tokens:
            return {'action_type': 'DISCARD', 'card_index': 0}
        else:
            return {'action_type': 'PLAY', 'card_index': 0}
