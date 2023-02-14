from build_hanabi_env import get_hanabi_env
from ftpubvec import RemaingCards


class HintMatrixHandCard(dict):
    def __init__(self, constants, rem_cards: RemaingCards, idx_ply: int,
                 idx_card: int):
        self.idx_ply = idx_ply
        self.idx_card = idx_card
        super().__init__(self.__init(constants, rem_cards))

    def __init_start_condition(self, constants) -> dict:
        """Initializes the HintMatrixHandCard first time"""
        hint_matrix_hand_card = {}

        list_one = [1 for _ in range(constants.num_ranks + 1)]
        for color in constants.colors:
            hint_matrix_hand_card.update({color: list_one.copy()})

        return hint_matrix_hand_card

    def __init(self, constants, rem_cards: RemaingCards) -> dict:
        """Updates the HintMatrixHandCard based on the rem_cards"""
        hint_matrix_hand_card = self.__init_start_condition(constants)

        for color, rem_card_per_color in rem_cards.items():
            for rank in range(constants.max_rank + 1):
                rem_card_per_card = rem_card_per_color[rank]

                # Wenn es keine Karten mehr gibt kann diese auch
                # nicht in der Hand sein
                if rem_card_per_card == 0:
                    hint_matrix_hand_card[color][rank] = 0

        return hint_matrix_hand_card


def main():
    hanabi_env = get_hanabi_env()
    observation = hanabi_env['player_observations'][0]
    rem_cards = RemaingCards(observation)
    hint_matrix_hand_card = HintMatrixHandCard(observation, rem_cards)
    print()


if __name__ == "__main__":
    main()
