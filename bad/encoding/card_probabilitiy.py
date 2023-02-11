from colortointconverter import ColorToIntConverter
from rem_cards_to_int_converter import NumRemCardsToIntConverter


class CardProbabilitiy:
    '''card'''

    def __init__(self, color: str, num_rem_cards: int) -> None:
        rem_cards_converter: NumRemCardsToIntConverter = NumRemCardsToIntConverter()
        self.num_rem_cards = rem_cards_converter.convert(num_rem_cards)
        color_converter: ColorToIntConverter = ColorToIntConverter()
        self.color = color_converter.convert(color)


def main():
    '''main'''
    card: CardProbabilitiy = CardProbabilitiy('R', 1)
    print(card.rank)
    print(card.color)


if __name__ == "__main__":
    main()
