from public_belief_hand_card import PublicBelfHandCard
from ftpubvec import RemaingCards
from hint_matrix_player import HintMatrixPlayer


class PublicBeliefPlayer(list):
    def __init__(self, observation, rem_cards: RemaingCards, 
                 hint_matrix_player: HintMatrixPlayer):
    
        self.num_hand_cards = 5
        super().__init__(self.init(observation, rem_cards, hint_matrix_player))

    def init(self, observation, rem_cards: RemaingCards,
             hint_matrix_player: HintMatrixPlayer) -> list:
        public_belief_player = [PublicBelfHandCard(observation, rem_cards, hint_matrix_player[idx_card]) 
                                for idx_card in range(self.num_hand_cards)]
                
        return public_belief_player
        