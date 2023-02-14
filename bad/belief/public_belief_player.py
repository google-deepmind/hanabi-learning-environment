from public_belief_hand_card import PublicBelfHandCard
from ftpubvec import RemaingCards
from hint_matrix_player import HintMatrixPlayer


class PublicBeliefPlayer(list):
    def __init__(self, constants, idx_ply: int, rem_cards: RemaingCards, 
                 hint_matrix_player: HintMatrixPlayer):
        self.idx_ply = idx_ply    
        super().__init__(self.init(constants, rem_cards, hint_matrix_player))

    def init(self, constants, rem_cards: RemaingCards,
             hint_matrix_player: HintMatrixPlayer) -> list:
        public_belief_player = [PublicBelfHandCard(constants, self.idx_ply, idx_card,
                                rem_cards, hint_matrix_player[idx_card]) 
                                for idx_card in range(constants.num_ply)]
                
        return public_belief_player
        