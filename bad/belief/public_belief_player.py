from public_belief_hand_card import PublicBelfHandCard
from ftpubvec import RemaingCards
from hint_matrix_player import HintMatrixPlayer
from likelihood_player import LikelihoodPlayer


class PublicBeliefPlayer(list):
    def __init__(self, constants, idx_ply: int, rem_cards: RemaingCards, 
                 hint_matrix_ply: HintMatrixPlayer, likelihood_ply: LikelihoodPlayer):
        self.idx_ply = idx_ply    
        super().__init__(self.init(constants, rem_cards, 
                                   hint_matrix_ply, likelihood_ply))

    def init(self, constants, rem_cards: RemaingCards,
             hint_matrix_ply: HintMatrixPlayer, 
             likelihood_ply: LikelihoodPlayer) -> list:

        public_belief_player = [PublicBelfHandCard(constants, self.idx_ply, idx_card,
                                rem_cards, hint_matrix_ply[idx_card], 
                                likelihood_ply[idx_card]) 
                                for idx_card in range(constants.num_ply)]
        return public_belief_player
        