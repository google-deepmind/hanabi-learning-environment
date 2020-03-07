// Copyright 2020 Anton Komissarov (anton.v.komissarov@gmail.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __HANABI_PARALLEL_GAME_H__
#define __HANABI_PARALLEL_GAME_H__

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "hanabi_card.h"
#include "hanabi_move.h"
#include "hanabi_game.h"
#include "hanabi_state.h"
#include "hanabi_observation.h"
#include "canonical_encoders.h"

namespace hanabi_learning_env {

class HanabiParallelGame {
 public:

  /** \brief Struct for batched observations.
   */
  struct HanabiBatchObservation {
    /** \brief Contruct a batched observation of given shape.
     *  \param shape Shape of the observation (n states x encoded observation length).
     */
    HanabiBatchObservation(const std::array<int, 2> shape) : shape(shape) {
      observation.resize(std::get<0>(shape) * std::get<1>(shape));
      legal_moves.resize(std::get<0>(shape));
      reward.resize(std::get<0>(shape));
      done.resize(std::get<0>(shape));
    }
    std::vector<int> observation;
    std::vector<std::vector<int>> legal_moves;
    std::vector<double> reward;
    std::vector<bool> done;
    std::array<int, 2> shape{0, 0}; // n_states x encoded_observation_length
  };

  /** \brief Construct a game with several parallel states.
   *  \param game_params Parameters of the game. See HanabiGame.
   *  \param n_states Number of parallel states.
   *  \param reset_state_on_game_end Whether the state should be automatically reset. Default true.
   */
  explicit HanabiParallelGame(const std::unordered_map<std::string, std::string>& game_params,
      const int n_states, const bool reset_state_on_game_end=true);

  /** \brief Make a step: apply moves to states and return observations for current players.
   *  \param batch_move Moves, one for each state.
   *  \return HanabiBatchObservation with all states.
   */
  HanabiBatchObservation Step(const std::vector<HanabiMove>& batch_move);
  
  /** \overload with moves encoded as move ids.
   */
  HanabiBatchObservation Step(const std::vector<int>& batch_move);

  HanabiBatchObservation ObservePlayer(const int player_idx);

  /** \brief Get the HanabiGame game.
   */
  const HanabiGame& GetGame() {return game_;}

  /** \brief Check for states that are terminal and create new ones instead of those.
   *  This function is called after each Step if reset_state_on_game_end_ is set to true. Otherwise
   *  user is responsible for dealing with terminal states.
   */
  void ResetFinishedStates();

 private:
  /** \brief Create a new state and deal the cards.
   */
  HanabiState NewState();

  HanabiGame game_;
  std::vector<std::vector<HanabiObservation>> observations_;
  std::vector<HanabiState> parallel_states_;
  std::vector<std::vector<int>> agent_player_mapping_;
  CanonicalObservationEncoder observation_encoder_;
  bool reset_state_on_game_end_ = true;
};

}  // namespace hanabi_learning_env

#endif
