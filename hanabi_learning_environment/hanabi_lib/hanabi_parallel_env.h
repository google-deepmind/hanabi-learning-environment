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

#ifndef __HANABI_PARALLEL_ENV_H__
#define __HANABI_PARALLEL_ENV_H__

#include <algorithm>
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

#include <iostream>

namespace hanabi_learning_env {

class HanabiParallelEnv {
 public:

  /** \brief Struct for batched observations.
   */
  struct HanabiEncodedBatchObservation {

    /** \brief Contruct a batched observation of given shape.
     *
     *  \param n_states Number of states.
     *  \param observation_len Length of a single flat encoded observation.
     *  \param max_moves Total number of possible moves.
     */
    HanabiEncodedBatchObservation(const int n_states, const int observation_len,
        const int max_moves)
      : observation(n_states * observation_len),
        legal_moves(n_states * max_moves, 0),
        scores(n_states),
        done(n_states),
        observation_shape({n_states, observation_len}),
        legal_moves_shape({n_states, max_moves}) {
    }
    std::vector<int> observation; //< Concatenated flat encoded observations.
    std::vector<int> legal_moves; //< Concatenated legal moves.
    std::vector<int> scores;      //< Concatenated scores.
    std::vector<bool> done;       //< Concatenated termination statuses.
    std::array<int, 2> observation_shape{0, 0}; //< Shape of batched observation (n_states x encoded_observation_length).
    std::array<int, 2> legal_moves_shape{0, 0}; //< Shape of legal moves (n_states x max_moves).
  };

  /** \brief Construct and environment with a single game with several parallel states.
   *
   *  \param game_params Parameters of the game. See HanabiGame.
   *  \param n_states Number of parallel states.
   */
  explicit HanabiParallelEnv(
      const std::unordered_map<std::string, std::string>& game_params,
      const int n_states);

  /** \brief Make a step: apply moves to states.
   *
   *  \param batch_move Moves, one for each state.
   */
  void ApplyBatchMove(
      const std::vector<HanabiMove>& batch_move, const int agent_id);
  
  /** \overload with moves encoded as move ids.
   */
  void ApplyBatchMove(
      const std::vector<int>& batch_move, const int agent_id);

  /** \brief Get observations for a specific agent.
   */
  HanabiEncodedBatchObservation ObserveAgent(const int agent_id);

  /** \brief Get a reference to the HanabiGame game.
   */
  const HanabiGame& GetGame() const {return game_;}
  // HanabiGame& GetGame() {return game_;}

  /** \brief Get a pointer to the HanabiGame game.
   */
  HanabiGame* GetGamePtr() {return &game_;}
  const HanabiGame* GetGamePtr() const {return &game_;}

  /** \brief Get shape of a single encoded observation.
   */
  std::vector<int> GetObservationShape() const {return observation_encoder_.Shape();};

  /** \brief Get a const reference to the parallel states.
   */
  const std::vector<HanabiState>& GetStates() const {return parallel_states_;}

  /** \brief Get the current score for each state.
   */
  std::vector<int> GetScores() const;

  /** \brief Get length of a single flattened encoded observation.
   */
  int GetObservationFlatLength() const;

  /** \brief Number of parallel states in this environment.
   */
  int GetNumStates() const {return n_states_;};

  /** \brief Number of possible moves for this a game in this environment.
   */
  int MaxMoves() const {return game_.MaxMoves();}

  /** \brief Check for states that are terminal and create new ones instead of those.
   *
   *  \param states           States to be reset.
   *  \param current_agent_id Id of the agent whose turn it is now.
   */
  void ResetStates(const std::vector<int>& states, const int current_agent_id);

  /** \brief Reset the environment, i.e. reset all states to initial.
   */
  void Reset();

 private:
  /** \brief Create a new state and deal the cards.
   *
   *  \return New HanabiState with cards dealt to players.
   */
  HanabiState NewState();

  HanabiGame game_;                                     //< Underlying instance of HanabiGame.
  std::vector<HanabiState> parallel_states_;            //< List with game states.
  std::vector<std::vector<int>> agent_player_mapping_;  //< List of players associated with each agent.
  CanonicalObservationEncoder observation_encoder_;     //< Observation encoder.
  const int n_states_ = 1;                              //< Number of parallel states.
};

}  // namespace hanabi_learning_env

#endif // __HANABI_PARALLEL_ENV_H__
