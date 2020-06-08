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


#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include "hanabi_state.h"
#include "util.h"
#include "hanabi_parallel_env.h"

hanabi_learning_env::HanabiParallelEnv::HanabiParallelEnv(
    const std::unordered_map<std::string,
    std::string>& game_params,
    const int n_states)
  : game_(HanabiGame(game_params)),
    observation_encoder_(&game_),
    n_states_(n_states)
{
  Reset();
}

void hanabi_learning_env::HanabiParallelEnv::Reset() {
  parallel_states_.clear();
  agent_player_mapping_.clear();

  const auto n_players = game_.NumPlayers();
  agent_player_mapping_.resize(n_players);
  for (int state_id = 0; state_id < n_states_; ++state_id) {
    parallel_states_.push_back(NewState());
    for (size_t agent_idx = 0; agent_idx < n_players; ++agent_idx) {
      agent_player_mapping_[agent_idx].push_back(
          (parallel_states_.back().CurPlayer() + agent_idx) % n_players);
    }
  }
}

hanabi_learning_env::HanabiState
hanabi_learning_env::HanabiParallelEnv::NewState() {
  HanabiState state(&game_);
  while (state.CurPlayer() == kChancePlayerId) {
    state.ApplyRandomChance();
  }
  return state;
}

std::vector<int> hanabi_learning_env::HanabiParallelEnv::GetScores() const {
  std::vector<int> scores;
  std::transform(parallel_states_.begin(), parallel_states_.end(),
                 std::back_inserter(scores),
                 [](const hanabi_learning_env::HanabiState& state) {
                    return state.Score(); });
  return scores;
}

void hanabi_learning_env::HanabiParallelEnv::ResetStates(
    const std::vector<int>& states,
    const int current_agent_id) {
  #pragma omp parallel for
  for (size_t idx = 0; idx < states.size(); ++idx) {
    const size_t state_idx = states[idx];
    auto& state = parallel_states_[state_idx];
    state = NewState();
    for (int player_idx = 0; player_idx < game_.NumPlayers(); ++player_idx) {
      const int agent_id =
        (current_agent_id + player_idx) % game_.NumPlayers();
      const int corresponding_player_id =
        (state.CurPlayer() + player_idx) % game_.NumPlayers();
      agent_player_mapping_[agent_id][state_idx] = corresponding_player_id;
    }
    REQUIRE(!parallel_states_[state_idx].IsTerminal());
  }
}

void hanabi_learning_env::HanabiParallelEnv::ApplyBatchMove(
    const std::vector<HanabiMove>& batch_move, const int agent_id) {
  const auto& player_ids = agent_player_mapping_[agent_id];
  #pragma omp parallel for
  for (size_t state_idx = 0; state_idx < parallel_states_.size(); ++state_idx) {
    const int cur_player = player_ids[state_idx];
    auto& state = parallel_states_[state_idx];
    REQUIRE(cur_player == state.CurPlayer());
    const auto& move = batch_move[state_idx];
    state.ApplyMove(move);
    int next_player = state.CurPlayer();
    while (next_player == kChancePlayerId) {
      state.ApplyRandomChance();
      next_player = state.CurPlayer();
    }
  }
}

template<typename T>
void print_enc_obs(std::vector<T> obs) {
  std::cout << "moves cpp: ";
  for (const auto x : obs) {
    std::cout << x;
  }
  std::cout << std::endl;
}

void hanabi_learning_env::HanabiParallelEnv::ApplyBatchMove(
    const std::vector<int>& batch_move, const int agent_id) {
  std::vector<HanabiMove> moves;
  std::transform(batch_move.begin(),
                 batch_move.end(),
                 std::back_inserter(moves),
                 [this](const int muid){ return game_.GetMove(muid); });
  ApplyBatchMove(moves, agent_id);
}

int hanabi_learning_env::HanabiParallelEnv::GetObservationFlatLength() const {
  const auto obs_shape = GetObservationShape();
  return std::accumulate(
            obs_shape.begin(), obs_shape.end(), 1, std::multiplies<int>());
}


hanabi_learning_env::HanabiParallelEnv::HanabiEncodedBatchObservation
hanabi_learning_env::HanabiParallelEnv::ObserveAgent(const int agent_id) {
  HanabiEncodedBatchObservation batch_observation(
      n_states_, GetObservationFlatLength(), MaxMoves());
  const auto player_ids = agent_player_mapping_[agent_id];
  #pragma omp parallel for
  for (size_t state_idx = 0; state_idx < parallel_states_.size(); ++state_idx) {
    const int player_idx = player_ids[state_idx];
    const auto& state = parallel_states_[state_idx];
    const HanabiObservation observation(state, player_idx);
    const auto encoded_observation = observation_encoder_.Encode(observation);
    auto vec_observ_iter = batch_observation.observation.begin() +
        state_idx * encoded_observation.size();
    vec_observ_iter = std::copy(encoded_observation.begin(),
                                encoded_observation.end(),
                                vec_observ_iter);
    // gather legal moves
    auto lm_iter =
        batch_observation.legal_moves.begin() + state_idx * MaxMoves();
    for (const auto& lm : state.LegalMoves(player_idx)) {
      *(lm_iter + game_.GetMoveUid(lm)) = 1;
    }
    batch_observation.scores[state_idx] = state.Score();
    batch_observation.done[state_idx] = state.IsTerminal();
  }
  return batch_observation;
}
