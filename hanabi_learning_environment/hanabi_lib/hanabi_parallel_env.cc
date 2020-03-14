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


#include "hanabi_parallel_env.h"
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <iostream>

hanabi_learning_env::HanabiParallelEnv::HanabiParallelEnv(
    const std::unordered_map<std::string,
    std::string>& game_params,
    const int n_states,
    const bool reset_state_on_game_end)
  : game_(HanabiGame(game_params)),
    observation_encoder_(&game_),
    reset_state_on_game_end_(reset_state_on_game_end)
{
  const auto n_players = game_.NumPlayers();
  agent_player_mapping_.resize(n_players);
  for (int state_id = 0; state_id < n_states; ++state_id) {
    parallel_states_.push_back(NewState());
    for (size_t agent_idx = 0; agent_idx < n_players; ++agent_idx) {
      agent_player_mapping_[agent_idx].push_back(
          (parallel_states_.back().CurPlayer() + agent_idx) % n_players);
    }
  }
  for (int player_id = 0; player_id < game_.NumPlayers(); ++player_id) {
    observations_.push_back({});
    for (const auto& state : parallel_states_) {
      observations_.back().push_back(HanabiObservation(state, player_id));
    }
  }
}

hanabi_learning_env::HanabiState hanabi_learning_env::HanabiParallelEnv::NewState() {
  HanabiState state(&game_);
  while (state.CurPlayer() == kChancePlayerId) {
    state.ApplyRandomChance();
  }
  return state;
}

void hanabi_learning_env::HanabiParallelEnv::ResetFinishedStates(const int agent_id) {
  for (size_t state_idx = 0; state_idx < parallel_states_.size(); ++state_idx) {
    auto& state = parallel_states_[state_idx];
    if (state.IsTerminal()) {
      state = NewState();
      for (int player_idx = 0; player_idx < game_.NumPlayers(); ++player_idx) {
        agent_player_mapping_[(player_idx + agent_id) % game_.NumPlayers()][state_idx] =
            (state.CurPlayer() + player_idx) % game_.NumPlayers();
        observations_[player_idx][state_idx] = HanabiObservation(state, player_idx);
      }
    }
  }
}

hanabi_learning_env::HanabiParallelEnv::HanabiBatchObservation
hanabi_learning_env::HanabiParallelEnv::ApplyBatchMove(
    const std::vector<HanabiMove>& batch_move, const int agent_id) {
  HanabiBatchObservation batch_observation(
      {static_cast<int>(parallel_states_.size()), observation_encoder_.Shape()[0]});
  const auto& player_ids = agent_player_mapping_[agent_id];
  #pragma omp parallel for
  for (size_t state_idx = 0; state_idx < parallel_states_.size(); ++state_idx) {
    const int cur_player = player_ids[state_idx];
    auto& state = parallel_states_[state_idx];
    const auto& move = batch_move[state_idx];
    state.ApplyMove(move);
    int next_player = state.CurPlayer();
    while (next_player == kChancePlayerId) {
      state.ApplyRandomChance();
      next_player = state.CurPlayer();
    }
    const auto& observation = observations_[next_player][state_idx];

    // put encoded observation into batch observation
    const std::vector<int> encoded_observation = observation_encoder_.Encode(observation);
    auto vec_observ_iter =
      batch_observation.observation.begin() + state_idx * encoded_observation.size();
    vec_observ_iter =
      std::copy(encoded_observation.begin(), encoded_observation.end(), vec_observ_iter);

    // gather legal moves
    auto& lms = batch_observation.legal_moves[state_idx];
    for (const auto& lm : state.LegalMoves(next_player)) {
      lms.push_back(game_.GetMoveUid(lm));
    }
    batch_observation.reward[state_idx] = state.Score();
    batch_observation.done[state_idx] = state.IsTerminal();
  }
  if (reset_state_on_game_end_) {
    ResetFinishedStates((agent_id + 1) % game_.NumPlayers());
    return ObserveAgent((agent_id + 1) % game_.NumPlayers());
  }
  return batch_observation;
}

hanabi_learning_env::HanabiParallelEnv::HanabiBatchObservation
hanabi_learning_env::HanabiParallelEnv::ApplyBatchMove(
    const std::vector<int>& batch_move, const int agent_id) {
  std::vector<HanabiMove> moves;
  std::transform(batch_move.begin(),
                 batch_move.end(),
                 std::back_inserter(moves),
                 [this](const int muid){ return game_.GetMove(muid); });
  return ApplyBatchMove(moves, agent_id);
}

int hanabi_learning_env::HanabiParallelEnv::GetObservationFlatLength() const {
  const auto obs_shape = GetObservationShape();
  return std::accumulate(obs_shape.begin(), obs_shape.end(), 1, std::multiplies<int>());
}

hanabi_learning_env::HanabiParallelEnv::HanabiBatchObservation
hanabi_learning_env::HanabiParallelEnv::ObserveAgent(const int agent_id) {
  HanabiBatchObservation batch_observation(
      {static_cast<int>(parallel_states_.size()), observation_encoder_.Shape()[0]});
  const auto player_ids = agent_player_mapping_[agent_id];
  for (size_t state_idx = 0; state_idx < parallel_states_.size(); ++state_idx) {
    const int player_idx = player_ids[state_idx];
    const auto& state = parallel_states_[state_idx];
    const auto& observation = observations_[player_idx][state_idx];
    const auto encoded_observation = observation_encoder_.Encode(observation);
    auto vec_observ_iter =
      batch_observation.observation.begin() + state_idx * encoded_observation.size();
    vec_observ_iter = std::copy(encoded_observation.begin(),
                                encoded_observation.end(),
                                vec_observ_iter);
    // gather legal moves
    auto& lms = batch_observation.legal_moves[state_idx];
    for (const auto& lm : state.LegalMoves(player_idx)) {
      lms.push_back(game_.GetMoveUid(lm));
    }
    batch_observation.reward[state_idx] = state.Score();
    batch_observation.done[state_idx] = state.IsTerminal();
  }
  return batch_observation;
}
