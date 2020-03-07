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


#include "hanabi_parallel_game.h"
#include <stdexcept>
#include <algorithm>
#include "hanabi_state.h"

hanabi_learning_env::HanabiParallelGame::HanabiParallelGame(const std::unordered_map<std::string, std::string>& game_params, const int n_states, const bool reset_state_on_game_end)
  : game_(HanabiGame(game_params)), observation_encoder_(&game_), reset_state_on_game_end_(reset_state_on_game_end)
{
  agent_player_mapping_.resize(game_.NumPlayers());
  for (int state_id = 0; state_id < n_states; ++state_id) {
    parallel_states_.push_back(NewState());
    for (size_t agent_idx = 0; agent_idx < game_.NumPlayers(); ++agent_idx) {
      agent_player_mapping_[agent_idx].push_back((parallel_states_.back().CurPlayer() + agent_idx) % game_.NumPlayers());
    }
  }
  for (int player_id = 0; player_id < game_.NumPlayers(); ++player_id) {
    observations_.push_back({});
    for (auto& state : parallel_states_) {
      observations_.back().push_back(HanabiObservation(state, player_id));
    }
  }
}

hanabi_learning_env::HanabiState hanabi_learning_env::HanabiParallelGame::NewState() {
  HanabiState state(&game_);
  while (state.CurPlayer() == kChancePlayerId) {
    state.ApplyRandomChance();
  }
  return state;
}

void hanabi_learning_env::HanabiParallelGame::ResetFinishedStates() {
  for (size_t state_idx = 0; state_idx < parallel_states_.size(); ++state_idx) {
    auto& state = parallel_states_[state_idx];
    if (state.IsTerminal()) {
      state = NewState();
      for (int player_idx = 0; player_idx < game_.NumPlayers(); ++player_idx) {
        agent_player_mapping_[player_idx][state_idx] = (state.CurPlayer() + player_idx) % game_.NumPlayers();
        observations_[player_idx][state_idx] = HanabiObservation(state, player_idx);
      }
    }
  }
}

hanabi_learning_env::HanabiParallelGame::HanabiBatchObservation
hanabi_learning_env::HanabiParallelGame::Step(const std::vector<HanabiMove>& batch_move) {
  int cur_player = parallel_states_[0].CurPlayer();
  HanabiBatchObservation batch_observation(
      {static_cast<int>(parallel_states_.size()), observation_encoder_.Shape()[0]});
  std::vector<int> encoded_observation;
  #pragma omp parallel for
  for (size_t state_idx = 0; state_idx < parallel_states_.size(); ++state_idx) {
    auto& state = parallel_states_[state_idx];
    // TODO: THINK: maybe it should be allowed that the agent plays as different players?
    if (state.CurPlayer() != cur_player) {
      throw std::runtime_error("Current player is not the same for different states.");
    }
    auto& move = batch_move[state_idx];
    auto& observation = observations_[state.CurPlayer()][state_idx];
    state.ApplyMove(move);
    // put encoded observation into batch observation
    encoded_observation = observation_encoder_.Encode(observation);
    auto vec_observ_iter = batch_observation.observation.begin() + state_idx * encoded_observation.size();
    vec_observ_iter = std::copy(encoded_observation.begin(), encoded_observation.end(), vec_observ_iter);

    // gather legal moves
    auto& lms = batch_observation.legal_moves[state_idx];
    for (const auto& lm : state.LegalMoves(cur_player)) {
      lms.push_back(game_.GetMoveUid(lm));
    }
    batch_observation.reward[state_idx] = state.Score();
    batch_observation.done[state_idx] = state.IsTerminal();
  }
  if (reset_state_on_game_end_) ResetFinishedStates();
  return batch_observation;
}

hanabi_learning_env::HanabiParallelGame::HanabiBatchObservation
hanabi_learning_env::HanabiParallelGame::ObservePlayer(const int player_idx) {
  HanabiBatchObservation batch_observation(
      {static_cast<int>(parallel_states_.size()), observation_encoder_.Shape()[0]});
  const auto& player_observations = observations_[player_idx];
  for (size_t state_idx = 0; state_idx < parallel_states_.size(); ++state_idx) {
    auto& observation = player_observations[state_idx];
    auto encoded_observation = observation_encoder_.Encode(observation);
    auto vec_observ_iter = batch_observation.observation.begin() + state_idx * encoded_observation.size();
    vec_observ_iter = std::copy(encoded_observation.begin(), encoded_observation.end(), vec_observ_iter);
  }
  return batch_observation;
}


hanabi_learning_env::HanabiParallelGame::HanabiBatchObservation
hanabi_learning_env::HanabiParallelGame::Step(const std::vector<int>& batch_move) {
  std::vector<HanabiMove> moves;
  std::transform(batch_move.begin(), batch_move.end(), moves.begin(), [this](const auto& a){return game_.GetMove(a);});
  return Step(moves);
}
