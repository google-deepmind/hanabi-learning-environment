# coding=utf-8
# Copyright 2018 The Dopamine Authors and Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# This file is a fork of the original Dopamine code incorporating changes for
# the multiplayer setting and the Hanabi Learning Environment.
#
"""An implementation of Prioritized Experience Replay (PER).

This implementation is based on the paper "Prioritized Experience Replay"
by Tom Schaul et al. (2015). Many thanks to Tom Schaul, John Quan, and Matteo
Hessel for providing useful pointers on the algorithm and its implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from third_party.dopamine import sum_tree
import gin.tf
import numpy as np
import replay_memory
import tensorflow as tf

DEFAULT_PRIORITY = 100.0


class OutOfGraphPrioritizedReplayMemory(replay_memory.OutOfGraphReplayMemory):
  """An Out of Graph Replay Memory for Prioritized Experience Replay.

  See replay_memory.py for details.
  """

  def __init__(self, num_actions, observation_size, stack_size, replay_capacity,
               batch_size, update_horizon=1, gamma=1.0):
    """This data structure does the heavy lifting in the replay memory.

    Args:
      num_actions: int, number of actions.
      observation_size: int, size of an input observation.
      stack_size: int, number of frames to use in state stack.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int, batch size.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
    """
    super(OutOfGraphPrioritizedReplayMemory, self).__init__(
        num_actions=num_actions,
        observation_size=observation_size, stack_size=stack_size,
        replay_capacity=replay_capacity, batch_size=batch_size,
        update_horizon=update_horizon, gamma=gamma)

    self.sum_tree = sum_tree.SumTree(replay_capacity)

  def add(self, observation, action, reward, terminal, legal_actions):
    """Adds a transition to the replay memory.

    Since the next_observation in the transition will be the observation added
    next there is no need to pass it.

    If the replay memory is at capacity the oldest transition will be discarded.

    Compared to OutOfGraphReplayMemory.add(), this version also sets the
    priority of dummy frames to 0.

    Args:
      observation: `np.array` uint8, (observation_size, observation_size).
      action: int, indicating the action in the transition.
      reward: float, indicating the reward received in the transition.
      terminal: int, acting as a boolean indicating whether the transition
                 was terminal (1) or not (0).
      legal_actions: Binary vector indicating legal actions (1 == legal).
    """
    if self.is_empty() or self.terminals[self.cursor() - 1] == 1:
      dummy_observation = np.zeros((self._observation_size))
      dummy_legal_actions = np.zeros((self._num_actions))
      for _ in range(self._stack_size - 1):
        self._add(dummy_observation, 0, 0, 0, dummy_legal_actions, priority=0.0)

    self._add(observation, action, reward, terminal, legal_actions,
              priority=DEFAULT_PRIORITY)

  def _add(self, observation, action, reward, terminal, legal_actions,
           priority=DEFAULT_PRIORITY):
    new_element_index = self.cursor()

    super(OutOfGraphPrioritizedReplayMemory, self)._add(
        observation, action, reward, terminal, legal_actions)

    self.sum_tree.set(new_element_index, priority)

  def sample_index_batch(self, batch_size):
    """Returns a batch of valid indices.

    Args:
      batch_size: int, number of indices returned.

    Returns:
      List of size batch_size containing valid indices.

    Raises:
      Exception: If the batch was not constructed after maximum number of tries.
    """
    indices = []
    allowed_attempts = replay_memory.MAX_SAMPLE_ATTEMPTS

    while len(indices) < batch_size and allowed_attempts > 0:
      index = self.sum_tree.sample()

      if self.is_valid_transition(index):
        indices.append(index)
      else:
        allowed_attempts -= 1

    if len(indices) != batch_size:
      raise Exception('Could only sample {} valid transitions'.format(
          len(indices)))
    else:
      return indices

  def set_priority(self, indices, priorities):
    """Sets the priority of the given elements according to Schaul et al.

    Args:
      indices: `np.array` of indices in range [0, replay_capacity).
      priorities: list of floats, the corresponding priorities.
    """
    assert indices.dtype == np.int32, ('Indices must be integers, '
                                       'given: {}'.format(indices.dtype))
    for i, memory_index in enumerate(indices):
      self.sum_tree.set(memory_index, priorities[i])

  def get_priority(self, indices, batch_size=None):
    """Fetches the priorities correspond to a batch of memory indices.

    For any memory location not yet used, the corresponding priority is 0.

    Args:
      indices: `np.array` of indices in range [0, replay_capacity).
      batch_size: int, requested number of items.
    Returns:
      The corresponding priorities.
    """
    if batch_size is None:
      batch_size = self._batch_size
    if batch_size != self._state_batch.shape[0]:
      self.reset_state_batch_arrays(batch_size)

    priority_batch = np.empty((batch_size), dtype=np.float32)

    assert indices.dtype == np.int32, ('Indices must be integers, '
                                       'given: {}'.format(indices.dtype))
    for i, memory_index in enumerate(indices):
      priority_batch[i] = self.sum_tree.get(memory_index)

    return priority_batch


@gin.configurable(blacklist=['observation_size', 'stack_size'])
class WrappedPrioritizedReplayMemory(replay_memory.WrappedReplayMemory):
  """In graph wrapper for the python Replay Memory.

  Usage:
    To add a transition:  run the operation add_transition_op
                          (and feed all the placeholders in add_transition_ph)

    To sample a batch:    Construct operations that depend on any of the
                          sampling tensors. Every sess.run using any of these
                          tensors will sample a new transition.

    When using staging:   Need to prefetch the next batch with each train_op by
                          calling self.prefetch_batch.

                          Everytime this op is called a new transition batch
                          would be prefetched.

  Attributes:
    # The following tensors are sampled randomly each sess.run
    states
    actions
    rewards
    next_states
    terminals

    add_transition_op:    tf operation to add a transition to the replay
                          memory. All the following placeholders need to be fed.
    add_obs_ph
    add_action_ph
    add_reward_ph
    add_terminal_ph
  """

  def __init__(self,
               num_actions,
               observation_size,
               stack_size,
               use_staging=True,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=1.0):
    """Initializes a graph wrapper for the python Replay Memory.

    Args:
      num_actions: int, number of possible actions.
      observation_size: int, size of an input observation.
      stack_size: int, number of frames to use in state stack.
      use_staging: bool, when True it would use a staging area to prefetch
        the next sampling batch.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.

    Raises:
      ValueError: If update_horizon is not positive.
      ValueError: If discount factor is not in [0, 1].
    """
    memory = OutOfGraphPrioritizedReplayMemory(num_actions, observation_size,
                                               stack_size, replay_capacity,
                                               batch_size, update_horizon,
                                               gamma)
    super(WrappedPrioritizedReplayMemory, self).__init__(
        num_actions,
        observation_size, stack_size, use_staging, replay_capacity, batch_size,
        update_horizon, gamma, wrapped_memory=memory)

  def tf_set_priority(self, indices, losses):
    """Sets the priorities for the given indices.

    Args:
      indices: tensor of indices (int32), size k.
      losses: tensor of losses (float), size k.

    Returns:
       A TF op setting the priorities according to Prioritized Experience
       Replay.
    """
    return tf.py_func(
        self.memory.set_priority, [indices, losses],
        [],
        name='prioritized_replay_set_priority_py_func')

  def tf_get_priority(self, indices):
    """Gets the priorities for the given indices.

    Args:
      indices: tensor of indices (int32), size k.

    Returns:
       A tensor (float32) of priorities.
    """
    return tf.py_func(
        self.memory.get_priority, [indices],
        [tf.float32],
        name='prioritized_replay_get_priority_py_func')
