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
"""The standard DQN replay memory.

This implementation is an out-of-graph replay memory + in-graph wrapper. It
supports vanilla n-step updates of the form typically found in the literature,
i.e. where rewards are accumulated for n steps and the intermediate trajectory
is not exposed to the agent. This does not allow, for example, performing
off-policy corrections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import math
import os
import pickle

import gin.tf
import numpy as np
import tensorflow as tf


# This constant determines how many iterations a checkpoint is kept for.
CHECKPOINT_DURATION = 4
MAX_SAMPLE_ATTEMPTS = 1000000


def invalid_range(cursor, replay_capacity, stack_size):
  """Returns an array with all the indices invalidated by cursor.

  It handles special cases in a circular buffer in the beginning and the end.

  Args:
    cursor: int, The position of the cursor.
    replay_capacity: int, The size of the replay memory.
    stack_size: int, The size of the stacks returned by the replay memory.

  Returns:
    `np.array` of size stack_size with the invalid indices.
  """
  assert cursor < replay_capacity
  return np.array(
      [(cursor - 1 + i) % replay_capacity for i in range(stack_size)])


class OutOfGraphReplayMemory(object):
  """A simple out of graph replay memory.

  Stores transitions (i.e. state, action, reward, next_state, terminal)
  efficiently when the states consist of stacks. The writing behaves like
  a FIFO buffer and the sampling is uniformly random.

  Attributes:
    add_count:  counter of how many transitions have been added.
    observations: `np.array`, circular buffer of observations.
    actions: `np.array`, circular buffer of actions.
    rewards: `np.array`, circular buffer of rewards.
    terminals: `np.array`, circular buffer of terminals.
    legal_actions: `np.array`, circular buffer of legal actions for hanabi.
    invalid_range: `np.array`, currently invalid indices.
  """

  def __init__(self, num_actions, observation_size, stack_size, replay_capacity,
               batch_size, update_horizon=1, gamma=1.0):
    """Data structure doing the heavy lifting.

    Args:
      num_actions: int, number of possible actions.
      observation_size: int, size of an input frame.
      stack_size: int, number of frames to use in state stack.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int, batch size.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: float, the discount factor.
    """
    self._observation_size = observation_size
    self._num_actions = num_actions
    self._replay_capacity = replay_capacity
    self._batch_size = batch_size
    self._stack_size = stack_size
    self._update_horizon = update_horizon
    self._gamma = gamma

    # When the horizon is > 1, we compute the sum of discounted rewards as a dot
    # product using the precomputed vector <gamma^0, gamma^1, ..., gamma^{n-1}>.
    self._cumulative_discount_vector = np.array(
        [math.pow(self._gamma, n) for n in range(update_horizon)],
        dtype=np.float32)

    # Create numpy arrays used to store sampled transitions.
    self.observations = np.empty(
        (replay_capacity, observation_size), dtype=np.uint8)
    self.actions = np.empty((replay_capacity), dtype=np.int32)
    self.rewards = np.empty((replay_capacity), dtype=np.float32)
    self.terminals = np.empty((replay_capacity), dtype=np.uint8)
    self.legal_actions = np.empty((replay_capacity, num_actions),
                                  dtype=np.float32)
    self.reset_state_batch_arrays(batch_size)
    self.add_count = np.array(0)

    self.invalid_range = np.zeros((self._stack_size))

  def add(self, observation, action, reward, terminal, legal_actions):
    """Adds a transition to the replay memory.

    Since the next_observation in the transition will be the observation added
    next there is no need to pass it.

    If the replay memory is at capacity the oldest transition will be discarded.

    Args:
      observation: `np.array` uint8, (observation_size).
      action: uint8, indicating the action in the transition.
      reward: float, indicating the reward received in the transition.
      terminal: uint8, acting as a boolean indicating whether the transition
                 was terminal (1) or not (0).
      legal_actions: Binary vector indicating legal actions (1 == legal).
    """
    if self.is_empty() or self.terminals[self.cursor() - 1] == 1:
      dummy_observation = np.zeros((self._observation_size))
      dummy_legal_actions = np.zeros((self._num_actions))
      for _ in range(self._stack_size - 1):
        self._add(dummy_observation, 0, 0, 0, dummy_legal_actions)
    self._add(observation, action, reward, terminal, legal_actions)

  def _add(self, observation, action, reward, terminal, legal_actions):
    cursor = self.cursor()
    self.observations[cursor] = observation
    self.actions[cursor] = action
    self.rewards[cursor] = reward
    self.terminals[cursor] = terminal
    self.legal_actions[cursor] = legal_actions
    self.add_count += 1
    self.invalid_range = invalid_range(self.cursor(), self._replay_capacity,
                                       self._stack_size)

  def is_empty(self):
    """Is the replay memory empty?"""
    return self.add_count == 0

  def is_full(self):
    """Is the replay memory full?"""
    return self.add_count >= self._replay_capacity

  def cursor(self):
    """Index to the location where the next transition will be written."""
    return self.add_count % self._replay_capacity

  def get_stack(self, array, index):
    """Returns the stack of array at the index.

    Args:
      array: `np.array`, to get the stack from.
      index: int, index to the first terminal in the stack to be returned.
    Returns:
      `Tensor` with shape (stack_size)
    """
    assert index >= 0
    assert index < self._replay_capacity
    if not self.is_full():
      assert index < self.cursor(), 'Index %i has not been added.' % index
      assert index >= self._stack_size - 1, ('Not enough elements to sample '
                                             'index %i' % index)
    # Fast slice read.
    if index >= self._stack_size - 1:
      stack = array[(index - self._stack_size + 1):(index + 1), ...]
    # Slow list read.
    else:
      indices = [(index - i) % self._replay_capacity
                 for i in reversed(range(self._stack_size))]
      stack = array[indices, ...]
    return stack

  def get_observation_stack(self, index):
    state = self.get_stack(self.observations, index)
    return np.transpose(state, [1, 0])

  def get_terminal_stack(self, index):
    return self.get_stack(self.terminals, index)

  def is_valid_transition(self, index):
    """Checks if the index contains a valid transition.

    The index range needs to be valid and it must not collide with the end of an
      episode.

    Args:
      index: int, index to the state in the transition. Note that next_state
        must also be valid.

    Returns:
      bool, True if transition is valid.

    """
    # Range checks
    if index < 0 or index >= self._replay_capacity:
      return False
    if not self.is_full():
      # The indices and next_indices must be smaller than the cursor.
      if index >= self.cursor() - self._update_horizon:
        return False
      # The first few indices contain the padding states of the first episode.
      if index < self._stack_size - 1:
        return False

    # Skip transitions that straddle the cursor.
    if index in set(self.invalid_range):
      return False

    # If there are terminal flags in any other frame other than the last one
    # the stack is not valid, so don't sample it.
    if self.get_terminal_stack(index)[:-1].any():
      return False

    return True

  def reset_state_batch_arrays(self, batch_size):
    self._next_state_batch = np.empty(
        (batch_size, self._observation_size, self._stack_size), dtype=np.uint8)
    self._state_batch = np.empty(
        (batch_size, self._observation_size, self._stack_size), dtype=np.uint8)

  def sample_index_batch(self, batch_size):
    """Returns a batch of valid indices.

    Args:
      batch_size: int, number of indices returned.

    Returns:
      list of batch_size, containing valid indices.

    Raises:
      Exception: If the batch was not constructed after maximum number of tries.
    """
    indices = []
    attempt_count = 0
    while len(indices) < batch_size and attempt_count < MAX_SAMPLE_ATTEMPTS:
      attempt_count += 1
      # index references the state and index + 1 points to next_state
      if self.is_full():
        index = np.random.randint(0, self._replay_capacity)
      else:
        # Can't start at 0 because the buffer is not yet circular
        index = np.random.randint(self._stack_size - 1, self.cursor() - 1)
      if self.is_valid_transition(index):
        indices.append(index)
    if len(indices) != batch_size:
      raise Exception('I tried %i times but only sampled %i valid transitions' %
                      (MAX_SAMPLE_ATTEMPTS, len(indices)))
    return indices

  def sample_transition_batch(self, batch_size=None, indices=None):
    """Returns a batch of transitions.

    Args:
      batch_size: int, number of transitions returned. If None the batch_size
        defined at init will be used.
      indices: list of ints. If not None, use the given indices instead of
        sampling them.

    Returns:
      Minibatch of transitions:  A tuple with elements state_batch,
        action_batch, reward_batch, next_state_batch and terminal_batch. The
        shape of state_batch and next_state_batch is (minibatch_size,
        observation_size, stack_size) and the rest of tensors have
        shape (minibatch_size)
    """
    if batch_size is None:
      batch_size = self._batch_size
    if batch_size != self._state_batch.shape[0]:
      self.reset_state_batch_arrays(batch_size)
    if not self.is_full():
      assert self.add_count >= batch_size, (
          'There is not enough to sample.'
          ' You need to call add at '
          'least %i (batch_size) times (currently it is %i)' % (batch_size,
                                                                self.add_count))
    if indices is None:
      indices = self.sample_index_batch(batch_size)
    assert len(indices) == batch_size

    action_batch = self.actions[indices]
    reward_batch = np.empty((batch_size), dtype=np.float32)
    terminal_batch = np.empty((batch_size), dtype=np.uint8)
    indices_batch = np.empty((batch_size), dtype=np.int32)
    next_legal_actions_batch = np.empty((batch_size, self._num_actions),
                                        dtype=np.float32)

    for batch_element, memory_index in enumerate(indices):
      indices_batch[batch_element] = memory_index

      self._state_batch[batch_element] = self.get_observation_stack(
          memory_index)

      # Compute indices in the replay memory up to n steps ahead.
      trajectory_indices = [(memory_index + j) % self._replay_capacity for
                            j in range(self._update_horizon)]

      # Determine if this trajectory segment contains a terminal state.
      terminals_in_trajectory = np.nonzero(self.terminals[trajectory_indices])
      if terminals_in_trajectory[0].size == 0:
        # If not, sum rewards along the trajectory, property discounted.
        terminal_batch[batch_element] = 0
        reward_batch[batch_element] = self._cumulative_discount_vector.dot(
            self.rewards[trajectory_indices])
      else:
        # Updates leading to a terminal state require a little more care, in
        # particular to avoid summing rewards past the end of the episode.
        terminal_batch[batch_element] = 1
        # Fetch smallest index corresponding to a terminal state.
        terminal_index = np.min(terminals_in_trajectory)

        truncated_discount_vector = (
            self._cumulative_discount_vector[0:terminal_index + 1])
        reward_batch[batch_element] = truncated_discount_vector.dot(
            self.rewards[trajectory_indices[0:terminal_index + 1]])

      bootstrap_state_index = (
          (memory_index + self._update_horizon) % self._replay_capacity)
      self._next_state_batch[batch_element] = (
          self.get_observation_stack(bootstrap_state_index))
      next_legal_actions_batch[batch_element] = (
          self.legal_actions[bootstrap_state_index])

    return (self._state_batch, action_batch, reward_batch,
            self._next_state_batch, terminal_batch, indices_batch,
            next_legal_actions_batch)

  def _generate_filename(self, checkpoint_dir, name, suffix):
    return os.path.join(checkpoint_dir, '{}_ckpt.{}.gz'.format(name, suffix))

  def save(self, checkpoint_dir, iteration_number):
    """Save the python replay memory attributes into a file.

    This method will save all the replay memory's state in a single file.

    Args:
      checkpoint_dir: str, directory where numpy checkpoint files should be
        saved.
      iteration_number: int, iteration_number to use as a suffix in naming numpy
        checkpoint files.
    """
    if not tf.gfile.Exists(checkpoint_dir):
      return
    for attr in self.__dict__:
      if not attr.startswith('_'):
        filename = self._generate_filename(checkpoint_dir, attr,
                                           iteration_number)
        with tf.gfile.Open(filename, 'wb') as f:
          with gzip.GzipFile(fileobj=f) as outfile:
            # Checkpoint numpy arrays directly with np.save to avoid excessive
            # memory usage. This is particularly important for the observations
            # data.
            if isinstance(self.__dict__[attr], np.ndarray):
              np.save(outfile, self.__dict__[attr], allow_pickle=False)
            else:
              pickle.dump(self.__dict__[attr], outfile)

      # After writing a checkpoint file, we garbage collect the checkpoint file
      # that is four versions old.
      stale_iteration_number = iteration_number - CHECKPOINT_DURATION
      if stale_iteration_number >= 0:
        stale_filename = self._generate_filename(checkpoint_dir, attr,
                                                 stale_iteration_number)
        try:
          tf.gfile.Remove(stale_filename)
        except tf.errors.NotFoundError:
          pass

  def load(self, checkpoint_dir, suffix):
    """Restores the object from bundle_dictionary and numpy checkpoints.

    Args:
      checkpoint_dir: str, directory where to read the numpy checkpointed files
        from.
      suffix: str, suffix to use in numpy checkpoint files.

    Raises:
      NotFoundError: if all expected files are not found in directory.
    """
    # We will first make sure we have all the necessary files available to avoid
    # loading a partially-specified (i.e. corrupted) replay buffer.
    for attr in self.__dict__:
      if attr.startswith('_'):
        continue
      filename = self._generate_filename(checkpoint_dir, attr, suffix)
      if not tf.gfile.Exists(filename):
        raise tf.errors.NotFoundError(None, None,
                                      'Missing file: {}'.format(filename))
    # If we've reached this point then we have verified that all expected files
    # are available.
    for attr in self.__dict__:
      if attr.startswith('_'):
        continue
      filename = self._generate_filename(checkpoint_dir, attr, suffix)
      with tf.gfile.Open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as infile:
          if isinstance(self.__dict__[attr], np.ndarray):
            self.__dict__[attr] = np.load(infile, allow_pickle=False)
          else:
            self.__dict__[attr] = pickle.load(infile)


@gin.configurable(blacklist=['observation_size', 'stack_size'])
class WrappedReplayMemory(object):
  """In-graph wrapper for the python replay memory.

  Usage:
    To add a transition:  run the operation add_transition_op
                          (and feed all the placeholders in add_transition_ph).

    To sample a batch:    Construct operations that depend on any of the
                          sampling tensors. Every sess.run using any of these
                          tensors will sample a new transition.

    When using staging:   Need to prefetch the next batch with each train_op by
                          calling self.prefetch_batch.

                          Everytime this op is called a new transition batch
                          would be prefetched.

  Attributes:
    The following tensors are sampled randomly each sess.run:
      states actions rewards next_states terminals
    add_transition_op:    tf operation to add a transition to the replay memory.
    All the following placeholders need to be fed. add_obs_ph add_action_ph
    add_reward_ph add_terminal_ph
  """

  def __init__(self,
               num_actions,
               observation_size,
               stack_size,
               use_staging=True,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=1.0,
               wrapped_memory=None):
    """Initializes a graph wrapper for the python replay memory.

    Args:
      num_actions: int, number of possible actions.
      observation_size: int, size of an input frame.
      stack_size: int, number of frames to use in state stack.
      use_staging: bool, when True it would use a staging area to prefetch the
        next sampling batch.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      wrapped_memory: The 'inner' memory data structure. Defaults to None, which
        creates the standard DQN replay memory.

    Raises:
      ValueError: If update_horizon is not positive.
      ValueError: If discount factor is not in [0, 1].
    """
    if replay_capacity < update_horizon + 1:
      raise ValueError('Update horizon (%i) should be significantly smaller '
                       'than replay capacity (%i).'
                       % (update_horizon, replay_capacity))
    if not update_horizon >= 1:
      raise ValueError('Update horizon must be positive.')
    if not 0.0 <= gamma <= 1.0:
      raise ValueError('Discount factor (gamma) must be in [0, 1].')

    # Allow subclasses to create self.memory.
    if wrapped_memory is not None:
      self.memory = wrapped_memory
    else:
      self.memory = OutOfGraphReplayMemory(
          num_actions, observation_size, stack_size,
          replay_capacity, batch_size, update_horizon, gamma)

    with tf.name_scope('replay'):
      with tf.name_scope('add_placeholders'):
        self.add_obs_ph = tf.placeholder(
            tf.uint8, [observation_size], name='add_obs_ph')
        self.add_action_ph = tf.placeholder(tf.int32, [], name='add_action_ph')
        self.add_reward_ph = tf.placeholder(
            tf.float32, [], name='add_reward_ph')
        self.add_terminal_ph = tf.placeholder(
            tf.uint8, [], name='add_terminal_ph')
        self.add_legal_actions_ph = tf.placeholder(
            tf.float32, [num_actions], name='add_legal_actions_ph')

      add_transition_ph = [
          self.add_obs_ph, self.add_action_ph, self.add_reward_ph,
          self.add_terminal_ph, self.add_legal_actions_ph
      ]

      with tf.device('/cpu:*'):
        self.add_transition_op = tf.py_func(
            self.memory.add, add_transition_ph, [], name='replay_add_py_func')

        self.transition = tf.py_func(
            self.memory.sample_transition_batch, [],
            [tf.uint8, tf.int32, tf.float32, tf.uint8, tf.uint8, tf.int32,
             tf.float32],
            name='replay_sample_py_func')

        if use_staging:
          # To hide the py_func latency use a staging area to pre-fetch the next
          # batch of transitions.
          (states, actions, rewards, next_states,
           terminals, indices, next_legal_actions) = self.transition
          # StagingArea requires all the shapes to be defined.
          states.set_shape([batch_size, observation_size, stack_size])
          actions.set_shape([batch_size])
          rewards.set_shape([batch_size])
          next_states.set_shape(
              [batch_size, observation_size, stack_size])
          terminals.set_shape([batch_size])
          indices.set_shape([batch_size])
          next_legal_actions.set_shape([batch_size, num_actions])

          # Create the staging area in CPU.
          prefetch_area = tf.contrib.staging.StagingArea(
              [tf.uint8, tf.int32, tf.float32, tf.uint8, tf.uint8, tf.int32,
               tf.float32])

          self.prefetch_batch = prefetch_area.put(
              (states, actions, rewards, next_states, terminals, indices,
               next_legal_actions))
        else:
          self.prefetch_batch = tf.no_op()

      if use_staging:
        # Get the sample_transition_batch in GPU. This would do the copy from
        # CPU to GPU.
        self.transition = prefetch_area.get()

      (self.states, self.actions, self.rewards, self.next_states,
       self.terminals, self.indices, self.next_legal_actions) = self.transition

      # Since these are py_func tensors, no information about their shape is
      # present. Setting the shape only for the necessary tensors
      self.states.set_shape([None, observation_size, stack_size])
      self.next_states.set_shape([None, observation_size, stack_size])

  def save(self, checkpoint_dir, iteration_number):
    """Save the underlying replay memory's contents in a file.

    Args:
      checkpoint_dir: str, directory from where to read the numpy checkpointed
        files.
      iteration_number: int, iteration_number to use as a suffix in naming
        numpy checkpoint files.
    """
    self.memory.save(checkpoint_dir, iteration_number)

  def load(self, checkpoint_dir, suffix):
    """Loads the replay memory's state from a saved file.

    Args:
      checkpoint_dir: str, directory from where to read the numpy checkpointed
        files.
      suffix: str, suffix to use in numpy checkpoint files.
    """
    self.memory.load(checkpoint_dir, suffix)
