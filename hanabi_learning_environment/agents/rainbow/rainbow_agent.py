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
"""Implementation of a Rainbow agent adapted to the multiplayer setting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import dqn_agent
import gin.tf
import numpy as np
import prioritized_replay_memory
import tensorflow as tf


slim = tf.contrib.slim


@gin.configurable
def rainbow_template(state,
                     num_actions,
                     num_atoms=51,
                     layer_size=512,
                     num_layers=1):
  r"""Builds a Rainbow Network mapping states to value distributions.

  Args:
    state: A `tf.placeholder` for the RL state.
    num_actions: int, number of actions that the RL agent can take.
    num_atoms: int, number of atoms to approximate the distribution with.
    layer_size: int, number of hidden units per layer.
    num_layers: int, number of hidden layers.

  Returns:
    net: A `tf.Graphdef` for Rainbow:
      `\theta : \mathcal{X}\rightarrow\mathbb{R}^{|\mathcal{A}| \times N}`,
      where `N` is num_atoms.
  """
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  net = tf.cast(state, tf.float32)
  net = tf.squeeze(net, axis=2)

  for _ in range(num_layers):
    net = slim.fully_connected(net, layer_size,
                               activation_fn=tf.nn.relu)
  net = slim.fully_connected(net, num_actions * num_atoms, activation_fn=None,
                             weights_initializer=weights_initializer)
  net = tf.reshape(net, [-1, num_actions, num_atoms])
  return net


@gin.configurable
class RainbowAgent(dqn_agent.DQNAgent):
  """A compact implementation of the multiplayer Rainbow agent."""

  @gin.configurable
  def __init__(self,
               num_actions=None,
               observation_size=None,
               num_players=None,
               num_atoms=51,
               vmax=25.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=500,
               update_period=4,
               target_update_period=500,
               epsilon_train=0.0,
               epsilon_eval=0.0,
               epsilon_decay_period=1000,
               learning_rate=0.000025,
               optimizer_epsilon=0.00003125,
               tf_device='/cpu:*'):
    """Initializes the agent and constructs its graph.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_size: int, size of observation vector.
      num_players: int, number of players playing this game.
      num_atoms: Int, the number of buckets for the value function distribution.
      vmax: float, maximum return predicted by a value distribution.
      gamma: float, discount factor as commonly used in the RL literature.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of stored transitions before training.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_train: float, final epsilon for training.
      epsilon_eval: float, epsilon during evaluation.
      epsilon_decay_period: int, number of steps for epsilon to decay.
      learning_rate: float, learning rate for the optimizer.
      optimizer_epsilon: float, epsilon for Adam optimizer.
      tf_device: str, Tensorflow device on which to run computations.
    """
    # We need this because some tools convert round floats into ints.
    vmax = float(vmax)
    self.num_atoms = num_atoms
    # Using -vmax as the minimum return is is wasteful, because all rewards are
    # positive -- but does not unduly affect performance.
    self.support = tf.linspace(-vmax, vmax, num_atoms)
    self.learning_rate = learning_rate
    self.optimizer_epsilon = optimizer_epsilon

    graph_template = functools.partial(rainbow_template, num_atoms=num_atoms)
    super(RainbowAgent, self).__init__(
        num_actions=num_actions,
        observation_size=observation_size,
        num_players=num_players,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        graph_template=graph_template,
        tf_device=tf_device)
    tf.logging.info('\t learning_rate: %f', learning_rate)
    tf.logging.info('\t optimizer_epsilon: %f', optimizer_epsilon)

  def _build_replay_memory(self, use_staging):
    """Creates the replay memory used by the agent.

    Rainbow uses prioritized replay.

    Args:
      use_staging: bool, whether to use a staging area in the replay memory.

    Returns:
      A replay memory object.
    """
    return prioritized_replay_memory.WrappedPrioritizedReplayMemory(
        num_actions=self.num_actions,
        observation_size=self.observation_size,
        stack_size=1,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma)

  def _reshape_networks(self):
    # self._q is actually logits now, rename things.
    # size of _logits: 1 x num_actions x num_atoms
    self._logits = self._q
    # size of _probabilities: 1 x num_actions x num_atoms
    self._probabilities = tf.contrib.layers.softmax(self._q)
    # size of _q: 1 x num_actions
    self._q = tf.reduce_sum(self.support * self._probabilities, axis=2)
    # Recompute argmax from q values. Ignore illegal actions.
    self._q_argmax = tf.argmax(self._q + self.legal_actions_ph, axis=1)[0]

    # size of _replay_logits: 1 x num_actions x num_atoms
    self._replay_logits = self._replay_qs
    # size of _replay_next_logits: 1 x num_actions x num_atoms
    self._replay_next_logits = self._replay_next_qt
    del self._replay_qs
    del self._replay_next_qt

  def _build_target_distribution(self):
    self._reshape_networks()
    batch_size = tf.shape(self._replay.rewards)[0]
    # size of rewards: batch_size x 1
    rewards = self._replay.rewards[:, None]
    # size of tiled_support: batch_size x num_atoms
    tiled_support = tf.tile(self.support, [batch_size])
    tiled_support = tf.reshape(tiled_support, [batch_size, self.num_atoms])
    # size of target_support: batch_size x num_atoms

    is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: batch_size x 1
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = gamma_with_terminal[:, None]

    target_support = rewards + gamma_with_terminal * tiled_support
    # size of next_probabilities: batch_size  x num_actions x num_atoms
    next_probabilities = tf.contrib.layers.softmax(
        self._replay_next_logits)

    # size of next_qt: 1 x num_actions
    next_qt = tf.reduce_sum(self.support * next_probabilities, 2)
    # size of next_qt_argmax: 1 x batch_size
    next_qt_argmax = tf.argmax(
        next_qt + self._replay.next_legal_actions, axis=1)[:, None]
    batch_indices = tf.range(tf.to_int64(batch_size))[:, None]
    # size of next_qt_argmax: batch_size x 2
    next_qt_argmax = tf.concat([batch_indices, next_qt_argmax], axis=1)
    # size of next_probabilities: batch_size x num_atoms
    next_probabilities = tf.gather_nd(next_probabilities, next_qt_argmax)
    return project_distribution(target_support, next_probabilities,
                                self.support)

  def _build_train_op(self):
    """Builds the training op for Rainbow.

    Returns:
      train_op: An op performing one step of training.
    """
    target_distribution = tf.stop_gradient(self._build_target_distribution())

    # size of indices: batch_size x 1.
    indices = tf.range(tf.shape(self._replay_logits)[0])[:, None]
    # size of reshaped_actions: batch_size x 2.
    reshaped_actions = tf.concat([indices, self._replay.actions[:, None]], 1)
    # For each element of the batch, fetch the logits for its selected action.
    chosen_action_logits = tf.gather_nd(self._replay_logits, reshaped_actions)

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_distribution,
        logits=chosen_action_logits)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=self.learning_rate,
        epsilon=self.optimizer_epsilon)

    update_priorities_op = self._replay.tf_set_priority(
        self._replay.indices, tf.sqrt(loss + 1e-10))

    target_priorities = self._replay.tf_get_priority(self._replay.indices)
    target_priorities = tf.math.add(target_priorities, 1e-10)
    target_priorities = 1.0 / tf.sqrt(target_priorities)
    target_priorities /= tf.reduce_max(target_priorities)

    weighted_loss = target_priorities * loss

    with tf.control_dependencies([update_priorities_op]):
      return optimizer.minimize(tf.reduce_mean(weighted_loss)), weighted_loss


def project_distribution(supports, weights, target_support,
                         validate_args=False):
  """Projects a batch of (support, weights) onto target_support.

  Based on equation (7) in (Bellemare et al., 2017):
    https://arxiv.org/abs/1707.06887
  In the rest of the comments we will refer to this equation simply as Eq7.

  This code is not easy to digest, so we will use a running example to clarify
  what is going on, with the following sample inputs:
    * supports =       [[0, 2, 4, 6, 8],
                        [1, 3, 4, 5, 6]]
    * weights =        [[0.1, 0.6, 0.1, 0.1, 0.1],
                        [0.1, 0.2, 0.5, 0.1, 0.1]]
    * target_support = [4, 5, 6, 7, 8]
  In the code below, comments preceded with 'Ex:' will be referencing the above
  values.

  Args:
    supports: Tensor of shape (batch_size, num_dims) defining supports for the
      distribution.
    weights: Tensor of shape (batch_size, num_dims) defining weights on the
      original support points. Although for the CategoricalDQN agent these
      weights are probabilities, it is not required that they are.
    target_support: Tensor of shape (num_dims) defining support of the projected
      distribution. The values must be monotonically increasing. Vmin and Vmax
      will be inferred from the first and last elements of this tensor,
      respectively. The values in this tensor must be equally spaced.
    validate_args: Whether we will verify the contents of the
      target_support parameter.

  Returns:
    A Tensor of shape (batch_size, num_dims) with the projection of a batch of
    (support, weights) onto target_support.

  Raises:
    ValueError: If target_support has no dimensions, or if shapes of supports,
      weights, and target_support are incompatible.
  """
  target_support_deltas = target_support[1:] - target_support[:-1]
  # delta_z = `\Delta z` in Eq7.
  delta_z = target_support_deltas[0]
  validate_deps = []
  supports.shape.assert_is_compatible_with(weights.shape)
  supports[0].shape.assert_is_compatible_with(target_support.shape)
  target_support.shape.assert_has_rank(1)
  if validate_args:
    # Assert that supports and weights have the same shapes.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(tf.equal(tf.shape(supports), tf.shape(weights))),
            [supports, weights]))
    # Assert that elements of supports and target_support have the same shape.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(
                tf.equal(tf.shape(supports)[1], tf.shape(target_support))),
            [supports, target_support]))
    # Assert that target_support has a single dimension.
    validate_deps.append(
        tf.Assert(
            tf.equal(tf.size(tf.shape(target_support)), 1), [target_support]))
    # Assert that the target_support is monotonically increasing.
    validate_deps.append(
        tf.Assert(tf.reduce_all(target_support_deltas > 0), [target_support]))
    # Assert that the values in target_support are equally spaced.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(tf.equal(target_support_deltas, delta_z)),
            [target_support]))

  with tf.control_dependencies(validate_deps):
    # Ex: `v_min, v_max = 4, 8`.
    v_min, v_max = target_support[0], target_support[-1]
    # Ex: `batch_size = 2`.
    batch_size = tf.shape(supports)[0]
    # `N` in Eq7.
    # Ex: `num_dims = 5`.
    num_dims = tf.shape(target_support)[0]
    # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
    # Ex: `clipped_support = [[[ 4.  4.  4.  6.  8.]]
    #                         [[ 4.  4.  4.  5.  6.]]]`.
    clipped_support = tf.clip_by_value(supports, v_min, v_max)[:, None, :]
    # Ex: `tiled_support = [[[[ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]]
    #                        [[ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]]]]`.
    tiled_support = tf.tile([clipped_support], [1, 1, num_dims, 1])
    # Ex: `reshaped_target_support = [[[ 4.]
    #                                  [ 5.]
    #                                  [ 6.]
    #                                  [ 7.]
    #                                  [ 8.]]
    #                                 [[ 4.]
    #                                  [ 5.]
    #                                  [ 6.]
    #                                  [ 7.]
    #                                  [ 8.]]]`.
    reshaped_target_support = tf.tile(target_support[:, None], [batch_size, 1])
    reshaped_target_support = tf.reshape(reshaped_target_support,
                                         [batch_size, num_dims, 1])
    # numerator = `|clipped_support - z_i|` in Eq7.
    # Ex: `numerator = [[[[ 0.  0.  0.  2.  4.]
    #                     [ 1.  1.  1.  1.  3.]
    #                     [ 2.  2.  2.  0.  2.]
    #                     [ 3.  3.  3.  1.  1.]
    #                     [ 4.  4.  4.  2.  0.]]
    #                    [[ 0.  0.  0.  1.  2.]
    #                     [ 1.  1.  1.  0.  1.]
    #                     [ 2.  2.  2.  1.  0.]
    #                     [ 3.  3.  3.  2.  1.]
    #                     [ 4.  4.  4.  3.  2.]]]]`.
    numerator = tf.abs(tiled_support - reshaped_target_support)
    quotient = 1 - (numerator / delta_z)
    # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
    # Ex: `clipped_quotient = [[[[ 1.  1.  1.  0.  0.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  1.  0.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  0.  1.]]
    #                           [[ 1.  1.  1.  0.  0.]
    #                            [ 0.  0.  0.  1.  0.]
    #                            [ 0.  0.  0.  0.  1.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  0.  0.]]]]`.
    clipped_quotient = tf.clip_by_value(quotient, 0, 1)
    # Ex: `weights = [[ 0.1  0.6  0.1  0.1  0.1]
    #                 [ 0.1  0.2  0.5  0.1  0.1]]`.
    weights = weights[:, None, :]
    # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))`
    # in Eq7.
    # Ex: `inner_prod = [[[[ 0.1  0.6  0.1  0.  0. ]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.1 0. ]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.  0.1]]
    #                     [[ 0.1  0.2  0.5  0.  0. ]
    #                      [ 0.   0.   0.   0.1 0. ]
    #                      [ 0.   0.   0.   0.  0.1]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.  0. ]]]]`.
    inner_prod = clipped_quotient * weights
    # Ex: `projection = [[ 0.8 0.0 0.1 0.0 0.1]
    #                    [ 0.8 0.1 0.1 0.0 0.0]]`.
    projection = tf.reduce_sum(inner_prod, 3)
    projection = tf.reshape(projection, [batch_size, num_dims])
    return projection
