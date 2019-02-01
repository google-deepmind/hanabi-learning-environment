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
"""Implementation of a DQN agent adapted to the multiplayer setting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random

import gin.tf
import numpy as np
import replay_memory
import tensorflow as tf


slim = tf.contrib.slim

Transition = collections.namedtuple(
    'Transition', ['reward', 'observation', 'legal_actions', 'action', 'begin'])


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
  """Returns the current epsilon parameter for the agent's e-greedy policy.

  Args:
    decay_period: float, the decay period for epsilon.
    step: Integer, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before training starts.
    epsilon: float, the epsilon value.

  Returns:
    A float, the linearly decaying epsilon value.
  """
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = np.clip(bonus, 0.0, 1.0 - epsilon)
  return epsilon + bonus


def dqn_template(state, num_actions, layer_size=512, num_layers=1):
  r"""Builds a DQN Network mapping states to Q-values.

  Args:
    state: A `tf.placeholder` for the RL state.
    num_actions: int, number of actions that the RL agent can take.
    layer_size: int, number of hidden units per layer.
    num_layers: int, Number of hidden layers.

  Returns:
    net: A `tf.Graphdef` for DQN:
      `\theta : \mathcal{X}\rightarrow\mathbb{R}^{|\mathcal{A}|}`
  """
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  net = tf.cast(state, tf.float32)
  net = tf.squeeze(net, axis=2)
  for _ in range(num_layers):
    net = slim.fully_connected(net, layer_size,
                               activation_fn=tf.nn.relu)
  net = slim.fully_connected(net, num_actions, activation_fn=None,
                             weights_initializer=weights_initializer)
  return net


@gin.configurable
class DQNAgent(object):
  """A compact implementation of the multiplayer DQN agent."""

  @gin.configurable
  def __init__(self,
               num_actions=None,
               observation_size=None,
               num_players=None,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=500,
               update_period=4,
               stack_size=1,
               target_update_period=500,
               epsilon_fn=linearly_decaying_epsilon,
               epsilon_train=0.02,
               epsilon_eval=0.001,
               epsilon_decay_period=1000,
               graph_template=dqn_template,
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.RMSPropOptimizer(
                   learning_rate=.0025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=1e-6,
                   centered=True)):
    """Initializes the agent and constructs its graph.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_size: int, size of observation vector.
      num_players: int, number of players playing this game.
      gamma: float, discount factor as commonly used in the RL literature.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of stored transitions before training.
      update_period: int, period between DQN updates.
      stack_size: int, number of observations to use as state.
      target_update_period: Update period for the target network.
      epsilon_fn: Function expecting 4 parameters: (decay_period, step,
        warmup_steps, epsilon), and which returns the epsilon value used for
        exploration during training.
      epsilon_train: float, final epsilon for training.
      epsilon_eval: float, epsilon during evaluation.
      epsilon_decay_period: int, number of steps for epsilon to decay.
      graph_template: function for building the neural network graph.
      tf_device: str, Tensorflow device on which to run computations.
      use_staging: bool, when True use a staging area to prefetch the next
        sampling batch.
      optimizer: Optimizer instance used for learning.
    """

    tf.logging.info('Creating %s agent with the following parameters:',
                    self.__class__.__name__)
    tf.logging.info('\t gamma: %f', gamma)
    tf.logging.info('\t update_horizon: %f', update_horizon)
    tf.logging.info('\t min_replay_history: %d', min_replay_history)
    tf.logging.info('\t update_period: %d', update_period)
    tf.logging.info('\t target_update_period: %d', target_update_period)
    tf.logging.info('\t epsilon_train: %f', epsilon_train)
    tf.logging.info('\t epsilon_eval: %f', epsilon_eval)
    tf.logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    tf.logging.info('\t tf_device: %s', tf_device)
    tf.logging.info('\t use_staging: %s', use_staging)
    tf.logging.info('\t optimizer: %s', optimizer)

    # Global variables.
    self.num_actions = num_actions
    self.observation_size = observation_size
    self.num_players = num_players
    self.gamma = gamma
    self.update_horizon = update_horizon
    self.cumulative_gamma = math.pow(gamma, update_horizon)
    self.min_replay_history = min_replay_history
    self.target_update_period = target_update_period
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.update_period = update_period
    self.eval_mode = False
    self.training_steps = 0
    self.batch_staged = False
    self.optimizer = optimizer

    with tf.device(tf_device):
      # Calling online_convnet will generate a new graph as defined in
      # graph_template using whatever input is passed, but will always share
      # the same weights.
      online_convnet = tf.make_template('Online', graph_template)
      target_convnet = tf.make_template('Target', graph_template)
      # The state of the agent. The last axis is the number of past observations
      # that make up the state.
      states_shape = (1, observation_size, stack_size)
      self.state = np.zeros(states_shape)
      self.state_ph = tf.placeholder(tf.uint8, states_shape, name='state_ph')
      self.legal_actions_ph = tf.placeholder(tf.float32,
                                             [self.num_actions],
                                             name='legal_actions_ph')
      self._q = online_convnet(
          state=self.state_ph, num_actions=self.num_actions)
      self._replay = self._build_replay_memory(use_staging)
      self._replay_qs = online_convnet(self._replay.states, self.num_actions)
      self._replay_next_qt = target_convnet(self._replay.next_states,
                                            self.num_actions)
      self._train_op = self._build_train_op()
      self._sync_qt_ops = self._build_sync_op()

      self._q_argmax = tf.argmax(self._q + self.legal_actions_ph, axis=1)[0]

    # Set up a session and initialize variables.
    self._sess = tf.Session(
        '', config=tf.ConfigProto(allow_soft_placement=True))
    self._init_op = tf.global_variables_initializer()
    self._sess.run(self._init_op)

    self._saver = tf.train.Saver(max_to_keep=3)

    # This keeps tracks of the observed transitions during play, for each
    # player.
    self.transitions = [[] for _ in range(num_players)]

  def _build_replay_memory(self, use_staging):
    """Creates the replay memory used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area for replaying.

    Returns:
      A replay memory object.
    """
    return replay_memory.WrappedReplayMemory(
        num_actions=self.num_actions,
        observation_size=self.observation_size,
        batch_size=32,
        stack_size=1,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma)

  def _build_target_q_op(self):
    """Build an op to be used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the target Q-value.
    """
    # Get the max q_value across the actions dimension.
    replay_next_qt_max = tf.reduce_max(self._replay_next_qt +
                                       self._replay.next_legal_actions, 1)
    # Calculate the sample Bellman update.
    #   Q_t = R_t + \gamma^N * Q'_t+1
    # where,
    #   Q'_t+1 is \argmax_a Q(S_t+1, a)
    #          (or) 0 if S_t is a terminal state,
    # and
    #   N is the update horizon (by default, N=1).
    return self._replay.rewards + self.cumulative_gamma * replay_next_qt_max * (
        1. - tf.cast(self._replay.terminals, tf.float32))

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_qs * replay_action_one_hot,
        reduction_indices=1,
        name='replay_chosen_q')

    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    return self.optimizer.minimize(tf.reduce_mean(loss))

  def _build_sync_op(self):
    """Build ops for assigning weights from online to target network.

    Returns:
      ops: A list of ops assigning weights from online to target network.
    """
    # Get trainable variables from online and target networks.
    sync_qt_ops = []
    trainables_online = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='Online')
    trainables_target = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='Target')
    for (w_online, w_target) in zip(trainables_online, trainables_target):
      # Assign weights from online to target network.
      sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
    return sync_qt_ops

  def begin_episode(self, current_player, legal_actions, observation):
    """Returns the agent's first action.

    Args:
      current_player: int, the player whose turn it is.
      legal_actions: `np.array`, actions which the player can currently take.
      observation: `np.array`, the environment's initial observation.

    Returns:
      A legal, int-valued action.
    """
    self._train_step()

    self.action = self._select_action(observation, legal_actions)
    self._record_transition(current_player, 0, observation, legal_actions,
                            self.action, begin=True)
    return self.action

  def step(self, reward, current_player, legal_actions, observation):
    """Stores observations from last transition and chooses a new action.

    Notifies the agent of the outcome of the latest transition and stores it
      in the replay memory, selects a new action and applies a training step.

    Args:
      reward: float, the reward received from its action.
      current_player: int, the player whose turn it is.
      legal_actions: `np.array`, actions which the player can currently take.
      observation: `np.array`, the most recent observation.

    Returns:
      A legal, int-valued action.
    """
    self._train_step()

    self.action = self._select_action(observation, legal_actions)
    self._record_transition(current_player, reward, observation, legal_actions,
                            self.action)
    return self.action

  def end_episode(self, final_rewards):
    """Signals the end of the episode to the agent.

    Args:
      final_rewards: `np.array`, the last rewards from the environment. Each
        player gets their own reward, which is the sum of the rewards since
        their last move.
    """
    self._post_transitions(terminal_rewards=final_rewards)

  def _record_transition(self, current_player, reward, observation,
                         legal_actions, action, begin=False):
    """Records the most recent transition data.

    Specifically, the data consists of (r_t, o_{t+1}, l_{t+1}, a_{t+1}), where
      r_t is the most recent reward (since our last action),
      o_{t+1} is the following observation,
      l_{t+1} are the legal actions from the corresponding state,
      a_{t+1} is the chosen action from that state.

    Args:
      current_player: int, the player experiencing the transition.
      reward: float, the received reward.
      observation: `np.array`, the player's observation.
      legal_actions: `np.array`, legal actions from this state.
      action: int, the selected action.
      begin: bool, if True, this is the beginning of an episode.
    """
    self.transitions[current_player].append(
        Transition(reward, np.array(observation, dtype=np.uint8, copy=True),
                   np.array(legal_actions, dtype=np.float32, copy=True),
                   action, begin))

  def _post_transitions(self, terminal_rewards):
    """Posts this episode to the replay memory.

    Each player has their own episode, which is posted separately.

    Args:
      terminal_rewards: `np.array`,terminal rewards for each player.
    """
    # We store each player's episode consecutively in the replay memory.
    for player in range(self.num_players):
      num_transitions = len(self.transitions[player])

      for index, transition in enumerate(self.transitions[player]):
        # Add: o_t, l_t, a_t, r_{t+1}, term_{t+1}
        final_transition = index == num_transitions - 1
        if final_transition:
          reward = terminal_rewards[player]
        else:
          reward = self.transitions[player][index + 1].reward

        self._store_transition(transition.observation, transition.action,
                               reward, final_transition,
                               transition.legal_actions)

      # Now that this episode has been stored, drop it from the transitions
      # buffer.
      self.transitions[player] = []

  def _select_action(self, observation, legal_actions):
    """Select an action from the set of allowed actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    will otherwise choose greedily from the current q-value estimates.

    Args:
      observation: `np.array`, the current observation.
      legal_actions: `np.array`, describing legal actions, with -inf meaning
        not legal.

    Returns:
      action: int, a legal action.
    """
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                self.min_replay_history, self.epsilon_train)

    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      legal_action_indices = np.where(legal_actions == 0.0)
      return np.random.choice(legal_action_indices[0])
    else:
      # Convert observation into a batch-based format.
      self.state[0, :, 0] = observation

      # Choose the action maximizing the q function for the current state.
      action = self._sess.run(self._q_argmax,
                              {self.state_ph: self.state,
                               self.legal_actions_ph: legal_actions})
      assert legal_actions[action] == 0.0, 'Expected legal action.'
      return action

  def _train_step(self):
    """Runs a single training step.

    Runs a training op if both:
    (1) A minimum number of frames have been added to the replay buffer.
    (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online to target network if training steps is a
    multiple of target update period.
    """
    if self.eval_mode:
      return

    # Run a training op.
    if (self._replay.memory.add_count >= self.min_replay_history and
        not self.batch_staged):
      self._sess.run(self._replay.prefetch_batch)
      self.batch_staged = True
    if (self._replay.memory.add_count > self.min_replay_history and
        self.training_steps % self.update_period == 0):
      self._sess.run([self._train_op, self._replay.prefetch_batch])
    # Sync weights.
    if self.training_steps % self.target_update_period == 0:
      self._sess.run(self._sync_qt_ops)
    self.training_steps += 1

  def _store_transition(self, observation, action, reward, is_terminal,
                        legal_actions):
    """Stores a transition during training mode.

    Executes a tf session and executes replay memory ops in order to store the
    following tuple in the replay buffer (last_observation, action, reward,
    is_terminal).

    Args:
      observation: `np.array`, observation.
      action: int, the action taken.
      reward: float, the reward.
      is_terminal: bool, indicating if the current state is a terminal state.
      legal_actions: Legal actions from the current state.
    """
    if not self.eval_mode:
      self._sess.run(
          self._replay.add_transition_op, {
              self._replay.add_obs_ph: observation,
              self._replay.add_action_ph: action,
              self._replay.add_reward_ph: reward,
              self._replay.add_terminal_ph: is_terminal,
              self._replay.add_legal_actions_ph: legal_actions
          })

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state.

    This is used for checkpointing. It will return a dictionary containing all
    non-TensorFlow objects (to be saved into a file by the caller), and it saves
    all TensorFlow objects into a checkpoint file.

    Args:
      checkpoint_dir: str, directory where TensorFlow objects will be saved.
      iteration_number: int, iteration number for naming the checkpoint file.

    Returns:
      A dictionary containing all of the agent's non-TensorFlow objects.
        If the checkpoint directory does not exist, will return None.
    """
    if not tf.gfile.Exists(checkpoint_dir):
      return None
    self._saver.save(
        self._sess,
        os.path.join(checkpoint_dir, 'tf_ckpt'),
        global_step=iteration_number)
    self._replay.save(checkpoint_dir, iteration_number)
    bundle_dictionary = {}
    bundle_dictionary['state'] = self.state
    bundle_dictionary['eval_mode'] = self.eval_mode
    bundle_dictionary['training_steps'] = self.training_steps
    bundle_dictionary['batch_staged'] = self.batch_staged
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint.

    Restores the agent's Python objects to those specified in bundle_dictionary,
    and restores the TensorFlow objects to those specified in the
    checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
      agent's state.

    Args:
      checkpoint_dir: str, path to the checkpoint saved by `tf.Save`.
      iteration_number: int, checkpoint version.
      bundle_dictionary: Dictionary containing this class's Python objects.

    Returns:
      A boolean indicating whether unbundling was successful.
    """
    try:
      # replay.load() will throw a GOSError if it does not find all the
      # necessary files, in which case we should abort the process.
      self._replay.load(checkpoint_dir, iteration_number)
    except tf.errors.NotFoundError:
      return False
    for key in self.__dict__:
      if key in bundle_dictionary:
        self.__dict__[key] = bundle_dictionary[key]
    self._saver.restore(self._sess, tf.train.latest_checkpoint(checkpoint_dir))
    return True
