# pylint: disable=missing-module-docstring, wrong-import-position, unused-variable, unused-argument, not-callable, invalid-name
import sys
import os
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.observation import Observation
from bad.bayesian_action import BayesianAction

class ActionNetwork():
    ''' action network '''

    def __init__(self) -> None:
        self.model = None

    def build(self, observation: Observation, max_action: int) -> None:
        '''build'''
        if self.model is None:
            shape = observation.to_array().shape
            self.model = tf.keras.Sequential([
                tf.keras.Input(shape=shape, name="input"),
                tf.keras.layers.Dense(384, activation="relu", name="layer1"),
                tf.keras.layers.Dense(384, activation="relu", name="layer2"),
                tf.keras.layers.Dense(max_action, activation='softmax', name='Output_Layer')
            ])
            opt = tf.keras.optimizers.Adam(learning_rate=0.01)
            self.model.compile(loss='categorical_crossentropy', optimizer=opt)

    def print_summary(self):
        '''print summary'''
        self.model.summary()

    def get_model_input(self, observation: Observation):
        '''get model input'''
        network_input = observation.to_array()
        reshaped = tf.reshape(network_input, [1, network_input.shape[0]])
        return reshaped

    def get_action(self, observation: Observation) -> BayesianAction:
        '''get action'''
       
        result = self.model(self.get_model_input(observation))
        return BayesianAction(result.numpy()[0])

    def train_step(self, x, y):
        '''train step'''
        model = self.model
        optimizer = self.model.optimizer
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        tf_y =  tf.constant(y)
        tf_x = self.get_model_input(x)
        # TODO: hier weitermachen
        return

        with tf.GradientTape() as tape:
            logits = model(tf_x, training=True)
            loss_value = loss_fn(tf_y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(tf_y, logits)

    def backpropagation(self, loss_policy: float) -> None:
        '''backpropagation'''

        model = self.model
        optimizer = self.model.optimizer

        tfconst = tf.constant(loss_policy)
        return

        # TODO: hier weitermachen
        with tf.GradientTape() as tape:
            grads = tape.gradient(tfconst, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
