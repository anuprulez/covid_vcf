import tensorflow as tf
from tensorflow import feature_column

import unicodedata
import re
import numpy as np
import os
import io
import time

import utils


class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, f_h=8, s_h=4):
        super(Encoder, self).__init__()

        self.hidden_layer1 = tf.keras.layers.Dense(
            units=f_h,
            activation=tf.nn.relu,
        )

        self.hidden_layer2 = tf.keras.layers.Dense(
            units=s_h,
            activation=tf.nn.relu,
        )

        self.output_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.relu
        )

    def call(self, input_features):
        a_f_h = self.hidden_layer1(input_features)
        a_s_h = self.hidden_layer2(a_f_h)
        return self.output_layer(a_s_h)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, orig_dim, f_h=4, s_h=8):
        super(Decoder, self).__init__()
        
        self.hidden_layer1 = tf.keras.layers.Dense(
            units=f_h,
            activation=tf.nn.relu,
        )
        
        self.hidden_layer2 = tf.keras.layers.Dense(
            units=s_h,
            activation=tf.nn.relu,
        )
        
        self.output_layer = tf.keras.layers.Dense(
            units=orig_dim,
            activation=tf.nn.relu
        )

    def call(self, encoded):
        a_f_h = self.hidden_layer1(encoded)
        a_s_h = self.hidden_layer2(a_f_h)
        return self.output_layer(a_s_h)


class Autoencoder(tf.keras.Model):
    def __init__(self, original_dim, intermediate_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(intermediate_dim)
        self.decoder = Decoder(original_dim)

    def call(self, input_features):
        code = self.encoder(input_features)
        return self.decoder(code)
        
    def loss(self, x, x_pred):
       return tf.math.reduce_mean(tf.losses.binary_crossentropy(x, x_pred))

    def grad(self, model, inputs):
        with tf.GradientTape() as tape:
            reconstruction = model(inputs)
            loss_value = self.loss(inputs, reconstruction)
        return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction
