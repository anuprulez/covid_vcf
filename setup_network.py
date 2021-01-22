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
    def __init__(self, intermediate_dim, vocab_size, embed_dim, f_h=16, s_h=8):
        super(Encoder, self).__init__()

        self.embedder = tf.keras.layers.Embedding(
            vocab_size,
            8,
            input_length=1,
            embeddings_initializer='glorot_uniform',
            trainable=True
        )
        
        '''self.embedder = tf.keras.layers.Dense(
            units=8,
            activation=tf.nn.relu,
        )'''

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
        #re_in_features = utils.encode_integers(self.embedder, input_features)
        a_f_h = self.hidden_layer1(input_features)
        a_s_h = self.hidden_layer2(a_f_h)
        return self.output_layer(a_s_h)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, orig_dim, f_h=8, s_h=16):
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
    def __init__(self, vocab_size, original_dim, intermediate_dim, embedding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(intermediate_dim, vocab_size, embedding_dim)
        self.decoder = Decoder(original_dim)

    def call(self, input_features):
        code = self.encoder(input_features)
        return self.decoder(code)
        
    def loss(self, x, x_pred):
       return tf.losses.binary_crossentropy(x, x_pred)
       #tf.losses.mean_squared_error(x, x_pred)

    def grad(self, model, inputs):
        with tf.GradientTape() as tape:
            reconstruction = model(inputs)
            re_input = utils.encode_integers(model.encoder.embedder, inputs)
            loss_value = self.loss(re_input, reconstruction)
        return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction
