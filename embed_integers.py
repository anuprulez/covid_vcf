import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time


EMBEDDING_DIM = 8
VOCAB_SIZE = 50000


class IntegerEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super(IntegerEncoder, self).__init__()

        self.embed_encoder = tf.keras.layers.Dense(
            units=EMBEDDING_DIM,
            activation=tf.nn.relu
        )

    def call(self, input_features):
        return self.embed_encoder(input_features)


class IntegerDecoder(tf.keras.layers.Layer):
    def __init__(self):
        super(IntegerDecoder, self).__init__()
        
        self.embed_decoder = tf.keras.layers.Dense(
            units=1,
            activation=tf.nn.relu
        )

    def call(self, encoded):
        return self.embed_decoder(encoded)


class IntegerAutoencoder(tf.keras.Model):
    def __init__(self):
        super(IntegerAutoencoder, self).__init__()
        self.encoder = IntegerEncoder()
        self.decoder = IntegerDecoder()

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed
        
    def loss(self, x, x_bar):
       return tf.losses.mean_squared_error(x, x_bar)

    def grad(self, model, inputs):
        with tf.GradientTape() as tape:
            reconstruction = model(inputs)
            loss_value = self.loss(inputs, reconstruction)
        return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction           
