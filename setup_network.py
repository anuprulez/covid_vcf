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

        self.output_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.relu
        )

    def call(self, input_features):
        a_f_h = self.hidden_layer1(input_features)
        a_s_h = self.hidden_layer2(a_f_h)
        return self.output_layer(a_s_h)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, original_dim, f_h=4, s_h=8):
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
            units=original_dim,
            activation=tf.nn.relu
        )

        self.output_layer = tf.keras.layers.Dense(
            units=original_dim,
            activation=tf.nn.relu
        )

    def call(self, encoded):
        a_f_h = self.hidden_layer1(encoded)
        a_s_h = self.hidden_layer2(a_f_h)
        return self.output_layer(a_s_h)


class Autoencoder(tf.keras.Model):
    def __init__(self, intermediate_dim, original_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(intermediate_dim=intermediate_dim)
        self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed

    '''def loss(self, model, original):
        reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
        return reconstruction_error'''
        
    def loss(self, x, x_bar):
       return tf.losses.mean_squared_error(x, x_bar)

    def grad(self, model, inputs):
        with tf.GradientTape() as tape:
            reconstruction = model(inputs)
            loss_value = self.loss(inputs, reconstruction)
        return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction

    '''def train(self, loss, model, opt, original):
        with tf.GradientTape() as tape:
            gradients = tape.gradient(loss(model, original), model.trainable_variables)
            gradient_variables = zip(gradients, model.trainable_variables)
            opt.apply_gradients(gradient_variables)'''
            
            
