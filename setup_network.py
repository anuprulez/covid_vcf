import tensorflow as tf

import unicodedata
import re
import numpy as np
import os
import io
import time

EMBEDDING_DIM = 16
VOCAB_SIZE = 50000


class IntegerEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super(IntegerEncoder, self).__init__()

        self.embed_encoder = tf.keras.layers.Embedding(
            VOCAB_SIZE, EMBEDDING_DIM
        )

    def call(self, input_features):
        return self.embed_encoder(input_features)
        

class Encoder(tf.keras.layers.Layer):
    def __init__(self, embedder, intermediate_dim, f_h=8, s_h=4, embedding_s=8):
        super(Encoder, self).__init__()

        self.embedder = embedder

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
        pos_reshape = self.__reshape(input_features[:, 0])
        qual_reshape = self.__reshape(input_features[:, 1])

        # transform POS to a vector
        pos_mat = self.embedder(pos_reshape)

        pos_mat = np.reshape(pos_mat, (pos_mat.shape[0], pos_mat.shape[2]))

        #print(pos_mat.shape)

        # transform QUAL to a vector
        qual_mat = self.embedder(qual_reshape)

        #print(pos_mat.shape)

        qual_mat = np.reshape(qual_mat, (qual_mat.shape[0], qual_mat.shape[2]))
        
        sliced_input_f = input_features[:, 2:]
        
        #print(sliced_input_f[0,:])
        
        concatenated_input_f = np.hstack((pos_mat, qual_mat, sliced_input_f))
        
        #print(concatenated_input_f.shape)
        
        a_f_h = self.hidden_layer1(concatenated_input_f)
        a_s_h = self.hidden_layer2(a_f_h)
        return self.output_layer(a_s_h)
        
    def __reshape(self, feature):     
        return np.reshape(feature, (feature.shape[0], 1))


class Decoder(tf.keras.layers.Layer):
    def __init__(self, orig_dim, f_h=4, s_h=16):
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
        self.embedder = IntegerEncoder()
        self.encoder = Encoder(self.embedder, intermediate_dim)
        self.decoder = Decoder(original_dim)

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed
        
    def loss(self, x, x_bar):
       return tf.losses.mean_squared_error(x, x_bar)
    
    def embed_raw_input(self, test_features):
        pos_reshape = self.__reshape(test_features[:, 0])
        qual_reshape = self.__reshape(test_features[:, 1])

        #print("Loss features...")
        # transform POS to a vector
        pos_mat = self.embedder(pos_reshape)
        
        pos_mat = np.reshape(pos_mat, (pos_mat.shape[0], pos_mat.shape[2]))
        
        #print(pos_mat.shape)
        
        # transform QUAL to a vector
        qual_mat = self.embedder(qual_reshape)
        
        qual_mat = np.reshape(qual_mat, (qual_mat.shape[0], qual_mat.shape[2]))
        
        sliced_input_f = test_features[:, 2:]
        
        #print(sliced_input_f[0,:].shape)
        
        concatenated_input_f = np.hstack((pos_mat, qual_mat, sliced_input_f))
        
        #print(concatenated_input_f.shape)
        #print("=======================")
        return concatenated_input_f
        
    def __reshape(self, feature):     
        return np.reshape(feature, (feature.shape[0], 1))

    def grad(self, model, inputs):
        with tf.GradientTape() as tape:
            reconstruction = model(inputs)
            re_input = self.embed_raw_input(inputs)
            #print(reconstruction.shape, re_input.shape)
            loss_value = self.loss(re_input, reconstruction)
        return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction, self.embedder   
