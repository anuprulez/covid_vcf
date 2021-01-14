import time
import sys
import os
import allel
import gzip
import glob
import random
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import numpy as np

import transform_variants
import setup_network


def read_files(path="data/sars-cov2.variants/*.gz", n_max_file=1000):
    file_names = glob.glob(path)
    random.shuffle(file_names)
    samples = dict()
    print("Preparing variants...")
    for idx in range(n_max_file):
        file_path = file_names[idx]
        file_name = file_path.split('/')[-1]
        df = allel.vcf_to_dataframe(file_path)
        samples[file_name] = list()
        try:
            for idx, i in enumerate(df["POS"].tolist()):
                variant = dict()
                variant[i] = "{}>{}".format(df["REF"][idx], df["ALT_1"][idx]) #(df["REF"][idx], df["ALT_1"][idx])
                samples[file_name].append(variant)
        except Exception as ex:
            continue
    return samples

            
def split_format_variants(samples, tr_test_split=0.2):

    s_names = list()
    variants_freq = dict()
    # split samples into train and test
    split_int = int(len(samples) * tr_test_split)
    train_data = dict(list(samples.items())[split_int:])
    test_data = dict(list(samples.items())[:split_int])

    assert len(train_data) == len(samples) - split_int
    assert len(test_data) == split_int
    
    tf_variants = transform_variants.TransformVariants()
    print("Train data...")
    tr_transformed_samples = tf_variants.get_variants(train_data)
    print(len(tr_transformed_samples))
    print("Test data...")
    te_transformed_samples = tf_variants.get_variants(test_data)
    print(len(te_transformed_samples))
    return tr_transformed_samples, te_transformed_samples
    
def train_autoencoder(train_data, test_data, batch_size=32, learning_rate=1e-2, epochs=10):

    autoencoder = setup_network.Autoencoder(intermediate_dim=2, original_dim=16)
    opt = tf.optimizers.Adam(learning_rate=learning_rate)

    training_features = np.asarray(train_data)[0]
    test_features = np.asarray(test_data)[0]

    print(training_features.shape)

    training_features = training_features.astype('float32')
    training_dataset = tf.data.Dataset.from_tensor_slices(training_features)
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.shuffle(training_features.shape[0])
    training_dataset = training_dataset.prefetch(batch_size * 4)

    writer = tf.summary.create_file_writer('tmp')

    epo_loss = np.zeros((epochs, 1))

    print("Start training...")

    with writer.as_default():
        with tf.summary.record_if(True):
            for epoch in range(epochs):
                loss = 0.0
                for step, batch_features in enumerate(training_dataset):
                    print(batch_features)
                    autoencoder.train(autoencoder.loss, autoencoder, opt, batch_features)
                    loss_values = autoencoder.loss(autoencoder, batch_features)
                    original = batch_features #tf.reshape(batch_features, (batch_features.shape[0], 1))
                    #reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)), (batch_features.shape[0], 1))
                    reconstructed = autoencoder(tf.constant(batch_features))
                    current_loss = loss_values.numpy()
                    #print(current_loss)
                    loss += current_loss
                mean_loss = loss / batch_size
                epo_loss[epoch] = mean_loss
                print("Epoch {} training loss: {}".format(epoch + 1, str(mean_loss)))


if __name__ == "__main__":
    start_time = time.time()
    samples = read_files()
    tr_data, te_data = split_format_variants(samples)
    train_autoencoder(tr_data, te_data)
    end_time = time.time()
    print("Program finished in {} seconds".format(str(end_time - start_time)))
