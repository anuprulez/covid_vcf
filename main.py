import time
import sys
import os
import allel
import gzip
import glob
import random
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import logging

import transform_variants
import setup_network
import post_processing
import utils


SEED = 32000
N_FILES = 50
N_EPOCHS = 1
BATCH_SIZE = 32
LR = 1e-4
TR_TE_SPLIT = 0.2

REF_DIM = 10
ALT_1_DIM = 5
EMBED_DIM = 0
ORIG_DIM = REF_DIM + ALT_1_DIM + 1 + EMBED_DIM + EMBED_DIM
I_DIM = 2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def read_files(path="data/sars-cov2.variants/*.gz", n_max_file=N_FILES):
    file_names = glob.glob(path)
    print("Total files: {}".format(str(len(file_names))))
    random.seed(SEED)
    random.shuffle(file_names)
    samples = dict()
    print("Preparing variants...")
    for idx in range(n_max_file):
        file_path = file_names[idx]
        file_name = file_path.split('/')[-1]
        df = allel.vcf_to_dataframe(file_path)
        callset = allel.read_vcf(file_path, fields=['AF'])
        try:
            AF = callset['variants/AF'][:, 0]
            samples[file_name] = list()
            for idx, i in enumerate(df["POS"].tolist()):
                variant = dict()
                variant[i] = "{}>{}>{}>{}".format(df["REF"][idx], df["ALT_1"][idx], df["QUAL"][idx], AF[idx])
                samples[file_name].append(variant)
        except Exception as ex:
            continue
    utils.save_as_json("data/samples.json", samples)
    #post_processing.pre_viz(samples)
    return samples

def split_format_variants(samples, tr_test_split=TR_TE_SPLIT):

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
    tr_transformed_samples, tr_pos_qual = tf_variants.get_variants(train_data, "train")

    print("Test data...")
    te_transformed_samples , _ = tf_variants.get_variants(test_data, "test")
    return tr_transformed_samples, te_transformed_samples, tr_pos_qual


def train_autoencoder(train_data, test_data, tr_pos_qual, batch_size=BATCH_SIZE, learning_rate=LR, num_epochs=N_EPOCHS):

    training_features = np.asarray(train_data)
    
    print(training_features.shape)
    
    test_features = np.asarray(test_data)

    print(test_features.shape)

    training_features = training_features.astype('float32')

    test_features = test_features.astype('float32')

    tr_epo_loss = np.zeros((num_epochs, 1))
    te_epo_loss = np.zeros((num_epochs, 1))

    print("Start training...")
    vocab_size = np.max(tr_pos_qual) + 1 # POS and QUAL
    autoencoder = setup_network.Autoencoder(vocab_size, ORIG_DIM, I_DIM, EMBED_DIM)
    optimizer = tf.optimizers.Adam(learning_rate=LR)
    steps = training_features.shape[0] / float(batch_size)
    global_step = tf.Variable(0)
    for epoch in range(num_epochs):
        tr_loss = list()
        te_loss = list()
        for x in range(0, len(training_features), batch_size):
            x_inp = training_features[x : x + batch_size][:, 2:]    
            loss_value, grads, reconstruction = autoencoder.grad(autoencoder, x_inp)
            optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables), global_step)
            #embedder = autoencoder.encoder.embedder
            #re_x_inp = utils.encode_integers(embedder, x_inp)
            c_tr_loss = autoencoder.loss(x_inp, reconstruction)
            #re_test_features = utils.encode_integers(embedder, test_features)
            c_te_loss = autoencoder.loss(test_features[:, 2:], autoencoder(test_features[:, 2:]))
            tr_loss.append(c_tr_loss)
            te_loss.append(c_te_loss)
        sample_f = test_features[:, 2:][0]
        #print(test_features[:, 2:][0,:])
        #print()
        #print(autoencoder(test_features[:, 2:])[0,:])
        #print("========================================")
        mean_tr_loss = np.mean(tr_loss)
        mean_te_loss = np.mean(te_loss)
        tr_epo_loss[epoch] = mean_tr_loss
        te_epo_loss[epoch] = mean_te_loss
        print("Epoch {}/{} training loss: {}".format(epoch + 1, num_epochs, str(np.round(mean_tr_loss, 4))))
        print("Epoch {}/{} test loss: {}".format(epoch + 1, num_epochs, str(np.round(mean_te_loss, 4))))
        print()
    np.savetxt("data/train_loss.txt", tr_epo_loss)
    np.savetxt("data/test_loss.txt", te_epo_loss)
    print("Post processing predictions...")
    #low_dim_test_predictions = autoencoder.encoder(test_features[:, 2:])
    #post_processing.transform_predictions(low_dim_test_predictions)
    test_data = test_features[:, 2:]
    post_processing.plot_true_pred(test_data, autoencoder(test_data))


if __name__ == "__main__":
    start_time = time.time()
    samples = read_files()
    tr_data, te_data, tr_pos_qual = split_format_variants(samples)
    train_autoencoder(tr_data, te_data, tr_pos_qual)
    
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
