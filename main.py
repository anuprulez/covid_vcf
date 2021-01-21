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
N_FILES = 100
N_EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
LOW_DIM = 2
TR_TE_SPLIT = 0.2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def read_files(path="data/sars-cov2.variants/*.gz", n_max_file=N_FILES):
    file_names = glob.glob(path)
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
    
    
    #autoencoder = tf_variants.collect_POS_QUAL(train_data)
    
    # learn POS and QUAL embedding

    print("Train data...")
    tr_transformed_samples = tf_variants.get_variants(train_data, "train")
    
    print("Test data...")
    te_transformed_samples = tf_variants.get_variants(test_data, "test")
    return tr_transformed_samples, te_transformed_samples


def train_autoencoder(train_data, test_data, batch_size=BATCH_SIZE, learning_rate=LR, num_epochs=N_EPOCHS):

    training_features = np.asarray(train_data)
    
    print(training_features.shape)
    
    test_features = np.asarray(test_data)

    print(test_features.shape)

    training_features = training_features.astype('float32')

    test_features = test_features.astype('float32')

    tr_epo_loss = np.zeros((num_epochs, 1))
    te_epo_loss = np.zeros((num_epochs, 1))

    print("Start training...")

    ORIG_DIM = 10 + 5 + 1 + 16 + 16
    autoencoder = setup_network.Autoencoder(ORIG_DIM, LOW_DIM)
    optimizer = tf.optimizers.Adam(learning_rate=LR)
    global_step = tf.Variable(0)

    for epoch in range(num_epochs):
        tr_loss = 0.0
        te_loss = 0.0
        for x in range(0, len(training_features), batch_size):
            x_inp = training_features[x : x + batch_size]
            loss_value, grads, reconstruction, embedder = autoencoder.grad(autoencoder, x_inp)
            optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables), global_step)
            re_x_inp = embed_test(embedder, x_inp)
            c_tr_loss = np.mean(autoencoder.loss(re_x_inp, reconstruction).numpy())
            re_test_features = embed_test(embedder, test_features)
            #print("Re test features")
            #print(re_test_features.shape)
            c_te_loss = np.mean(autoencoder.loss(re_test_features, autoencoder(test_features)).numpy())
            tr_loss += c_tr_loss
            te_loss += c_te_loss
        mean_tr_loss = tr_loss / batch_size
        mean_te_loss = te_loss / batch_size
        tr_epo_loss[epoch] = mean_tr_loss
        te_epo_loss[epoch] = mean_te_loss
        print("Epoch {} training loss: {}".format(epoch + 1, str(np.round(mean_tr_loss, 4))))
        print("Epoch {} test loss: {}".format(epoch + 1, str(np.round(mean_te_loss, 4))))
        print()
    #print("Post processing predictions...")
    #low_dim_test_predictions = autoencoder.encoder(test_features)
    #post_processing.transform_predictions(low_dim_test_predictions)
    
def embed_test(embedder, feature):

    pos_reshape = np.reshape(feature[:, 0], (feature[:, 0].shape[0], 1))
    qual_reshape = np.reshape(feature[:, 1], (feature[:, 1].shape[0], 1))

    #print("Test features...")
    # transform POS to a vector
    pos_mat = embedder(pos_reshape)
        
    pos_mat = np.reshape(pos_mat, (pos_mat.shape[0], pos_mat.shape[2]))
        
    #print(pos_mat.shape)
        
    # transform QUAL to a vector
    qual_mat = embedder(qual_reshape)
        
    qual_mat = np.reshape(qual_mat, (qual_mat.shape[0], qual_mat.shape[2]))
        
    sliced_input_f = feature[:, 2:]
        
    #print(sliced_input_f[0,:])
        
    concatenated_input_f = np.hstack((pos_mat, qual_mat, sliced_input_f))
        
    #print(concatenated_input_f.shape)
    #print("=======================")
    return concatenated_input_f
    

if __name__ == "__main__":
    start_time = time.time()
    samples = read_files()
    tr_data, te_data = split_format_variants(samples)
    train_autoencoder(tr_data, te_data)
    
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
