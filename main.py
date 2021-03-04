import time
import sys
import os
#import allel
import gzip
import glob
import random
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import logging
import h5py

import transform_variants
import setup_network
import post_processing
import utils
import fetch_clades_mutations


SEED = 32000
N_FILES = 1000
N_EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-4
TR_TE_SPLIT = 0.2

REF_DIM = 5
ALT_1_DIM = 2
POS_AF = 2
ORIG_DIM = POS_AF + REF_DIM + ALT_1_DIM
I_DIM = 2
MODEL_SAVE_PATH = "data/saved_models/model"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
BOSTON_DATA_PATH = "data/boston_vcf/bos_by_sample.tsv"
COG_20201120 = "data/boston_vcf/cog_20201120_by_sample.tsv"


def read_files(clades_mutations, path=BOSTON_DATA_PATH):
    """
    
    """
    print("Extracting data from tabular variants file...")
    take_cols = ["Sample", "POS", "REF", "ALT", "AF"]
    by_sample_dataframe = pd.read_csv(path, sep="\t")
    by_sample_dataframe_take_cols = by_sample_dataframe[take_cols]
    samples_dict = dict()
    sample_name = ""
    #clades_mutations = []
    for idx in range(len(by_sample_dataframe_take_cols)):
        sample_row = by_sample_dataframe_take_cols.take([idx])
        sample_name = sample_row["Sample"].values[0]
        check_var = "{}>{}>{}".format(sample_row["REF"].values[0], sample_row["POS"].values[0], sample_row["ALT"].values[0])
        # exclude signature mutations from known clades such as 19A, 19B .. in search of novel mutations.
        if check_var not in clades_mutations:
            variant = "{}>{}>{}>{}".format(sample_row["POS"].values[0], sample_row["REF"].values[0], sample_row["ALT"].values[0], sample_row["AF"].values[0])
            sample_name = sample_row["Sample"].values[0] 
            if sample_name not in samples_dict:
                samples_dict[sample_name] = list()
            samples_dict[sample_name].append(variant)
    #assert len(by_sample_dataframe_take_cols[by_sample_dataframe_take_cols["Sample"] == sample_name]) == len(samples_dict[sample_name])
    utils.save_as_json("data/samples_dict.json", samples_dict)    
    return samples_dict
    

def split_format_variants(samples, tr_test_split=TR_TE_SPLIT):
    s_names = list()
    variants_freq = dict()
    # split samples into train and test
    split_int = int(len(samples) * tr_test_split)
    train_data = dict(list(samples.items())[split_int:])
    test_data = dict(list(samples.items())[:split_int])
    assert len(train_data) == len(samples) - split_int
    assert len(test_data) == split_int
    
    #var_freq = post_processing.pre_viz(train_data)
    
    #var_freq = post_processing.pre_viz(test_data)
    
    tf_variants = transform_variants.TransformVariants()
    print("Train data...")
    tr_transformed_samples = tf_variants.get_variants(train_data, "train")
    
    print("Test data...")
    te_transformed_samples = tf_variants.get_variants(test_data, "test")
    return tr_transformed_samples, te_transformed_samples


def balance_train_data(train_data, batch_size, scaler):

    pos_categories = {"0": "0.0-0.2", "1": "0.2-0.4", "2": "0.4-0.6", "3": "0.6-0.8", "4": "0.8-1.0"} 
    #{"0": "0-5000", "1": "5000-10000", "2": "10000-15000", "3": "15000-20000", "4": "20000-25000", "5": "25000-30000"}
    af_categories = {"0": "0.0-0.2", "1": "0.2-0.4", "2": "0.4-0.6", "3": "0.6-0.8", "4": "0.8-1.0"}
    balanced_tr_data = list()
    while len(balanced_tr_data) < batch_size:
        random.shuffle(train_data)
        for i, item in enumerate(train_data):
            get_rand_pos = random.choice((list(pos_categories.keys())))
            get_rand_af = random.choice((list(af_categories.keys())))
            rand_pos = pos_categories[get_rand_pos]
            min_pos, max_pos = rand_pos.split("-")
            rand_af = af_categories[get_rand_af]
            min_af, max_af = rand_af.split("-")
            if (item[1] > float(min_af) and item[1] <= float(max_af)): 
                balanced_tr_data.append(item)
    balanced_tr_data = np.asarray(balanced_tr_data)
    return balanced_tr_data


def get_variants_categories(train_data):
    var_categories = list()
    for item in train_data:
        var_name = utils.get_var(item, POS_AF, REF_DIM)
        var_categories.append(var_name)
    return list(set(var_categories))
    
def balance_var(train_data, var_categories, batch_size):
    balanced_tr_data = list()
    while len(balanced_tr_data) < batch_size:
        rand_var = random.choice(var_categories)
        for i, item in enumerate(train_data):
            var_n = utils.get_var(item, POS_AF, REF_DIM)
            if var_n == rand_var:
                balanced_tr_data.append(item)
                break
    return np.asarray(balanced_tr_data)


def train_autoencoder(train_data, test_data, batch_size=BATCH_SIZE, learning_rate=LR, num_epochs=N_EPOCHS):

    training_features = np.asarray(train_data)
    print(training_features.shape)
    test_features = np.asarray(test_data)
    print(test_features.shape)
    training_features = training_features.astype('float32')
    test_features = test_features.astype('float32')
    
    training_features, test_features, scaler = utils.transform_integers(training_features, test_features)

    print(training_features.shape)
    print(test_features.shape)

    var_categories = get_variants_categories(training_features)

    tr_iter = int(training_features.shape[0] / float(batch_size))
    '''print("Balancing training batches...")
    balanced_training_features = list()
    for i in range(0, tr_iter):
        bal_tr = balance_var(training_features, var_categories, batch_size)
        balanced_training_features.append(bal_tr)
    balanced_training_features = np.asarray(balanced_training_features)'''

    tr_epo_loss = np.zeros((num_epochs, 1))
    te_epo_loss = np.zeros((num_epochs, 1))
    print("Start training...")
    autoencoder = setup_network.Autoencoder(ORIG_DIM, I_DIM)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=LR)
    #tf.keras.optimizers.SGD(learning_rate=LR)
    #tf.optimizers.Adam(learning_rate=LR)
    steps = training_features.shape[0] / float(batch_size)
    global_step = tf.Variable(0)
    
    rand_pos = np.random.randint(test_features.shape[0])
    for epoch in range(num_epochs):
        tr_loss = list()
        te_loss = list()
        for x in range(0, len(training_features), batch_size):
            #for x in range(0, len(training_features), batch_size):
            #x_inp = training_features[x : x + batch_size]
            #for i in range(0, tr_iter):
            x_inp = training_features[x : x + batch_size]
            #x_inp = balance_var(training_features, var_categories, batch_size)
            #sys.exit()
            loss_value, grads, reconstruction = autoencoder.grad(autoencoder, x_inp)
            optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables), global_step)
            c_tr_loss = autoencoder.loss(x_inp, reconstruction)
            c_te_loss = autoencoder.loss(test_features, autoencoder(test_features))
            #print(c_tr_loss, c_te_loss)
            tr_loss.append(c_tr_loss)
            te_loss.append(c_te_loss)
        
        print(test_features[rand_pos,:])
        print()
        print(autoencoder(test_features)[rand_pos,:].numpy())
        mean_tr_loss = np.mean(tr_loss)
        mean_te_loss = np.mean(te_loss)
        tr_epo_loss[epoch] = mean_tr_loss
        te_epo_loss[epoch] = mean_te_loss
        print("Epoch {}/{} training loss: {}".format(epoch + 1, num_epochs, str(np.round(mean_tr_loss, 4))))
        print("Epoch {}/{} test loss: {}".format(epoch + 1, num_epochs, str(np.round(mean_te_loss, 4))))
        print("========================================")
        print()
    np.savetxt("data/train_loss.txt", tr_epo_loss)
    np.savetxt("data/test_loss.txt", te_epo_loss)
    print("Post processing predictions...")
    autoencoder.save(MODEL_SAVE_PATH)
    h5f = h5py.File('data/test_data.h5', 'w')
    h5f.create_dataset('test_data', data=test_features)
    #low_dim_test_predictions = autoencoder.encoder(test_features)
    #post_processing.transform_predictions(low_dim_test_predictions)
    #post_processing.plot_losses()
    #post_processing.plot_true_pred(test_features, autoencoder(test_features))


if __name__ == "__main__":
    start_time = time.time()
    u_all_clades_mutations = fetch_clades_mutations.get_nuc_clades()
    samples = read_files(u_all_clades_mutations)
    tr_data, te_data = split_format_variants(samples)
    #train_autoencoder(tr_data, te_data)
    
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
