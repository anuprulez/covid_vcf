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

import transform_variants


def read_files(path="data/sars-cov2.variants/*.gz", n_max_file=10):
    file_names = glob.glob(path)
    random.shuffle(file_names)
    samples = dict()
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
    print("Test data...")
    te_transformed_samples = tf_variants.get_variants(test_data)


if __name__ == "__main__":
    start_time = time.time()
    samples = read_files()
    split_format_variants(samples)
    end_time = time.time()
    print("Program finished in {} seconds".format(str(end_time - start_time)))
