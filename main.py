import time
import sys
import os
#import allel
import gzip
import glob
import random
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import logging
import h5py

import transform_variants
import post_processing
import utils
import cluster_variants


AF_CUTOFF = 0.8

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
BOSTON_DATA_PATH = "data/boston_vcf/bos_by_sample.tsv"
COG_20201120 = "data/boston_vcf/cog_20201120_by_sample.tsv"
#FULL_LIST = ['19A', '20H/501Y.V2', '20A', '20E (EU1)', '20D', '20I/501Y.V1', '20B', '20F', '19B', '20J/501Y.V3', '20G', '20C']
CLADES_EXCLUDE_LIST = ['19A', '19B', '20A', '20B', '20C', '20D', '20E (EU1)', '20F', '20G', '20H/501Y.V2', '20I/501Y.V1', '20J/501Y.V3']
#CLADES_EXCLUDE_LIST = ['19A'] # '19A', '19B', '20A', '20B', '20C', '20D', '20E (EU1)', '20F', '20G', '20H/501Y.V2', '20I/501Y.V1', '20J/501Y.V3'
CLADES_MUTATIONS = "data/clades/parsed_nuc_clades.json"


def read_files(path=BOSTON_DATA_PATH):
    """
    
    """
    print("Extracting data from tabular variants file...")
    take_cols = ["Sample", "POS", "REF", "ALT", "AF"]
    by_sample_dataframe = pd.read_csv(path, sep="\t")
    by_sample_dataframe_take_cols = by_sample_dataframe[take_cols]
    samples_name_idx = dict()
    sample_counter = 1
    samples_dict = dict()
    selected_mutations = utils.include_mutations(utils.read_json(CLADES_MUTATIONS), CLADES_EXCLUDE_LIST)
    for idx in range(len(by_sample_dataframe_take_cols)):
        sample_row = by_sample_dataframe_take_cols.take([idx])
        check_var = "{}>{}>{}".format(sample_row["REF"].values[0], sample_row["POS"].values[0], sample_row["ALT"].values[0])
        AF = float(sample_row["AF"].values[0])
        # exclude signature mutations from known clades such as 19A, 19B .. in search of novel mutations.
        if check_var not in selected_mutations and AF < AF_CUTOFF:
            variant = "{}>{}>{}>{}>{}".format(sample_row["Sample"].values[0], sample_row["POS"].values[0], sample_row["REF"].values[0], sample_row["ALT"].values[0], AF)
            sample_name = sample_row["Sample"].values[0]
            if sample_name not in samples_name_idx:
                samples_name_idx[sample_name] = sample_counter
                sample_counter += 1
            if sample_name not in samples_dict:
                samples_dict[sample_name] = list()
            samples_dict[sample_name].append(variant)
    utils.save_as_json("data/samples_dict.json", samples_dict)
    utils.save_as_json("data/samples_name_idx.json", samples_name_idx)
    print("Clades excluded: {}".format(",".join(CLADES_EXCLUDE_LIST)))
    print("Total samples: {}".format(str(len(samples_dict))))
    return samples_dict, samples_name_idx


def encode_variants(samples, samples_name_idx):    
    print("Encoded variants...")
    variants = transform_variants.TransformVariants()
    transformed_samples = variants.get_variants(samples, samples_name_idx, "train")
    features = np.asarray(transformed_samples)
    features = features.astype('float32')
    #features = utils.transform_integers(features)
    cluster_variants.transform_variants(features, samples_name_idx, BOSTON_DATA_PATH)

  
if __name__ == "__main__":
    start_time = time.time()
    samples, samples_name_idx = read_files()
    encode_variants(samples, samples_name_idx)
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
