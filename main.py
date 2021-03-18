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

'''
BOSTON_DATA_PATH = "data/boston_vcf/bos_by_sample.tsv"  # 9249 mutations
COG_UK_2020_09_17 = "data/cog_uk_vcf/cog_20200917_by_sample.tsv"
COG_20201120 = "data/cog_uk_vcf/cog_20201120_by_sample.tsv"
'''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
BOSTON_DATA_PATH = "data/boston_vcf/bos_by_sample.tsv" 
COG_UK_2020_09_17 = "data/cog_uk_vcf/cog_20200917_by_sample.tsv"
COG_20201120 = "data/cog_uk_vcf/cog_20201120_by_sample.tsv"
#FULL_LIST = ['19A', '20H/501Y.V2', '20A', '20E (EU1)', '20D', '20I/501Y.V1', '20B', '20F', '19B', '20J/501Y.V3', '20G', '20C']
CLADES_EXCLUDE_LIST = [] #['19A', '19B', '20A', '20B', '20C', '20D', '20E (EU1)', '20F', '20G', '20H/501Y.V2', '20I/501Y.V1', '20J/501Y.V3']
#CLADES_EXCLUDE_LIST = ['19A'] # '19A', '19B', '20A', '20B', '20C', '20D', '20E (EU1)', '20F', '20G', '20H/501Y.V2', '20I/501Y.V1', '20J/501Y.V3'
#                               [''     '19B', '20A', '20B', '20C',        '20E (EU1)', '20F', '20G', '20H/501Y.V2', '20I/501Y.V1']
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
    max_len_REF = 0
    max_len_ALT = 0
    AF_CUTOFF = 1.0
    #selected_mutations = utils.include_mutations(utils.read_json(CLADES_MUTATIONS), CLADES_EXCLUDE_LIST)
    selected_mutations = utils.get_clades_pos_alt()
    for idx in range(len(by_sample_dataframe_take_cols)):
        sample_row = by_sample_dataframe_take_cols.take([idx])
        check_var = "{}>{}>{}".format(sample_row["REF"].values[0], sample_row["POS"].values[0], sample_row["ALT"].values[0])
        check_var_pos_alt = "{}>{}".format(sample_row["POS"].values[0], sample_row["ALT"].values[0])
        AF = float(sample_row["AF"].values[0])
        REF = sample_row["REF"].values[0]
        POS = sample_row["POS"].values[0]
        sample_name = sample_row["Sample"].values[0]
        ALT = sample_row["ALT"].values[0]
        # exclude signature mutations from known clades such as 19A, 19B .. in search of novel mutations.
        if check_var_pos_alt not in selected_mutations and AF <= AF_CUTOFF:
            variant = "{}>{}>{}>{}>{}".format(sample_name, POS, REF, ALT, AF)
            if max_len_ALT < len(ALT):
                max_len_ALT = len(ALT)
            if max_len_REF < len(REF):
                max_len_REF = len(REF) 
            
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
    print("Max length of REF and ALT: {}, {}".format(str(max_len_REF), str(max_len_ALT)))
    return samples_dict, samples_name_idx, max_len_REF, max_len_ALT


def encode_variants(samples, samples_name_idx, max_REF, max_ALT):    
    print("Encode and transform variants...")
    variants = transform_variants.TransformVariants()
    transformed_samples = variants.get_variants(samples, samples_name_idx, max_REF, max_ALT, "train")
    features = np.asarray(transformed_samples)
    features = features.astype('float32')
    #features = utils.transform_integers(features)
    cluster_variants.transform_variants(features, samples_name_idx)

  
if __name__ == "__main__":
    start_time = time.time()
    samples, samples_name_idx, max_REF, max_ALT = read_files()
    encode_variants(samples, samples_name_idx, max_REF, max_ALT)
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
