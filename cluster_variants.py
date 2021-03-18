import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import OPTICS
from sklearn import metrics


import utils

SAVE_PATH = "data/outputs/Boston/clades/"

#SAVE_PATH = "data/outputs/Pre-COG-UK/all/"

#SAVE_PATH = "data/outputs/Pre-COG-UK/all/"


def transform_variants(samples_data, samples_name_idx):
    detailed_samples = utils.read_json("data/train_n_variants.json")
    sample_names = utils.read_json("data/train_all_variants.json")
    n_samples = list()
    s_name_df = list()
    s_idx_df = list()
    var_name_df = list()
    var_pos_df = list()
    var_af_df = list()
    var_ref_df = list()
    var_alt_df = list()
    for i, sample_count in enumerate(detailed_samples):
        idx = int(list(sample_count.values())[0])
        sample_name = list(sample_count.keys())[0]
        sample_variants = sample_names[sample_name]
        _, var_sidx, var_pos, var_name, var_af, var_ref, var_alt = utils.deserialize(sample_variants, sample_name, samples_name_idx)
        s_name_df.extend(np.repeat(sample_name, idx))
        s_idx_df.extend(var_sidx)
        var_name_df.extend(var_name)
        var_pos_df.extend(var_pos)
        var_af_df.extend(var_af)
        var_ref_df.extend(var_ref)
        var_alt_df.extend(var_alt)
    cluster_mutations(samples_data, s_name_df, s_idx_df, var_name_df, var_pos_df, var_af_df, var_ref_df, var_alt_df, samples_name_idx)


def find_optimal_clusters(features):
    clustering_type = OPTICS(min_samples=2, min_cluster_size=2)
    cluster_labels = clustering_type.fit_predict(features)
    print("Number of unique clusters: {}".format(str(len(list(set(cluster_labels))))))
    cluster_silhouette_score = metrics.silhouette_score(features, cluster_labels, metric='euclidean')
    return cluster_labels


def extract_co_occuring_samples(mutations_df, mutations_labels):
    u_mutation_labels = list(set(mutations_labels))
    clusters_samples = list()
    c_co_occurring_mut_path = "{}clusters_of_co-occurring_samples.csv".format(SAVE_PATH)
    for c_label in u_mutation_labels:
        cluster_rows = mutations_df[mutations_df["Cluster"] == c_label]
        if len(cluster_rows) > 0:
            s_dict = cluster_rows["Index"].to_dict()
            l_samples = list(s_dict.values())
            clusters_samples.append({c_label: l_samples})
    # make sizes of list of samples uniform to enable clustering
    mat_samples = list()
    for i, item in enumerate(clusters_samples):
        l_samples = list(item.values())[0]
        mat_samples.append(np.sort(l_samples))
    assert len(mat_samples) == len(list(set(mutations_labels)))
    sample_cluster_indices = dict()
    print("Finding overlap within mutations and samples...")
    for i, rowi in enumerate(mat_samples):
        for j, rowj in enumerate(mat_samples):
            if i != j:
                if len(rowi) <= len(rowj):
                    intersection = np.sort(list(set(rowi).intersection(set(rowj))))
                    # One set of samples is completely present in another set of samples 
                    # having different sets of mutations
                    if len(intersection) == len(rowi):
                        key = ",".join(str(v) for v in intersection)
                        if key not in sample_cluster_indices:
                            sample_cluster_indices[key] = list()
                        if i not in sample_cluster_indices[key]:
                            sample_cluster_indices[key].extend([i])
                        if j not in sample_cluster_indices[key]:
                            sample_cluster_indices[key].extend([j])
    utils.save_as_json("data/samples_clusters.json", sample_cluster_indices)
    cluster_ctr = 0
    merge_clusters = list()
    by_mutations = list()
    for item in sample_cluster_indices:
        samples = [int(v) for v in item.split(",")]
        l_samples = len(samples)
        clusters = sample_cluster_indices[item]
        for c in clusters:
            l_names = list()
            l_af = list()
            cluster_df = mutations_df[mutations_df["Cluster"] == int(c)]
            clean_df_csv = utils.clean_cluster(cluster_df.to_csv())
            row = clean_df_csv[0]
            merged_row = row[2:5]
            merged_row.extend([row[6]])
            for s in samples:
                cluster_sample = cluster_df[cluster_df["Index"] == s].to_csv()
                cluster = utils.clean_cluster(cluster_sample, cluster_ctr)
                merge_clusters.extend(cluster)
                _, af, name = utils.clean_cluster_by_mutation(cluster_sample)
                l_names.append(name)
                l_af.append(af)
            merged_row.extend([min(l_af), max(l_af), len(l_names), ",".join(l_names)])
            by_mutations.extend([merged_row])
        cluster_ctr += 1
    
    ## TODO: Fix by-mutations report
    
    co_occur_variants_df = pd.DataFrame(merge_clusters, columns=["Sample", "Index", "REF", "POS", "ALT", "AF", "ClusterMutations", "Cluster"])
    
    co_occur_variants_df = co_occur_variants_df.astype({'Cluster': 'int', 'POS': 'int'})
    save_co_occur_variants_df = co_occur_variants_df.drop(["Index", "ClusterMutations"], axis=1)
    save_co_occur_variants_df = save_co_occur_variants_df.sort_values(by=["Cluster", "POS"], ascending=[True, True])
    utils.save_dataframe(save_co_occur_variants_df, c_co_occurring_mut_path)
    

def cluster_mutations(features, s_name_df, s_idx_df, var_name_df, var_pos_df, var_af_df, var_ref_df, var_alt_df, samples_name_idx):

    features_uniform = ["REF", "POS", "ALT"]
    column_names = ["Sample", "Index", "REF", "POS", "ALT", "AF",  "Cluster"]
    excluded_mutations_path = "{}excluded_mutations.csv".format(SAVE_PATH)
    cluster_mutations_path = "{}clusters_of_mutations.csv".format(SAVE_PATH)
    cluster_mutations_singletons_path = "{}clusters_of_mutations_with_singletons.csv".format(SAVE_PATH)
    
    print("Shape of features: ({},{})".format(str(features.shape[0]), str(features.shape[1])))
    print("Clustering mutations...")
    cluster_labels = find_optimal_clusters(features)

    # create dataframe of variants
    variants_df = pd.DataFrame(list(zip(s_name_df, s_idx_df, var_ref_df, var_pos_df, var_alt_df, var_af_df, cluster_labels)), columns=column_names, index=None)
    
    # separate dataframes with > 1 mutation clusters and singleton clusters
    clustered_mut_df, single_mut_df = utils.remove_single_mutation(variants_df, features_uniform)

    # save singleton mutations
    single_mut_df = single_mut_df.astype({'Cluster': 'int', 'POS': 'int'})
    single_mut_df = single_mut_df.sort_values(by=["Cluster"], ascending=[True])
    utils.save_dataframe(single_mut_df, excluded_mutations_path)
    
    clustered_mut_df = clustered_mut_df.astype({'Cluster': 'int', 'POS': 'int'})
    clustered_mut_df = clustered_mut_df.sort_values(by=["Cluster"], ascending=[True])
    
    # set serial numbers to all clusters
    ordered_c_labels = utils.set_serial_cluster_numbers(clustered_mut_df["Cluster"])
    clustered_mut_df["Cluster"] = ordered_c_labels

    # check if clusters are uniform
    utils.check_uniform_clusters(clustered_mut_df, len(list(set(ordered_c_labels))))
    utils.save_dataframe(clustered_mut_df, cluster_mutations_path)

    # merge clusters with more than one mutation with singleton clusters
    clusters_with_singletons_df = pd.concat([clustered_mut_df, single_mut_df], ignore_index=True)
    utils.save_dataframe(clusters_with_singletons_df, cluster_mutations_singletons_path)

    # merge clusters
    merged_clusters_df = merge_all_clusters(clusters_with_singletons_df)

    # extract same POS using all clusters with different REF and ALT
    create_dataset_with_same_pos(merged_clusters_df)

    # get repeated samples with same mutations using clustered mutations dataframe
    extract_co_occuring_samples(clustered_mut_df, ordered_c_labels)


def merge_all_clusters(clusters_with_singletons_df):
    merge_clusters = list()
    column_names = ["REF", "POS", "ALT", "MinAF", "MaxAF", "NumSamples", "SampleNames", "MutationCluster"]
    merged_c_mutations_path = "{}merged_clusters_of_mutations.csv".format(SAVE_PATH)
    
    n_u_clusters = list(set(clusters_with_singletons_df["Cluster"].to_list()))
    for clstr in n_u_clusters:
        cluster_df = clusters_with_singletons_df[clusters_with_singletons_df["Cluster"] == clstr]
        max_af = np.max(cluster_df["AF"])
        min_af = np.min(cluster_df["AF"])
        sample_names = cluster_df["Sample"].to_list()
        to_csv = utils.clean_cluster(cluster_df.to_csv())
        row = to_csv[0][2:5]
        row.extend([min_af, max_af, len(sample_names), ",".join(sample_names)])
        row.extend([to_csv[0][-1]])
        merge_clusters.extend([row])
    merge_clusters_df = pd.DataFrame(merge_clusters, columns=column_names)
    merge_clusters_df = merge_clusters_df.astype({'MinAF': 'float64', 'MaxAF': 'float64', 'POS': 'int'})
    merge_clusters_df = merge_clusters_df.sort_values(by=["POS"], ascending=[True])
    utils.save_dataframe(merge_clusters_df, merged_c_mutations_path)
    return merge_clusters_df
    

def create_dataset_with_same_pos(all_clusters_df):
    same_pos_df = None
    unique_POS = np.sort(list(set(all_clusters_df["POS"].to_list())))
    for POS in unique_POS:
        POS_rows = all_clusters_df[all_clusters_df["POS"] == POS]
        if len(POS_rows) > 1:
            if same_pos_df is None:
                same_pos_df = POS_rows
            else:
                same_pos_df = pd.concat([same_pos_df, POS_rows], ignore_index=True)
    same_pos_df = same_pos_df.astype({'POS': 'int'})
    same_pos_df = same_pos_df.sort_values(by=["POS"], ascending=[True])
    utils.save_dataframe(same_pos_df, "{}same_pos_diff_ref_alt.csv".format(SAVE_PATH))
