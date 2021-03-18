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

save_path = "data/outputs/Boston/all/"



color_dict = {0: "red", 1: "green", 2: "blue"}
N_C = 3

def transform_variants(samples_data, samples_name_idx, BOSTON_DATA_PATH):
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
    cluster_mutations(samples_data, s_name_df, s_idx_df, var_name_df, var_pos_df, var_af_df, var_ref_df, var_alt_df, samples_name_idx, BOSTON_DATA_PATH)


def find_optimal_clusters(features):
    clustering_type = OPTICS(min_samples=2, min_cluster_size=2)
    cluster_labels = clustering_type.fit_predict(features)
    print("Number of unique clusters: {}".format(str(len(list(set(cluster_labels))))))
    cluster_silhouette_score = metrics.silhouette_score(features, cluster_labels, metric='euclidean')
    return cluster_labels
    
def extract_co_occuring_samples(mutations_df, mutations_labels):
    u_mutation_labels = list(set(mutations_labels))
    clusters_samples = list()
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
    ## TODO: Fix by mutations report. Commented out
    '''by_mut_df = pd.DataFrame(by_mutations, columns=["REF", "POS", "ALT", "ClusterMutations", "MinAF", "MaxAF", "NumberOfSamples", "Samples"])
    by_mut_df["NumberOfSamples"] = by_mut_df["NumberOfSamples"].astype(int)
    by_mut_df = by_mut_df.sort_values(by=["NumberOfSamples", "POS"], ascending=[False, True])
    by_mut_df.to_csv("data/by_mutations.csv")
    
    final_by_mutations = by_mut_df.drop(["SampleIndex", "ClusterMutations"], axis=1)
    final_by_mutations["NumberOfSamples"] = final_by_mutations["NumberOfSamples"].astype(int)
    final_by_mutations["POS"] = final_by_mutations["POS"].astype(int)
    final_by_mutations["AF"] = final_by_mutations["AF"].astype(float)
    final_by_mutations = final_by_mutations.sort_values(by=["AF"], ascending=[False])
    final_by_mutations.to_csv("data/final_by_mutations.csv")'''
    
    co_occur_variants_df = pd.DataFrame(merge_clusters, columns=["Sample", "SampleIndex", "REF", "POS", "ALT", "AF", "ClusterMutations", "Cluster"])
    co_occur_variants_df["Cluster"] = co_occur_variants_df["Cluster"].astype(int)
    co_occur_variants_df["POS"] = co_occur_variants_df["POS"].astype(int)
    
    co_occur_variants_df = co_occur_variants_df.sort_values(by=["Cluster"], ascending=[True])    
    save_co_occur_variants_df = co_occur_variants_df.drop(["SampleIndex", "ClusterMutations"], axis=1)
    save_co_occur_variants_df = save_co_occur_variants_df.sort_values(by=["Cluster"], ascending=[True])
    #final_df.to_csv("{}clusters_of_co-occurring_samples.csv".format(save_path), index=False)
    utils.save_dataframe(save_co_occur_variants_df, "{}clusters_of_co-occurring_samples.csv".format(save_path))
    

def cluster_mutations(features, s_name_df, s_idx_df, var_name_df, var_pos_df, var_af_df, var_ref_df, var_alt_df, samples_name_idx, BOSTON_DATA_PATH):
    print("Shape of features: ({},{})".format(str(features.shape[0]), str(features.shape[1])))
    print("Clustering mutations...")
    cluster_labels = find_optimal_clusters(features)

    variants_df = pd.DataFrame(list(zip(s_name_df, s_idx_df, var_ref_df, var_pos_df, var_alt_df, var_af_df, cluster_labels)), columns=["Sample", "Index", "REF", "POS", "ALT", "AF",  "Cluster"], index=None)
    variants_df["Cluster"] = variants_df["Cluster"].astype(str)
    
    # separate dataframes with > 1 mut clusters and singleton clusters
    clustered_mut_df, single_mut_df = utils.remove_single_mutation(variants_df, ["REF", "POS", "ALT"])

    # save singleton mutations
    #single_mut_df.to_csv("{}excluded_mutations.csv".format(save_path), index=False)
    utils.save_dataframe(single_mut_df, "{}excluded_mutations.csv".format(save_path))

    clustered_mut_df["Cluster"] = clustered_mut_df["Cluster"].astype(int)
    clustered_mut_df["POS"] = clustered_mut_df["POS"].astype(int)
    clustered_mut_df = clustered_mut_df.sort_values(by=["Cluster"], ascending=[True])
    ordered_c_labels = utils.set_serial_cluster_numbers(clustered_mut_df["Cluster"])
    clustered_mut_df["Cluster"] = ordered_c_labels

    # check if clusters are uniform
    utils.check_uniform_clusters(clustered_mut_df, len(list(set(ordered_c_labels))))
    #clustered_mut_df.to_csv("{}clusters_of_mutations.csv".format(save_path), index=False)
    utils.save_dataframe(clustered_mut_df, "{}clusters_of_mutations.csv".format(save_path))

    # merge clusters with more than one mutation with singleton clusters
    clusters_with_singletons_df = pd.concat([clustered_mut_df, single_mut_df], ignore_index=True)
    #clusters_with_singletons_df.to_csv("{}clusters_of_mutations_with_singletons.csv".format(save_path), index=False)
    utils.save_dataframe(clusters_with_singletons_df, "{}clusters_of_mutations_with_singletons.csv".format(save_path))

    # merge clusters
    merged_clusters_df = merge_all_clusters(clusters_with_singletons_df)

    # extract same POS using all clusters with different REF and ALT
    create_dataset_with_same_pos(merged_clusters_df)

    # get repeated samples with same mutations using clustered mutations dataframe
    extract_co_occuring_samples(clustered_mut_df, ordered_c_labels)


def merge_all_clusters(clusters_with_singletons_df):
    merge_clusters = list()
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
    merge_clusters_df = pd.DataFrame(merge_clusters, columns=["REF", "POS", "ALT", "MinAF", "MaxAF", "NumSamples", "SampleNames", "MutationCluster"])
    merge_clusters_df["MinAF"] = merge_clusters_df["MinAF"].astype(float)
    merge_clusters_df["MaxAF"] = merge_clusters_df["MaxAF"].astype(float)
    merge_clusters_df["POS"] = merge_clusters_df["POS"].astype(int)
    merge_clusters_df = merge_clusters_df.sort_values(by=["POS"], ascending=[True])
    #merge_clusters_df.to_csv("{}merged_clusters_of_mutations.csv".format(save_path), index=False)
    utils.save_dataframe(merge_clusters_df, "{}merged_clusters_of_mutations.csv".format(save_path))
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
    same_pos_df["POS"] = same_pos_df["POS"].astype(int)
    same_pos_df = same_pos_df.sort_values(by=["POS"], ascending=[True])
    #same_pos_df.to_csv("{}same_pos_diff_ref_alt.csv".format(save_path), index=False)
    utils.save_dataframe(same_pos_df, "{}same_pos_diff_ref_alt.csv".format(save_path))
