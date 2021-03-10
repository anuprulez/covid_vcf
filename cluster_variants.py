import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import TruncatedSVD, FastICA, SparsePCA, NMF
from sklearn.manifold import SpectralEmbedding
from sklearn import metrics


import utils

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
    cluster(samples_data, s_name_df, s_idx_df, var_name_df, var_pos_df, var_af_df, var_ref_df, var_alt_df, samples_name_idx, BOSTON_DATA_PATH)


def find_optimal_clusters(low_dimensional_features, number_iter=10, cluster_step=10, initial_clusters=200):
    clustering_perf = dict()
    for i in range(0, number_iter):
        print("Clustering iteration: {}/{}".format(str(i + 1), str(number_iter)))
        clustering_type = MiniBatchKMeans(n_clusters=initial_clusters, random_state=32) # reassignment_ratio=200.0
        cluster_labels = clustering_type.fit_predict(low_dimensional_features)
        print("Number of unique clusters: {}".format(str(len(list(set(cluster_labels))))))
        cluster_silhouette_score = metrics.silhouette_score(low_dimensional_features, cluster_labels, metric='euclidean')
        print("Silhouette score: {}".format(str(np.round(cluster_silhouette_score, 2))))
        clustering_perf[i] = (cluster_silhouette_score, cluster_labels, initial_clusters)
        initial_clusters += cluster_step
        print()
    best_perf = 0.0
    idx = 0
    for key in clustering_perf:
        if clustering_perf[key][0] > best_perf:
            best_perf = clustering_perf[key][0]
            idx = key
    best_clustering = clustering_perf[idx]
    print(best_clustering[0], best_clustering[2])
    return best_clustering[1], best_clustering[0]

def cluster(features, s_name_df, s_idx_df, var_name_df, var_pos_df, var_af_df, var_ref_df, var_alt_df, samples_name_idx, BOSTON_DATA_PATH, path_plot_df="data/test_clusters.csv"):
    print("Clustering...")
    print(features.shape)
    #print(s_name_df, var_name_df, var_pos_df, var_af_df)
    decomposition = NMF(n_components=2, max_iter=10000, init='nndsvda', random_state=32)
    #decomposition = TruncatedSVD(n_components=2, n_iter=5, random_state=42)
    #decomposition = FastICA(n_components=2)
    #decomposition = SpectralEmbedding(n_components=2)
    #decomposition = SparsePCA(n_components=2)
    low_dimensional_features = decomposition.fit_transform(features)
    
    #predict the labels of clusters
    cluster_labels, _ = find_optimal_clusters(low_dimensional_features)
    
    print(len(cluster_labels))
    # cluster_labels = DBSCAN(eps=0.9).fit_predict(low_dimensional_features)
    # AgglomerativeClustering(n_clusters=10).fit_predict(low_dimensional_features) 
    # DBSCAN(eps=0.5).fit_predict(features)
    clusters = list()
    x = list()
    y = list()
    for i, l in enumerate(cluster_labels):
        pred_val = low_dimensional_features[i]
        x.append(pred_val[0])
        y.append(pred_val[-1])
        clusters.append(l)

    scatter_df = pd.DataFrame(list(zip(s_name_df, s_idx_df, var_ref_df, var_pos_df, var_alt_df, var_af_df, clusters, x, y, var_name_df)), columns=["Sample", "Index", "REF", "POS", "ALT", "AF",  "Cluster", "x", "y", "annotations"], index=None)
    scatter_df["Cluster"] = scatter_df["Cluster"].astype(str)
    
    clean_scatter_df = utils.remove_single_mutation(scatter_df, "POS")
    
    fig = px.scatter(clean_scatter_df,
        x="x",
        y="y",
        color="Cluster",
        hover_data=['annotations'],
        size=np.repeat(1, len(clean_scatter_df))
    )
    
    clean_scatter_df = clean_scatter_df.drop(["x", "y", "annotations"], axis=1)
    clean_scatter_df["Cluster"] = clean_scatter_df["Cluster"].astype(int)
    clean_scatter_df = clean_scatter_df.sort_values(by=["Cluster", "Index", "REF", "ALT", "POS", "AF"])
    clean_scatter_df.to_csv(path_plot_df)
    
    #utils.reconstruct_with_original(sorted_df, BOSTON_DATA_PATH)
    
    #plotly.offline.plot(fig, filename='data/cluster_variants.html')
    #fig.show()
