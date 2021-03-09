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

import utils

color_dict = {0: "red", 1: "green", 2: "blue"}
N_C = 3

def transform_variants(samples_data):
    detailed_samples = utils.read_json("data/train_n_variants.json")
    sample_names = utils.read_json("data/train_all_variants.json")
    n_samples = list()
    s_name_df = list()
    var_name_df = list()
    var_pos_df = list()
    var_af_df = list()
    var_ref_df = list()
    var_alt_df = list()
    for i, sample_count in enumerate(detailed_samples):
        idx = int(list(sample_count.values())[0])
        sample_name = list(sample_count.keys())[0]
        sample_variants = sample_names[sample_name]
        _, var_pos, var_name, var_af, var_ref, var_alt = utils.deserialize(sample_variants, sample_name)
        s_name_df.extend(np.repeat(sample_name, idx))
        var_name_df.extend(var_name)
        var_pos_df.extend(var_pos)
        var_af_df.extend(var_af)
        var_ref_df.extend(var_ref)
        var_alt_df.extend(var_alt)
        #print(np.repeat(sample_name, idx), var_pos, var_name, var_af)
        #print(var_name_df)
        #print("---------------")
    #print(s_name_df, var_name_df, var_pos_df, var_af_df)
    #print(len(s_name_df), len(var_name_df), len(var_pos_df), len(var_af_df))
    cluster(samples_data, s_name_df, var_name_df, var_pos_df, var_af_df, var_ref_df, var_alt_df)


def cluster(features, s_name_df, var_name_df, var_pos_df, var_af_df, var_ref_df, var_alt_df, path_plot_df="data/test_clusters.csv"):
    print("Clustering...")
    #print(features.shape)
    #print(s_name_df, var_name_df, var_pos_df, var_af_df)
    decomposition = NMF(n_components=2, max_iter=500, init='nndsvd')
    #decomposition = TruncatedSVD(n_components=2, n_iter=5, random_state=42)
    #decomposition = FastICA(n_components=2)
    #decomposition = SpectralEmbedding(n_components=2)
    #decomposition = SparsePCA(n_components=2)
    low_dimensional_features = decomposition.fit_transform(features)
    
    #predict the labels of clusters
    clustering_type = MiniBatchKMeans(n_clusters=100)
    cluster_labels = clustering_type.fit_predict(low_dimensional_features)
    #print(len(cluster_labels))
    # cluster_labels = DBSCAN(eps=0.9).fit_predict(low_dimensional_features)
    # AgglomerativeClustering(n_clusters=10).fit_predict(low_dimensional_features) 
    # DBSCAN(eps=0.5).fit_predict(features)
    clusters = list()
    x = list()
    y = list()
    for i, l in enumerate(cluster_labels):
        if l >= 0:
            pred_val = low_dimensional_features[i]
            x.append(pred_val[0])
            y.append(pred_val[-1])
            clusters.append(l)
    print(len(clusters))
    scatter_df = pd.DataFrame(list(zip(s_name_df, var_ref_df, var_pos_df, var_alt_df, var_af_df, clusters, x, y, var_name_df)), columns=["Sample name", "REF", "POS", "ALT", "AF",  "Cluster", "x", "y", "annotations"])
    scatter_df["Cluster"] = scatter_df["Cluster"].astype(str)
    fig = px.scatter(scatter_df,
        x="x",
        y="y",
        color="Cluster",
        hover_data=['annotations'],
        size=np.repeat(1, len(clusters))
    )
    
    scatter_df = scatter_df.drop(["x", "y", "annotations"], axis=1)
    scatter_df["Cluster"] = scatter_df["Cluster"].astype(int)
    sorted_df = scatter_df.sort_values(by=["Cluster", "REF", "ALT", "POS", "AF"])
    sorted_df.to_csv(path_plot_df)
    plotly.offline.plot(fig, filename='data/cluster_variants.html')
    #fig.show()
