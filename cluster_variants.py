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
    n_sample_variants = list()
    s_name_df = list()
    var_name_df = list()
    var_pos_df = list()
    var_encoded_df = list()
    var_af_df = list()
    for i, sample_count in enumerate(detailed_samples):
        idx = int(list(sample_count.values())[0])
        sample_name = list(sample_count.keys())[0]
        sample_variants = sample_names[sample_name]
        n_sample_variants.append(len(sample_variants))
        _, var_pos, var_name, var_af = utils.deserialize(sample_variants, sample_name)
        s_name_df.extend(np.repeat(sample_name, idx))
        var_name_df.extend(var_name)
        var_pos_df.extend(var_pos)
        var_af_df.extend(var_af)
        x = x + idx
    cluster(samples_data, var_name_df, s_name_df, var_name_df, var_pos_df, var_af_df)


def cluster(features, pt_annotations, n_samples, var_name_df, var_pos_df, var_af_df, path_plot_df="data/test_clusters.csv"):
    print("Clustering...")
    print(features.shape)
    decomposition = NMF(n_components=2, iter=1000, init='nndsvd')
    #decomposition = TruncatedSVD(n_components=2, n_iter=5, random_state=42)
    #decomposition = FastICA(n_components=2)
    #decomposition = SpectralEmbedding(n_components=2)
    #decomposition = SparsePCA(n_components=2)
    low_dimensional_features = decomposition.fit_transform(features)
    print(low_dimensional_features.shape)
    
    #predict the labels of clusters
    clustering_type = MiniBatchKMeans(n_clusters=10)
    cluster_labels = clustering_type.fit_predict(low_dimensional_features)
    
    # cluster_labels = DBSCAN(eps=0.9).fit_predict(low_dimensional_features)
    # AgglomerativeClustering(n_clusters=10).fit_predict(low_dimensional_features) 
    # DBSCAN(eps=0.5).fit_predict(features)
    clusters = list()
    x = list()
    y = list()
    for i, l in enumerate(cluster_labels):
        if l >= 0:
            clusters.append(l)
            pred_val = low_dimensional_features[i]
            x.append(pred_val[0])
            y.append(pred_val[-1])
    scatter_df = pd.DataFrame(list(zip(n_samples, var_name_df, var_pos_df, var_af_df, x, y, pt_annotations, clusters)), columns=["sample_name", "variant", "POS", "AF", "x", "y", "annotations", "clusters"])
    scatter_df["clusters"] = scatter_df["clusters"].astype(str)
    fig = px.scatter(scatter_df,
        x="x",
        y="y",
        color="clusters",
        hover_data=['annotations'],
        size=np.repeat(1, len(clusters))
    )
    scatter_df = scatter_df.sort_values(by="clusters")
    scatter_df.to_csv(path_plot_df)
    plotly.offline.plot(fig, filename='data/cluster_variants.html')
    #fig.show()
