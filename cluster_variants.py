import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import SpectralClustering

import utils

color_dict = {0: "red", 1: "green", 2: "blue"}
N_C = 3

def transform_variants(pred_test):
    train_count_var = utils.read_json("data/train_n_variants.json")
    sample_names = utils.read_json("data/samples_dict.json")
    n_samples = list()
    n_sample_variants = list()
    sample_var_summary = list()
    s_name_df = list()
    var_name_df = list()
    var_pos_df = list()
    var_x_df = list()
    var_y_df = list()
    var_af_df = list()
    x = 0
    for i, c_pred in enumerate(train_count_var):
        idx = list(c_pred.values())[0]
        sample_name = list(c_pred.keys())[0]
        sample_variants = sample_names[sample_name]
        n_sample_variants.append(len(sample_variants))
        annot, var_pos, var_name, var_af = utils.deserialize(sample_variants, sample_name)

        sample_var_summary.append(annot)
        n_samples.append(sample_name)
        x_val = pred_test[x: x + idx, 0]
        y_val = pred_test[x: x + idx:, 1]

        s_name_df.extend(np.repeat(sample_name, idx))
        var_name_df.extend(var_name)
        var_pos_df.extend(var_pos)
        var_x_df.extend(x_val)
        var_y_df.extend(y_val)
        var_af_df.extend(var_af)
        x = idx

    # save predicted df
    pred_df = pd.DataFrame(list(zip(s_name_df, var_name_df, var_pos_df, var_af, var_x_df, var_y_df)), columns=["sample_name", "variant", "POS", "AF", "x", "y"])
    pred_df.to_csv("data/predicted_var.csv")
    cluster(pred_test, var_name_df, s_name_df, var_name_df, var_pos_df, var_af_df)


def cluster(features, pt_annotations, n_samples, var_name_df, var_pos_df, var_af_df, path_plot_df="data/test_clusters.csv"):
    print("Clustering...")
    print(features)
    print(features.shape)
    svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
    low_dimensional_features = svd.fit_transform(features)
    print(low_dimensional_features.shape)
    kmeans = KMeans(n_clusters=10)
    #predict the labels of clusters
    cluster_labels = kmeans.fit_predict(low_dimensional_features)
    colors = list()
    x = list()
    y = list()
    for i, l in enumerate(cluster_labels):
        colors.append(int(l))
        pred_val = low_dimensional_features[i]
        x.append(pred_val[0])
        y.append(pred_val[1])
    scatter_df = pd.DataFrame(list(zip(n_samples, var_name_df, var_pos_df, var_af_df, x, y, pt_annotations, colors)), columns=["sample_name", "variant", "POS", "AF", "x", "y", "annotations", "clusters"])    
    scatter_df = scatter_df.sort_values(by="clusters")
    fig = px.scatter(scatter_df,
        x="x",
        y="y",
        color="clusters",
        hover_data=['annotations'],
        size=np.repeat(10, len(colors))
    )
    scatter_df.to_csv(path_plot_df)
    plotly.offline.plot(fig, filename='data/cluster_variants.html')
    fig.show()
