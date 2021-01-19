import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import utils

color_dict = {0: "red", 1: "green", 2: "blue"}

def deserialize(var_lst, sample_name):
    var_txt = ""
    #var_txt += "Sample name: {} <br>".format(sample_name)
    #var_txt += "Num of variants: {} <br>".format(str(len(var_lst)))
    for i, item in enumerate(var_lst):
        key = list(item.keys())[0]
        val = list(item.values())[0]
        var_txt += " {}->{} <br>".format(key, val)
    return var_txt

def transform_predictions(pred_test):
    test_count_var = utils.read_json("data/test_n_variants.json")
    sample_names = utils.read_json("data/samples.json")
    n_samples = list()
    n_sample_variants = list()
    sample_var_summary = list()
    summary_test_pred = np.zeros((len(test_count_var), pred_test.shape[1]))
    x = 0
    for i, c_pred in enumerate(test_count_var):
        idx = list(c_pred.values())[0]
        sample_name = list(c_pred.keys())[0]
        sample_variants = sample_names[sample_name]
        n_sample_variants.append(len(sample_variants))
        sample_var_summary.append(deserialize(sample_variants, sample_name))
        n_samples.append(sample_name)
        sliced_pred = pred_test
        x_val = sliced_pred[x: x + idx, 0] 
        y_val = sliced_pred[x: x + idx:, 1] 
        summary_test_pred[i,:] = [np.mean(x_val), np.mean(y_val)]
        x = idx
    cluster(summary_test_pred, sample_var_summary, n_samples, n_sample_variants)

def cluster(features, pt_annotations, n_samples, n_sample_variants, path_plot_df="data/test_clusters.csv"):

    #Initialize the class object
    kmeans = KMeans(n_clusters=len(color_dict))
    #predict the labels of clusters
    cluster_labels = kmeans.fit_predict(features)
    colors = list()
    x = list()
    y = list()
    for i, l in enumerate(cluster_labels):
        colors.append(int(l))
        pred_val = features[i]
        x.append(pred_val[0])
        y.append(pred_val[1])
    
    scatter_df = pd.DataFrame(list(zip(n_samples, x, y, n_sample_variants, pt_annotations, colors)), columns=["sample_name", "x", "y", "# variants", "annotations", "clusters"])
    
    scatter_df = scatter_df.sort_values(by="clusters")
    
    scatter_df.to_csv(path_plot_df)

    fig = px.scatter(scatter_df,
        x="x",
        y="y",
        color="clusters",
        hover_data=['annotations'],
        size=np.repeat(20, len(colors))
    )

    plotly.offline.plot(fig, filename='data/cluster_variants.html')

    fig.show()
