import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans

import utils

color_dict = {0: "red", 1: "green", 2: "blue"}

def deserialize(var_lst, sample_name):
    var_txt = ""
    var_pos = list()
    var_name = list()
    var_qual = list()
    var_af = list()
    for i, item in enumerate(var_lst):
        key = list(item.keys())[0]
        val = list(item.values())[0]
        
        var_split = val.split(">")
        ref, alt, qual, af = var_split[0], var_split[1], var_split[2], var_split[3]
        
        var_txt += "{}>{}>{}>{}>{} <br>".format(key, ref, alt, qual, af)
        
        var_pos.append(key)
        var_qual.append(qual)
        var_af.append(af)
        var_name.append("{}>{}>{} <br>".format(key, ref, alt))
    return var_txt, var_pos, var_name, var_qual, var_af

def transform_predictions(pred_test):
    test_count_var = utils.read_json("data/test_n_variants.json")
    sample_names = utils.read_json("data/samples.json")
    n_samples = list()
    n_sample_variants = list()
    sample_var_summary = list()
    summary_test_pred = np.zeros((len(test_count_var), pred_test.shape[1]))
    
    s_name_df = list()
    var_name_df = list()
    var_pos_df = list()
    var_x_df = list()
    var_y_df = list()
    var_qual_df = list()
    var_af_df = list()
    x = 0
    for i, c_pred in enumerate(test_count_var):
        idx = list(c_pred.values())[0]
        sample_name = list(c_pred.keys())[0]
        sample_variants = sample_names[sample_name]
        n_sample_variants.append(len(sample_variants))
        annot, var_pos, var_name, var_qual, var_af = deserialize(sample_variants, sample_name)

        sample_var_summary.append(annot)
        n_samples.append(sample_name)
        x_val = pred_test[x: x + idx, 0] 
        y_val = pred_test[x: x + idx:, 1]
        summary_test_pred[i,:] = [np.mean(x_val), np.mean(y_val)]

        s_name_df.extend(np.repeat(sample_name, idx))
        var_name_df.extend(var_name)
        var_pos_df.extend(var_pos)
        var_x_df.extend(x_val.numpy())
        var_y_df.extend(y_val.numpy())
        var_qual_df.extend(var_qual)
        var_af_df.extend(var_af)
        x = idx
    
    # save predicted df
    pred_df = pd.DataFrame(list(zip(s_name_df, var_name_df, var_pos_df, var_qual, var_af, var_x_df, var_y_df)), columns=["sample_name", "variant", "POS", "Qual", "AF", "x", "y"])
    pred_df.to_csv("data/predicted_var.csv")
    
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
