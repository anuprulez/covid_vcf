import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import utils

color_dict = {0: "red", 1: "green", 2: "blue"}
N_C = 5 #len(color_dict)

def pre_viz(samples):
    POS = dict()
    variants = dict()
    afs = dict()
    quality = dict()
    for sample in samples:
        l_variants = samples[sample]
        for index, item in enumerate(l_variants):
            ref_var = item.split(">")
            pos, ref, alt_1, allel_freq = int(ref_var[0]), ref_var[1], ref_var[2], float(ref_var[3])
            if not pos in POS:
                POS[pos] = 0
            POS[pos] += 1
            
            var = "{}>{}".format(ref, alt_1)
            if not var in variants:
                variants[var] = 0
            variants[var] += 1
            
            if not allel_freq in afs:
                afs[allel_freq] = 0
            afs[allel_freq] += 1
            
    _ = plot_freq(POS, "POS", "Frequency of variant positions across samples", "data/pos_freq")
    qty_var_df = plot_freq(variants, "Variants", "Frequency of variants across samples", "data/var_freq")
    _ = plot_freq(afs, "AF", "Frequency of AF across samples", "data/afs_freq")
    return qty_var_df
            

def plot_freq(qty_freq, f_name=None, title=None, file_name=None):
    qty, count = qty_freq.keys(), qty_freq.values()
    qty_freq_df = pd.DataFrame(list(zip(qty, count)), columns=[f_name, "Count"])
    
    qty_freq_df = qty_freq_df.sort_values(by=f_name)
    
    print(qty_freq_df)
    
    fig = px.bar(qty_freq_df, x=f_name, y='Count',
        title=title,
        color_discrete_sequence =['blue']*len(qty_freq_df)
    )
    
    f_html = "{}.html".format(file_name)
    f_csv = "{}.csv".format(file_name)
    
    qty_freq_df.to_csv(f_csv)
    
    fig.update_traces(marker_color='blue')
    plotly.offline.plot(fig, filename=f_html)
    fig.show()
    return qty_freq_df
    
def get_layout(plt_title):
    layout = go.Layout(xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text='Features',
        )),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text='Number of samples',
        )
    ),
        title=plt_title
    )
    
    return layout
    
def plot_true_pred(true, predicted):
    
    fig_true = go.Figure(data=go.Heatmap(
                   z=true,
                   hoverongaps = False), layout=get_layout("True test samples"))
    fig_true.show()
    
    fig_pred = go.Figure(data=go.Heatmap(
                   z=predicted,
                   hoverongaps = False), layout=get_layout("Predicted test samples"))
    fig_pred.show()
    plotly.offline.plot(fig_true, filename="data/true_matrix.html")
    plotly.offline.plot(fig_pred, filename="data/predicted_matrix.html")

def deserialize(var_lst, sample_name):
    var_txt = ""
    var_pos = list()
    var_name = list()
    var_af = list()
    for i, item in enumerate(var_lst):
        var_split = item.split(">")
        pos, ref, alt, af = var_split[0], var_split[1], var_split[2], var_split[3]
        var_txt += "{}>{}>{}>{}>{} <br>".format(sample_name, pos, ref, alt, af)
        var_pos.append(pos)
        var_af.append(af)
        var_name.append("{}>{}>{}>{}>{} <br>".format(sample_name, pos, ref, alt, af))
    return var_txt, var_pos, var_name, var_af

def transform_predictions(pred_test):
    test_count_var = utils.read_json("data/test_n_variants.json")
    sample_names = utils.read_json("data/samples_dict.json")
    n_samples = list()
    n_sample_variants = list()
    sample_var_summary = list()
    pred_test = pred_test.numpy()
    s_name_df = list()
    var_name_df = list()
    var_pos_df = list()
    var_x_df = list()
    var_y_df = list()
    var_af_df = list()
    x = 0
    for i, c_pred in enumerate(test_count_var):
        idx = list(c_pred.values())[0]
        sample_name = list(c_pred.keys())[0]
        sample_variants = sample_names[sample_name]
        n_sample_variants.append(len(sample_variants))
        annot, var_pos, var_name, var_af = deserialize(sample_variants, sample_name)

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
    #Initialize the class object
    kmeans = KMeans(n_clusters=N_C)

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
    
    scatter_df = pd.DataFrame(list(zip(n_samples, var_name_df, var_pos_df, var_af_df, x, y, pt_annotations, colors)), columns=["sample_name", "variant", "POS", "AF", "x", "y", "annotations", "clusters"])    
    scatter_df = scatter_df.sort_values(by="clusters")
    fig = px.scatter(scatter_df,
        x="x",
        y="y",
        color="clusters",
        hover_data=['annotations'],
        size=np.repeat(20, len(colors))
    )
    scatter_df.to_csv(path_plot_df)
    plotly.offline.plot(fig, filename='data/cluster_variants.html')

    fig.show()
    
def plot_loss(losses, title):
    epochs = np.arange(1, len(losses))
    df_loss = pd.DataFrame(list(zip(epochs, losses)), columns=["Epochs", "Loss"])    
    fig = px.line(df_loss, x="Epochs", y="Loss", title=title)
    plotly.offline.plot(fig, filename='data/{}.html'.format(title))
    fig.show()
    
def plot_losses():
    tr_loss = utils.read_txt("data/train_loss.txt")
    te_loss = utils.read_txt("data/test_loss.txt")
    print(tr_loss)
    print(te_loss)
    plot_loss(tr_loss, "Train loss")
    plot_loss(te_loss, "Test loss")   
