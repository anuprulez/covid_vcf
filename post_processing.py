import json
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import utils

color_dict = {0: "red", 1: "green", 2: "blue"}


def transform_predictions(pred_test):
    #cluster(pred_test)
    test_count_var = utils.read_json("data/test_n_variants.json")
    summary_test_pred = np.zeros((len(test_count_var), pred_test.shape[1]))
    print(test_count_var)
    for c_pred in test_count_var:
        
    
    


def cluster(features):

    #Initialize the class object
    kmeans = KMeans(n_clusters=3)

    #predict the labels of clusters
    cluster_labels = kmeans.fit_predict(features)

    u_labels = np.unique(cluster_labels)
    
    colors = list()
    x = list()
    y = list()
    for i, l in enumerate(cluster_labels):
        colors.append(color_dict[l])
        pred_val = features[i].numpy()
        x.append(pred_val[0])
        y.append(pred_val[1])

    plt.scatter(x, y, color=colors)
    plt.grid(True)
    plt.show()
