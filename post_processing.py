import json
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

color_dict = {0: "red", 1: "green", 2: "blue"}

def transform_predictions(pred_test):
    cluster(pred_test)


def cluster(features):

    #Initialize the class object
    kmeans = KMeans(n_clusters = 3)

    #predict the labels of clusters
    cluster_labels = kmeans.fit_predict(df)

    u_labels = np.unique(cluster_labels)
    
    colors = list()
    for l in cluster_labels:
        colors.append(color_dict[l])
        
    print(cluster_labels)
    
    print(colors)

    #plotting the results:
    '''for i in u_labels:
        plt.scatter(features[label == i , 0] , features[label == i , 1] , label = i)
    plt.legend()
    plt.show()'''
