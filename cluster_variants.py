import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, FeatureAgglomeration, AgglomerativeClustering, DBSCAN, MeanShift
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
    cluster_mutations(samples_data, s_name_df, s_idx_df, var_name_df, var_pos_df, var_af_df, var_ref_df, var_alt_df, samples_name_idx, BOSTON_DATA_PATH)


def find_optimal_clusters(features, number_iter=1, cluster_step=20, initial_clusters=220):
    clustering_perf = dict()
    for i in range(0, number_iter):
        print("Clustering iteration: {}/{}".format(str(i + 1), str(number_iter)))
        
        clustering_type = KMeans(n_clusters=initial_clusters, tol=0.0, algorithm='full', random_state=32) # max_iter=500, n_init=250
        
        #FeatureAgglomeration(n_clusters=initial_clusters, linkage='complete')
        
        #KMeans(n_clusters=initial_clusters, precompute_distances=True, tol=0.0, algorithm='full', random_state=32) 
        
        #MiniBatchKMeans(n_clusters=initial_clusters, random_state=32)
        
        cluster_labels = clustering_type.fit_predict(features) #clustering_type.fit_predict(features)

        #print(clustering_type.cluster_centers_)
        
        print("Sum of squared distances of samples to their closest cluster center: {}".format(str(np.round(clustering_type.inertia_, 4))))
        
        print("Number of unique clusters: {}".format(str(len(list(set(cluster_labels))))))
        cluster_silhouette_score = metrics.silhouette_score(features, cluster_labels, metric='euclidean')
        print("Silhouette score: {}".format(str(cluster_silhouette_score)))
        print()
        clustering_perf[i] = (cluster_silhouette_score, cluster_labels, initial_clusters)
        initial_clusters += cluster_step
    best_perf = 0.0
    idx = 0
    for key in clustering_perf:
        if clustering_perf[key][0] > best_perf:
            best_perf = clustering_perf[key][0]
            idx = key
    best_clustering = clustering_perf[idx]
    print(best_clustering[0], best_clustering[2])
    return best_clustering[1], best_clustering[0]


def cluster_samples(mutations_df, mutations_labels):
    u_mutation_labels = list(set(mutations_labels))
    clusters_samples = list()
    max_n_samples = 0
    for c_label in u_mutation_labels:
        cluster_rows = mutations_df[mutations_df["Cluster"] == c_label]
        if len(cluster_rows) > 0:
            s_dict = cluster_rows["Index"].to_dict()
            l_samples = list(s_dict.values())
            clusters_samples.append({c_label: l_samples})
            len_samples = len(l_samples)
            if len_samples > max_n_samples:
                max_n_samples = len_samples
    # make sizes of list of samples uniform to enable clustering
    mat_samples = np.zeros((len(clusters_samples), max_n_samples))
    for i, item in enumerate(clusters_samples):
        l_samples = list(item.values())[0]
        mat_samples[i, :len(l_samples)] = l_samples

    clustering_type = KMeans(n_clusters=10, tol=0.0, algorithm='full', random_state=32)
    cluster_labels = clustering_type.fit_predict(mat_samples)
    
    mat_samples_df = pd.DataFrame(mat_samples)
    mat_samples_df['Cluster'] = cluster_labels
    mat_samples_df["Cluster"] = mat_samples_df["Cluster"].astype(str)
    
    for c in list(set(cluster_labels)):
        rows = mat_samples_df[mat_samples_df["Cluster"] == str(c)]
        if len(rows) > 0:
            print(c)
            print(rows)
            print()
    
    sorted_mat_samples_df = mat_samples_df.sort_values(by=["Cluster"])
    sorted_mat_samples_df.to_csv("data/cluster_samples.csv")
    


def cluster_mutations(features, s_name_df, s_idx_df, var_name_df, var_pos_df, var_af_df, var_ref_df, var_alt_df, samples_name_idx, BOSTON_DATA_PATH, path_plot_df="data/test_clusters.csv"):
    print("Clustering...")
    print(features.shape)
    #print(s_name_df, var_name_df, var_pos_df, var_af_df)
    #decomposition = NMF(n_components=2, max_iter=1000, init='nndsvda', alpha=0.0001, l1_ratio=0.5, random_state=32) #tol=0.00000001,
    #decomposition = TruncatedSVD(n_components=2, n_iter=5, random_state=42)
    #decomposition = FastICA(n_components=2)
    #decomposition = SpectralEmbedding(n_components=2)
    #decomposition = SparsePCA(n_components=2)
    #low_dimensional_features = features #decomposition.fit_transform(features)
    
    #print(decomposition.reconstruction_err_, decomposition.n_iter_)
    
    #import sys
    #sys.exit()
    
    #predict the labels of clusters
    cluster_labels, _ = find_optimal_clusters(features)
    
    print(len(cluster_labels))
    # cluster_labels = DBSCAN(eps=0.9).fit_predict(low_dimensional_features)
    # AgglomerativeClustering(n_clusters=10).fit_predict(low_dimensional_features) 
    # DBSCAN(eps=0.5).fit_predict(features)
    #clusters = list()
    #x = list()
    #y = list()
    #for i, l in enumerate(cluster_labels):
        #pred_val = low_dimensional_features[i]
        #x.append(pred_val[0])
        #y.append(pred_val[-1])
        #clusters.append(l)

    #scatter_df = pd.DataFrame(list(zip(s_name_df, s_idx_df, var_ref_df, var_pos_df, var_alt_df, var_af_df, clusters, x, y, var_name_df)), columns=["Sample", "Index", "REF", "POS", "ALT", "AF",  "Cluster", "x", "y", "annotations"], index=None)
    scatter_df = pd.DataFrame(list(zip(s_name_df, s_idx_df, var_ref_df, var_pos_df, var_alt_df, var_af_df, cluster_labels, var_name_df)), columns=["Sample", "Index", "REF", "POS", "ALT", "AF",  "Cluster", "annotations"], index=None)
    scatter_df["Cluster"] = scatter_df["Cluster"].astype(str)
    
    clean_scatter_df = utils.remove_single_mutation(scatter_df, "POS")
    
    '''fig = px.scatter(clean_scatter_df,
        x="x",
        y="y",
        color="Cluster",
        hover_data=['annotations'],
        size=np.repeat(1, len(clean_scatter_df))
    )'''
    
    #clean_scatter_df = clean_scatter_df.drop(["x", "y", "annotations"], axis=1)
    clean_scatter_df = clean_scatter_df.drop(["annotations"], axis=1)
    clean_scatter_df["Cluster"] = clean_scatter_df["Cluster"].astype(int)
    clean_scatter_df = clean_scatter_df.sort_values(by=["Cluster", "Index", "REF", "ALT", "POS", "AF"])
    clean_scatter_df.to_csv(path_plot_df)
    
    cluster_samples(clean_scatter_df, cluster_labels)
    
    #utils.reconstruct_with_original(sorted_df, BOSTON_DATA_PATH)
    
    #plotly.offline.plot(fig, filename='data/cluster_variants.html')
    #fig.show()
