import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def save_as_json(filepath, data):
    with open(filepath, 'w') as fp:
        json.dump(data, fp)


def read_json(path):
    with open(path, 'r') as fp:
        f_content = json.loads(fp.readline())
        return f_content


def include_mutations(mutations, include_list):
    f_mutations = list()
    for clade in mutations:
        if clade in include_list:
            f_mutations.extend(mutations[clade])
    return f_mutations


def deserialize(var_lst, sample_name, samples_name_idx):
    var_txt = ""
    var_sidx = list()
    var_pos = list()
    var_name = list()
    var_af = list()
    var_alt = list()
    var_ref = list()
    for i, item in enumerate(var_lst):
        var_split = item.split(">")
        s_idx, pos, ref, alt, af = var_split[0], var_split[1], var_split[2], var_split[3], var_split[4]
        var_txt += "{}>{}>{}>{}>{}>{} <br>".format(sample_name, s_idx, pos, ref, alt, af)
        var_sidx.append(samples_name_idx[s_idx])
        var_ref.append(ref)
        var_alt.append(alt)
        var_pos.append(pos)
        var_af.append(af)
        var_name.append("{}>{}>{}>{}>{}>{} <br>".format(sample_name, s_idx, pos, ref, alt, af))
    return var_txt, var_sidx, var_pos, var_name, var_af, var_ref, var_alt


def transform_integers(train_data):
    print(train_data)
    scaler = RobustScaler(with_centering=False)
    #tr_feature_sname = feature_reshape(train_data[:, 0])
    tr_feature_pos = feature_reshape(train_data[:, 0])
    #transformed_sname = scaler.fit_transform(tr_feature_sname)
    transformed_pos = scaler.fit_transform(tr_feature_pos)
    #train_data_transformed = np.hstack((transformed_sname, transformed_pos, train_data[:, 2:]))
    train_data_transformed = np.hstack((transformed_pos, train_data[:, 1:]))
    print(train_data_transformed)
    return train_data_transformed


def reconstruct_with_original(cluster_df, original_file):
    original_df = pd.read_csv(original_file, sep="\t")
    for idx in range(len(cluster_df)):
        sample_row = cluster_df.take([idx])
        sample_name = sample_row["Sample"].values[0]
        sample_pos = sample_row["POS"].values[0]
        #original_df_samples = original_df[original_df["Sample"] == sample_name and ]
        #for idy in range(len(original_df)):


def remove_single_mutation(dataframe, key):
    frequency_pos = dataframe[key].value_counts().to_dict()
    #print(frequency_pos)
    single_freq = list()
    #dataframe = dataframe[~dataframe[key].value_counts() == 1] 
    #return dataframe
    for key in frequency_pos:
        value = int(frequency_pos[key])
        if value == 1:
            dataframe = dataframe[~((dataframe['REF'] == key[0]) & (dataframe['POS'] == key[1]) & (dataframe['ALT'] == key[2]))]
    return dataframe
    '''print(key)
    if frequency_pos[key] == 1:
    single_freq.append(key)
    return dataframe[~dataframe[key].isin(single_freq)]'''

def set_serial_cluster_numbers(cluster_labels):
    cluster_labels = cluster_labels.tolist()
    u_clusters = np.sort(list(set(cluster_labels)))
    ordered_labels = list()
    replacement_indices = dict()
    for i, c_label in enumerate(u_clusters):
        replacement_indices[c_label] = i
    for c_label in cluster_labels:
        ordered_labels.append(replacement_indices[c_label])
    return ordered_labels
    
def plot_mat(samples_distance_matrix):
    plt.matshow(samples_distance_matrix)
    plt.colorbar()
    plt.show()
    
def clean_cluster(cluster, new_cluster_num):
    
    cluster = cluster.split("\n")[1:]
    cluster = cluster[0:len(cluster)-1]
    cleaned_cluster = list()
    for item in cluster:
        item = item.split(",")
        l = len(item)
        clean_row = item[1:]
        clean_row.append(str(new_cluster_num))
        cleaned_cluster.append(clean_row)
    return cleaned_cluster
                        

def feature_reshape(feature):     
    return np.reshape(feature, (feature.shape[0], 1))


def read_txt(file_path):
    with open(file_path, "r") as fr:
        data = fr.read()
    loss = list()
    for ln in data.split("\n"):
        if ln not in ['']:
            loss.append(float(ln))
    return loss


def get_var(item, POS_AF=2, REF_DIM=10):
    ref = item[POS_AF: POS_AF + REF_DIM]
    alt = item[POS_AF + REF_DIM:]
    get_ref_non_0 = ",".join([str(i) for i in ref[np.where(ref!=0)]])
    get_alt_non_0 = ",".join([str(i) for i in alt[np.where(alt!=0)]])
    var_name = "{}>{}".format(get_ref_non_0, get_alt_non_0)
    return var_name
