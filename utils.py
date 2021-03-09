import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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
    
def deserialize(var_lst, sample_name):
    var_txt = ""
    var_pos = list()
    var_name = list()
    var_af = list()
    var_alt = list()
    var_ref = list()
    for i, item in enumerate(var_lst):
        var_split = item.split(">")
        pos, ref, alt, af = var_split[0], var_split[1], var_split[2], var_split[3]
        var_txt += "{}>{}>{}>{}>{} <br>".format(sample_name, pos, ref, alt, af)
        var_ref.append(ref)
        var_alt.append(alt)
        var_pos.append(pos)
        var_af.append(af)
        var_name.append("{}>{}>{}>{}>{} <br>".format(sample_name, pos, ref, alt, af))
    return var_txt, var_pos, var_name, var_af, var_ref, var_alt


def transform_integers(train_data):
    scaler = MinMaxScaler()
    tr_feature = feature_reshape(train_data[:, 0])
    tr_feature_transformed = scaler.fit_transform(tr_feature)
    train_data_transformed = np.hstack((tr_feature_transformed, train_data[:, 1:]))
    return train_data_transformed


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
