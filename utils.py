import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def save_as_json(filepath, data):
    with open(filepath, 'w') as fp:
        json.dump(data, fp)


def read_json(path):
    with open(path, 'r') as fp:
        f_content = json.loads(fp.readline())
        return f_content


def transform_integers(train_data, test_data):
    scaler = MinMaxScaler()
    tr_feature = feature_reshape(train_data[:, 0])
    scaler.fit(tr_feature)
    tr_feature_transformed = scaler.transform(tr_feature)
    train_data_transformed = np.hstack((tr_feature_transformed, train_data[:, 1:]))
    
    te_feature = feature_reshape(test_data[:, 0])
    te_feature_transformed = scaler.transform(te_feature)
    test_data_transformed = np.hstack((te_feature_transformed, test_data[:, 1:]))
    return train_data_transformed, test_data_transformed


def encode_integers(embedder, features):
    
    pos_reshape = feature_reshape(features[:, 0])
    qual_reshape = feature_reshape(features[:, 1])
    # transform POS integer to a vector
    pos_mat = embedder(pos_reshape)
    
    pos_mat = tf.clip_by_value(
        pos_mat, 0.0, 1.0, name=None
    )

    pos_mat = np.reshape(pos_mat, (pos_mat.shape[0], pos_mat.shape[2]))
    # transform QUAL integer to a vector
    qual_mat = embedder(qual_reshape)
    
    qual_mat = tf.clip_by_value(
        qual_mat, 0.0, 1.0, name=None
    )
    
    qual_mat = np.reshape(qual_mat, (qual_mat.shape[0], qual_mat.shape[2]))
    np.hstack((pos_mat, qual_mat, features[:, 2:]))
    return features


def feature_reshape(feature):     
    return np.reshape(feature, (feature.shape[0], 1))
