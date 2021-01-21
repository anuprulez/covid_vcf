import json
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import embed_integers

import utils


N_SEQ_ENCODER = [0.25, 0.5, 0.75, 1.0]

class TransformVariants:

    def __init__(self):
        """ Init method. """
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(np.array(['a','c','g','t','z']))

    def get_variants(self, samples, typ):
        print("Transforming variants...")
        sample_n_variants = list()
        encoded_samples = list()
        num_v_sample = list()
        pos_qual = list()
        for s_idx, sample in enumerate(samples):
            positions = list()
            variants = list()
            variants = dict()
            l_variants = samples[sample]
            if len(l_variants) > 0:
                #pos_qual.extend(self.get_POS_QUAL(l_variants))
                #embedder = self.embed_POS(pos_qual)
                transformed_variants = self.transform_variants(l_variants)
                encoded_samples.extend(transformed_variants)
                variants[sample] = len(transformed_variants)
                sample_n_variants.append(variants)
                num_v_sample.append(transformed_variants.shape[0])
        assert np.sum(num_v_sample) == len(encoded_samples)
        print("Num transformed rows for {} samples: {}".format(str(s_idx + 1), str(len(encoded_samples))))

        utils.save_as_json("data/{}_n_variants.json".format(typ), sample_n_variants)
        return encoded_samples


    def string_to_array(self, n_seq):
        n_seq = n_seq.lower()
        n_seq = re.sub('[^acgt]', 'z', n_seq)
        n_seq_arr = np.array(list(n_seq))
        return n_seq_arr  


    def encode_nucleotides(self, seq, encoded_AGCT=N_SEQ_ENCODER):
        encoded_seq = self.ordinal_encoder(self.string_to_array(seq))
        return encoded_seq


    def ordinal_encoder(self, my_array):
        integer_encoded = self.label_encoder.transform(my_array)
        float_encoded = integer_encoded.astype(float)
        float_encoded[float_encoded == 0] = 0.25 # A
        float_encoded[float_encoded == 1] = 0.50 # C
        float_encoded[float_encoded == 2] = 0.75 # G
        float_encoded[float_encoded == 3] = 1.00 # T
        float_encoded[float_encoded == 4] = 0.00 # anything else
        return float_encoded


    def transform_variants(self, variants, n_features=3, max_len_ref=10, max_len_alt=5):
        encoded_sample = np.zeros((len(variants), max_len_ref + max_len_alt + 3))
        for index, item in enumerate(variants):
            pos = list(item.keys())[0]
            var = list(item.values())[0]
            ref_var = var.split(">")
            ref, alt_1, qual, allel_freq = ref_var[0], ref_var[1], ref_var[2], ref_var[3]
            encoded_sample[index, 0:1] = [pos]
            encoded_sample[index, 1:2] = [qual]
            encoded_sample[index, 2:3] = [allel_freq]
            if len(ref) <= max_len_ref and len(alt_1) <= max_len_alt:
                encoded_ref = self.encode_nucleotides(ref, max_len_ref)
                encoded_alt = self.encode_nucleotides(alt_1, max_len_alt)
                n_e_ref = np.concatenate((encoded_ref, np.zeros(max_len_ref - len(encoded_ref))), axis=None)
                n_e_alt = np.concatenate((encoded_alt, np.zeros(max_len_alt - len(encoded_alt))), axis=None)
                encoded_sample[index, 3:max_len_ref + 3] = n_e_ref
                encoded_sample[index, max_len_ref + 3: max_len_ref + max_len_alt + 3] = n_e_alt
        return encoded_sample
    '''def collect_POS_QUAL(self, samples):
        pos_qual = list()
        for s_idx, sample in enumerate(samples):
            l_variants = samples[sample]
            if len(l_variants) > 0:
                pos_qual.extend(self.get_POS_QUAL(l_variants))
        autoencoder = self.embed_POS(pos_qual)


    def get_POS_QUAL(self, variants):
        samples_POS_QUAL = list()
        for index, item in enumerate(variants):
            pos = list(item.keys())[0]
            var = list(item.values())[0]
            ref_var = var.split(">")
            ref, alt_1, qual, allel_freq = ref_var[0], ref_var[1], ref_var[2], ref_var[3]
            samples_POS_QUAL.append(pos)
            samples_POS_QUAL.append(int(float(qual)))
        return samples_POS_QUAL


    def embed_POS(self, POS_QUAL, num_epochs=10):

        split_num = int(0.2 * len(POS_QUAL))
        batch_size = 32
        #POS_QUAL = np.unique(POS_QUAL)
        training_features = POS_QUAL[split_num:]
        test_features = POS_QUAL[:split_num]
        
        training_features = np.asarray(training_features)

        test_features = np.asarray(test_features)
        
        training_features = training_features.astype('int32')

        test_features = test_features.astype('int32')
        
        training_features = np.reshape(training_features, (training_features.shape[0], 1))
        test_features = np.reshape(test_features, (test_features.shape[0], 1))
        
        print(training_features.shape)
        print(test_features.shape)

        tr_epo_loss = np.zeros((num_epochs, 1))
        te_epo_loss = np.zeros((num_epochs, 1))

        autoencoder = embed_integers.IntegerAutoencoder()
        optimizer = tf.optimizers.Adam(learning_rate=0.001)
        global_step = tf.Variable(0)
        
        print("Training integer embedder...")

        for epoch in range(num_epochs):
            tr_loss = 0.0
            te_loss = 0.0
            for x in range(0, len(training_features), batch_size):
                x_inp = training_features[x : x + batch_size]
                #print(x_inp)
                #print()
                loss_value, grads, reconstruction = autoencoder.grad(autoencoder, x_inp)
                #print(reconstruction)
                optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables), global_step)
                c_tr_loss = np.mean(autoencoder.loss(x_inp, reconstruction).numpy())
                c_te_loss = np.mean(autoencoder.loss(test_features, autoencoder(test_features)).numpy())
                tr_loss += c_tr_loss
                te_loss += c_te_loss
            mean_tr_loss = tr_loss / batch_size
            mean_te_loss = te_loss / batch_size
            tr_epo_loss[epoch] = mean_tr_loss
            te_epo_loss[epoch] = mean_te_loss
            print("Epoch {} training loss: {}".format(epoch + 1, str(np.round(mean_tr_loss, 4))))
            print("Epoch {} test loss: {}".format(epoch + 1, str(np.round(mean_te_loss, 4))))
            print()
        return autoencoder'''
