import json
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

import utils


N_SEQ_ENCODER = [0.25, 0.5, 0.75, 1.0]

class TransformVariants:

    def __init__(self):
        """ Init method. """
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(np.array(['a','c','g','t','z']))

    def get_variants(self, samples, samples_name_idx, typ):
        print("Transforming variants...")
        sample_n_variants = list()
        encoded_samples = list()
        num_v_sample = list()
        samples_save = dict()
        for s_idx, sample in enumerate(samples):
            positions = list()
            variants = list()
            variants = dict()
            l_variants = samples[sample]
            if len(l_variants) > 0:
                #print(sample, samples_name_idx[sample])
                transformed_variants, v_list = self.transform_variants(l_variants, samples_name_idx[sample])
                encoded_samples.extend(transformed_variants)
                variants[sample] = len(transformed_variants)
                sample_n_variants.append(variants)
                samples_save[sample] = v_list
                num_v_sample.append(transformed_variants.shape[0])
            assert np.sum(num_v_sample) == len(encoded_samples)
        print("Num transformed rows for {} samples: {}".format(str(s_idx + 1), str(len(encoded_samples))))

        utils.save_as_json("data/{}_n_variants.json".format(typ), sample_n_variants)
        utils.save_as_json("data/{}_all_variants.json".format(typ), samples_save)
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

    def transform_variants(self, variants, name_idx, max_len_ref=2, max_len_alt=2):
        encoded_sample = list()
        v_list = list()
        for index, item in enumerate(variants):
            ref_var = item.split(">")
            s_name, pos, ref, alt_1, allel_freq = ref_var[0], ref_var[1], ref_var[2], ref_var[3], ref_var[4]
            if len(ref) <= max_len_ref and len(alt_1) <= max_len_alt:
                sample = list()
                encoded_ref = self.encode_nucleotides(ref, max_len_ref)
                encoded_alt = self.encode_nucleotides(alt_1, max_len_alt)
                n_e_ref = np.concatenate((encoded_ref, np.zeros(max_len_ref - len(encoded_ref))), axis=None)
                n_e_alt = np.concatenate((encoded_alt, np.zeros(max_len_alt - len(encoded_alt))), axis=None)
                #sample = np.hstack(([name_idx, pos, allel_freq], n_e_ref.tolist(), n_e_alt.tolist()))
                #sample = np.hstack(([pos, allel_freq], n_e_ref.tolist(), n_e_alt.tolist()))
                sample = np.hstack(([pos], n_e_ref.tolist(), n_e_alt.tolist()))
                sample = [float(x) for x in sample]
                encoded_sample.append(sample)
                v_list.append(item)
        return np.asarray(encoded_sample), v_list
