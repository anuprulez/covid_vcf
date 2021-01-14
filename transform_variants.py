import numpy as np
import re
from sklearn.preprocessing import LabelEncoder


class TransformVariants:

    def __init__(self):
        """ Init method. """
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(np.array(['a','c','g','t','z']))

    def get_variants(self, samples):
        print("Transforming variants...")
        encoded_samples = list()
        for sample in samples:
            positions = list()
            variants = list()
            l_variants = samples[sample]
            '''for item in l_variants:
                pos = list(item.keys())[0]
                var = list(item.values())[0]
                if var not in variants_freq:
                    variants_freq[var] = 1
                else:
                    variants_freq[var] += 1
            print(variants_freq)'''
            transformed_variants = self.transform_variants(l_variants)
            encoded_samples.append(transformed_variants)
        return encoded_samples 

    def string_to_array(self, my_string):
        my_string = my_string.lower()
        my_string = re.sub('[^acgt]', 'z', my_string)
        my_array = np.array(list(my_string))
        return my_array  

    def encode_nucleotides(self, seq, encoded_AGCT=[0.25, 0.5, 0.75, 1.0]):
        encoded_seq = self.ordinal_encoder(self.string_to_array(seq))
        return encoded_seq

    def ordinal_encoder(self, my_array):
        integer_encoded = self.label_encoder.transform(my_array)
        float_encoded = integer_encoded.astype(float)
        float_encoded[float_encoded == 0] = 0.25 # A
        float_encoded[float_encoded == 1] = 0.50 # C
        float_encoded[float_encoded == 2] = 0.75 # G
        float_encoded[float_encoded == 3] = 1.00 # T
        float_encoded[float_encoded == 4] = 0.00 # anything else, z
        return float_encoded

    def transform_variants(self, variants, n_features=3, max_len_ref=10, max_len_alt=5):
        #print(variants)
        encoded_sample = np.zeros((len(variants), max_len_ref+max_len_alt+1))
        for index, item in enumerate(variants):
            pos = list(item.keys())[0]
            var = list(item.values())[0]
            encoded_sample[index, 0:1] = [pos]
            ref_var = var.split(">")
            ref, alt_1 = ref_var[0], ref_var[1]
            if len(ref) <= max_len_ref and len(alt_1) <= max_len_alt:
                encoded_ref = self.encode_nucleotides(ref, max_len_ref)
                encoded_alt = self.encode_nucleotides(alt_1, max_len_alt)
                n_e_ref = np.concatenate((encoded_ref, np.zeros(max_len_ref - len(encoded_ref))), axis=None)
                n_e_alt = np.concatenate((encoded_alt, np.zeros(max_len_alt - len(encoded_alt))), axis=None)
                encoded_sample[index, 1:max_len_ref + 1] = n_e_ref
                encoded_sample[index, max_len_ref + 1:max_len_ref + 1 + max_len_alt] = n_e_alt
        #print(encoded_sample)
        return encoded_sample
