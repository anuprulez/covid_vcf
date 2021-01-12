import numpy as np



class TransformVariants:

    def __init__(self):
        """ Init method. """
        
    def get_variants(self, samples):
        for sample in samples:
            positions = list()
            variants = list()
            print(sample)
            l_variants = samples[sample]
            #print(l_variants)
            '''for item in l_variants:
                pos = list(item.keys())[0]
                var = list(item.values())[0]
                if var not in variants_freq:
                    variants_freq[var] = 1
                else:
                    variants_freq[var] += 1
            print(variants_freq)'''
            #print("=====================")
            transformed_variants = self.transform_variants(l_variants)
    
    def transform_variants(self, variants):
        print(variants)
        print("=====================")
    
        
