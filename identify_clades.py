import json
import pandas as pd

clades = "https://raw.githubusercontent.com/nextstrain/ncov/557efcbe06641af72c28d3f7f0fa511b6f16e89c/defaults/clades.tsv"
Clusters_path = "data/test_clusters.csv"


# Test sample names: 
# BurkinaFaso/4307/2020
# Pakistan/Gilgit1/2020
# PapuaNewGuinea/10/2020
# PapuaNewGuinea/7/2020
# Ecuador/USFQ-556/2020
# India/MH-1-27/2020


def get_data(path, sep):
    dataframe = pd.read_csv(path, sep=sep)
    return dataframe
    
def find_clades_in_clusters():
    clades_df = get_data(clades, "\t")
    print(clades_df)
    clusters_df = get_data(Clusters_path, ",")
    print(clusters_df)
    clades_present = list()
    clades_missing = list()
    for i in range(len(clades_df)):
        sample_row = clades_df.take([i])
        clade_name = sample_row["clade"].values[0]
        POS = sample_row["site"].values[0]
        ALT = sample_row["alt"].values[0]
        row = clusters_df[(clusters_df["POS"] == int(POS)) & (clusters_df["ALT"] == ALT)]
        print(clade_name, POS, ALT, len(row))
        if len(row) > 0:
            if clade_name not in clades_present:
                clades_present.append([clade_name, str(POS), ALT])
        else:
            if clade_name not in clades_missing:
                clades_missing.append([clade_name, str(POS), ALT])
        #print(row)
        #print("-------------------")
    print(clades_present)
    print(clades_missing)
    clades_present_df = pd.DataFrame(clades_present, columns=["Clade name", "POS", "ALT"])
    clades_present_df.to_csv("data/clades_present_df.csv", sep=",")
    
    clades_missing_df = pd.DataFrame(clades_missing, columns=["Clade name", "POS", "ALT"])
    clades_missing_df.to_csv("data/clades_missing_df.csv", sep=",")
    
        
    
    
if __name__ == "__main__":
    find_clades_in_clusters()
