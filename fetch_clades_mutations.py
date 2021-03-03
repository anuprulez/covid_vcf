import json

CLADES_PATH = "data/clades/samples_clades_nuc.json"


def read_phylogenetic_data(json_file="data/clades/ncov_global_2021-01-27_17-42.json"):
    with open(json_file, "r") as fp:
        data = json.loads(fp.read())
    tree = data["tree"]
    clade_info = dict()
    recursive_branch(tree, clade_info)
    with open(CLADES_PATH, "w") as fread:
        fread.write(json.dumps(clade_info))


def recursive_branch(obj, clade_info):
    if "branch_attrs" in obj:
        branch_attr = obj["branch_attrs"]
        if "children" in obj:
            child = obj["children"]
            for item in child:
                if "name" in item and "branch_attrs" in item:
                    item_branch_attr = item["branch_attrs"]
                    if "mutations" in item_branch_attr:
                        if "nuc" in item_branch_attr["mutations"]:
                            if "node_attrs" in item:
                                if "clade_membership" in item["node_attrs"]:
                                    clade_info[item["name"]] = list()
                                    clade_info[item["name"]].append(item["node_attrs"]["clade_membership"])
                                    clade_info[item["name"]].append({"nuc": item_branch_attr["mutations"]["nuc"]})
                recursive_branch(item, clade_info)
            
    else:
        return None
        
def nuc_parser(lst_nuc):
    parsed_nuc = list()
    last_pos = None
    for i, item in enumerate(lst_nuc):
        ref_pos_alt = [char for char in item]
        ref = ref_pos_alt[0]
        alt = ref_pos_alt[-1]
        pos = "".join(ref_pos_alt[1:len(ref_pos_alt) - 1])
        if last_pos is not None:
            if last_pos + 1 == pos and :
                continue
        
        last_pos = pos
        print(i, item, ref_pos_alt, ref, pos, alt)
    
        
def get_nuc_clades():
    with open(CLADES_PATH, "r") as cf:
        clades = json.loads(cf.read())
    clades_nuc = dict()
    for key in clades:
        clade_name = clades[key][0]["value"]
        clade_nuc = clades[key][1]["nuc"]
        #print(key, clades[key], clade_name, clade_nuc)
        if clade_name not in clades_nuc:
            clades_nuc[clade_name] = list()
        parsed_nuc = nuc_parser(clade_nuc)
        clades_nuc[clade_name].extend(parsed_nuc)
        #print("-------")
    #print("===========================")
    print(clades_nuc)




