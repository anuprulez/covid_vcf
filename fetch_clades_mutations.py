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
    prev_pos = 0
    chained_ref = list()
    chained_alt = list()
    chained_pos = list()
    for i, item in enumerate(lst_nuc):
        #print(prev_pos)
        
        ref_pos_alt = [char for char in item]
        c_ref = ref_pos_alt[0]
        c_alt = ref_pos_alt[-1]
        c_pos = int("".join(ref_pos_alt[1:len(ref_pos_alt) - 1]))
        if c_ref != "-" and c_alt != "-":
            p_nuc = "{}>{}>{}".format(c_ref, c_pos, c_alt)
            parsed_nuc.append(p_nuc)
        else:
            chained_ref.append(c_ref)
            chained_alt.append(c_alt)
            chained_pos.append(str(c_pos))
            if prev_pos == c_pos - 1:
                print("Inside else if")
                print(prev_pos, c_pos)
                print(len(chained_ref))
                print("====")
                if i == len(lst_nuc) - 1:
                    print("End of list")
                    p_nuc = "{}>{}>{}".format("".join(chained_ref), " ".join(chained_pos), "".join(chained_alt))
                    parsed_nuc.append(p_nuc)
                    chained_ref = list()
                    chained_alt = list()
                    chained_pos = list()
            else:
                if len(chained_ref) > 0:
                    print("Inside else")
                    print(len(chained_ref))
                    assert len(chained_ref) == len(chained_pos)
                    assert len(chained_ref) == len(chained_alt)
                    p_nuc = "{}>{}>{}".format("".join(chained_ref), " ".join(chained_pos), "".join(chained_alt))
                    parsed_nuc.append(p_nuc)
                    chained_ref = list()
                    chained_alt = list()
                    chained_pos = list()
                else:
                    p_nuc = "{}>{}>{}".format(c_ref, c_pos, c_alt)
                    parsed_nuc.append(p_nuc)
        prev_pos = c_pos       
        #print(i, item, ref_pos_alt, ref, pos, alt)
    return list(set(parsed_nuc))
    
        
def get_nuc_clades():
    with open(CLADES_PATH, "r") as cf:
        clades = json.loads(cf.read())
    clades_nuc = dict()
    for key in clades:
        clade_name = clades[key][0]["value"]
        clade_nuc = clades[key][1]["nuc"]
        if clade_name not in clades_nuc:
            clades_nuc[clade_name] = list()
        #print(clade_name)
        if clade_name == "20F":
            print(clade_nuc)
            parsed_nuc = nuc_parser(clade_nuc)
            print()
            print(parsed_nuc)
            print("---------------")
            clades_nuc[clade_name].extend(parsed_nuc)
    #print(clades_nuc)
    with open("data/clades/parsed_nuc.json", "w") as fread:
        fread.write(json.dumps(clades_nuc))




