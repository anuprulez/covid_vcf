import json


def read_phylogenetic_data(json_file="data/clades/ncov_global_2021-01-27_17-42.json"):
    with open(json_file, "r") as fp:
        data = json.loads(fp.read())
    tree = data["tree"]
    clade_info = dict()
    recursive_branch(tree, clade_info)
    with open("data/clades/samples_clades_nuc.json", "w") as fread:
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



