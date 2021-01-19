import json


def save_as_json(filepath, data):
    with open(filepath, 'w') as fp:
        json.dump(data, fp)
        

def read_json(path):
    with open(path, 'r') as fp:
        f_content = json.loads(fp.readline())
        return f_content

