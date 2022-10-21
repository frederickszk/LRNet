import yaml
import os

def load_yaml(path):
    current_dir = os.getcwd()
    file = open(os.path.join(current_dir, path), 'r', encoding='utf-8')
    string = file.read()
    dict_yaml = yaml.safe_load(string)
    return dict_yaml
