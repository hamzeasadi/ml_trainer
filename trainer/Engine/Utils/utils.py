"""
general utility functions
"""


import os
from typing import List, Optional, Any
import pickle
import json
import csv
from dataclasses import dataclass





def crtdir(path:str):
    if not os.path.exists(path):
        os.makedirs(path)




@dataclass
class Paths:
    project_root:str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    data_path:str = os.path.join(project_root, "data")

    dataset_path:str = os.path.join(data_path, "dataset")
    model_path:str = os.path.join(data_path, "model")
    report_path:str = os.path.join(data_path, "report")
    config_path:str = os.path.join(data_path, "config")

    def crtdir(self, path:str):
        crtdir(path)
    
    def initpath(self):
        for path_name, path_dir in self.__dict__.items():
            if os.path.expanduser("~") in path_dir:
                self.crtdir(path_dir)




def save_pickle(data:Any, save_path:str, file_name:str):
    crtdir(save_path)
    with open(os.path.join(save_path, file_name), "wb") as pickle_file:
        pickle.dump(obj=data, file=pickle_file)




def load_pickle(file_path:str):
    with open(file_path, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data














if __name__ == "__main__":
    print(__file__)

    paths = Paths()
    paths.initpath()
