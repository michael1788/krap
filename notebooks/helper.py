import pickle
import pandas as pd
import numpy as np

def save_obj(savepath, obj):
    """
    Save data with pickle
    Parameters
    ----------
    - savepath (str)
    - obj: data to pickle
    """

    savepath = savepath.replace(".pkl", "")
    with open(f"{savepath}.pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    """
    Load data saved with pickle
    Parameters
    ----------
    - path (str): path to the pickle data file
    """
    path = path.replace('.pkl', '')
    with open(f'{path}.pkl', 'rb') as f:
        return pickle.load(f)
    
