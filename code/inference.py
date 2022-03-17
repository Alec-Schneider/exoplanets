import logging 
import argparse
import os 
import sys 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, Dataset
import torch.utils.data
import torchvision

from typing import Callable

from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from types import SimpleNamespace


def model_fn(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = torch.load(model_path)
    
    return loaded_model.to(device)


def input_fn(request_body, request_content_type):
    
    if request_content_type == "application/json":
        data = json.loads(request_body)
        
    if isinstance(data, list) and len(data) > 0 and not isinstance(data[0], str):
        pass
    else:
            raise ValueError("Unsupported input type. Input type can be a list of ints or floats. \
                             I got {}".format(data))
    
    input_data = torch.tensor(data)
    return input_data.float()

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
 
    input_data = input_data.to(device)
    with torch.no_grad():
        y = model(input_data.float)
        preds = outputs.detach().cpu().numpy() 
        prediction = np.round(preds,0).flatten()[0]
        print("=============== inference result =================")
        print(prediction)
    return prediction