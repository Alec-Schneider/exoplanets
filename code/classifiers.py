import logging 
import os 
import sys 
import numpy as np 
import pandas as pd 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
from sklearn.model_selection import train_test_split


class BinaryClassifier(nn.Module):
    """
    Binary Classifier using only Linear layers
    """
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.l_1 = nn.Linear(3197, 5000)
        self.l_2 = nn.Linear(5000, 10000)
        self.l_3 = nn.Linear(10000, 5000)
        self.l_4 = nn.Linear(5000, 2500)
        self.l_5 = nn.Linear(2500, 250)
        self.l_6 = nn.Linear(250, 1)

        
    def forward(self, x):
        x = F.relu(self.l_1(x))
        x = F.relu(self.l_2(x))
        x = F.relu(self.l_3(x))
        x = F.relu(self.l_4(x))
        x = F.relu(self.l_5(x))
        x = torch.sigmoid(self.l_6(x))
        return  x
    
    
class ShallowBinaryClassifier(nn.Module):
    """
    Binary Classifier with two linear layers and BatchNorm layer
    """
    def __init__(self):
        super(ShallowBinaryClassifier, self).__init__()
        self.l_1 = nn.Linear(3197, 5000)
        self.b_norm1 = nn.BatchNorm1d(5000)
        self.l_2 = nn.Linear(5000, 250)
        self.l_3 = nn.Linear(250, 1)

        
    def forward(self, x):
        x = F.relu(self.l_1(x))
        x = F.relu(self.b_norm1(x))
        x = F.relu(self.l_2(x))
        x = torch.sigmoid(self.l_3(x))
        return  x
    

class CNN(nn.Module):
    """
    1D Convolutional Neural Netowrk with BatchNorm
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 6, 3)
        self.b_norm1 = nn.BatchNorm1d(6)
        self.conv2 = nn.Conv1d(6, 16, 3)
        self.b_norm2 = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(16 * 354, 500)
        self.fc2 = nn.Linear(500, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Transform the input shape of the batch 
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        x = F.max_pool1d(F.relu(self.conv1(x)), 3)
        x = self.b_norm1(x) 
        x = F.max_pool1d(F.relu(self.conv2(x)), 3)
        x = self.b_norm2(x)
        x = x.view(-1, 16 * 354)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x
    

class CNN2(nn.Module):
    """
    1D Convolutional Neural Netowrk with BatchNorm and Dropout
    """
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv1d(2, 8, 3)
        self.b_norm1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.b_norm2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 32, 3)
        self.b_norm3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 64, 3)
        self.b_norm4 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 38, 250)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(250, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Transform the input shape of the batch 
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        x = F.max_pool1d(F.relu(self.conv1(x)), 3)
        x = self.b_norm1(x) 
        x = F.max_pool1d(F.relu(self.conv2(x)), 3)
        x = self.b_norm2(x)
        x = F.max_pool1d(F.relu(self.conv3(x)), 3)
        x = self.b_norm3(x)
        x = F.max_pool1d(F.relu(self.conv4(x)), 3)
        x = self.b_norm4(x)
        x = x.view(-1, 64 * 38)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))        
        return x