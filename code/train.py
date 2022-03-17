import logging 
import argparse
import os 
import sys 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

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

logger = logging.getLogger(__name__) 
logger.setLevel(logging.DEBUG) 
logger.addHandler(logging.StreamHandler(sys.stdout)) 



def get_data_loader(root, batch_size, train=True):
    logger.info("Get data loader")
    
    if train:
        dataset = ExoplanetDataset(root, train=True)
    else:
        dataset = ExoplanetDataset(root, train=False)
    
    tensor_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return tensor_dataloader


class ExoplanetDataset(Dataset):
    """
    Dataset PyTorch class to retrieve the training and test data of the Exoplanents dataset
    """
    
    def  __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.x_train_file = "data_train_x_scaled_filt.txt"
        self.y_train_file = "data_train_y.csv"
        self.x_test_file = "data_test_x_scaled_filt.txt"
        self.y_test_file = "data_test_y.csv"
        
        if self.train:
            logger.info("Loading Training Data")
            self.x_train = self.load_features(self.x_train_file)
            self.y_train = pd.read_csv(os.path.join(self.root, self.y_train_file))[['LABEL']]
            
            
            pos_idx = np.where(self.y_train == 1)[0]
            neg_idx = np.where(self.y_train == 0)[0]
            x_pos = self.x_train[pos_idx]
            x_neg = self.x_train[neg_idx]
            # Add rotations for the positives to the dataset to upsample
            num_rotations = 100
            for i in range(len(x_pos)):
                 for r in range(num_rotations):
                        rotated_row = np.roll(x_pos[i,:], shift = r, axis=0)
                        self.x_train = np.vstack([self.x_train, rotated_row[np.newaxis, :, :]])
            
            self.y_train = np.vstack([self.y_train, 
                                      np.array([1] * len(x_pos) * num_rotations).reshape(-1,1)])\
                            .reshape(-1)
            self.len = len(self.y_train)
            
        else:
            logger.info("Loading Test Data")
            self.x_test = self.load_features(self.x_test_file)
            self.y_test = pd.read_csv(os.path.join(self.root, self.y_test_file))['LABEL'].values
            self.len = len(self.y_test)
            
    def __getitem__(self, idx):
        if self.train:
            features = torch.tensor(self.x_train[idx])
            labels = torch.tensor(self.y_train[idx])
        
        else:
            features = torch.tensor(self.x_test[idx])
            labels = torch.tensor(self.y_test[idx])
            
        return (features, labels)
    
    def __len__(self):
        return self.len
    
    def load_features(self, file):
        loaded_arr = np.loadtxt(os.path.join(self.root, file))
        features = loaded_arr.reshape(
            loaded_arr.shape[0], loaded_arr.shape[1] // 2, 2)
        
        return features
        
        
class CNN(nn.Module):
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

    
def train_model(args):
    """
    args: NameSpace type with model training arguments
    
    return: Trained PyTorch Model
    """
     
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Set the seed
    torch.manual_seed(args.seed)
    
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        
    train_loader = get_data_loader(args.data_dir, args.batch_size, train=True)
    test_loader = get_data_loader(args.data_dir, args.test_batch_size, train=False)
    
    model = CNN()
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs + 1):
        
        total_loss = 0
        model.train()
        
        for step, batch in enumerate(train_loader):
            feats = batch[0].to(device)
            labels = batch[1].to(device)
            
            model.zero_grad()
            
            outputs = model(feats.float())
          
            loss = criterion(torch.squeeze(outputs, 1).float(), labels.float())
            total_loss += loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step()
            
            if step % args.log_interval == 0: 
                logger.info( 
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format( 
                        epoch, 
                        step * len(batch[0]), 
                        len(train_loader.sampler), 
                        100.0 * step / len(train_loader), 
                        loss.item(), 
                    ) 
                ) 
        logger.info("Average training loss: %f\n", total_loss / len(train_loader)) 
        test(model, test_loader, device)
        
    logger.info("Saving tuned model")
    
    model_2_save = model.module if hasattr(model, "module") else model 
     # ... train `model`, then save it to `model_dir`
    with open(os.path.join(args.model_dir, f'{args.model_name}.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
        
    return model


def test(model, test_loader, device):
    
    def get_correct_count(preds, labels): 
        pred_flat = np.round(preds,0).flatten() 
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat), len(labels_flat) 

    model.eval() 
    _, eval_accuracy = 0, 0 
    total_correct = 0 
    total_count = 0 


    with torch.no_grad(): 
        for batch in test_loader: 

            b_input_ids = batch[0].to(device) 
            b_labels = batch[1].to(device) 

            outputs = model(b_input_ids.float()) 
#             preds = outputs[0] 
            preds = outputs.detach().cpu().numpy() 
            label_ids = b_labels.to("cpu").numpy() 
                         

            num_correct, num_count = get_correct_count(preds, label_ids) 
            total_correct += num_correct 
            total_count += num_count 
    accuracy = total_correct/total_count
    logger.info("Test set: Accuracy: %f\n", accuracy) 

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size", type=int, default=24, metavar="N", help="input batch size for training (default: 24)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=16, metavar="N", help="input batch size for testing (default: 16)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 2)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")

    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--model-name", type=str, default="torch_model", help="Name to save model as.")
    
    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
 

    train_model(parser.parse_args())