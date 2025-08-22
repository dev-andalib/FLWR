"""FedAVGids: A Flower / PyTorch app."""
from collections import OrderedDict

from collections import OrderedDict
from typing import List
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
import pandas as pd

from datasets import load_dataset
from flwr_datasets.partitioner import DirichletPartitioner
from datasets import Dataset
from datasets import disable_caching

"""datatesetingfile: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
#from flwr_datasets.partitioner import IidPartitioner
#from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=42, hidden_dims=[64, 48, 32], latent_dim=20):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], latent_dim),
            nn.ReLU()
        )
        
        # Decoder (mirrored architecture)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()  # Use Tanh or Identity if your features are not scaled to [0,1]
        )
        
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

# fds = None  # if you use caching, can keep this
from sklearn.model_selection import StratifiedKFold


def train_test_split(partition, test_fraction, seed):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]
    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test



def load_data(partition_id: int, num_partitions: int, server: bool = False):   
    base_path = os.path.dirname(__file__)
    data_files = os.path.join(base_path, "processed_train_test_network.csv")
    dataset = pd.read_csv(data_files)

    if server:
        # feature_columns = [col for col in dataset.column_names if ((col != "type") or (col != "label"))]
        x = dataset.drop(columns=["type", "label"])
        #x = dataset.drop(columns=["Label"])
        
        y = dataset["type"]
        X_test_tensor = torch.tensor(x.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y.values, dtype=torch.float32)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        return test_loader , test_loader
    
    df = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    partitioner = DirichletPartitioner(num_partitions=num_partitions, 
                                       alpha=350,
                                       partition_by="type",)
    partitioner.dataset = dataset
    partition = partitioner.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    train_ds = partition_train_test["train"]
    test_ds = partition_train_test["test"]
    
    
    feature_columns = [col for col in train_ds.column_names if col not in ("type", "label")]
    #feature_columns = [col for col in train_ds.column_names if col not in ("Label")]
    
    print("Selected feature columns:", feature_columns)
    print("Number of features:", len(feature_columns))
# solve by converting to pandas DataFrame
    train_ds = train_ds.to_pandas()
    test_ds = test_ds.to_pandas()

    X_train_tensor = torch.tensor(train_ds[feature_columns].values, dtype=torch.float32)
    X_test_tensor = torch.tensor(test_ds[feature_columns].values, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_ds["type"].values, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_ds["type"].values, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return train_loader, test_loader

def test(net, testloader, device, threshold=None):
    """Validate the AutoEncoder model and compute classification-like metrics."""
    net.to(device)
    net.eval()

    all_errors = []
    all_labels = []
    total_loss = 0.0

    criterion = torch.nn.BCELoss(reduction="mean")  # For reconstruction loss

    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)

            reconstructed = net(X_batch)

            # Compute reconstruction error per sample
            errors = torch.mean((X_batch - reconstructed) ** 2, dim=1).unsqueeze(1)  # [B, 1]
            loss = criterion(reconstructed, X_batch)
            total_loss += loss.item()

            all_errors.append(errors.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    # Flatten arrays
    all_errors_np = np.vstack(all_errors)
    all_labels_np = np.vstack(all_labels).astype(int).flatten()

    # Compute threshold if not provided
    if threshold is None:
        threshold = np.percentile(all_errors_np, 85)
    print(f"Threshold for anomaly detection: {threshold}")

    # Predict using threshold
    all_preds_np = (all_errors_np > threshold).astype(int).flatten()

    # Debug output
    print("Test preds", np.unique(all_preds_np, return_counts=True))
    print("Test labels", np.unique(all_labels_np, return_counts=True))

    # Metrics
    avg_loss = total_loss / len(testloader)
    accuracy = np.mean(all_preds_np == all_labels_np)
    precision = precision_score(all_labels_np, all_preds_np, zero_division=0)
    recall = recall_score(all_labels_np, all_preds_np, zero_division=0)
    f1 = f1_score(all_labels_np, all_preds_np, zero_division=0)

    cm = confusion_matrix(all_labels_np, all_preds_np)
    FP = np.clip(cm.sum(axis=0) - np.diag(cm), 0, None)
    FN = np.clip(cm.sum(axis=1) - np.diag(cm), 0, None)
    TP = np.clip(np.diag(cm), 0, None)
    TN = np.clip(cm.sum() - (FP + FN + TP), 0, None)
    fpr = FP / (FP + TN + 1e-10)
    fpr = fpr.mean()

    try:
        auc = roc_auc_score(all_labels_np, all_errors_np)
    except Exception as e:
        print(f"Warning: AUC calculation failed: {e}")
        auc = float("nan")

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "fpr": fpr,
        "auc": auc,
    }

    return metrics


def train(net, trainloader, epochs, device):
    net.to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()

    running_loss = 0.0
    for _ in range(epochs):
        for X_batch, _ in trainloader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            reconstructed = net(X_batch)
            loss = criterion(reconstructed, X_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    return avg_trainloss


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
