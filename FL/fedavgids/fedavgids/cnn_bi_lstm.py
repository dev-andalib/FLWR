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

class CNN_BiLSTM_Net(nn.Module):
    def __init__(self, input_dim = 42):
        super(CNN_BiLSTM_Net, self).__init__()
        input_dim = 42
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm1 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=16, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(32, 1)  # single output for binary classification

    def forward(self, x):
        x = x.unsqueeze(1)             # [batch, 1, features]
        x = F.relu(self.conv1(x))      # [batch, 32, features]
        x = self.pool(x)               # [batch, 32, features/2]

        x = x.permute(0, 2, 1)        # [batch, seq_len, features]
        x, _ = self.lstm1(x)           # [batch, seq_len, 64]
        x, _ = self.lstm2(x)           # [batch, seq_len, 32]

        x = x[:, -1, :]                # last time step: [batch, 32]
        x = self.fc(x)                 # [batch, 1]

        return x  # raw logits (no sigmoid here)


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
        y = dataset["label"]
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
    print("Selected feature columns:", feature_columns)
    print("Number of features:", len(feature_columns))


# solve by converting to pandas DataFrame
    train_ds = train_ds.to_pandas()
    test_ds = test_ds.to_pandas()

    X_train_tensor = torch.tensor(train_ds[feature_columns].values, dtype=torch.float32)
    X_test_tensor = torch.tensor(test_ds[feature_columns].values, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_ds["label"].values, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_ds["label"].values, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return train_loader, test_loader




# def load_data(partition_id: int, num_partitions: int):
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     csv_path = os.path.join(script_dir, "preprocessed_dataset_binary.csv")

#     df_full = pd.read_csv(csv_path)
#     df_full.dropna(inplace=True)

#     # Get features and target
#     X = df_full.drop("Label", axis=1)
#     y = df_full["Label"]

#     # Perform stratified K-Fold partitioning
#     skf = StratifiedKFold(n_splits=num_partitions, shuffle=True, random_state=42)
#     all_splits = list(skf.split(X, y))

#     if partition_id >= num_partitions:
#         raise ValueError(f"partition_id {partition_id} out of range for {num_partitions} partitions.")

#     train_idx, _ = all_splits[partition_id]
#     X_part = X.iloc[train_idx]
#     y_part = y.iloc[train_idx]

#     # Train-test split within the partition
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_part, y_part, test_size=0.2, random_state=42, stratify=y_part
#     )

#     # Convert to tensors
#     X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
#     X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
#     y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

#     # Create DataLoaders
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

#     # Optional: log label distribution
#     print(f"[Client {partition_id}] Label distribution in train: {y_train.value_counts().to_dict()}")
#     print(f"[Client {partition_id}] Label distribution in test: {y_test.value_counts().to_dict()}")

#     return train_loader, test_loader

# def load_data(partition_id: int, num_partitions: int):
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     csv_path = os.path.join(script_dir, "preprocessed_dataset_binary.csv")

#     df_full = pd.read_csv(csv_path)
#     df_full.dropna(inplace=True)

#     X = df_full.drop("Label", axis=1)
#     y = df_full["Label"]

#     if num_partitions == 1:
#         # Centralized case: use whole dataset with stratified train/test split
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, stratify=y, random_state=42
#         )
#     else:
#         # Federated case: stratified partitioning for clients
#         skf = StratifiedKFold(n_splits=num_partitions, shuffle=True, random_state=42)
#         all_splits = list(skf.split(X, y))

#         if partition_id >= len(all_splits):
#             raise ValueError(f"partition_id {partition_id} is out of bounds.")

#         train_idx, _ = all_splits[partition_id]
#         X_part = X.iloc[train_idx]
#         y_part = y.iloc[train_idx]

#         X_train, X_test, y_train, y_test = train_test_split(
#             X_part, y_part, test_size=0.2, stratify=y_part, random_state=42
#         )

#     # Convert to tensors
#     X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
#     X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
#     y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

#     # Create DataLoaders
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

#     print(f"[Client {partition_id}] Train label dist: {y_train.value_counts().to_dict()}")
#     print(f"[Client {partition_id}] Test label dist: {y_test.value_counts().to_dict()}")

#     return train_loader, test_loader



def test(net, testloader, device):
    """Validate the binary classification model and compute various metrics."""
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    net.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)  # Ensure shape [B, 1]

            outputs = net(X_batch)
            outputs = outputs.view(-1, 1)  # Ensure shape [B, 1] for logits

            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    # Concatenate all batches
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds).astype(int).flatten()
    all_labels = np.vstack(all_labels).astype(int).flatten()
    print("Test preds", np.unique(all_preds, return_counts=True))
    print("Test labels", np.unique(all_labels, return_counts=True))


    avg_loss = total_loss / len(testloader)
    accuracy = np.mean(all_preds == all_labels)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Confusion matrix and FPR
    cm = confusion_matrix(all_labels, all_preds)
    FP = np.clip(cm.sum(axis=0) - np.diag(cm), 0, None)
    FN = np.clip(cm.sum(axis=1) - np.diag(cm), 0, None)
    TP = np.clip(np.diag(cm), 0, None)
    TN = np.clip(cm.sum() - (FP + FN + TP), 0, None)
    fpr = FP / (FP + TN + 1e-10)
    fpr = fpr.mean()

    # AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
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
   

    """Train the model on the training set for binary classification."""
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    running_loss = 0.0

    for _ in range(epochs):
        for X_batch, y_batch in trainloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)  # Ensure shape [batch_size, 1]

            optimizer.zero_grad()
            outputs = net(X_batch)

            # Make sure output is also [batch_size, 1]
            if outputs.shape != y_batch.shape:
                outputs = outputs.view(-1, 1)

            loss = criterion(outputs, y_batch)
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
