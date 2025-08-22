"""zero-trust-fed-cnn-bilstm-app: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from datasets import load_dataset #dataset_loader
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Normalize, ToTensor

import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder, label_binarize
import pandas as pd
import os

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

class Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)



fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "preprocessed_dataset.csv")

    dataset = load_dataset("csv", data_files=csv_path)
    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = dataset["train"]

    partition = partitioner.load_partition(partition_id)
    df = partition.with_format("pandas")[:]
    df.dropna(inplace=True)

    X = df.drop("Label", axis=1)
    y = df["Label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    return train_loader, test_loader

def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0

    for _ in range(epochs):
         for X_batch, y_batch in trainloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    return avg_trainloss

def test(net, testloader, device):
    """Validate the model on the test set and compute various metrics."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = net(X_batch)
            batch_loss = criterion(outputs, y_batch)
            loss += batch_loss.item()
            
            probs = torch.softmax(outputs, dim=1)  # get probabilities
            preds = outputs.argmax(dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    avg_loss = loss / len(testloader)
    accuracy = np.mean(all_preds == all_labels)
    
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # False Positive Rate calculation (FPR = FP / (FP + TN))
    cm = confusion_matrix(all_labels, all_preds)

    # FP = cm.sum(axis=0) - np.diag(cm)  # False Positives per class
    # TN = cm.sum() - (cm.sum(axis=1) + FP + np.diag(cm))
    # fpr_per_class = FP / (FP + TN + 1e-10)  # avoid division by zero
    # fpr = np.mean(fpr_per_class)  # average FPR over classes
    FP = np.clip(cm.sum(axis=0) - np.diag(cm), 0, None)
    FN = np.clip(cm.sum(axis=1) - np.diag(cm), 0, None)
    TP = np.clip(np.diag(cm), 0, None)
    TN = np.clip(cm.sum() - (FP + FN + TP), 0, None)

    denominator = FP + TN + 1e-10
    fpr_per_class = FP / denominator
    fpr = np.mean(fpr_per_class)
    # AUC calculation - only valid for multi-class if using one-vs-rest
    try:
        # Assume labels are integer encoded from 0 to num_classes-1
        all_labels_one_hot = label_binarize(all_labels, classes=np.arange(all_probs.shape[1]))
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except Exception as e:
        print(f"Warning: AUC calculation failed: {e}")
        auc = float('nan')
    
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

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
