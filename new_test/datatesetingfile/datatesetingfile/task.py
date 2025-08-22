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


class Net(nn.Module):
    def __init__(self, input_dim = 42):
        super(Net, self).__init__()
        
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


# fds = None  # Cache FederatedDataset


# def load_data(partition_id: int, num_partitions: int):
#     """Load partition CIFAR10 data."""
#     # Only initialize `FederatedDataset` once
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         fds = FederatedDataset(
#             dataset="uoft-cs/cifar10",
#             partitioners={"train": partitioner},
#         )
#     partition = fds.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     pytorch_transforms = Compose(
#         [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )

#     def apply_transforms(batch):
#         """Apply transforms to the partition from FederatedDataset."""
#         batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#         return batch

#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=32)
#     return trainloader, testloader

####################### customized for new test ##########################
fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int):   
    base_path = os.path.dirname(__file__)
    data_files = os.path.join(base_path, "processed_train_test_network.csv")

    dataset = pd.read_csv(data_files)
    df = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # #pre-processing
    # # Convert all columns except 'Label' to numeric (coerce errors to NaN)
    # for col in df.columns:
    #     if col != "Label":
    #         df[col] = pd.to_numeric(df[col], errors='coerce')

    

    #disable_caching()
    dataset = Dataset.from_pandas(df) 

    partitioner = DirichletPartitioner(num_partitions=num_partitions, 
                                       alpha=0.5,
                                       partition_by="type",)
        
    
    
    partitioner.dataset = dataset

   
    partition = partitioner.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    train_ds = partition_train_test["train"]
    test_ds = partition_train_test["test"]
    
    feature_columns = [col for col in train_ds.column_names if ((col != "type") or (col != "label"))]
# solve by converting to pandas DataFrame
    train_ds = train_ds.to_pandas()
    test_ds = test_ds.to_pandas()

    X_train_tensor = torch.tensor(train_ds[feature_columns].values, dtype=torch.float32)
    X_test_tensor = torch.tensor(test_ds[feature_columns].values, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_ds["label"].values, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_ds["label"].values, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader

# ############################### NEW ###########################
# def load_datasets(partition_id: int, num_partitions: int):
#     fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
#     partition = fds.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     pytorch_transforms = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )

#     def apply_transforms(batch):
#         # Instead of passing transforms to CIFAR10(..., transform=transform)
#         # we will use this function to dataset.with_transform(apply_transforms)
#         # The transforms object is exactly the same
#         batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#         return batch

#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
#     valloader = DataLoader(partition_train_test["test"], batch_size=32)
#     testset = fds.load_split("test").with_transform(apply_transforms)
#     testloader = DataLoader(testset, batch_size=32)
#     return trainloader, valloader, testloader
#####################################################################


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
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


    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Evaluate the model on the test set."""
    net.to(device)
    net.eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)  # Ensure shape [batch, 1]

            outputs = net(X_batch)

            # Ensure output shape matches target
            if outputs.shape != y_batch.shape:
                outputs = outputs.view(-1, 1)

            batch_loss = criterion(outputs, y_batch)
            loss += batch_loss.item()

            # Apply sigmoid to get probabilities, then round to get binary predictions
            preds = torch.sigmoid(outputs)
            preds_binary = (preds >= 0.5).float()
            correct += (preds_binary == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    avg_loss = loss / len(testloader)
    return avg_loss, accuracy



def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
