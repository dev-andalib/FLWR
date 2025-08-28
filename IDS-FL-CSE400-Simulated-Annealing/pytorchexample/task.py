from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from pytorchexample.utility import save_metrics_to_json, save_sa, print_msg, file_handle
from sklearn.feature_selection import SelectKBest, chi2
from typing import Tuple


class Net(nn.Module):
    def __init__(self, input_dim=42):  # Removed num_classes since we need 1 output for binary
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=16, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(32, 1)  # Single output for binary classification

    def forward(self, x):
        x = x.unsqueeze(1)             # [batch, 1, features]
        x = F.relu(self.conv1(x))      # [batch, 32, features]
        x = self.pool(x)               # [batch, 32, features/2]

        x = x.permute(0, 2, 1)        # [batch, seq_len, features]
        x, _ = self.lstm1(x)           # [batch, seq_len, 64]
        x, _ = self.lstm2(x)           # [batch, seq_len, 32]

        x = x[:, -1, :]                # last time step: [batch, 32]
        x = self.fc(x)                 # [batch, 1]
        
        return x  # Return raw logits for BCEWithLogitsLoss

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

fds = None  # Cache FederatedDataset
path = 'Ids_dataset/UNSW_NB15_binary_label_is_label.parquet'

def load_data(partition_id: int, num_partitions: int, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Custom data division mechanism to divide data into partitions, selecting top 20 features using chi-squared test.
    Returns train, validation, and test DataLoaders.
    """
    global path, fds
    batch_size = batch_size if batch_size > 0 else 32  # Default batch size
    
    if fds is None:
        ds = load_dataset("parquet", data_files=path, split="train")
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = ds
        fds = partitioner
        

    # Load the partitioned dataset
    dataset = fds.load_partition(partition_id).with_format("pandas")[:]
    
    X = dataset.drop("label", axis=1)
    y = dataset["label"]
    
    # Convert multiclass labels to binary (0 or 1)
    y = (y != 0).astype(int)  # Assuming positive class is anything non-zero, negative class is 0

    # Apply chi-squared feature selection to select top 20 features
    # selector = SelectKBest(score_func=chi2, k=20)
    # X_selected = selector.fit_transform(X, y)
    # selected_features = X.columns[selector.get_support()].tolist()
    
    # # Create new DataFrame with selected features
    # X = X[selected_features]

    # Create train, validation, and test splits
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)  # Float for BCE
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train(net, trainloader, valloader, epochs, learning_rate, device, temp, client):
    """Train the model on the training set and validate on the validation set."""
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss for binary classification
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()

    total_epoch_losses = []
    total_val_losses = []

    best_val_loss = float('inf')
    best_metrics = {}

    for epoch in range(epochs):
        running_loss = 0.0
        all_preds_epoch = []
        all_labels_epoch = []

        for batch in trainloader:
            features, labels = batch
            features = features.to(device).float()
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float().detach().cpu().numpy()
            all_preds_epoch.extend(preds.flatten())
            all_labels_epoch.extend(labels.cpu().numpy().flatten())

        avg_train_loss = running_loss / len(trainloader)
        total_epoch_losses.append(avg_train_loss)

        # Validation phase
        net.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in valloader:
                features, labels = batch
                features = features.to(device).float()
                labels = labels.to(device).float()

                outputs = net(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                val_preds.extend(preds.flatten())
                val_labels.extend(labels.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(valloader)
        total_val_losses.append(avg_val_loss)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, zero_division=0)
        val_rec = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_cm = confusion_matrix(val_labels, val_preds, labels=[0, 1])  # [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = val_cm.ravel()  # Extract individual elements

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_metrics = {
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
                "cm_tn": int(tn),  # True Negatives
                "cm_fp": int(fp),  # False Positives
                "cm_fn": int(fn),  # False Negatives
                "cm_tp": int(tp)   # True Positives
            }

        net.train()
        
        # save_metrics_to_json(best_metrics, f"Epoch {epoch+1}/{epochs} - Val Metrics", cid, output_folder="E:/New_IDS - Copy/results/train/")
        # save_metrics_to_json(avg_train_loss, f"Train Loss", cid, output_folder="E:/New_IDS - Copy/results/train/")

        client_accept = file_handle(client,  best_metrics, temp)
    return best_metrics, client_accept

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    net.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in testloader:
            features, labels = batch
            features = features.to(device).float()
            labels = labels.to(device).float()

            outputs = net(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Calculate metrics
    avg_loss = total_loss / len(testloader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])  # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()  # Extract individual elements

    output_dict = {
        "acc": acc,
        "rec": rec,
        "prec": prec,
        "f1": f1,
        "cm_tn": int(tn),  # True Negatives
        "cm_fp": int(fp),  # False Positives
        "cm_fn": int(fn),  # False Negatives
        "cm_tp": int(tp)   # True Positives
    }
    
    



    
    

    return avg_loss, len(testloader), output_dict



