from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, TensorDataset, random_split

import flwr 
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()

###################################################################################

#load data
data = pd.read_parquet('CIC-ToN-IoT-V2.parquet')

data = data.drop(columns=['Attack'])

print(data.shape)
data_x = data.iloc[:, :-1].values  
data_y = data.iloc[:, -1].values 


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_x = scaler.fit_transform(data_x)





from sklearn.model_selection import StratifiedKFold
def start_k_fold(client_loader):
    n_splits = 5  #fold num
    for x_client, y_client in client_data:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for t_index, v_index in skf.split(x_client, y_client):
            x_train, x_val = x_client[t_index], x_client[v_index]
            y_train, y_val = y_client[t_index], y_client[v_index]

        # Convert to tensors
            x_train_t = torch.tensor(x_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.long)
            x_val_t = torch.tensor(x_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.long)

        # Create datasets and loaders
            train_data = TensorDataset(x_train_t, y_train_t)
            val_data = TensorDataset(x_val_t, y_val_t)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

            client_loaders.append((train_loader, val_loader))


def simply_div(client_loader):
    for x_client, y_client in client_data:
    # Split client's local data into train and val (80/20)
        x_train_c, x_val_c, y_train_c, y_val_c = train_test_split(
        x_client, y_client, test_size=0.2, stratify=y_client, random_state=42
    )

    # Convert to tensors
        x_train_t = torch.tensor(x_train_c, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_c, dtype=torch.long)
        x_val_t = torch.tensor(x_val_c, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_c, dtype=torch.long)

    # Create loaders
        train_ds = TensorDataset(x_train_t, y_train_t)
        val_ds = TensorDataset(x_val_t, y_val_t)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        client_loaders.append((train_loader, val_loader))


# splitting

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.3, stratify=data_y, random_state=42)


batch_size = 32
# Convert test data to tensors
x_test_t = torch.tensor(x_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Create test dataset and loader
test_data = TensorDataset(x_test_t, y_test_t)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Number of clients
num_clients = 10

# Calculate the number of samples per client
samples_per_client = len(x_train) // num_clients

# Shuffle the training data
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

# Split data among clients
client_data = []
for i in range(num_clients):
    start_idx = i * samples_per_client
    end_idx = (i + 1) * samples_per_client if i != num_clients - 1 else len(x_train)
    x_client = x_train[start_idx:end_idx]
    y_client = y_train[start_idx:end_idx]
    client_data.append((x_client, y_client))









client_loaders = []

simply_div(client_loaders)


class MLP(nn.Module):
    def __init__(self, i_size, cls_num, cls_type='binary'):
        super(MLP, self).__init__()
        self.i_size = i_size
        self.cls_num = cls_num
        self.cls_type = cls_type
        
        
        self.dense1 = nn.Linear(i_size, 24)
        self.dense2 = nn.Linear(24, 16)
        
        if cls_type == 'binary':
            self.output_layer = nn.Linear(16, 1)
        else:
            self.output_layer = nn.Linear(16, cls_num)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.output_layer(x)
        
        if self.cls_type == 'binary':
            x = torch.sigmoid(x)
        else:
            
            x = F.log_softmax(x, dim=1)
        return x



class CNN(nn.Module):
    def __init__(self, i_size, cls_num, cls_type='binary'):
        super(CNN, self).__init__()
        self.i_size = i_size  
        self.cls_num = cls_num
        self.cls_type = cls_type
 
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=24, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3, padding=1)

        
        self.pool = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.dropout = nn.Dropout(0.3)
         
        
        
        self._calculate_conv_output_size()
        
        self.fc1 = nn.Linear(self.conv_output_size, 24)
        
        
        if cls_type == 'binary':
            self.output_layer = nn.Linear(24, 1)
        else:
            self.output_layer = nn.Linear(24, cls_num)

    def _calculate_conv_output_size(self):
        # Create a dummy tensor with the shape of the input
        dummy_input = torch.ones(1, 1, self.i_size)  # Batch size of 1, 1 channel, input size
        dummy_output = self._forward_convolution(dummy_input)
        
        # Calculate the size of the output before flattening (number of features)
        self.conv_output_size = dummy_output.view(-1).size(0)

    def _forward_convolution(self, x):
        # Forward pass through convolution and pooling layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        
        x = x.unsqueeze(1) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        
        
        x = x.view(x.size(0), -1)  
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        
        if self.cls_type == 'binary':
            x = torch.sigmoid(x)
        else:
            
            x = F.log_softmax(x, dim=1)
        
        return x

      

    def forward(self, x):
        
        x = x.unsqueeze(1) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        
        
        x = x.view(x.size(0), -1)  
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        
        if self.cls_type == 'binary':
            x = torch.sigmoid(x)
        else:
            
            x = F.log_softmax(x, dim=1)
        
        return x


class DNN(nn.Module):
    def __init__(self, i_size, cls_num, cls_type='binary'):
        super(DNN, self).__init__()
        self.i_size = i_size
        self.cls_num = cls_num
        self.cls_type = cls_type
        
        
        
        # layers
        self.fc1 = nn.Linear(i_size, 24)
        self.fc2 = nn.Linear(24, 48)
        self.bn1 = nn.BatchNorm1d(48)
        self.fc3 = nn.Linear(48, 24)
        self.dropout1 = nn.Dropout(0.3)
        
        
        
        
        if cls_type == 'binary':
            self.output_layer = nn.Linear(24, 1)
        else:
            self.output_layer = nn.Linear(24, cls_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.bn1(x))
        x = F.relu(self.fc3(x))
        x = self.dropout1(x)
        x = self.output_layer(x)


        
        if self.cls_type == 'binary':
            x = torch.sigmoid(x)
        else:
            
            x = F.log_softmax(x, dim=1)
        
        return x




class LSTM(nn.Module):
    def __init__(self, i_size, cls_num, cls_type='binary'):
        super(LSTM, self).__init__()
        self.i_size = i_size
        self.cls_num = cls_num
        self.cls_type = cls_type
        
        
        
        self.lstm1 = nn.LSTM(input_size=i_size, hidden_size=24, batch_first=True, bidirectional=False)
        
        self.lstm2 = nn.LSTM(input_size=24, hidden_size=16, batch_first=True, bidirectional=False)
        
        self.fc = nn.Linear(16, 8)
        
        if cls_type == 'binary':
            self.output_layer = nn.Linear(8, 1)
        else:
            self.output_layer = nn.Linear(8, cls_num)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x, _ = self.lstm1(x)  
        
        x, _ = self.lstm2(x)  
        
        x = x[:, -1, :] 
        
        x = F.relu(self.fc(x))
        x = self.output_layer(x)
        
        if self.cls_type == 'binary':
            x = torch.sigmoid(x)
        else:
            
            x = F.log_softmax(x, dim=1)
        
        return x


class Concat(nn.Module):
    def __init__(self, cls_num, cls_type = 'binary'):
        super(Concat, self).__init__()
        self.cls_num = cls_num
        self.cls_type = cls_type

        self.fc1 = nn.Linear(4, 10)
        
        if cls_type == 'binary':
            self.output_layer = nn.Linear(10, 1)
        else:
            self.output_layer = nn.Linear(10, cls_num)


    def forward(self, x):
        
        

        x = F.relu(self.fc1(x))
        x = self.output_layer(x)

        if self.cls_type == 'binary':
            x = torch.sigmoid(x)
        else:
            
            x = F.log_softmax(x, dim=1)

        return x

class DIT(nn.Module):
    def __init__(self, mlp, dnn, cnn, lstm, meta_learner):
        super().__init__()
        self.mlp = mlp
        self.dnn = dnn
        self.cnn = cnn
        self.lstm = lstm
        self.meta = meta_learner

    def forward(self, x):
        out1 = self.mlp(x)
        out2 = self.dnn(x)
        out3 = self.cnn(x)
        out4 = self.lstm(x)

        
        combined = torch.cat([out1, out2, out3, out4], dim=1)
        
        final_output = self.meta(combined)
        return final_output


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flwr")



def train(net, trainloader, epochs, cls_type = 'binary', device=None):
    net.to(device)
    net.train()

    
    
    
    criterion = torch.nn.BCELoss() if cls_type == 'binary' else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    for _ in range(epochs):
        for inputs, labels in trainloader:
            labels = labels.view(-1, 1)
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.float() if cls_type == 'binary' else labels)
            loss.backward()
            optimizer.step()








from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def test(net, testloader, cls_type='binary', device=None):
    net.to(device)
    net.eval()

    criterion = torch.nn.BCELoss() if cls_type == 'binary' else torch.nn.CrossEntropyLoss()

    label_col = []
    pred_col = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in testloader:
            labels = labels.view(-1, 1) 
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = net(inputs)

            loss = criterion(outputs, labels.float() if cls_type == 'binary' else labels)
            total_loss += loss.item()

            if cls_type == 'binary':
                preds = (outputs > 0.5).int().cpu().numpy()
                labels_np = labels.int().cpu().numpy()
            else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()

            pred_col.extend(preds.flatten())
            label_col.extend(labels_np.flatten())

    total_loss = total_loss / len(testloader)
    accuracy = (np.array(pred_col) == np.array(label_col)).mean()
    precision = precision_score(label_col, pred_col)
    recall = recall_score(label_col, pred_col)
    f1 = f1_score(label_col, pred_col)

    # FPR
    tn, fp, fn, tp = confusion_matrix(label_col, pred_col).ravel()
    fpr = fp / (fp + tn)

    tup = (total_loss, accuracy, precision, recall, f1, fpr)

    
    
    return tup









class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, testloader, binary='binary'):
        global device
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.binary = binary
        self.device = device
        
        

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict(zip(self.net.state_dict().keys(), [torch.tensor(p).to(self.device) for p in parameters]))
        self.net.load_state_dict(state_dict, strict=True)

    
    def evaluate(self, parameters, config):
        loss, accuracy, precision, recall, f1, fpr = test(
    self.net, self.testloader, cls_type=self.binary, device=self.device
)
        return float(loss), len(self.testloader.dataset), {
                                                          "accuracy": float(accuracy),
                                                          "precision": float(precision),
                                                          "recall" : float(recall),
                                                          "f1" : float(f1),
                                                          "fpr" : float(fpr)
                                                         }

       

    def fit(self, parameters, config):            
        self.set_parameters(parameters)
        epochs = config.get("epochs", 1)
        train(self.net, self.trainloader, epochs, cls_type = self.binary, device=self.device)
        
        val_loss, val_acc, val_prec, val_rec, val_f1, val_fpr = test(
    self.net, self.valloader, cls_type=self.binary, device=self.device
)


        
        return self.get_parameters({}), len(self.trainloader.dataset), {
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "val_prec" : float(val_prec),
        "val_f1": float(val_f1),
        "val_fpr": float(val_fpr),
        "val_rcl": float(val_rec)}
   
cid_counter = -1

def client_fn(context: Context) -> Client:
    
    global cid_counter
    cid_counter += 1  # increments each time client_fn is called
    cid = cid_counter
    assert cid < len(client_loaders), f"cid {cid} exceeds client_loader size"
    
    size = 77
    cls_num = 2
    cls_type = 'binary'
    
    mlp = MLP(i_size=size, cls_num=cls_num, cls_type=cls_type)
    dnn = DNN(i_size=size, cls_num=cls_num, cls_type=cls_type)
    cnn = CNN(i_size=size, cls_num=cls_num, cls_type=cls_type)
    lstm = LSTM(i_size=size, cls_num=cls_num, cls_type=cls_type)

    
    
    meta = Concat(cls_num=cls_num, cls_type=cls_type)
    
   
    net = DIT(mlp, dnn, cnn, lstm, meta)

    
    # Get client's train and val loaders from global list
    train_loader, val_loader = client_loaders[int(cid)]
    
    # Create FlowerClient with ensemble model and loaders
    return FlowerClient(net, train_loader, val_loader, test_loader, binary='binary').to_client()



client = ClientApp(client_fn=client_fn)

# Start simulation

def aggregate_evaluate_metrics(metrics):
    num_clients = len(metrics)

    avg_acc = sum(m[1]["accuracy"] for m in metrics) / num_clients
    avg_prec = sum(m[1]["precision"] for m in metrics) / num_clients
    avg_rec = sum(m[1]["recall"] for m in metrics) / num_clients
    avg_f1 = sum(m[1]["f1"] for m in metrics) / num_clients
    avg_fpr = sum(m[1]["fpr"] for m in metrics) / num_clients

    return {
        "accuracy": avg_acc,
        "precision": avg_prec,
        "recall": avg_rec,
        "f1": avg_f1,
        "fpr": avg_fpr
    }

def aggregate_fit_metrics(metrics):
    num_clients = len(metrics)
    
    avg_loss = sum(m[1]["val_loss"] for m in metrics) / num_clients
    avg_acc = sum(m[1]["val_accuracy"] for m in metrics) / num_clients
    avg_prec = sum(m[1]["val_prec"] for m in metrics) / num_clients
    avg_rcl = sum(m[1]["val_rcl"] for m in metrics) / num_clients
    avg_f1 = sum(m[1]["val_f1"] for m in metrics) / num_clients
    avg_fpr = sum(m[1]["val_fpr"] for m in metrics) / num_clients
    return {
        "loss" : avg_loss,
        "accuracy": avg_acc,
        "f1": avg_f1,
        "precision": avg_prec,
        "recall": avg_rcl,
        "fpr": avg_fpr
    }






from flwr.server.strategy import FedAvg
from flwr.common import FitIns

class Cus_FedAvg(FedAvg):
    def configure_fit(self, server_round, parameters, client_manager):
        config = {"epochs": 5} 
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients,
        )
        return [(client, FitIns(parameters, config)) for client in clients]







strategy = Cus_FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=10,
    min_evaluate_clients=8,
    min_available_clients=10,
    fit_metrics_aggregation_fn=aggregate_fit_metrics,
    evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
)




if device == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}

else:
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}



def server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)



server = ServerApp(server_fn=server_fn)


run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=10,
    backend_config=backend_config,
)