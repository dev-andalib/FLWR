import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd





import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict



# load 

data= pd.read_csv("cic.csv")
d = data.copy()
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
print(data.shape)


data_x = data.iloc[:, :-1].values  
data_y = data.iloc[:, -1].values 

x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.3, stratify=data_y, random_state=42
)

# Convert test set to tensors
x_test_t = torch.tensor(x_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)
test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=64, shuffle=False)



from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cls_type = 'binary'
size = 78
cls_num = 2





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
         
        
        
        
        
        self.fc1 = nn.Linear(self.fc1_size(i_size), 24)
        
        
        if cls_type == 'binary':
            self.output_layer = nn.Linear(24, 1)
        else:
            self.output_layer = nn.Linear(24, cls_num)

    
    def fc1_size(self, input_size):
        
        # Start with the input size
        x = torch.zeros(1, 1, input_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
           
            
        return x.view(1, -1).size(1)

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

def validate(net, testloader, cls_type='binary', device=None):
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




for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
    print(f"\nFold {fold + 1}/5")

    # Split train and validation sets
    x_tr, x_val = x_train[train_idx], x_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(torch.tensor(x_tr, dtype=torch.float32),
                                            torch.tensor(y_tr, dtype=torch.long)), batch_size=64, shuffle=True)

    val_loader = DataLoader(TensorDataset(torch.tensor(x_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.long)), batch_size=64, shuffle=False)

    # Define model
    mlp = MLP(i_size=size, cls_num=cls_num, cls_type=cls_type)
    dnn = DNN(i_size=size, cls_num=cls_num, cls_type=cls_type)
    cnn = CNN(i_size=size, cls_num=cls_num, cls_type=cls_type)
    lstm = LSTM(i_size=size, cls_num=cls_num, cls_type=cls_type)
    meta = Concat(cls_num=cls_num, cls_type=cls_type)

    net = DIT(mlp, dnn, cnn, lstm, meta)

    # Train model
    train(net, train_loader, epochs=10, cls_type=cls_type, device=device)

    # Validate model
    val_metrics = validate(net, val_loader, cls_type=cls_type, device=device)
    acc = val_metrics[1]  # accuracy is the 2nd element

    cv_scores.append(acc)
    print(f"Fold {fold + 1} validation accuracy: {acc:.4f}")

# Final average score
print(f"\nAverage CV validation accuracy: {np.mean(cv_scores):.4f}")



full_train_loader = DataLoader(
    TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.long)),
    batch_size=64,
    shuffle=True
)


mlp = MLP(i_size=size, cls_num=cls_num, cls_type=cls_type)
dnn = DNN(i_size=size, cls_num=cls_num, cls_type=cls_type)
cnn = CNN(i_size=size, cls_num=cls_num, cls_type=cls_type)
lstm = LSTM(i_size=size, cls_num=cls_num, cls_type=cls_type)
meta = Concat(cls_num=cls_num, cls_type=cls_type)

net = DIT(mlp, dnn, cnn, lstm, meta)

# Train final model
train(net, full_train_loader, epochs=5, cls_type=cls_type, device=device)





# Evaluate on 30% test set
metrics = validate(net, test_loader, cls_type=cls_type, device=device)

# Display results
print("\nTest Set Performance:")
print(f"Loss:      {metrics[0]:.4f}")
print(f"Accuracy:  {metrics[1]:.4f}")
print(f"Precision: {metrics[2]:.4f}")
print(f"Recall:    {metrics[3]:.4f}")
print(f"F1-Score:  {metrics[4]:.4f}")
print(f"FPR:       {metrics[5]:.4f}")