
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




# Loading the dataset


print("Preprocessing steps completed.")




data = pd.read_csv("cic.csv")



data_x = data.iloc[:500, :-1].values  
data_y = data.iloc[:500, -1].values 


pca_cic = {"dt":[], "nb":[], "lg":[], "dnn":[], "cnn":[], "rnn":[]}
lda_cic = {"dt":[], "nb":[], "lg":[], "dnn":[], "cnn":[], "rnn":[]}
ae_cic  = {"dt":[], "nb":[], "lg":[], "dnn":[], "cnn":[], "rnn":[]}







from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def PCA_FUNC(data):
    pca = PCA(n_components=20, svd_solver='auto')  
    x_pca = pca.fit_transform(data)  # output to be used 

    # # explained variance
    # var = pca.explained_variance_ratio_
    # print("Data Variance captured by PCA:", var)


    # plt.figure(figsize=(8,6))
    # plt.plot(range(1, len(var) + 1), var.cumsum(), marker='o', linestyle='--')
    # plt.title('Cumulative Explained Variance by PC')
    # plt.xlabel('Number of Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.grid(True)
    # plt.show()
    
    
    # cumulative_variance = var.cumsum()
    # components_for_95 = (cumulative_variance >= 0.95).argmax() + 1
    # print(f"Number of components required to explain 95% of the variance: {components_for_95}")
    return x_pca
    


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def LDA_FUNC(x, y):
    lda = LDA(n_components=1)  # 1 as binary classification 
    x_lda = lda.fit_transform(x, y) # output to be used 
    
    
    # var = lda.explained_variance_ratio_
    # print("Explained variance by the LDA component:", var)
    
    
    # print("LDA-transformed data shape:", x_lda.shape)
    
    
    # import matplotlib.pyplot as plt
    
    # # Visualize  transformed data 
    # plt.scatter(x_lda, y, c=y, cmap='viridis')
    # plt.title('LDA Projection')
    # plt.xlabel('LDA Component 1')
    # plt.ylabel('Class Labels')
    # plt.colorbar()
    # plt.show()
    return x_lda


class AutoEncoder(nn.Module):
    def __init__(self, input_size=38, bottleneck_size=20):
        super(AutoEncoder, self).__init__()
        
        # Encoder 
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),  
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(64, 30), 
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(30, bottleneck_size),
            nn.ReLU()
        )
        
        # Decoder 
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 30),  
            nn.ReLU(),
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, input_size), 
            nn.ReLU()  #
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_AE(net, x_train, epochs=3, batch_size=32, lr=0.001, wd=1e-4):
    
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    
    
    criterion = nn.MSELoss()  
    
    
    net.to(device)
    net.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(x_train), batch_size):
            batch_data = torch.tensor(x_train[i:i+batch_size], dtype=torch.float32).to(device)
            
            optimizer.zero_grad()  
            
            
            outputs = net(batch_data)
            
            
            loss = criterion(outputs, batch_data)
            epoch_loss += loss.item()
            
            
            loss.backward()
            optimizer.step()
        
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(x_train):.4f}")
    return net




# Decision Tree
from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    def __init__(self, max_depth=None, random_state=42):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def fit(self, x_train, y_train):
        train_shallow(self.model, x_train, y_train)

    def test(self, x_test, y_test):
        return test_shallow(self.model, x_test, y_test)
    


# Logistic Regression

from sklearn.linear_model import LogisticRegression

class LogisticRegressor:
    def __init__(self, max_iter=100, solver='lbfgs', random_state=42):
        self.model = LogisticRegression(solver=solver, max_iter=max_iter, random_state=random_state)

    def fit(self, x_train, y_train):
        train_shallow(self.model, x_train, y_train)

    def test(self, x_test, y_test):
        return test_shallow(self.model, x_test, y_test)
    




# Naive Bayes
from sklearn.naive_bayes import GaussianNB

class NaiveBayes:
    def __init__(self):
        self.model = GaussianNB()

    def fit(self, x_train, y_train):
        train_shallow(self.model, x_train, y_train)

    def test(self, x_test, y_test):
        return test_shallow(self.model, x_test, y_test)
    


from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, log_loss

# Train Model
def train_shallow(model, x_train, y_train):
    model.fit(x_train, y_train)


# Test Model, Calculate Metrics
def test_shallow(model, x_test, y_test):
    
    pred_col = model.predict(x_test)
    prob_col = model.predict_proba(x_test)[:, 1]  
    loss = log_loss(y_test, prob_col)
    
    # Confusion matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_test, pred_col).ravel()
    
    # Calculate metrics
    acc = (tp + tn) / (tp + tn + fp + fn)  # Accuracy
    dr = tp / (tp + fn)  # detection rate / recall
    if (tp + fp) > 0:
        prec = tp / (tp + fp)  # precision
    else:
        prec = 0
    f1 = 2 * prec * dr / (prec + dr)  # F1 
    fpr = fp / (fp + tn)  # False Positive Rate
    
    # AUC 
    auc = roc_auc_score(y_test, prob_col)
    
    # # ROC  plotting
    # fpr_curve, tpr_curve, _ = roc_curve(y_test, prob_col)
    # plt.figure()
    # plt.plot(fpr_curve, tpr_curve, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='lower right')
    # plt.show()
    
    return loss, acc, dr, prec, f1, fpr, auc



#########################################################################################
# DFF
class DNN(nn.Module):
    def __init__(self, input_s, H_L_sizes=[20, 20, 20], output_s=1, dropout_rate=0.2):
        super(DNN, self).__init__()
        
        
        layers = []
        
        # input layer
        layers.append(nn.Linear(input_s, H_L_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))  #  for regularization

        # Hidden layers
        for i in range(1, len(H_L_sizes)):
            layers.append(nn.Linear(H_L_sizes[i-1], H_L_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Dropout for regularization
        
        # Output layer 
        layers.append(nn.Linear(H_L_sizes[-1], output_s))
        layers.append(nn.Sigmoid())  

        # Combine layers 
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)






class CNN(nn.Module):
    def __init__(self, input_size, num_filters=20, kernel_size=3, output_size=1, dropout_rate=0.2):
        super(CNN, self).__init__()

        self.input_size = input_size
        
            

        # Convolutional layers with padding
        if input_size < 10:
            kernel_size = 1    
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size)


        
        if input_size < 10:
            kernel_size = 1
        else:
            kernel_size -= 1
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size)


        
        if input_size < 10:
            kernel_size = 1
        else:
            kernel_size -= 1
        self.conv3 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size)



        
        # Max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Fully connected layer
        self.fc1 = nn.Linear(self.fc1_size(input_size), output_size)
        self.output = nn.Sigmoid()
        # Dropout layer 
        self.dropout = nn.Dropout(dropout_rate)

    

    def forward(self, x):
        x = x.unsqueeze(1) 
        
        if self.input_size >= 10:
            x = F.relu(self.conv1(x))  
            x = self.pool(x) 
            
            x = F.relu(self.conv2(x))  
            x = self.pool(x) 
    
            x = F.relu(self.conv3(x))  
            x = self.pool(x)  
    
        else:
            x = F.relu(self.conv1(x))  
            
        
        
        x = x.view(x.size(0), -1)            
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.output(x)
        

    def fc1_size(self, input_size):
        
        # Start with the input size
        x = torch.zeros(1, 1, input_size)
        if self.input_size >= 10:
            x = self.conv1(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.pool(x)
        else:
            x = self.conv1(x)
           
            
        return x.view(1, -1).size(1)
    





 # RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=1, dropout_rate=0.2):
        super(RNN, self).__init__()

        # LSTM Layer with 2 layers as specified
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)


        # Dropout layer 
        self.dropout = nn.Dropout(dropout_rate)

        # fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        # Pass through LSTM layers
        x = x.unsqueeze(1) 
        x, (hn, cn) = self.lstm1(x) 
        x = self.dropout(x)
        x, (hn, cn) = self.lstm2(x)
        x = self.dropout(x)
        # Use the last hidden state 
        x = hn[-1]

        # output
        x = torch.sigmoid(self.fc(x))
        return x







def train(net, trainloader, epochs = 50, cls_type = 'binary', device=None):
    net.to(device)
    net.train()

    
    
    
    criterion = torch.nn.BCELoss() if cls_type == 'binary' else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    for _ in range(epochs):
        for inputs, labels in trainloader:
            labels = labels.view(-1, 1)
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.float() if cls_type == 'binary' else labels)
            loss.backward()
            optimizer.step()
    return net




def test(net, val_loader, cls_type='binary', device=None):
    net.to(device)
    net.eval()

    criterion = torch.nn.BCELoss() if cls_type == 'binary' else torch.nn.CrossEntropyLoss()

    label_col = []
    pred_col = []
    prob_col = []  
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = labels.view(-1, 1) 
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = net(inputs)

            loss = criterion(outputs, labels.float() if cls_type == 'binary' else labels)
            total_loss += loss.item()

            if cls_type == 'binary':
                preds = (outputs > 0.5).int().cpu().numpy()
                prob = outputs.cpu().numpy()[:, 0]  
                labels_np = labels.int().cpu().numpy()
            
            else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  
                labels_np = labels.cpu().numpy()

            pred_col.extend(preds.flatten())
            label_col.extend(labels_np.flatten())
            prob_col.extend(prob.flatten())

    total_loss = total_loss / len(val_loader)

    
    

    
    tn, fp, fn, tp = confusion_matrix(label_col, pred_col).ravel()
    
    # Calculate metrics
    acc = (tp + tn) / (tp + tn + fp + fn)  # Accuracy
    dr = tp / (tp + fn)  # detection rate / recall
    if (tp + fp) > 0:
        prec = tp / (tp + fp)  # precision
    else:
        prec = 0
    f1 = 2 * prec * dr / (prec + dr)  # F1 
    fpr = fp / (fp + tn)  # False Positive Rate    
    
    auc = roc_auc_score(label_col, prob_col)

    # # ROC plotting
    # fpr_curve, tpr_curve, _ = roc_curve(label_col, prob_col)
    # plt.figure()
    # plt.plot(fpr_curve, tpr_curve, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='lower right')
    # plt.show()

    
    tup = (total_loss, acc, dr, prec, f1, fpr, auc)

    return tup



AE = AutoEncoder(77)
x_train, _, _, _ = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
AE = train_AE(AE, x_train)



from sklearn.model_selection import StratifiedKFold


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []


for fold, (train_idx, val_idx) in enumerate(skf.split(data_x, data_y)):
    print(f"\nFold {fold + 1}/5")

    # Split train and validation sets
    x_tr, x_val = data_x[train_idx], data_x[val_idx]
    y_tr, y_val = data_y[train_idx], data_y[val_idx]

    

    
    # Feature Extraction Used 
    
    print("\nUSING PCA / PRINCIPAL COMPONENT ANALYSIS")
    x = PCA_FUNC(x_tr)
    val = PCA_FUNC(x_val)

    train_loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32),
                                            torch.tensor(y_tr, dtype=torch.long)), batch_size=64, shuffle=True)

    val_loader = DataLoader(TensorDataset(torch.tensor(val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.long)), batch_size=64, shuffle=False)

    
    # PCA with DT
    print("Metrics for Decision Tree")
    dt = DecisionTree(max_depth=5)
    dt.fit(x, y_tr)
    loss, acc, dr, prec, f1, fpr, auc = dt.test(val, y_val)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    pca_cic["dt"].append([loss, acc, dr, prec, f1, fpr, auc])

    
    # PCA with NB
    print(" Metrics for Naive Bayes")
    nb = NaiveBayes()
    nb.fit(x, y_tr)
    loss, acc, dr, prec, f1, fpr, auc = nb.test(val, y_val)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    pca_cic["nb"].append([loss, acc, dr, prec, f1, fpr, auc])
    
    # PCA with LR    
    print(" Metrics for Logistic Regression")
    lg = LogisticRegressor(max_iter=500)
    lg.fit(x, y_tr)
    loss, acc, dr, prec, f1, fpr, auc = lg.test(val, y_val)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    pca_cic["lg"].append([loss, acc, dr, prec, f1, fpr, auc])

    
    # DEEP MODELS
    
    # PCA with DNN  
    print("Metrics for Deep Feed Forward Network")
    dnn = DNN(20)
    trained_m = train(dnn, train_loader, device=device)
    loss, acc, dr, prec, f1, fpr, auc = test(trained_m, val_loader, device=device)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    pca_cic["dnn"].append([loss, acc, dr, prec, f1, fpr, auc])

    # PCA with CNN  
    print("Metrics for Convolution Neural Network")
    cnn = CNN(20)
    trained_m = train(cnn, train_loader, device=device)
    loss, acc, dr, prec, f1, fpr, auc = test(trained_m, val_loader, device=device)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    pca_cic["cnn"].append([loss, acc, dr, prec, f1, fpr, auc])

    # PCA with RNN  
    print("Metrics for Recurrent Neural Network")
    rnn = RNN(20)
    trained_m = train(rnn, train_loader, device=device)
    loss, acc, dr, prec, f1, fpr, auc = test(trained_m, val_loader, device=device)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    pca_cic["rnn"].append([loss, acc, dr, prec, f1, fpr, auc])




########################################################################################################################################################################################################




    print("\nUSING LDA / LINEAR DISCRIMINANT ANALYSIS")
    x = LDA_FUNC(x_tr, y_tr)
    val = LDA_FUNC(x_val, y_val)
    train_loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32),
                                            torch.tensor(y_tr, dtype=torch.long)), batch_size=64, shuffle=True)

    val_loader = DataLoader(TensorDataset(torch.tensor(val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.long)), batch_size=64, shuffle=False)
    
    # LDA with DT
    print(" Metrics for Decision Tree")
    dt = DecisionTree(max_depth=5)
    dt.fit(x, y_tr)
    loss, acc, dr, prec, f1, fpr, auc = dt.test(val, y_val)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    lda_cic["dt"].append([loss, acc, dr, prec, f1, fpr, auc])

    # LDA with NB
    print(" Metrics for Naive Bayes")
    nb = NaiveBayes()
    nb.fit(x, y_tr)
    loss, acc, dr, prec, f1, fpr, auc = nb.test(val, y_val)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    lda_cic["nb"].append([loss, acc, dr, prec, f1, fpr, auc])
    
    # LDA with LR    
    print(" Metrics for Logistic Regression")
    lg = LogisticRegressor(max_iter=500)
    lg.fit(x, y_tr)
    loss, acc, dr, prec, f1, fpr, auc = lg.test(val, y_val)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    lda_cic["lg"].append([loss, acc, dr, prec, f1, fpr, auc])

    # DEEP MODELS
    
    # LDA with DNN  
    print("Metrics for Deep Feed Forward Network")
    dnn = DNN(1)
    trained_m = train(dnn, train_loader, device=device)
    loss, acc, dr, prec, f1, fpr, auc = test(trained_m, val_loader, device=device)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    lda_cic["dnn"].append([loss, acc, dr, prec, f1, fpr, auc])

    # LDA with CNN  
    print("Metrics for Convolution Neural Network")
    cnn = CNN(1, kernel_size = 1)
    trained_m = train(cnn, train_loader, device=device)
    loss, acc, dr, prec, f1, fpr, auc = test(trained_m, val_loader, device=device)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    lda_cic["cnn"].append([loss, acc, dr, prec, f1, fpr, auc])

    # LDA with RNN  
    print("Metrics for Recurrent Neural Network")
    rnn = RNN(1)
    trained_m = train(rnn, train_loader, device=device)
    loss, acc, dr, prec, f1, fpr, auc = test(trained_m, val_loader, device=device)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    lda_cic["rnn"].append([loss, acc, dr, prec, f1, fpr, auc])




#########################################################################################################################################################################################################






    
    print("\nUSING AE / Auto Encoder")
    AE.eval()
    with torch.no_grad():
        # TO TENSOR
        x_tr_tensor = torch.tensor(x_tr, dtype=torch.float32).to(device)
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
        # ENCODING                 
        deep_x = AE.encode(x_tr_tensor)
        deep_x_val = AE.encode(x_val_tensor)
        
        # TO NUMPY
        x = deep_x.cpu().numpy()
        val = deep_x_val.cpu().numpy()

        train_loader = DataLoader(TensorDataset(deep_x,  torch.tensor(y_tr, dtype=torch.long)), batch_size=64, shuffle=True)

        val_loader = DataLoader(TensorDataset(deep_x_val,torch.tensor(y_val, dtype=torch.long)), batch_size=64, shuffle=False)
    
    # AE with DT
    print(" Metrics for Decision Tree")
    dt = DecisionTree(max_depth=5)
    dt.fit(x, y_tr)
    loss, acc, dr, prec, f1, fpr, auc = dt.test(val, y_val)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    ae_cic["dt"].append([loss, acc, dr, prec, f1, fpr, auc])

    # AE with NB
    print(" Metrics for Naive Bayes")
    nb = NaiveBayes()
    nb.fit(x, y_tr)
    loss, acc, dr, prec, f1, fpr, auc = nb.test(val, y_val)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    ae_cic["nb"].append([loss, acc, dr, prec, f1, fpr, auc])
    
    # AE with LR    
    print(" Metrics for Logistic Regression")
    lg = LogisticRegressor(max_iter=500)
    lg.fit(x, y_tr)
    loss, acc, dr, prec, f1, fpr, auc = lg.test(val, y_val)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    ae_cic["lg"].append([loss, acc, dr, prec, f1, fpr, auc])


    # DEEP MODELS
    
    # AE with DNN  
    print("Metrics for Deep Feed Forward Network")
    dnn = DNN(20)
    trained_m = train(dnn, train_loader, device=device)
    loss, acc, dr, prec, f1, fpr, auc = test(trained_m, val_loader, device=device)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    ae_cic["dnn"].append([loss, acc, dr, prec, f1, fpr, auc])

    # AE with CNN  
    print("Metrics for Convolution Neural Network")
    cnn = CNN(20)
    trained_m = train(cnn, train_loader, device=device)
    loss, acc, dr, prec, f1, fpr, auc = test(trained_m, val_loader, device=device)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    ae_cic["cnn"].append([loss, acc, dr, prec, f1, fpr, auc])

    # AE with RNN  
    print("Metrics for Recurrent Neural Network")
    rnn = RNN(20)
    trained_m = train(rnn, train_loader, device=device)
    loss, acc, dr, prec, f1, fpr, auc = test(trained_m, val_loader, device=device)
    print(f"Loss: {loss:.4f} , Accuracy: {acc:.4f}, DR (Recall): {dr:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}, FPR: {fpr:.4f}, AUC: {auc:.4f}")
    ae_cic["rnn"].append([loss, acc, dr, prec, f1, fpr, auc]) 





# Function to calculate averages for each metric
def calc_avj(dictt):
    avg = {}
    for met, results in dictt.items():
        avg[met] = {}
        
        # Convert the nested list to a numpy array
        results_arr = np.array(results)  # Shape: (n_folds, 7)
        
        # Calculate the average for each metric across all folds
        avg[met] = {
            "Loss": np.mean(results_arr[:, 0]),
            "Accuracy": np.mean(results_arr[:, 1]),
            "DR (Recall)": np.mean(results_arr[:, 2]),
            "Precision": np.mean(results_arr[:, 3]),
            "F1 Score": np.mean(results_arr[:, 4]),
            "FPR": np.mean(results_arr[:, 5]),
            "AUC": np.mean(results_arr[:, 6])
        }
    return avg

# Calculate averages for PCA, LDA, and AE
avg_pca = calc_avj(pca_cic)
avg_lda = calc_avj(lda_cic)
avg_ae = calc_avj(ae_cic)

# Function to print the averages in a readable format
def print_avg(avj, met_name):
    print(f"\nAverage metrics for {met_name}:")
    for model, metrics in avj.items():
        print(f"\n{model}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

# Printing the averaged results
print_avg(avg_pca, "PCA")
print_avg(avg_lda, "LDA")
print_avg(avg_ae, "AE")
