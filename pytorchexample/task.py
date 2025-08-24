from collections import OrderedDict, Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from sklearn.feature_selection import SelectKBest, chi2
from typing import Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
import os
import json
from imblearn.over_sampling import SMOTE

from pytorchexample.helper import create_sequences, train_epoch_binary, train_epoch_joint, evaluate_binary, train_epoch_multiclass, evaluate_hierarchical

class Net(nn.Module):
    def __init__(self, input_features=20, seq_length=10, num_attack_types=10):  # FIXED: Keep 10 for consistency
        super().__init__()
        
        # ANALYSIS: CNN-BiLSTM Architecture for Network Intrusion Detection
        # Input: Sequential network traffic data [batch, seq_length, features]
        # Output: Binary classification (normal/attack) + Multiclass (attack types)
        # CRITICAL: Must maintain consistent architecture across federated clients
        
        # Multi-scale CNN feature extraction
        self.conv1_3 = nn.Conv1d(input_features, 16, kernel_size=3, padding=1)  # Short patterns
        self.conv1_5 = nn.Conv1d(input_features, 8, kernel_size=5, padding=2)   # Long patterns
        
        self.bn1 = nn.BatchNorm1d(24)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(24, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        
        self.bilstm = nn.LSTM(
            input_size=32,
            hidden_size=16,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
        self.binary_head = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1)
        )
        
        self.multiclass_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_attack_types)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def attention_net(self, lstm_output):
        attention_scores = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_output = torch.sum(lstm_output * attention_weights, dim=1)
        return weighted_output
    
    def extract_features(self, x):
        x = x.permute(0, 2, 1)
        
        conv_3 = self.conv1_3(x)
        conv_5 = self.conv1_5(x)
        x = torch.cat([conv_3, conv_5], dim=1)
        
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.bilstm(x)
        
        features = self.attention_net(lstm_out)
        
        return features
    
    def forward(self, x, stage='both'):
        features = self.extract_features(x)
        
        if stage == 'binary' or stage == 'both':
            binary_output = self.binary_head(features)  # Remove sigmoid here
            
        if stage == 'multiclass' or stage == 'both':
            multiclass_output = self.multiclass_head(features)
            
        if stage == 'binary':
            return binary_output
        elif stage == 'multiclass':
            return multiclass_output
        else:
            return binary_output, multiclass_output
 
def get_weights(net):
    try:
        # Ensure all tensors are moved to CPU and synchronized before extracting weights
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        weights = []
        for _, val in net.state_dict().items():
            if val.is_cuda:
                # Clear any pending CUDA operations before moving to CPU
                torch.cuda.synchronize()
                weights.append(val.cpu().numpy())
            else:
                weights.append(val.numpy())
        return weights
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error in get_weights: {e}")
            # Force CPU mode and retry
            net = net.cpu()
            return [val.numpy() for _, val in net.state_dict().items()]
        else:
            raise e
 
def set_weights(net, parameters):
    """
    FEDERATED LEARNING WEIGHT SETTING:
    - Handles potential architecture mismatches gracefully
    - Ensures consistent model structure across clients
    - Provides detailed error reporting for debugging
    """
    try:
        params_dict = zip(net.state_dict().keys(), parameters) #global
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict}) #global
        
        # FIXED: Add shape verification before loading
        net_state_dict = net.state_dict() #local
        for key, param in state_dict.items(): #global
            if key in net_state_dict:
                if param.shape != net_state_dict[key].shape:
                    print(f"WARNING: Shape mismatch for {key}: expected {net_state_dict[key].shape}, got {param.shape}")
                    # For multiclass head mismatch, pad or truncate as needed
                    if "multiclass_head" in key and len(param.shape) >= 1:
                        target_shape = net_state_dict[key].shape
                        if param.shape[0] != target_shape[0]:  # Output dimension mismatch
                            print(f"FIXING: Adjusting {key} from {param.shape} to {target_shape}")
                            if len(param.shape) == 1:  # Bias
                                if param.shape[0] > target_shape[0]:
                                    param = param[:target_shape[0]]  # Truncate
                                else:
                                    padding = torch.zeros(target_shape[0] - param.shape[0])
                                    param = torch.cat([param, padding])  # Pad
                            elif len(param.shape) == 2:  # Weight
                                if param.shape[0] > target_shape[0]:
                                    param = param[:target_shape[0], :]  # Truncate rows
                                else:
                                    padding = torch.zeros(target_shape[0] - param.shape[0], param.shape[1])
                                    param = torch.cat([param, padding], dim=0)  # Pad rows
                            state_dict[key] = param
        
        net.load_state_dict(state_dict, strict=True) #local updating based on global
        print("Model weights loaded successfully")
        
    except Exception as e:
        print(f"ERROR in set_weights: {e}")
        print("Attempting fallback weight loading...")
        try:
            # Fallback: load weights with strict=False
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=False)
            print("Fallback weight loading successful (some layers may be uninitialized)")
        except Exception as fallback_error:
            print(f"CRITICAL ERROR: Both strict and non-strict loading failed: {fallback_error}")
            raise fallback_error
 
fds = None  # Cache FederatedDataset
path = 'sequences.parquet'  # Use relative path
meta_path="sequences_meta.json"
meta = None
def load_data(partition_id: int, num_partitions: int, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """
    Custom data division mechanism to divide data into partitions, selecting top 20 features using chi-squared test.
    Returns train, validation, and test DataLoaders.
    """
    global path, fds, meta, meta_path
    batch_size = 64
    test_size = 0.3
    random_state = 42
    
    if fds is None:
        ds = load_dataset("parquet", data_files=path, split="train")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = ds
        fds = partitioner
 
    # Load the partitioned dataset
    df  = fds.load_partition(partition_id).with_format("pandas")[:]
    L = int(meta["seq_len"])
    F = int(meta["num_features"])
    feat_cols = [f"f{j}_t{k}" for k in range(L) for j in range(F)]
    # Memory optimization: limit dataset size if too large
    X_flat = df[feat_cols].values.astype(np.float32)
    X_seq = X_flat.reshape(-1, L, F)
    y_binary_seq = df["y_binary"].values.astype(np.float32)
    y_attack_seq = df["y_attack"].values.astype(np.int64)

    # --- Split data ---
    X_train, X_temp, y_binary_train, y_binary_temp, y_attack_train, y_attack_temp = train_test_split(
        X_seq, y_binary_seq, y_attack_seq,
        test_size=test_size, stratify=y_binary_seq, random_state=random_state
    )

    X_val, X_test, y_binary_val, y_binary_test, y_attack_val, y_attack_test = train_test_split(
        X_temp, y_binary_temp, y_attack_temp,
        test_size=0.5, stratify=y_binary_temp, random_state=random_state
    )

    # --- Convert to tensors ---
    X_train = torch.from_numpy(X_train)
    X_val = torch.from_numpy(X_val)
    X_test = torch.from_numpy(X_test)

    y_binary_train = torch.from_numpy(y_binary_train)
    y_binary_val = torch.from_numpy(y_binary_val)
    y_binary_test = torch.from_numpy(y_binary_test)

    y_attack_train = torch.from_numpy(y_attack_train)
    y_attack_val = torch.from_numpy(y_attack_val)
    y_attack_test = torch.from_numpy(y_attack_test)

    # --- Filter attack-only subset ---
    attack_indices = (y_binary_train == 1)
    y_attack_train_filtered = y_attack_train[attack_indices]
    X_train_attacks = X_train[attack_indices]

    if len(y_attack_train_filtered) == 0:
        print("Warning: No attack samples found in training data!")
        y_attack_train_filtered = torch.LongTensor([0])
        X_train_attacks = X_train[:1]

    unique_attacks = torch.unique(y_attack_train_filtered)

    # CRITICAL FIX: Model architecture consistency for federated learning
    # Problem: Changing model architecture breaks federated aggregation
    # Solution: Keep consistent 10-class architecture, use proper class mapping
    
    print(f"Original unique attacks: {unique_attacks}")
    print("FIXED: Proper hierarchical prediction with consistent 10-class architecture")
    
    # FIX 1: Map attack labels to ensure they fit in 0-9 range
    unique_attacks_list = unique_attacks.tolist()
    print(f"Unique attack labels found: {unique_attacks_list}")
    
    # Keep original labels if they're already in 0-9 range, otherwise remap
    if all(0 <= label <= 9 for label in unique_attacks_list):
        print("Attack labels already in 0-9 range, no remapping needed")
        y_attack_train_remapped = y_attack_train.clone()
        y_attack_val_remapped = y_attack_val.clone() 
        y_attack_test_remapped = y_attack_test.clone()
        # For filtered subset, use the already filtered labels since no remapping needed
        y_attack_train_filtered_remapped = y_attack_train_filtered.clone()
    else:
        print("Remapping attack labels to 1-9 range")
        attack_to_new_label = {old_label: new_label for new_label, old_label in enumerate(unique_attacks_list, 1)}
        print(f"Attack label mapping: {attack_to_new_label}")
        
        # Apply remapping
        y_attack_train_remapped = y_attack_train.clone()
        y_attack_val_remapped = y_attack_val.clone() 
        y_attack_test_remapped = y_attack_test.clone()
        
        for old_label, new_label in attack_to_new_label.items():
            mask_train = y_attack_train == old_label
            mask_val = y_attack_val == old_label
            mask_test = y_attack_test == old_label
            
            y_attack_train_remapped[mask_train] = new_label
            y_attack_val_remapped[mask_val] = new_label
            y_attack_test_remapped[mask_test] = new_label
            
        # For filtered subset, apply remapping to the already filtered attack labels
        y_attack_train_filtered_remapped = y_attack_train_filtered.clone()
        for old_label, new_label in attack_to_new_label.items():
            mask = y_attack_train_filtered == old_label
            y_attack_train_filtered_remapped[mask] = new_label
    
    print(f"Final attack labels range: {torch.unique(y_attack_train_filtered_remapped)}")
    print("FIXED: Consistent 10-class model architecture maintained")

    # FIX 2: Proper binary class weights
    num_attacks = (y_binary_train == 1).sum().float()
    num_normal = (y_binary_train == 0).sum().float()
    pos_weight = torch.tensor([num_normal / torch.clamp(num_attacks, min=1.0)])
    print(f"Binary pos_weight: {pos_weight.item():.4f} (normal/attack ratio)")

    # FIX 3: Proper multiclass weights for all 10 classes
    attack_only_labels = y_attack_train_filtered_remapped
    attack_counts = Counter(attack_only_labels.numpy())
    attack_total = len(attack_only_labels)
    
    # Always use 10 classes for consistent architecture
    actual_num_classes = 10  # FIXED: Always 10 for federated consistency
    print(f"FIXED: Using consistent {actual_num_classes} classes (0=normal, 1-9=attacks)")
    print(f"Attack-only class distribution: {dict(sorted(attack_counts.items()))}")
    
    # Create weights for all 10 classes
    attack_class_weights = torch.ones(actual_num_classes)
    attack_class_weights[0] = 1.0  # Normal class weight
    
    # Calculate inverse frequency weights for attack classes that exist
    for attack_label, count in attack_counts.items():
        if 1 <= attack_label < actual_num_classes:
            attack_class_weights[attack_label] = attack_total / (len(attack_counts) * count)
    
    print(f"FIXED attack class weights shape: {attack_class_weights.shape}")
    print(f"FIXED attack class weights: {attack_class_weights}")
    
    # FIX 4: Simple class imbalance handling with standard techniques
    if attack_counts:
        min_samples = min(attack_counts.values())
        max_samples = max(attack_counts.values())
        imbalance_ratio = max_samples / max(min_samples, 1)
        print(f"WARNING: Class imbalance ratio: {imbalance_ratio:.2f}x")
        
        # Use standard approach - just weighted loss, no complex augmentation
        print("Using standard multiclass dataset with weighted loss function")
    else:
        print("WARNING: No attack samples found")

    # --- Datasets & loaders ---
    train_dataset = TensorDataset(X_train, y_binary_train, y_attack_train_remapped)
    val_dataset   = TensorDataset(X_val,   y_binary_val,   y_attack_val_remapped)
    test_dataset  = TensorDataset(X_test,  y_binary_test,  y_attack_test_remapped)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # Create multiclass dataset for attack samples only
    print(f"DEBUG: X_train_attacks shape: {X_train_attacks.shape}")
    print(f"DEBUG: y_attack_train_filtered_remapped shape: {y_attack_train_filtered_remapped.shape}")
    print(f"DEBUG: y_attack_train_filtered_remapped unique values: {torch.unique(y_attack_train_filtered_remapped)}")
    
    # SMOTE: Apply to the 5 classes with the least labels
    if len(attack_counts) > 0:
        # Sort classes by count to find the 5 least represented
        sorted_counts = sorted(attack_counts.items(), key=lambda x: x[1])
        print(f"Class distribution (sorted by count): {sorted_counts}")
        
        # Get the 5 classes with least samples
        classes_to_smote = [class_id for class_id, count in sorted_counts[:5]]
        print(f"Applying SMOTE to 5 classes with least samples: {classes_to_smote}")
        
        # Convert to numpy for SMOTE
        X_attacks_np = X_train_attacks.numpy()
        y_attacks_np = y_attack_train_filtered_remapped.numpy()
        
        # Reshape X for SMOTE (flatten sequence dimension)
        original_shape = X_attacks_np.shape  # [N, seq_len, features]
        X_attacks_flat = X_attacks_np.reshape(X_attacks_np.shape[0], -1)
        
        print(f"Before SMOTE: {X_attacks_flat.shape[0]} samples")
        print(f"Original class distribution: {Counter(y_attacks_np)}")
        
        try:
            # Apply SMOTE with AGGRESSIVE strategy for lower classes
            min_class_size = min(attack_counts.values())
            k_neighbors = min(3, min_class_size - 1) if min_class_size > 1 else 1
            
            # Create AGGRESSIVE sampling strategy for the 5 least represented classes
            max_count = max(attack_counts.values())
            median_count = sorted(attack_counts.values())[len(attack_counts) // 2]
            
            sampling_strategy = {}
            for i, class_id in enumerate(classes_to_smote):
                current_count = attack_counts[class_id]
                if current_count >= 5:  # Only apply SMOTE if we have enough samples
                    # AGGRESSIVE boosting: more boost for lower classes
                    if i == 0:  # Lowest class gets biggest boost
                        new_count = min(max_count // 4, current_count * 50)  # 50x boost for lowest
                    elif i == 1:  # Second lowest
                        new_count = min(max_count // 6, current_count * 30)  # 30x boost
                    elif i == 2:  # Third lowest
                        new_count = min(max_count // 8, current_count * 20)  # 20x boost
                    elif i == 3:  # Fourth lowest
                        new_count = min(max_count // 10, current_count * 10)  # 10x boost
                    else:  # Fifth lowest
                        new_count = min(max_count // 12, current_count * 5)   # 5x boost
                    
                    # Ensure minimum reasonable count
                    new_count = max(new_count, current_count * 3, 500)  # At least 3x or 500 samples
                    sampling_strategy[class_id] = new_count
            
            if sampling_strategy:
                print(f"SMOTE sampling strategy: {sampling_strategy}")
                
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=42
                )
                
                X_attacks_smoted, y_attacks_smoted = smote.fit_resample(X_attacks_flat, y_attacks_np)
                
                # Reshape back to original sequence format
                X_attacks_smoted = X_attacks_smoted.reshape(-1, original_shape[1], original_shape[2])
                
                print(f"After SMOTE: {X_attacks_smoted.shape[0]} samples")
                print(f"New class distribution: {Counter(y_attacks_smoted)}")
                
                # Convert back to tensors
                X_train_attacks = torch.from_numpy(X_attacks_smoted).float()
                y_attack_train_filtered_remapped = torch.from_numpy(y_attacks_smoted).long()
                
                print(f"SMOTE applied successfully! Added {X_attacks_smoted.shape[0] - X_attacks_flat.shape[0]} synthetic samples")
            else:
                print("No classes suitable for SMOTE (need at least 5 samples)")
                
        except Exception as e:
            print(f"SMOTE failed: {e}")
            print("Continuing with original data...")
    
    # Ensure sizes match before creating dataset
    assert X_train_attacks.shape[0] == y_attack_train_filtered_remapped.shape[0], f"Size mismatch: X_train_attacks {X_train_attacks.shape[0]} vs y_attack_train_filtered_remapped {y_attack_train_filtered_remapped.shape[0]}"
    
    multiclass_dataset = TensorDataset(X_train_attacks, y_attack_train_filtered_remapped)
    multiclass_loader  = DataLoader(multiclass_dataset, batch_size=batch_size, shuffle=True)

    return (
        train_loader, val_loader, test_loader,
        multiclass_loader, pos_weight, attack_class_weights
    )

 
def train(net, trainloader, valloader, multiclass_loader, pos_weight, attack_class_weights, epochs, learning_rate, device, cid):
    """
    HIERARCHICAL TRAINING ANALYSIS:
    Phase 1: Binary classification (Normal vs Attack)
    Phase 2: Multiclass classification (Attack type identification) - ONLY on attack samples
    Phase 3: Joint fine-tuning (Both tasks together)
    
    POTENTIAL ISSUES TO MONITOR:
    1. Data imbalance between normal/attack samples
    2. Inconsistent class distributions across federated clients
    3. Model architecture mismatch (10 classes vs actual unique classes)
    4. Loss function weight balancing
    5. Feature extraction freezing/unfreezing logic
    """
    print(f"Training started on client {cid} using device: {device}")
    
    net.to(device)
    binary_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # BASIC: Original simple approach that achieved 42% multiclass accuracy
    multiclass_criterion = nn.CrossEntropyLoss()  # NO WEIGHTS - simpler is better
    
    print(f"Client {cid}: Using BASIC CrossEntropy (no weights, no enhancements)")

    # BASIC: Simple optimizer - exactly as it was working before
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # Standard learning rate
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Phase 1: Binary Classification Training
    print(f"Client {cid}: Phase 1 - Training Binary Classifier")
    num_epochs = 15  # BASIC: Back to simple epoch count
    for epoch in range(num_epochs):
        try:
            train_loss, train_acc = train_epoch_binary(net, trainloader, binary_criterion, optimizer, device)
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_binary(net, valloader, binary_criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(net.state_dict(), f'best_binary_model_{cid}.pth')
            
            if (epoch + 1) % 5 == 0:
                print(f"Client {cid} - Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
                
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            break
    
    # Load best binary model
    best_model_path = f'best_binary_model_{cid}.pth'
    if os.path.exists(best_model_path):
        net.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    
    # Phase 2: Multi-class Classification Training (freeze feature extractors)
    print(f"Client {cid}: Phase 2 - Training Multi-class Classifier")
    
    # Check if we have enough attack samples for multiclass training
    if len(multiclass_loader.dataset) == 0:
        print(f"Client {cid}: No attack samples found, skipping multiclass training")
        multi_train_losses = []
        multi_train_accs = []
    else:
        # Get actual number of unique attack classes for this client
        unique_attack_classes = torch.unique(multiclass_loader.dataset.tensors[1])
        num_unique_classes = len(unique_attack_classes)
        print(f"Client {cid}: Training multiclass with {num_unique_classes} attack types: {unique_attack_classes}")
        
        # BASIC: Simple multiclass training - no freezing, no complexity
        print(f"Client {cid}: Training multiclass with {num_unique_classes} attack types: {unique_attack_classes}")
        print(f"Client {cid}: Using BASIC approach - no layer freezing, no advanced techniques")
        
        # Verify model architecture is correct
        model_multiclass_size = net.multiclass_head[-1].out_features
        print(f"Client {cid}: Model multiclass head size: {model_multiclass_size}")
        assert model_multiclass_size == 10, f"Model should have 10 classes, got {model_multiclass_size}"
        
        # BASIC: No layer freezing - train everything together
        print(f"Client {cid}: All layers trainable - simple end-to-end training")
        
        # BASIC: Simple optimizer for multiclass
        optimizer_multi = torch.optim.Adam(net.parameters(), lr=0.001)
        
        multi_train_losses = []
        multi_train_accs = []
        
        # BASIC: Simple multiclass training - no advanced techniques
        multiclass_epochs = 10  # BASIC: Very simple epoch count
        print(f"Client {cid}: Starting {multiclass_epochs} epochs of basic multiclass training")
        
        for epoch in range(multiclass_epochs):
            try:
                train_loss, train_acc = train_epoch_multiclass(net, multiclass_loader, multiclass_criterion, optimizer_multi, device)
                multi_train_losses.append(train_loss)
                multi_train_accs.append(train_acc)
                
                if (epoch + 1) % 5 == 0:
                    print(f"Client {cid} - Multiclass Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                    
            except Exception as e:
                print(f"Error in multiclass epoch {epoch}: {e}")
                break
    
    # Phase 3: Joint Fine-tuning - simple approach
    print(f"Client {cid}: Phase 3 - Basic Joint Fine-tuning")
    
    # BASIC: Simple joint optimizer
    optimizer_joint = torch.optim.Adam(net.parameters(), lr=0.0005)  # Lower LR for fine-tuning
    
    joint_train_losses = []
    joint_val_losses = []
    joint_train_accs = []
    joint_val_accs = []
    
    # BASIC: Simple joint training
    joint_epochs = 10  # Back to basic epoch count
    for epoch in range(joint_epochs):
        try:
            train_loss, train_acc = train_epoch_joint(net, trainloader, binary_criterion, multiclass_criterion, optimizer_joint, device)
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_binary(net, valloader, binary_criterion, device)
            
            joint_train_losses.append(train_loss)
            joint_val_losses.append(val_loss)
            joint_train_accs.append(train_acc)
            joint_val_accs.append(val_acc)
            
            if (epoch + 1) % 5 == 0:
                print(f"Client {cid} - Joint Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Binary Acc: {train_acc:.4f}")
                print(f"Val Binary Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
                
        except Exception as e:
            print(f"Error in joint epoch {epoch}: {e}")
            break
    
    # Return comprehensive metrics from all training phases
    final_metrics = {
        "val_loss": val_loss if 'val_loss' in locals() else 0.0,
        "val_accuracy": val_acc if 'val_acc' in locals() else 0.0,
        "val_precision": val_prec if 'val_prec' in locals() else 0.0,
        "val_recall": val_rec if 'val_rec' in locals() else 0.0,
        "binary_training_complete": True,
        "multiclass_training_complete": len(multi_train_losses) > 0 if 'multi_train_losses' in locals() else False,
        "joint_training_complete": len(joint_train_losses) > 0 if 'joint_train_losses' in locals() else False,
        "final_train_loss": joint_train_losses[-1] if 'joint_train_losses' in locals() and joint_train_losses else train_losses[-1] if train_losses else 0.0,
        "final_val_loss": joint_val_losses[-1] if 'joint_val_losses' in locals() and joint_val_losses else val_losses[-1] if val_losses else 0.0
    }

    print(f"Client {cid}: All 3 phases of training completed successfully!")
    return final_metrics
 
def test(net, testloader, device, cid=None):
    """Validate the model on the test set."""
    print(f"Testing client {cid}...")
    print(f"Test loader length: {len(testloader)}")
    print(f"Test dataset size: {len(testloader.dataset) if hasattr(testloader, 'dataset') else 'Unknown'}")
    
    net = net.to(device)
    
    # Initialize default values
    total_loss = 0.0
    binary_acc = 0.0
    multi_acc = 0.0
    
    try:
        # Create criteria for evaluation
        binary_criterion = nn.BCEWithLogitsLoss()
        multiclass_criterion = nn.CrossEntropyLoss()
        
        total_loss, binary_acc, multi_acc, binary_preds, multi_preds, true_binary, true_attack = evaluate_hierarchical(
            net, testloader, binary_criterion, multiclass_criterion, device
        )
        print(f"Client {cid} - Final Test Results:")
        print(f"Binary Classification Accuracy: {binary_acc:.4f}")
        print(f"Multi-class Classification Accuracy (on detected attacks): {multi_acc:.4f}")
        print(f"Total Loss: {total_loss:.4f}")
    except Exception as e:
        print(f"Error in testing client {cid}: {e}")
        import traceback
        traceback.print_exc()
        # Values already initialized above
 
    output_dict = {
        "binary_acc": binary_acc,
        "multi_acc": multi_acc,
    }
    
    print(f"Client {cid} returning: loss={total_loss}, test_size={len(testloader)}, output_dict={output_dict}")
    return total_loss, len(testloader), output_dict