from turtle import pd
import numpy as np
import torch
import json

def create_sequences(X, y_binary, y_attack_cat, seq_length=10):
    num_samples = len(X) - seq_length + 1
    # Use float32 to reduce memory usage
    X_seq = np.zeros((num_samples, seq_length, X.shape[1]), dtype=np.float32)
    y_binary_seq = np.zeros(num_samples, dtype=np.float32)
    y_attack_cat_seq = np.zeros(num_samples, dtype=np.int64)
    
    for i in range(num_samples):
        X_seq[i] = X[i:i+seq_length].astype(np.float32)
        y_binary_seq[i] = y_binary[i+seq_length-1]
        y_attack_cat_seq[i] = y_attack_cat[i+seq_length-1]
    
    return X_seq, y_binary_seq, y_attack_cat_seq

def train_epoch_joint(model, loader, binary_criterion, multiclass_criterion, optimizer, device):
    """
    JOINT TRAINING ANALYSIS:
    - Combines binary and multiclass losses
    - Only applies multiclass loss to attack samples
    - Balances loss contributions with weighting factor
    
    FIXED ISSUES:
    - Consistent threshold logic (0.0 for logits)
    - Proper loss weighting between tasks
    - Better handling of batches with no attacks
    """
    model.train()
    total_loss = 0
    binary_correct = 0
    total = 0
    multiclass_batches = 0
    
    for batch_x, batch_y_binary, batch_y_attack in loader:
        batch_x = batch_x.to(device)
        batch_y_binary = batch_y_binary.to(device)
        batch_y_attack = batch_y_attack.to(device)
        
        optimizer.zero_grad()
        
        binary_output, multiclass_output = model(batch_x, stage='both')
        binary_output = binary_output.squeeze()
        
        # Binary classification loss (always computed)
        binary_loss = binary_criterion(binary_output, batch_y_binary)
        
        # Multiclass loss (only for attack samples)
        attack_mask = batch_y_binary == 1
        if attack_mask.sum() > 0:
            attack_labels = batch_y_attack[attack_mask]
            attack_predictions = multiclass_output[attack_mask]
            multiclass_loss = multiclass_criterion(attack_predictions, attack_labels)
            # ENHANCED: Adaptive loss balancing - increase multiclass weight over time
            epoch_progress = min(1.0, multiclass_batches / 1000)  # Gradually increase weight
            multiclass_weight = 0.2 + 0.3 * epoch_progress  # Start at 0.2, increase to 0.5
            total_loss_batch = binary_loss + multiclass_weight * multiclass_loss
            multiclass_batches += 1
        else:
            total_loss_batch = binary_loss
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        # FIXED: Consistent threshold with other functions
        predicted = (binary_output > 0.0).float()  # Use 0.0 for logits
        binary_correct += (predicted == batch_y_binary).sum().item()
        total += batch_y_binary.size(0)
    
    avg_loss = total_loss / len(loader)
    avg_acc = binary_correct / total
    print(f"Joint training: {multiclass_batches}/{len(loader)} batches had attacks")
    
    return avg_loss, avg_acc

def train_epoch_binary(model, loader, criterion, optimizer, device):
    """
    BINARY CLASSIFICATION TRAINING ANALYSIS:
    - Uses BCEWithLogitsLoss (expects raw logits, not sigmoid)
    - Threshold at 0.0 for logits (equivalent to 0.5 for probabilities)
    - Critical: Model should NOT apply sigmoid in forward pass
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y_binary, _ in loader:
        batch_x, batch_y_binary = batch_x.to(device), batch_y_binary.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x, stage='binary').squeeze()
        loss = criterion(outputs, batch_y_binary)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # FIXED: Use 0.0 threshold for logits (BCEWithLogitsLoss expects raw logits)
        predicted = (outputs > 0.0).float()  # Threshold at 0 for logits, not 0.5
        correct += (predicted == batch_y_binary).sum().item()
        total += batch_y_binary.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate_binary(model, loader, criterion, device):
    """
    BINARY EVALUATION ANALYSIS:
    - Consistent threshold logic with training (0.0 for logits)
    - Proper precision/recall calculation for imbalanced datasets
    - F1-score as primary metric for attack detection
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    tp = fp = tn = fn = 0
    
    with torch.no_grad():
        for batch_x, batch_y_binary, _ in loader:
            batch_x, batch_y_binary = batch_x.to(device), batch_y_binary.to(device)
            
            outputs = model(batch_x, stage='binary').squeeze()
            loss = criterion(outputs, batch_y_binary)
            
            total_loss += loss.item()
            # FIXED: Consistent threshold logic with training
            predicted = (outputs > 0.0).float()  # Use 0.0 threshold for logits
            correct += (predicted == batch_y_binary).sum().item()
            total += batch_y_binary.size(0)
            
            # Calculate confusion matrix elements
            tp += ((predicted == 1) & (batch_y_binary == 1)).sum().item()
            fp += ((predicted == 1) & (batch_y_binary == 0)).sum().item()
            tn += ((predicted == 0) & (batch_y_binary == 0)).sum().item()
            fn += ((predicted == 0) & (batch_y_binary == 1)).sum().item()
    
    accuracy = correct / total
    
    # FIXED: Robust precision/recall calculation for edge cases
    if tp + fn == 0:  # No actual attacks in data
        recall = 1.0 if tp + fp == 0 else 0.0  # Perfect if no predictions either
    else:
        recall = tp / (tp + fn)
        
    if tp + fp == 0:  # No attack predictions
        precision = 1.0 if tp + fn == 0 else 0.0  # Perfect if no actual attacks
    else:
        precision = tp / (tp + fp)
    
    # F1-score calculation with edge case handling
    if precision + recall == 0:
        f1 = 0.0  # Both precision and recall are 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    print(f"Binary Classification Metrics: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    return total_loss / len(loader), accuracy, precision, recall, f1

def train_epoch_multiclass(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x, stage='multiclass')
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)
    
    return total_loss / len(loader), correct / total

def evaluate_hierarchical(model, loader, binary_criterion, multiclass_criterion, device):
    """
    HIERARCHICAL EVALUATION ANALYSIS:
    Critical Issues Found:
    1. Inconsistent threshold logic between binary training/evaluation
    2. Poor multiclass performance due to class imbalance
    3. Hierarchical prediction logic needs improvement
    
    Fixes Applied:
    - Consistent 0.0 threshold for logits
    - Better handling of edge cases
    - Improved hierarchical decision making
    - Added loss calculation
    """
    model.eval()
    binary_predictions = []
    multiclass_predictions = []
    true_binary = []
    true_attack = []
    total_binary_loss = 0
    total_multiclass_loss = 0
    multiclass_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y_binary, batch_y_attack in loader:
            batch_x = batch_x.to(device)
            batch_y_binary = batch_y_binary.to(device)
            batch_y_attack = batch_y_attack.to(device)
            
            binary_output, multiclass_output = model(batch_x, stage='both')
            
            # Calculate binary loss
            binary_loss = binary_criterion(binary_output.squeeze(), batch_y_binary)
            total_binary_loss += binary_loss.item()
            
            # Calculate multiclass loss (only for attack samples)
            attack_mask = batch_y_binary == 1
            if attack_mask.sum() > 0:
                attack_labels = batch_y_attack[attack_mask]
                attack_predictions = multiclass_output[attack_mask]
                multiclass_loss = multiclass_criterion(attack_predictions, attack_labels)
                total_multiclass_loss += multiclass_loss.item()
                multiclass_batches += 1
            
            # FIXED: Consistent threshold logic
            binary_pred = (binary_output.squeeze() > 0.0).float()  # Use 0.0 for logits
            _, multiclass_raw_pred = multiclass_output.max(1)
            
            # IMPROVED HIERARCHICAL PREDICTION LOGIC:
            # Step 1: Binary classifier decides normal vs attack
            # Step 2: If attack predicted, multiclass classifier determines attack type
            # Step 3: If normal predicted, force multiclass to 0 (normal)
            multiclass_pred = torch.where(
                binary_pred == 0,  # If binary predicts normal
                torch.zeros_like(multiclass_raw_pred),  # Force multiclass to 0 (normal)
                multiclass_raw_pred  # Otherwise use multiclass prediction
            )
            
            binary_predictions.extend(binary_pred.cpu().numpy())
            multiclass_predictions.extend(multiclass_pred.cpu().numpy())
            true_binary.extend(batch_y_binary.cpu().numpy())
            true_attack.extend(batch_y_attack.cpu().numpy())
    
    binary_predictions = np.array(binary_predictions)
    multiclass_predictions = np.array(multiclass_predictions)
    true_binary = np.array(true_binary)
    true_attack = np.array(true_attack)
    
    # Calculate average losses
    avg_binary_loss = total_binary_loss / len(loader)
    avg_multiclass_loss = total_multiclass_loss / max(multiclass_batches, 1)
    total_loss = avg_binary_loss + avg_multiclass_loss
    
    # Binary classification accuracy
    binary_accuracy = np.mean(binary_predictions == true_binary)
    
    # IMPROVED: Multiclass accuracy calculation
    # Only evaluate on samples where:
    # 1. Binary correctly identified as attack (true positive cases)
    # 2. AND ground truth is actually attack
    correct_attack_detection = (binary_predictions == 1) & (true_binary == 1)
    
    if correct_attack_detection.sum() > 0:
        # Calculate accuracy only on correctly detected attacks
        multiclass_accuracy = np.mean(
            multiclass_predictions[correct_attack_detection] == true_attack[correct_attack_detection]
        )
        print(f"Multiclass evaluated on {correct_attack_detection.sum()} correctly detected attacks")
    else:
        multiclass_accuracy = 0.0
        print("No correctly detected attacks for multiclass evaluation")
    
    # Additional diagnostics
    total_attacks = np.sum(true_binary == 1)
    detected_attacks = np.sum(binary_predictions == 1)
    print(f"Total attacks: {total_attacks}, Detected: {detected_attacks}")
    print(f"Binary recall: {np.sum(correct_attack_detection) / max(total_attacks, 1):.4f}")
    print(f"Binary loss: {avg_binary_loss:.4f}, Multiclass loss: {avg_multiclass_loss:.4f}")
    
    return total_loss, binary_accuracy, multiclass_accuracy, binary_predictions, multiclass_predictions, true_binary, true_attack
