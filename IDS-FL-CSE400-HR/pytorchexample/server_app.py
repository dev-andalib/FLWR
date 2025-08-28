"""pytorchexample: A Flower / PyTorch app."""

from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Context, Metrics, ndarrays_to_parameters, Parameters, FitRes, EvaluateRes, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from pytorchexample.task import get_weights, Net, test, set_weights
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import pandas as pd
from functools import reduce

class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy that handles models with different multiclass head sizes."""
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[any, FitRes]],
        failures: List[Union[Tuple[any, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, any]]:
        """Aggregate fit results using weighted average, handling different model shapes."""
        
        if not results:
            return None, {}
        
        # Convert results to weights
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Get reference model structure (first client) - FIXED: Consistent architecture
        reference_weights = weights_results[0][0]
        reference_model = Net(num_attack_types=10)  # FIXED: Keep 10 for federated consistency
        reference_state_dict = reference_model.state_dict()
        reference_keys = list(reference_state_dict.keys())
        
        print(f"Server Round {server_round}: Aggregating {len(weights_results)} client updates")
        print(f"FIXED: Using consistent 10-class architecture across all clients")
        
        # Aggregate parameters layer by layer
        aggregated_weights = []
        
        for i, (param_name, reference_param) in enumerate(zip(reference_keys, reference_weights)):
            if "multiclass_head" in param_name and param_name.endswith(".weight"):
                # Handle multiclass head weight specially - use reference shape
                print(f"Aggregating multiclass layer {param_name} with reference shape {reference_param.shape}")
                
                # Collect all client weights for this layer that match reference shape
                compatible_weights = []
                compatible_counts = []
                
                for client_weights, num_examples in weights_results:
                    if i < len(client_weights) and client_weights[i].shape == reference_param.shape:
                        compatible_weights.append(client_weights[i] * num_examples)
                        compatible_counts.append(num_examples)
                
                if compatible_weights:
                    # Weighted average of compatible weights
                    total_examples = sum(compatible_counts)
                    aggregated_param = sum(compatible_weights) / total_examples
                else:
                    # No compatible weights, use reference
                    aggregated_param = reference_param
                    
                aggregated_weights.append(aggregated_param)
                
            elif "multiclass_head" in param_name and param_name.endswith(".bias"):
                # Handle multiclass head bias specially
                print(f"Aggregating multiclass bias {param_name} with reference shape {reference_param.shape}")
                
                compatible_weights = []
                compatible_counts = []
                
                for client_weights, num_examples in weights_results:
                    if i < len(client_weights) and client_weights[i].shape == reference_param.shape:
                        compatible_weights.append(client_weights[i] * num_examples)
                        compatible_counts.append(num_examples)
                
                if compatible_weights:
                    total_examples = sum(compatible_counts)
                    aggregated_param = sum(compatible_weights) / total_examples
                else:
                    aggregated_param = reference_param
                    
                aggregated_weights.append(aggregated_param)
                
            else:
                # Standard aggregation for other layers
                layer_weights = []
                layer_counts = []
                
                for client_weights, num_examples in weights_results:
                    if i < len(client_weights):
                        layer_weights.append(client_weights[i] * num_examples)
                        layer_counts.append(num_examples)
                
                if layer_weights:
                    total_examples = sum(layer_counts)
                    aggregated_param = sum(layer_weights) / total_examples
                    aggregated_weights.append(aggregated_param)
                else:
                    aggregated_weights.append(reference_param)
        
        # Convert back to parameters
        aggregated_parameters = ndarrays_to_parameters(aggregated_weights)
        
        # Aggregate metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        
        return aggregated_parameters, metrics_aggregated


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Calculate weighted average of metrics across clients, summing confusion matrix elements."""
    total_examples = sum([num for num, _ in metrics])

    weighted_binary_acc= sum([metric["binary_acc"] * num for num, metric in metrics])
    weighted_multi_acc = sum([metric["multi_acc"] * num for num, metric in metrics])

    # Calculate weighted averages for rates
    accuracy_avg = weighted_binary_acc / total_examples
    weighted_multi_acc = weighted_multi_acc / total_examples

    return {
        "accuracy_avg_binary": accuracy_avg,
        "accuracy_avg_multi": weighted_multi_acc,
    }

def fit_weighted_avg(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Calculate weighted average of training/validation metrics across clients, summing confusion matrix elements."""
    total_examples = sum([num for num, _ in metrics])

    weighted_val_losses = sum([metric["val_loss"] * num for num, metric in metrics])
    weighted_accuracies = sum([metric["val_accuracy"] * num for num, metric in metrics])
    weighted_recalls = sum([metric["val_recall"] * num for num, metric in metrics])
    weighted_precisions = sum([metric["val_precision"] * num for num, metric in metrics])

    # Calculate weighted averages for rates
    val_loss_avg = weighted_val_losses / total_examples
    accuracy_avg = weighted_accuracies / total_examples
    recall_avg = weighted_recalls / total_examples
    precision_avg = weighted_precisions / total_examples


    return {
        "val_loss": val_loss_avg,
        "val_accuracy": accuracy_avg,
        "val_precision": precision_avg,
        "val_recall": recall_avg,
    }


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    server_device = context.run_config["server-device"]
    batch_size = context.run_config["batch-size"]
    # Load Parquet file with Pandas
    # path = './Ids_dataset/UNSW_NB15_binary_test.parquet'
    # if not os.path.exists(path):
    #     print(f"Warning: Test data file {path} not found. Using dummy data for server evaluation.")
    #     # Create dummy test data if file doesn't exist
    #     X_test = torch.randn(100, 20, dtype=torch.float32)  # 100 samples, 20 features
    #     y_binary_test = torch.randint(0, 2, (100,), dtype=torch.float32)
    #     y_attack_test = torch.randint(0, 9, (100,), dtype=torch.long)
    # else:
    #     df = pd.read_parquet(path)
    #     # Ensure we have the required columns
    #     if 'label' not in df.columns:
    #         raise ValueError("Test data must contain 'label' column")
        
    #     # Convert to tensors
    #     X_test = torch.tensor(df.drop("label", axis=1).to_numpy(), dtype=torch.float32)
    #     y_binary_test = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)
    #     # Add dummy attack labels if not present
    #     y_attack_test = torch.zeros(len(y_binary_test), dtype=torch.long)

    # # Create TensorDataset and DataLoader with proper format
    # test_dataset = TensorDataset(X_test, y_binary_test, y_attack_test)
    # testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model parameters with correct architecture (10 classes for natural labels 0-9)
    ndarrays = get_weights(Net(num_attack_types=10))
    parameters = ndarrays_to_parameters(ndarrays)
    

    # Define the strategy
    strategy = CustomFedAvg(
        min_available_clients=context.run_config["min_available_clients"],
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        fit_metrics_aggregation_fn=fit_weighted_avg,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)