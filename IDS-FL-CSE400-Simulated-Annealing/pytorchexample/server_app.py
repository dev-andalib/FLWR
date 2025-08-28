"""pytorchexample: A Flower / PyTorch app."""

import flwr 
from typing import List, Tuple, Dict, Union
from flwr.common import Context, ndarrays_to_parameters, FitRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pytorchexample.utility import print_msg
from pytorchexample.task import get_weights, Net, set_weights, test
import torch


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, test_size, output_dict = test(net, testloader, device=device) # dummy temp and prev acc send for now 
        return loss, output_dict

    return evaluate


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Calculate weighted average of metrics across clients, summing confusion matrix elements."""
    total_examples = sum([num for num, _ in metrics])

    weighted_accuracies = sum([metric["acc"] * num for num, metric in metrics])
    weighted_recalls = sum([metric["rec"] * num for num, metric in metrics])
    weighted_precisions = sum([metric["prec"] * num for num, metric in metrics])
    tn_sum = sum([metric["cm_tn"] for num, metric in metrics])
    fp_sum = sum([metric["cm_fp"] for num, metric in metrics])
    fn_sum = sum([metric["cm_fn"] for num, metric in metrics])
    tp_sum = sum([metric["cm_tp"] for num, metric in metrics])

    # Calculate weighted averages for rates
    accuracy_avg = weighted_accuracies / total_examples
    recall_avg = weighted_recalls / total_examples
    precision_avg = weighted_precisions / total_examples

    # Calculate F1 score from summed confusion matrix
    if (tp_sum + fp_sum) == 0:
        precision = 0.0  # Handle division by zero
    else:
        precision = tp_sum / (tp_sum + fp_sum)
    
    if (tp_sum + fn_sum) == 0:
        recall = 0.0  # Handle division by zero
    else:
        recall = tp_sum / (tp_sum + fn_sum)
    
    if (precision + recall) == 0:
        f1_avg = 0.0  # Handle division by zero
    else:
        f1_avg = 2 * (precision * recall) / (precision + recall)

    return {
        "acc": accuracy_avg,
        "rec": recall_avg,
        "prec": precision_avg,
        "f1": f1_avg,
        "cm_tn": int(tn_sum),  # Sum, not average
        "cm_fp": int(fp_sum),
        "cm_fn": int(fn_sum),
        "cm_tp": int(tp_sum)
    }

def fit_weighted_avg(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Calculate weighted average of training/validation metrics across clients, summing confusion matrix elements."""
    total_examples = sum([num for num, _ in metrics])

    weighted_val_losses = sum([metric["val_loss"] * num for num, metric in metrics])
    weighted_accuracies = sum([metric["val_accuracy"] * num for num, metric in metrics])
    weighted_recalls = sum([metric["val_recall"] * num for num, metric in metrics])
    weighted_precisions = sum([metric["val_precision"] * num for num, metric in metrics])
    tn_sum = sum([metric["cm_tn"] for num, metric in metrics])
    fp_sum = sum([metric["cm_fp"] for num, metric in metrics])
    fn_sum = sum([metric["cm_fn"] for num, metric in metrics])
    tp_sum = sum([metric["cm_tp"] for num, metric in metrics])

    # Calculate weighted averages for rates
    val_loss_avg = weighted_val_losses / total_examples
    accuracy_avg = weighted_accuracies / total_examples
    recall_avg = weighted_recalls / total_examples
    precision_avg = weighted_precisions / total_examples

    # Calculate F1 score from summed confusion matrix
    if (tp_sum + fp_sum) == 0:
        precision = 0.0  # Handle division by zero
    else:
        precision = tp_sum / (tp_sum + fp_sum)
    
    if (tp_sum + fn_sum) == 0:
        recall = 0.0  # Handle division by zero
    else:
        recall = tp_sum / (tp_sum + fn_sum)
    
    if (precision + recall) == 0:
        f1_avg = 0.0  # Handle division by zero
    else:
        f1_avg = 2 * (precision * recall) / (precision + recall)

    return {
        "val_loss": val_loss_avg,
        "val_accuracy": accuracy_avg,
        "val_precision": precision_avg,
        "val_recall": recall_avg,
        "val_f1": f1_avg,
        "cm_tn": int(tn_sum),  # Sum, not average
        "cm_fp": int(fp_sum),
        "cm_fn": int(fn_sum),
        "cm_tp": int(tp_sum)
    }


# simulated annealing temp

class SA(FedAvg):
    def __init__(self, *, start_temp=10.0, cooling=0.99, **kwargs):
        super().__init__(**kwargs)
        self.temp = start_temp
        self.cooling = cooling
    

    def configure_fit(self, server_round: int, parameters, client_manager):
    

        
        fit_ins = super().configure_fit(server_round, parameters, client_manager)

        
        for client, fin in fit_ins:
            
            cfg = dict(fin.config or {})
            
            # Add the 'temp' value
            cfg["temp"] = float(self.temp)
            
            # Assign the updated dictionary back
            fin.config = cfg

        
        self.temp *= self.cooling

        
        return fit_ins
    


    def aggregate_fit(self, server_round: int, results: List[Tuple[flwr.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[flwr.server.client_proxy.ClientProxy, FitRes], BaseException]],) -> Tuple[flwr.common.Parameters, dict]:
        
        #  new list to hold  results from accepted clients
        accepted_results = []
        for client_proxy, fit_res in results:
            
            if fit_res.metrics.get("accept", False):
                accepted_results.append((client_proxy, fit_res))
                
        # no clients accepted, return  old parameters
        if not accepted_results:
            return None, {}

        #CALL ORIGINAL FedAvg AGGREGATION
        print_msg(len(accepted_results))
        return super().aggregate_fit(server_round, accepted_results, failures)
    





def server_fn(context: Context):
    
    """Construct components that set the ServerApp behaviour."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    server_device = context.run_config["server-device"]
    batch_size = context.run_config["batch-size"]


    # Load Parquet file with Pandas
    path = 'Ids_dataset/UNSW_NB15_binary_test.parquet'
    df = pd.read_parquet(path)

    # Convert to tensors
    X_test = torch.tensor(df.drop("label", axis=1).to_numpy(), dtype=torch.float32)
    y_test = torch.tensor(df["label"].to_numpy(), dtype=torch.float32).view(-1, 1)

    # Create TensorDataset and DataLoader
    test_dataset = TensorDataset(X_test, y_test)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)
    
    
    # Define the strategy
    strategy = SA(
        start_temp=10.0,
        cooling=0.9,
       
        min_available_clients=context.run_config["min_available_clients"],
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        fit_metrics_aggregation_fn=fit_weighted_avg,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)