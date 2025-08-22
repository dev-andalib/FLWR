"""zero-trust-fed-cnn-bilstm-app: A Flower / PyTorch app."""

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from zero_trust_fed_cnn_bilstm_app.task import Net, get_weights , set_weights, load_data, test
import json
import os
def get_evaluate_fn(testloader, device):
    """Return a function for centralized evaluation on the server."""
    def evaluate(server_round, parameters_ndarrays, config):
        net = Net(input_dim=78, num_classes=15)  # Match your model init params
        set_weights(net, parameters_ndarrays)
        net.to(device)
        net.eval()

        metrics = test(net, testloader, device)
        loss = metrics["loss"]

        # Prefix metrics if needed (optional)
        metrics_prefixed = {f"centralized_{k}": v for k, v in metrics.items()}
        #JSON PART
        # Prepare result dictionary
        if server_round ==0:
            result = {
                "Starting New": "NN and FedAVG",
                "round": server_round,
                "loss": loss,
                **metrics_prefixed
                
            }
        else:
            result = {
                "round": server_round,
                "loss": loss,
                **metrics_prefixed
            }

        # File path to save results
        results_file = "server_metrics.json"

        # If file exists, load and append
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(result)

        # Write back to JSON
        with open(results_file, "w") as f:
            json.dump(data, f, indent=4)
        return loss, metrics_prefixed

    return evaluate




def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    classification = 15
    features = 78
    net = Net(input_dim=features, num_classes=classification)
    # Initialize model parameters
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    ### NEW
     # Load and prepare your centralized test data on server side
    _, testloader = load_data(partition_id=0, num_partitions=1)  # all data as one partition for global test
    # Ensure testloader is available for evaluation
    if testloader is None:
        raise ValueError("Test data loader is not available. Ensure it is loaded correctly.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_fn=get_evaluate_fn(testloader, device=device),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
