"""FedAVGids: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg,FedAdam,FedProx
#from fedavgids.AutoEncoder import AutoEncoder, get_weights , set_weights, load_data, test

from fedavgids.cnn_bi_lstm import CNN_BiLSTM_Net, get_weights , set_weights, load_data, test
#from fedavgids.task import Net, get_weights , set_weights, load_data, test
import json
import os
import torch


def get_evaluate_fn(testloader, device):
    """Return a function for centralized evaluation on the server."""
    def evaluate(server_round, parameters_ndarrays, config):
        net = CNN_BiLSTM_Net()  # Match your model init params
        set_weights(net, parameters_ndarrays)
        net.to(device)
        net.eval()

        metrics = test(net, testloader, device)
        loss = metrics["loss"]

        # Prefix metrics if needed (optional)
        metrics_prefixed = {f"centralized_{k}": v for k, v in metrics.items()}
        #JSON PART
        fl_model_checklist = {
            "Convolutional layers": "AutoEncoder",
            "Dataset": "ToN_ToT",
            "Server Round": 50,
            "Number of Clients": 10,

        }

        # Prepare result dictionary
        if server_round ==0:
            result = {
                "Starting New": "AutoEncoder and FedAVG",
                **fl_model_checklist,
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
        results_file = "binary _CNN_LSTM_ToN_ToT.json"

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

    net = CNN_BiLSTM_Net()
    # Initialize model parameters
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)


    _, testloader = load_data(partition_id=0, num_partitions=1,server = True)  # all data as one partition for global test
    # Ensure testloader is available for evaluation
    if testloader is None:
        raise ValueError("Test data loader is not available. Ensure it is loaded correctly.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Define strategy
    strategy = FedAdam(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        #proximal_mu=0.001,  # ✅ This is required! # Only for FedProx
        evaluate_fn=get_evaluate_fn(testloader, device=device),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
