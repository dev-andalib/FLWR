"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from pytorchexample.task import get_weights, load_data, set_weights, test, train, Net
from pytorchexample.utility import get_class_distribution
from pytorchexample.utility import print_msg

# Define Flower Client

class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, testloader, local_epochs, learning_rate, client_id):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.client_id = str(client_id)
        
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

    def fit(self, parameters, config):       
        set_weights(self.net, parameters)
        temp = float(config.get("temp", 0.0))
        results, client_accept = train(self.net, self.trainloader, self.valloader, self.local_epochs, self.learning_rate, self.device, temp, self.client_id)
        results["accept"] = client_accept
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, n, metrics,  = test(self.net, self.testloader, self.device)
        return float(loss), int(n), metrics


                                                         

def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
   
    batch_size = context.run_config["batch-size"]
    trainloader, valloader, testloader = load_data(partition_id, num_partitions, batch_size)
      
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    
    # Return Client instance
    return FlowerClient(trainloader, valloader, testloader, local_epochs, learning_rate, partition_id).to_client()


# Flower ClientApp
app = ClientApp(client_fn = client_fn)