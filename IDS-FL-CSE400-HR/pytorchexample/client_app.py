"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from pytorchexample.task import get_weights, load_data, set_weights, test, train, Net
from pytorchexample.getdist import get_class_distribution

# Define Flower Client

class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, testloader, multiclass_loader, pos_weight, attack_class_weights, local_epochs, learning_rate, cid: int):
        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Client {cid} using device: {self.device}")
        
        self.net = Net().to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.multiclass_loader = multiclass_loader
        self.pos_weight = pos_weight.to(self.device) if pos_weight is not None else None
        self.attack_class_weights = attack_class_weights.to(self.device) if attack_class_weights is not None else None
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.cid = cid
        

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(self.net, self.trainloader, self.valloader, self.multiclass_loader, 
                       self.pos_weight, self.attack_class_weights, self.local_epochs, 
                       self.learning_rate, self.device, self.cid)
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, test_size, output_dict = test(self.net, self.testloader, self.device, self.cid)
        return float(loss), test_size, output_dict


                                                         

def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    trainloader, valloader, testloader, multiclass_loader, pos_weight, attack_class_weights = load_data(partition_id, num_partitions, batch_size)
    # get_class_distribution(partition_id, trainloader, "Training data class distribution")
    # get_class_distribution(partition_id, valloader, "Validation data class distribution")
    # get_class_distribution(partition_id, testloader, "Test data class distribution")    
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(trainloader, valloader, testloader, multiclass_loader, pos_weight, attack_class_weights, local_epochs, learning_rate, partition_id).to_client()


# Flower ClientApp
app = ClientApp(client_fn = client_fn)