Federated Intrusion Detection System with Flower and Simulated Annealing
This repository contains the implementation of a robust, decentralized Intrusion Detection System (IDS) that leverages the power of Federated Learning (FL). The system is built using the Flower (flwr) framework to manage the distributed learning process.

This project explores multiple advanced strategies, including a novel optimization technique using Simulated Annealing to accelerate model convergence, and a Hierarchical Model that performs a two-stage classification for more granular threat analysis.

🚀 Key Features
Decentralized Learning: Trains a global IDS model on distributed data sources without centralizing sensitive network traffic data, ensuring data privacy and security.

Robust Threat Detection: By aggregating insights from multiple networks, the model learns to identify a wider and more diverse range of cyber threats.

Flower Framework: Utilizes the accessible and powerful Flower framework to implement the federated learning server and clients with ease.

Simulated Annealing for Faster Convergence: Implements a custom federated averaging strategy that incorporates the principles of Simulated Annealing. This allows the model to converge more quickly by making it more exploratory at the beginning of the training process.

Hierarchical Model: Implements a two-stage classification pipeline, first identifying threats (binary classification) and then categorizing them (multi-class classification) for more detailed and efficient analysis.

Scalable Architecture: The system is designed to be scalable, allowing for the addition of new clients with minimal overhead.

⚙️ How It Works
The project explores three primary approaches for training the federated IDS.

1. Standard Federated Learning with Flower
In the standard setup, we use the FedAvg strategy provided by Flower. The process is as follows:

Initialization: A central server initializes a global model and sends it to a random subset of clients.

Local Training: Each selected client trains the model on its local intrusion detection dataset for a few epochs.

Model Aggregation: The clients send their updated model weights back to the server.

Global Model Update: The server aggregates the received weights (e.g., by averaging them) to produce an improved global model.

Iteration: This process is repeated for a set number of rounds until the global model converges.

2. Federated Learning with Simulated Annealing
This approach modifies the server-side aggregation strategy to speed up convergence. The core idea is to treat the federated learning process as an optimization problem where we are trying to find the best set of global model weights.

The key difference lies in the aggregation step:

Initial High Temperature: At the beginning of the training, a "temperature" variable is set to a high value. In this state, the server is more likely to accept model updates from clients, even if they temporarily decrease the global model's accuracy. This exploratory behavior helps the model escape local minima.

Probabilistic Acceptance: The decision to accept a "worse" set of aggregated weights is probabilistic and depends on the current temperature and how much worse the new model is.

Cooling Schedule: As the training rounds progress, the temperature is gradually decreased according to a cooling schedule.

Final Convergence: In the later stages of training, when the temperature is low, the server becomes highly selective and only accepts model updates that result in a clear improvement. This ensures that the model stabilizes and converges to a high-quality solution.

3. Hierarchical Federated Learning (HFL)
This implementation uses a two-stage approach to first detect the presence of an intrusion and then classify the specific type of attack. This hierarchical structure can lead to better performance and more efficient resource usage.

Stage 1: Binary Classification (Anomaly Detection):

A primary global model is trained using federated learning (either FedAvg or with Simulated Annealing).

The goal of this model is simple: to classify network traffic as either 'Normal' or 'Attack'. This acts as a high-level filter.

Stage 2: Multi-Class Classification (Attack Categorization):

If a piece of traffic is classified as 'Attack' by the first model, it is then passed to a second, more specialized global model.

This second model is also trained using federated learning but on a dataset that only contains attack instances.

Its purpose is to perform multi-class classification to determine the specific category of the attack (e.g., DDoS, PortScan, Malware, WebAttack).

This method allows the first model to be lightweight and fast for general traffic, while the second, more complex model is only engaged when a potential threat is detected.

🏛️ System Architecture
The architecture consists of a central server and multiple distributed clients.

Central Server (Aggregator):

Orchestrates the entire federated learning process.

Initializes the global model(s).

Selects clients for each training round.

Aggregates model updates using one of the available strategies.

Evaluates the performance of the global model(s).

Clients (Data Nodes):

Represent individual organizations or networks where intrusion detection data is generated.

Each client holds its own private dataset.

Receives the global model(s) from the server, trains it on local data, and sends the updated weights back.

<img width="870" height="287" alt="image" src="https://github.com/user-attachments/assets/797efb5c-06be-459d-a41c-19155786592a" />

🛠️ Getting Started
Follow these instructions to get the project up and running on your local machine.

Prerequisites
Python 3.8+

pip

Installation
Clone the repository:

git clone https://github.com/your-username/federated-ids.git
cd federated-ids

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required packages:

pip install -r requirements.txt

The requirements.txt file should include:

flwr
numpy
tensorflow  # or torch
scikit-learn
pandas

▶️ Usage
To run the simulation, you need to start the federated learning server and then one or more clients.

Start the Server:
Open a terminal and run the server script. You can choose the strategy and model type.

For the standard FedAvg strategy:

python server.py --strategy=fedavg

For the Simulated Annealing strategy:

python server.py --strategy=sa

To use the Hierarchical model (can be combined with a strategy):

python server.py --strategy=sa --model=hierarchical

Start the Clients:
Open a new terminal for each client you want to run. The client script will adapt based on the model type requested by the server.

# Start the first client
python client.py --client-id=1

# Open another terminal and start the second client
python client.py --client-id=2

The clients will connect to the server and begin the federated training process.

📊 Results and Evaluation
The effectiveness of the system is evaluated based on the following metrics:

Model Accuracy: The final accuracy of the global model on a held-out test set. For the hierarchical model, this will include accuracy for both the binary and multi-class models.

Convergence Speed: The number of communication rounds required to reach a target accuracy level. We expect the Simulated Annealing approach to converge significantly faster than standard FedAvg.

Privacy: By design, the system ensures that raw data never leaves the client's premises.

After running the simulation, you can analyze the logs and generated plots to compare the performance of the different strategies and models.

🤝 Contributing
Contributions are welcome! If you have ideas for improvements or find any issues, please open an issue or submit a pull request.

Fork the Project.

Create your Feature Branch (git checkout -b feature/AmazingFeature).

Commit your Changes (git commit -m 'Add some AmazingFeature').

Push to the Branch (git push origin feature/AmazingFeature).

Open a Pull Request.

📄 License
This project is distributed under the MIT License. See LICENSE for more information.

