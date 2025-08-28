<div align="center">

🛡️ Federated Intrusion Detection System 🛡️
with Flower, Simulated Annealing & Hierarchical Models
</div>

<p align="center">
<img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python">
<img alt="Framework" src="https://img.shields.io/badge/Framework-Flower-orange?style=for-the-badge">
<img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
<img alt="Status" src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge">
</p>

This repository contains the implementation of a robust, decentralized Intrusion Detection System (IDS) that leverages the power of Federated Learning (FL). The system is built using the Flower (flwr) framework and explores multiple advanced strategies for efficient and accurate threat detection without compromising data privacy.

📚 Table of Contents
Key Features

How It Works

Standard Federated Learning

Federated Learning with Simulated Annealing

Hierarchical Federated Learning

System Architecture

Getting Started

Usage

Results and Evaluation

Contributing

License

🚀 Key Features
🏡 Decentralized Learning: Trains a global IDS model on distributed data sources without centralizing sensitive network traffic data, ensuring data privacy and security.

🔎 Robust Threat Detection: By aggregating insights from multiple networks, the model learns to identify a wider and more diverse range of cyber threats.

🌸 Flower Framework: Utilizes the accessible and powerful Flower framework to implement the federated learning server and clients with ease.

🔥 Simulated Annealing for Faster Convergence: Implements a custom federated averaging strategy that incorporates the principles of Simulated Annealing to accelerate model convergence and avoid local minima.

ιε Hierarchical Model: Implements a two-stage classification pipeline, first identifying threats (binary classification) and then categorizing them (multi-class classification) for more detailed and efficient analysis.

📈 Scalable Architecture: The system is designed to be scalable, allowing for the addition of new clients with minimal overhead.

⚙️ How It Works
The project explores three primary approaches for training the federated IDS.

1. Standard Federated Learning with Flower
In the standard setup, we use the FedAvg strategy provided by Flower. The process is as follows:

Initialization: A central server initializes a global model and sends it to a random subset of clients.

Local Training: Each selected client trains the model on its local intrusion detection dataset.

Model Aggregation: The clients send their updated model weights back to the server.

Global Model Update: The server aggregates the received weights (e.g., by averaging them) to produce an improved global model.

Iteration: This process is repeated for a set number of rounds until the global model converges.

2. Federated Learning with Simulated Annealing
This approach modifies the server-side aggregation strategy to speed up convergence. The core idea is to treat the federated learning process as an optimization problem where we are trying to find the best set of global model weights.

Initial High Temperature: At the beginning of training, a "temperature" is set high. In this state, the server is more likely to accept model updates from clients, even if they temporarily decrease the global model's accuracy. This exploratory behavior helps the model escape local minima.

Probabilistic Acceptance: The decision to accept a "worse" set of aggregated weights is probabilistic and depends on the current temperature.

Cooling Schedule: As training progresses, the temperature is gradually decreased.

Final Convergence: In later stages, when the temperature is low, the server becomes highly selective and only accepts model updates that result in a clear improvement, ensuring convergence to a high-quality solution.

3. Hierarchical Federated Learning (HFL)
This implementation uses a two-stage approach to first detect the presence of an intrusion and then classify the specific type of attack.

Stage 1: Binary Classification (Anomaly Detection):

A primary global model is trained to classify network traffic as either 'Normal' or 'Attack'. This acts as a high-level filter.

Stage 2: Multi-Class Classification (Attack Categorization):

Traffic flagged as 'Attack' is passed to a second, more specialized global model.

This second model is trained only on attack instances to perform multi-class classification (e.g., DDoS, PortScan, Malware).

This method allows the first model to be lightweight and fast, while the second, more complex model is only engaged when a potential threat is detected.

🏛️ System Architecture
The architecture consists of a central server that orchestrates the learning process and multiple distributed clients that hold the data.

graph TD
    subgraph Federated Learning Network
        Server(🌸 Central Server <br> Aggregator + Strategy Manager)
        subgraph Distributed Clients
            Client1(📱 Client 1 <br> Local Data)
            Client2(💻 Client 2 <br> Local Data)
            ClientN(🌐 Client N <br> Local Data)
        end
    end

    Server -- "1. Send Global Model" --> Client1
    Server -- "1. Send Global Model" --> Client2
    Server -- "1. Send Global Model" --> ClientN

    Client1 -- "2. Return Trained Weights" --> Server
    Client2 -- "2. Return Trained Weights" --> Server
    ClientN -- "2. Return Trained Weights" --> Server

    style Server fill:#f9f,stroke:#333,stroke-width:2px
    style Client1 fill:#bbf,stroke:#333,stroke-width:2px
    style Client2 fill:#bbf,stroke:#333,stroke-width:2px
    style ClientN fill:#bbf,stroke:#333,stroke-width:2px

🛠️ Getting Started
Follow these instructions to get the project up and running on your local machine.

Prerequisites
Python 3.8+

pip

Installation
Clone the repository:

git clone https://github.com/your-username/federated-ids.git
cd federated-ids

Create and activate a virtual environment:

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

📊 Results and Evaluation
The effectiveness of the system is evaluated based on:

Model Accuracy: The final accuracy of the global model(s) on a held-out test set.

Convergence Speed: The number of communication rounds required to reach a target accuracy level.

Privacy: By design, the system ensures that raw data never leaves the client's premises.

🤝 Contributing
Contributions are welcome! If you have ideas for improvements or find any issues, please open an issue or submit a pull request.

Fork the Project.

Create your Feature Branch (git checkout -b feature/AmazingFeature).

Commit your Changes (git commit -m 'Add some AmazingFeature').

Push to the Branch (git push origin feature/AmazingFeature).

Open a Pull Request.

📄 License
This project is distributed under the MIT License. See LICENSE for more information.
