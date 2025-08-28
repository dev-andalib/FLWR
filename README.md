<div align="center">
🛡️ Federated Intrusion Detection System
With Flower, Simulated Annealing & Hierarchical Models
</div> <p align="center"> <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python"> <img alt="Framework" src="https://img.shields.io/badge/Framework-Flower-orange?style=for-the-badge"> <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"> <img alt="Status" src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"> </p>
📖 Overview

This project implements a robust, decentralized Intrusion Detection System (IDS) using Federated Learning (FL). Built on the Flower (flwr) framework, it integrates advanced strategies like Simulated Annealing and Hierarchical Models for scalable, private, and accurate threat detection.

📚 Table of Contents

🚀 Key Features

⚙️ How It Works

Standard Federated Learning

Simulated Annealing Strategy

Hierarchical Federated Learning

🏛️ System Architecture

🛠️ Getting Started

▶️ Usage

📊 Results and Evaluation

🤝 Contributing

📄 License

🚀 Key Features

🏡 Decentralized Learning
Train global models without centralizing sensitive data, ensuring privacy and compliance.

🔎 Robust Threat Detection
Aggregates patterns from diverse sources to detect a wider range of cyber threats.

🌸 Powered by Flower
Leverages the simplicity and power of the Flower
 federated learning framework.

🔥 Simulated Annealing Optimization
Enhances convergence speed and robustness with a custom aggregation strategy.

🧠 Hierarchical Models
Combines binary anomaly detection with multi-class attack classification for deeper insights.

📈 Scalable Design
Easily add or remove clients without restructuring the architecture.

⚙️ How It Works

This project explores three federated learning strategies:

1. Standard Federated Learning with Flower

FedAvg strategy is used.

Clients train on local data and return weights.

Server aggregates them to update the global model.

Repeated for multiple rounds until convergence.

2. Federated Learning with Simulated Annealing

Initial Exploration: High "temperature" allows probabilistic acceptance of worse models.

Cooling Schedule: Gradually reduces exploration as training progresses.

Final Convergence: Accepts only improvements, converging on optimal weights.

3. Hierarchical Federated Learning (HFL)

Stage 1 - Binary Classification:
Detects whether network traffic is normal or an attack.

Stage 2 - Multi-Class Classification:
If an attack is detected, a second model categorizes it (e.g., DDoS, PortScan, Malware).

🏛️ System Architecture

A simplified view of the federated learning system:
graph TD
    subgraph Federated Learning Network
        Server[🌸 Central Server<br>Aggregator + Strategy Manager]
        subgraph Distributed Clients
            Client1[📱 Client 1<br>Local Data]
            Client2[💻 Client 2<br>Local Data]
            ClientN[🌐 Client N<br>Local Data]
        end
    end

    Server -->|1. Send Global Model| Client1
    Server -->|1. Send Global Model| Client2
    Server -->|1. Send Global Model| ClientN

    Client1 -->|2. Return Trained Weights| Server
    Client2 -->|2. Return Trained Weights| Server
    ClientN -->|2. Return Trained Weights| Server




🛠️ Getting Started
🔧 Prerequisites

Python 3.8+

pip

📦 Installation

# Clone the repository
git clone https://github.com/your-username/federated-ids.git
cd federated-ids

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


📋 Requirements
flwr
numpy
tensorflow  # or torch
scikit-learn
pandas


▶️ Usage
1. Start the Federated Server
# Standard FedAvg
python server.py --strategy=fedavg

# Simulated Annealing
python server.py --strategy=sa

# Hierarchical Model + Strategy
python server.py --strategy=sa --model=hierarchical

2. Start One or More Clients
# Client 1
python client.py --client-id=1

# Client 2
python client.py --client-id=2

# Add more as needed...


📊 Results and Evaluation

The system is evaluated based on:

Model Accuracy: Performance on a held-out test set.

Convergence Speed: Communication rounds required to reach a target accuracy.

Privacy Preservation: Raw data stays on client devices, by design.

🤝 Contributing

We welcome contributions! 🚀

Fork the project

Create a new feature branch:
git checkout -b feature/AmazingFeature

Commit your changes:
git commit -m 'Add AmazingFeature'

Push to your branch:
git push origin feature/AmazingFeature

Open a Pull Request

📄 License

This project is licensed under the MIT License.
