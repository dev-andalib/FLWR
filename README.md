🛡️ Federated Intrusion Detection System (IDS)

A privacy-preserving and scalable Intrusion Detection System built using Federated Learning, powered by the Flower framework. This project integrates advanced optimization techniques like Simulated Annealing and layered Hierarchical Learning to improve threat detection across distributed environments without sharing raw data.

<p align="center"> <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python"> <img alt="Framework" src="https://img.shields.io/badge/Framework-Flower-orange?style=flat-square"> <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square"> <img alt="Status" src="https://img.shields.io/badge/Project-Active-brightgreen?style=flat-square"> </p>




## ✨ Features

📡 Decentralized Threat Detection
Detects intrusions using models trained across multiple data sources — no central data collection required.

🌸 Built on Flower
Utilizes the Flower framework to orchestrate federated learning with minimal configuration.

🔥 Simulated Annealing Optimization
Custom aggregation strategy enables faster convergence and exploration of better global models.

🧠 Hierarchical Learning Architecture
Two-stage pipeline:

Stage 1: Binary classifier detects normal vs. anomalous traffic.

Stage 2: Multi-class classifier categorizes attack type.

📈 Scalable & Modular Design
Add new clients with minimal setup. Compatible with both TensorFlow and PyTorch.

## 🧠 How It Works

This system supports three federated learning strategies:

 1. Standard Federated Learning (FedAvg)
    Server distributes a global model to a subset of clients.
    Clients train locally on private data.
    Updates are sent back to the server and averaged.
    Process repeats until convergence.

 2. Federated Learning with Simulated Annealing
    Adds probabilistic acceptance of suboptimal models based on a "temperature" parameter.
    Encourages exploration during early rounds and stability during later ones.
    Helps avoid local minima and speeds up convergence.

  3. Hierarchical Federated Learning (HFL)
     Stage 1: A lightweight binary model filters traffic (normal vs. attack).
     Stage 2: A secondary model handles attack classification (e.g., DDoS, PortScan).
     Improves overall efficiency and reduces computation overhead.


## 🏗️ Architecture Overview
graph TD
    Server[🌸 Federated Server<br>Orchestration & Aggregation]
    Client1[Client 1<br>Local IDS Data]
    Client2[Client 2<br>Local IDS Data]
    ClientN[Client N<br>Local IDS Data]

    Server --> Client1
    Server --> Client2
    Server --> ClientN

    Client1 -->|Model Updates| Server
    Client2 -->|Model Updates| Server
    ClientN -->|Model Updates| Server


## ⚙️ Setup
   ✅ Requirements

       Python 3.8+
       pip

   📦 Installation
       Clone the repo
       git clone https://github.com/your-username/federated-ids.git
       cd federated-ids

    Set up a virtual environment
       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

    Install dependencies
       pip install -r requirements.txt


    Dependencies:
       flwr, tensorflow or torch, scikit-learn, pandas, numpy

## 🚀 Running the System
   Start the Server
   FedAvg Strategy
   python server.py --strategy fedavg

## Simulated Annealing Strategy
   python server.py --strategy sa

## Hierarchical Model
   python server.py --strategy sa --model hierarchical

   Launch Clients

   Open a separate terminal for each client:

   python client.py --client-id 1
   python client.py --client-id 2
# Add more clients as needed

##📊 Evaluation Metrics

Accuracy — Measured on a held-out test set across multiple clients.
Convergence Speed — Number of communication rounds to reach target accuracy.
Privacy — No raw data is shared between clients or with the server.

##🧪 Example Use Case

Imagine a global network of intrusion detection sensors deployed across enterprise data centers. With this system:
Each data center trains on its own traffic logs.
No logs leave the site.
The central server aggregates only model updates.
Together, they collaboratively train a global intrusion detection model.

##🤝 Contributing

Contributions are welcome!

# Fork the project
git checkout -b feature/YourFeature
# Make changes and commit
git commit -m "Add YourFeature"
# Push and open a PR
git push origin feature/YourFeature

📄 License

This project is licensed under the MIT License.
