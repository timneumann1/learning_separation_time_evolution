################################################
################# IMPORTS ######################
################################################

import numpy as np
import scipy
import itertools
import random
from typing import List
import scipy.special
import pennylane as qml
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
import pickle

################################################
################# PARAMETERS ###################
################################################

parser = argparse.ArgumentParser()

parser.add_argument("--hamiltonian_label", type=str, choices=['heisenberg', 'antiferro_XY', 'z', 'ising'], required=True)

args = parser.parse_args()

################# Data/Model ###################

hamiltonian_label = args.hamiltonian_label
assert hamiltonian_label in ['heisenberg','antiferro_XY','ising','z']

folder = f'experiments/opt/{hamiltonian_label}'
os.makedirs(folder, exist_ok=True)

log_path = os.path.join(folder, "output_prediction.log")
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

################################################
################# DATA LOADING #################
################################################

with open(f"experiments/{hamiltonian_label}/data_classical.pkl", "rb") as f:
    data = pickle.load(f)
    data_x = data['data_x']
    data_y = data['data_y']
    
with open(f"experiments/{hamiltonian_label}/data_quantum.pkl", "rb") as f:
    data = pickle.load(f)
    data_pauli = data['data_pauli']
    
with open(f"experiments/{hamiltonian_label}/data_alpha.pkl", "rb") as f:
    data = pickle.load(f)
    alpha = data['alpha']

################################################
################# DATA LOADING #################
################################################

n_epochs = 5000
weight_decay = 1e-5
patience = 500

class PauliNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def model_run(lr_, batch_size):
    X = torch.tensor(data_x, dtype=torch.float32) # train neural network on raw bitstrings as input (no access to quantum data)
    Y = torch.tensor(data_y, dtype=torch.float32).view(-1, 1) # Define ground truth comparable to LASSO model

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = PauliNN(X_train.shape[1]) # input to the model is the dimension of the qubit system 
    optimizer = optim.Adam(model.parameters(), lr=lr_ , weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    loss_fn = nn.MSELoss()
    losses = []
    
    counter = 0
    best_loss = np.inf

    for epoch in range(n_epochs):  # Model Training
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), f'experiments/opt/{hamiltonian_label}/best_model.pt')
            counter = 0
        else:
            counter += 1
            if counter > patience:
                print("Early stopping")
                break
    ### Model Prediction ###

    model.load_state_dict(torch.load(f"experiments/opt/{hamiltonian_label}/best_model.pt"))

    with torch.no_grad():
        y_pred = model(X_test)
        mse = ((y_pred - Y_test)**2).mean()
        print(f"Neural Network MSE Loss on test data: {mse:.6f}")

    r2 = r2_score(Y_test, y_pred)
    print(f"R^2 score of Neural Network:{np.mean(r2)}\n")

    comparison3 = np.vstack([y_pred[:100].squeeze().numpy(), Y_test[:100].squeeze().numpy()])

    ### Plots ###

    plt.figure(figsize=(12, 2))
    ax = sns.heatmap(
        comparison3,
        cmap="coolwarm",
        center=0,
        cbar=True,
        yticklabels=[r"$y_{pred}$", r"$y_{true}$"],
        linewidths=0.5,
        linecolor='white'
    )

    cbar = ax.collections[0].colorbar
    cbar.set_label("Expectation value", fontsize=12)

    ax.set_title(f"Comparison Heatmap for {hamiltonian_label} Hamiltonian on 10 qubits: $y_{{pred}}$ vs $y_{{true}}$ (Neural Network)", fontsize=14, pad=10)
    plt.tight_layout()

    file_name = os.path.join(folder, f"experiment_{hamiltonian_label}_lr{lr_}_batch_size{batch_size}.png")
    plt.savefig(file_name, dpi=300)
    plt.close()

    plt.figure(figsize=(12, 2))
    plt.plot(losses, linewidth=1.5)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.yscale("log")
    plt.title(f"Training Loss over Epochs for Neural Network", fontsize=14, pad=10)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    file_name = os.path.join(folder, f"loss_plot_{hamiltonian_label}_lr{lr_}_batch_size{batch_size}.png")
    plt.savefig(file_name, dpi=300)
    plt.close()
    
for lr_ in [1e-4, 1e-3,1e-2, 1e-1]:
    for batch_size in [32,128,512,3240]:
        model_run(lr_, batch_size)


