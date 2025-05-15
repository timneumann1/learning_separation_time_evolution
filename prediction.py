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

import functions

################################################
################# PARAMETERS ###################
################################################

parser = argparse.ArgumentParser()

parser.add_argument("--hamiltonian_label", type=str, choices=['heisenberg', 'antiferro_XY', 'z', 'ising'], required=True)
parser.add_argument("--n_qubits", type=int, required=True)
parser.add_argument("--rows", type=int, required=True)
parser.add_argument("--cols", type=int, required=True)
parser.add_argument("--B", type=float, required=True) # The parameter B here refers to the parameter B' in the report
parser.add_argument("--analytical", required=False)

args = parser.parse_args()

################# Data/Model ###################

hamiltonian_label = args.hamiltonian_label
assert hamiltonian_label in ['heisenberg','antiferro_XY','ising','z']
n_qubits = args.n_qubits
B = args.B # regularization parameter
n_data = 4500 # create an overdetermined system for LR and LASSO, since number of observables is poly, hence, data generation is efficient quantumly
K = 5 # cross-validation parameter

n_epochs = 5000 # for neural network
lr_ = 0.001
weight_decay = 1e-5

folder = f'experiments/{hamiltonian_label}'
os.makedirs(folder, exist_ok=True)

log_path = os.path.join(folder, "output_prediction.log")
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

################################################
################# DATA LOADING #################
################################################

with open(f"{folder}/data_classical.pkl", "rb") as f:
    data = pickle.load(f)
    data_x = data['data_x']
    data_y = data['data_y']
    
with open(f"{folder}/data_quantum.pkl", "rb") as f:
    data = pickle.load(f)
    data_pauli = data['data_pauli']
    
with open(f"{folder}/data_alpha.pkl", "rb") as f:
    data = pickle.load(f)
    alpha = data['alpha']

################################################
################ LASSO #########################
################################################

print("\n\n ################ RESULTS ###############\n\n")

print("### LASSO ###\n\n")

### Model Training ###

w_stars, mses, mse_trains, Y_tests, y_preds = functions.lasso_training(B, data_pauli, data_y, K)
w_star = w_stars[np.argmin(mses)] # choose best of the K values from K-fold validation

print(f"MSE loss of LASSO training: {mse_trains[np.argmin(mses)]}\n")
print(f"|w^*| = {np.linalg.norm(w_star)}\n")
print(f"|w^* - alpha| = {np.linalg.norm(w_star-alpha)}\n")

comparison = np.vstack([w_star[:250], alpha[:250]])

y_test = Y_tests[np.argmin(mses)]  # select data from best of the K runs in LASSO cross-validation
y_pred = y_preds[np.argmin(mses)]
print(f"MSE Loss of LASSO regression on test data set: {np.argmin(mses)}\n")  

print(f"Relative absolute error compared to ground truth range: {np.sqrt(np.min(mses)):.5f} over [{np.min(y_test):.5f}, {np.max(y_test):.5f}]")

comparison2 = np.vstack([y_pred[:100], y_test[:100]])

### Plots ###

plt.figure(figsize=(12, 2))
ax = sns.heatmap(
    comparison,
    cmap="coolwarm",
    center=0,
    cbar=True,
    yticklabels=[r"$w^*$", r"$\alpha$"],
    linewidths=0.5,
    linecolor='white'
)
ax.set_title(f"Comparison Heatmap for {hamiltonian_label} Hamiltonian on {n_qubits} qubits: $w^*$ vs. $\\alpha$ ", fontsize=14, pad=10)
plt.tight_layout()

file_name = os.path.join(folder, f"experiment_{hamiltonian_label}_alpha.png")
plt.savefig(file_name, dpi=300)
plt.close()

plt.figure(figsize=(12, 2))
ax = sns.heatmap(
    comparison2,
    cmap="coolwarm",
    center=0,
    cbar=True,
    yticklabels=[r"$y_{pred}$", r"$y_{true}$"],
    linewidths=0.5,
    linecolor='white'
)

cbar = ax.collections[0].colorbar
cbar.set_label("Expectation value", fontsize=12)

ax.set_title(f"Comparison Heatmap for {hamiltonian_label} Hamiltonian on {n_qubits} qubits: $y_{{pred}}$ vs. $y_{{true}}$ (LASSO)", fontsize=14, pad=10)
plt.tight_layout()

file_name = os.path.join(folder, f"experiment_{hamiltonian_label}_pred.png")
plt.savefig(file_name, dpi=300)
plt.close()

###################################################
############## NEURAL NETWORK #####################
###################################################

print("\n\n ### Neural Network ###\n\n")

X = torch.tensor(data_x, dtype=torch.float32) # train neural network on raw bitstrings as input (no access to quantum data)
Y = torch.tensor(data_y, dtype=torch.float32).view(-1, 1) # Define ground truth comparable to LASSO model

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = functions.PauliNN(X_train.shape[1]) # input to the model is the dimension of the qubit system 
optimizer = optim.Adam(model.parameters(), lr=lr_ , weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

loss_fn = nn.MSELoss()
losses = []

for epoch in range(n_epochs):  # Model Training
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred, Y_train)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

### Model Prediction ###

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

ax.set_title(f"Comparison Heatmap for {hamiltonian_label} Hamiltonian on {n_qubits} qubits: $y_{{pred}}$ vs $y_{{true}}$ (Neural Network)", fontsize=14, pad=10)
plt.tight_layout()

file_name = os.path.join(folder, f"experiment_{hamiltonian_label}_NN.png")
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

file_name = os.path.join(folder, f"loss_plot_{hamiltonian_label}.png")
plt.savefig(file_name, dpi=300)
plt.close()