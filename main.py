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
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys

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

################# System #######################
hamiltonian_label = args.hamiltonian_label
assert hamiltonian_label in ['heisenberg','antiferro_XY','ising','z']
n_qubits = args.n_qubits
lattice_rows, lattice_cols = args.rows, args.cols
assert lattice_rows * lattice_cols == n_qubits

################# Time Evolution ###############
trotter_steps = 3
trotter_order = 1
trotter_time = 1.0

################# Observable ###################
k_local = 3
pauli_operators = ['X', 'Y', 'Z']

################# Data/Model ###################

B = args.B # regularization parameter
n_data = 4500 # create an overdetermined system for LR and LASSO, since number of observables is poly, hence, data generation is efficient quantumly
K = 5 # cross-validation parameter

n_epochs = 2000 # for neural network
lr_ = 0.001

################# Device ###################
number_shots = 400
analytic = args.analytical is not None

if analytic:
    dev = qml.device("default.mixed", wires=n_qubits)
    print("Performing analytical simulation. \n")
else:
    dev = qml.device("lightning.qubit", wires=n_qubits, shots = number_shots) # increase shots to decrease sampling error


folder = f'experiments/{hamiltonian_label}_{lattice_rows}x{lattice_cols}_B{B}'
os.makedirs(folder, exist_ok=True)

log_path = os.path.join(folder, "output.log")
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

################################################
################# TIME EVOLUTION ###############
################################################

print("\n\n ################ HAMILTONIAN ###############\n\n")
H_evolution = functions.hamiltonian(hamiltonian_label, lattice_rows, lattice_cols)

print(f"Time evolution {hamiltonian_label} Hamiltonian with {len(H_evolution)} terms is given by: \n {H_evolution}\n")

pauli_strings = functions.kloc_pauli(n_qubits, k_local, pauli_operators) # generate Pauli strings

alpha = np.array([random.uniform(-1, 1) for _ in range(len(pauli_strings))]) # Create normalized array of random coefficients from [-1,1]
# alpha = alpha / np.linalg.norm(alpha)

observable_terms = [functions.pauli_observable(p) for p in pauli_strings]

print("\n\n ################ OBSERVABLE ###############\n\n")

print(f"Number of coefficient terms: {n_qubits} choose {k_local} * 3^{k_local} = {alpha.shape[0]} vs. number of data points: {n_data}\n")
print(f"Observable Terms (first 25 terms):{observable_terms[:25]}\n")
print(f"alpha (first 25 terms)= {alpha[:25]}\n")

assert (scipy.special.binom(n_qubits,k_local)*3**k_local == len(observable_terms))

@qml.qnode(dev)
def sample_pauli_evolved_state(x_bitstring, H_evolution, trotter_time, trotter_steps, trotter_order, observables = None):
    qml.BasisState(x_bitstring, wires=range(n_qubits))  # define initial state rho_x = |x⟩⟨x| 
    # Compute rho_H(x) = U rho_x U^* = e^-iHT rho_x e^iHt 
    qml.TrotterProduct(hamiltonian = H_evolution, time = trotter_time, n = trotter_steps, order = trotter_order)
    if analytic:
        return qml.density_matrix(wires = range(n_qubits)) # analytical result using density matrices
    else:
        return [qml.expval(p) for p in observables] # sampling error is introduced by defining the number of shots

data_x = np.array([np.random.randint(0, 2, size=n_qubits) for _ in range(n_data)])
data_pauli = np.array([np.zeros(len(observable_terms)) for _ in range(len(data_x))])
data_y = np.zeros(data_x.shape[0])

for i, string in enumerate(data_x):
    if analytic:
        rho_x = sample_pauli_evolved_state(string, H_evolution, trotter_time, trotter_steps, trotter_order) # density matrix
        data_pauli[i] = [np.trace(rho_x@qml.matrix(p_j, wire_order=range(n_qubits))).real for p_j in observable_terms] # we can cast to real in order to mitigate rounding errors since we know that all exp. values of Pauli strings are real
    else:
        data_pauli[i] = sample_pauli_evolved_state(string, H_evolution, trotter_time, trotter_steps, trotter_order, observable_terms)
    
    data_y[i] = alpha@data_pauli[i] # compute expectation value of observable
    
print("\n\n ################ DATA ###############\n\n")
print(f"x-data (first 5): {data_x[:5]}\n")
print(f"Expectation values of operator (first 5):\n{data_y[:5]}\n")
print(f"Expectation values of individual Pauli operators (first data point, first 5):\n{data_pauli[0][:5]}\n")


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

file_name = os.path.join(folder, "plot1.png")
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

file_name = os.path.join(folder, "plot2.png")
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
optimizer = optim.Adam(model.parameters(), lr=lr_)
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
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

### Model Prediction ###

with torch.no_grad():
    y_pred = model(X_test)
    mse = ((y_pred - Y_test)**2).mean()
    print(f"\n Neural Network MSE Loss on test data: {mse:.6f}")

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

file_name = os.path.join(folder, "plot3.png")
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

file_name = os.path.join(folder, "loss_plot.png")
plt.savefig(file_name, dpi=300)
plt.close()

