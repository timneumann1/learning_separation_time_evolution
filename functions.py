import numpy as np
import scipy
import itertools
import random
from typing import List
import scipy.special
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


def hamiltonian(hamiltonian_label, lattice_rows, lattice_cols):
    '''This function returns the Hamiltonian corresponding 
        defining the time evolution of the system.
        
    Inputs:
    _________________
    hamiltonian_label: str
        Name of the Hamiltonian to be implemented
    lattice_rows: int
        Number of rows in the spin lattice
    lattice_cols: int
        Number of columns in the spin lattice
        
    Returns: 
    _________________
    hamiltonian: qml object
        PennyLane Hamiltonian
    
    '''

    if hamiltonian_label=="heisenberg":    # 2D antiferromagnetic Heisenberg model

        heisenberg_operators = []
        heisenberg_coeffs = []

        for r in range(lattice_rows):
            for c in range(lattice_cols):
                qubit_idx = lattice_cols*r+c
                # Create horiztonal Heisenberg interaction
                if qubit_idx % lattice_cols < lattice_cols-1:
                    heisenberg_coeffs.extend(np.random.uniform(-1, 1, 3))
                    heisenberg_operators.append(qml.X(qubit_idx)@qml.X(qubit_idx+1))
                    heisenberg_operators.append(qml.Y(qubit_idx)@qml.Y(qubit_idx+1))
                    heisenberg_operators.append(qml.Z(qubit_idx)@qml.Z(qubit_idx+1))
                # Create vertical Heisenberg interaction
                if qubit_idx < lattice_cols*(lattice_rows-1):
                    heisenberg_coeffs.extend(np.random.uniform(-1, 1, 3))
                    heisenberg_operators.append(qml.X(qubit_idx)@qml.X(qubit_idx+lattice_cols))
                    heisenberg_operators.append(qml.Y(qubit_idx)@qml.Y(qubit_idx+lattice_cols))
                    heisenberg_operators.append(qml.Z(qubit_idx)@qml.Z(qubit_idx+lattice_cols))

        H_evolution = qml.Hamiltonian(heisenberg_coeffs, heisenberg_operators)

    if hamiltonian_label=="antiferro_XY":    # antiferromagnetic XY model

        heisenberg_operators = []
        heisenberg_coeffs = []

        for r in range(lattice_rows):
            for c in range(lattice_cols):
                qubit_idx = lattice_cols*r+c
                # Create horiztonal interactions
                if qubit_idx % lattice_cols < lattice_cols-1:
                    heisenberg_coeffs.extend(np.random.uniform(-1, 1, 2))
                    heisenberg_operators.append(qml.X(qubit_idx)@qml.X(qubit_idx+1))
                    heisenberg_operators.append(qml.Y(qubit_idx)@qml.Y(qubit_idx+1))
                # Create vertical interaction
                if qubit_idx < lattice_cols*(lattice_rows-1):
                    heisenberg_coeffs.extend(np.random.uniform(-1, 1, 2))
                    heisenberg_operators.append(qml.X(qubit_idx)@qml.X(qubit_idx+lattice_cols))
                    heisenberg_operators.append(qml.Y(qubit_idx)@qml.Y(qubit_idx+lattice_cols))

        H_evolution = qml.Hamiltonian(heisenberg_coeffs, heisenberg_operators)
        
    if hamiltonian_label=="z":    # Z magnetic field Hamiltonian
        heisenberg_operators = []
        heisenberg_coeffs = []
        coeff = np.random.uniform(-1, 1, 1)
        for qubit_idx in range(lattice_cols*lattice_rows):
            heisenberg_coeffs.extend(coeff)
            heisenberg_operators.append(qml.Z(qubit_idx))
       
        H_evolution = qml.Hamiltonian(heisenberg_coeffs, heisenberg_operators)
        
    if hamiltonian_label=="ising":    # 2D Ising model

        heisenberg_operators = []
        heisenberg_coeffs = []

        for r in range(lattice_rows):
            for c in range(lattice_cols):
                qubit_idx = lattice_cols*r+c
                # Create horiztonal Ising interactions
                if qubit_idx % lattice_cols < lattice_cols-1:
                    heisenberg_coeffs.extend(np.random.uniform(-1, 1, 1))
                    heisenberg_operators.append(qml.Z(qubit_idx)@qml.Z(qubit_idx+1))
                # Create vertical Ising interaction
                if qubit_idx < lattice_cols*(lattice_rows-1):
                    heisenberg_coeffs.extend(np.random.uniform(-1, 1, 1))
                    heisenberg_operators.append(qml.Z(qubit_idx)@qml.Z(qubit_idx+lattice_cols))
        # Create local magnetic field
        heisenberg_coeffs.extend(np.random.uniform(-1, 1, lattice_cols*lattice_rows))
        for qubit_idx in range(lattice_cols*lattice_rows):
            heisenberg_operators.append(qml.Z(qubit_idx))

        H_evolution = qml.Hamiltonian(heisenberg_coeffs, heisenberg_operators) 
        
    return H_evolution


############## OBSERVABLE ################

def kloc_pauli(n_qubits, k_local, pauli_operators):
    '''
    This function generates all possible k-local Pauli strings on n_qubits qubits
        
    Inputs:
    _________________
    n_qubits: int
        Number of qubits
    k_local: int
        Locality of observable
    pauli_operators: 
        List of Pauli operators
        
    Returns: 
    _________________
    pauli_strings: np.array
        Array of all possible k-local Pauli strings on n_qubits qubits
    '''
    
    pauli_strings = np.full(int(scipy.special.binom(n_qubits,k_local)*3**k_local),'', dtype=f'<U{n_qubits}')
    # All combinations of k_local qubit positions
    positions = list(itertools.combinations(range(n_qubits), k_local))
    #print(len(pauli_strings),len(positions))
    assert len(pauli_strings) == 3**k_local*len(positions)
    # All combinations of Pauli operators on these positions
    pauli_idx = 0
    for i, pos in enumerate(positions):
        for j, ops in enumerate(itertools.product(pauli_operators, repeat=k_local)):
            pauli_op = ['I'] * n_qubits
            for idx, qubit in enumerate(pos):
                pauli_op[qubit] = ops[idx]
            pauli_strings[pauli_idx]=''.join(pauli_op) #3**k_local*i+j
            pauli_idx +=1
    return pauli_strings


def pauli_observable(pauli_str):
    '''
    This function generates a single obserable for measurement.
        
    Inputs:
    _________________
    pauli_str: np.array
        Some Pauli string
  
    Returns: 
    _________________
    operation: qml object
        Operation corresponding to the specified Pauli string
    '''
    op = None
    for wire, char in enumerate(pauli_str):
        if char == 'I':
            continue
        gate = {'X': qml.PauliX, 'Y': qml.PauliY, 'Z': qml.PauliZ}[char](wire)
        op = gate if op is None else op @ gate
    operation = op if op is not None else qml.Identity(0)
    return operation

def lasso_training(B, data_pauli, data_y, K):
    '''
    Lasso regression with regularization parameter B to find w with small weight such that w * data_pauli(data_x) = data_y(data_x)

        
    Inputs:
    _________________
    B: int
        hyperparameter for Lasso regularization (coefficient of penalty term)
    data_pauli: 2d np.array
        Contains the (unweighted) Pauli string expectation values for each evolved state in the training data
    data_y: np.array
        Contains the observable expectation values
    K: int
        Parameter for K-fold validation
        
    Returns: 
    _________________
    w_star: np.array
        Array of coefficients approximating coefficient vector alpha sparsely
    '''

    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    mse_list = []
    r2_scores = []
    mse_trains = []
    w_stars = []
    Y_tests = []
    y_preds = []

    for train_idx, test_idx in kf.split(data_pauli):
        X_train, X_test = data_pauli[train_idx], data_pauli[test_idx]
        y_train, y_test = data_y[train_idx], data_y[test_idx]
        if B == 0: 
            model = LinearRegression(fit_intercept=False)
        else:
            model = Lasso(alpha=B, fit_intercept=False, max_iter=100000)
        model.fit(X_train, y_train)
        w_star = model.coef_
        y_pred_train = model.predict(X_train)
        mse_train = ((y_pred_train - y_train) ** 2).mean()
        y_pred = model.predict(X_test)
        mse = ((y_pred - y_test) ** 2).mean()
        r2 = r2_score(y_test, y_pred)

        mse_list.append(mse)
        r2_scores.append(r2)        
        mse_trains.append(mse_train)
        w_stars.append(w_star)
        Y_tests.append(y_test)
        y_preds.append(y_pred)

    print(f"Cross-validated MSE:{mse_list}\n ")
    print(f"Mean CV MSE:{np.mean(mse_list)}\n")
    
    print(f"Cross-validated R^2 Scores:{r2_scores}\n ")
    print(f"Mean R^2 score:{np.mean(r2_scores)}\n")
    
    return w_stars, mse_list, mse_trains, Y_tests, y_preds

class PauliNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)




