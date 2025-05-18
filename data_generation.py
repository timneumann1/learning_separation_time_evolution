################################################
################# IMPORTS ######################
################################################

import numpy as np
import scipy
import random
import scipy.special
import pennylane as qml
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


################# Device ###################
number_shots = 400
analytic = args.analytical is not None
n_data = 4500 # create an overdetermined system for LR and LASSO, since number of observables is poly, hence, data generation is efficient quantumly

if analytic:
    dev = qml.device("default.mixed", wires=n_qubits)
    print("Performing analytical simulation. \n")
else:
    dev = qml.device("lightning.qubit", wires=n_qubits, shots = number_shots) # increase shots to decrease sampling error


folder = f'experiments/{hamiltonian_label}'
os.makedirs(folder, exist_ok=True)

log_path = os.path.join(folder, "output_data_generation.log")
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

data_classical = {'data_x': data_x, 'data_y': data_y}
with open(f"{folder}/data_classical.pkl", "wb") as f:
    pickle.dump(data_classical, f)
    
data_quantum = {'data_pauli': data_pauli, 'data_y': data_y}
with open(f"{folder}/data_quantum.pkl", "wb") as f:
    pickle.dump(data_quantum, f)
    
data_alpha = {'alpha': alpha}
with open(f"{folder}/data_alpha.pkl", "wb") as f:
    pickle.dump(data_alpha, f)


