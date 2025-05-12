import subprocess


##### B ########

# NOTE: The hyperparameter B here refers to the hyperparameter B' in the report

B_values = [0, 1e-7, 1e-4, 1e-1]
base_args = [
    "--hamiltonian_label", "heisenberg",
    "--n_qubits", "10",
    "--rows", "2",
    "--cols", "5"
]

for B in B_values:
    print(f"\n=== Running with B = {B} ===")
    args = ["python", "main.py"] + base_args + ["--B", str(B)] 
    print(args)
    subprocess.run(args)

##### Analytical ########
print(f"\n=== Running with analytic=True ===")

base_args = [
    "--hamiltonian_label", "heisenberg",
    "--n_qubits", "6",
    "--rows", "2",
    "--cols", "3",
    "--B", "1e-4"
]

args = ["python", "main.py"] + base_args + ["--analytical", str(True)]
subprocess.run(args)

'''
##### Data Generation #####

hamiltonian_labels = ['antiferro_XY', 'z', 'ising']
base_args = [
    "--B", "1e-6",
    "--n_qubits", "10",
    "--rows", "2",
    "--cols", "5"
]

for hamiltonian in hamiltonian_labels:
    print(f"\n=== Running with Hamiltonian = {hamiltonian} === on {rows}x{cols} lattice ===")
    args = ["python", "main.py"] + base_args + ["--hamiltonian_label", hamiltonian]
    subprocess.run(args)
'''