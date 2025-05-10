import subprocess
'''
##### B ########
B_values = [1e-4, 1e-3, 1e-2, 1e-1]
base_args = [
    "--hamiltonian_label", "heisenberg",
    "--n_qubits", "8",
    "--rows", "2",
    "--cols", "4"
]

for B in B_values:
    print(f"\n=== Running with B = {B} ===")
    args = ["python", "main.py"] + base_args + ["--B", str(B)]
    subprocess.run(args)
    
'''
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

hamiltonian_labels = ['heisenberg', 'antiferro_XY', 'z', 'ising']
base_args = [
    "--B", "1e-4"
]

for hamiltonian in hamiltonian_labels:
    for lattice in [[2,5],[2,6]]:
        rows = lattice[0]
        cols = lattice[1]
        n_qubits = rows * cols
        print(f"\n=== Running with Hamiltonian = {hamiltonian} === on {rows}x{cols} lattice ===")
        args = ["python", "main.py"] + base_args + ["--n_qubits", str(n_qubits)] + ["--hamiltonian_label", hamiltonian] + ["--rows", str(rows)] + ["--cols", str(cols)]
        subprocess.run(args)
'''
