import subprocess

'''
##### B ########
B_values = [1e-4, 1e-3, 1e-2]
base_args = [
    "--hamiltonian_label", "heisenberg2d",
    "--n_qubits", "10",
    "--rows", "2",
    "--cols", "5"
]

for B in B_values:
    print(f"\n=== Running with B = {B} ===")
    args = ["python", "main.py"] + base_args + ["--B", str(B)]
    subprocess.run(args)

'''

##### Data Generation #####

hamiltonian_labels = ['heisenberg2d', 'antiferro_XY']
base_args = [
    "--n_qubits", "10",
    "--B", "1e-3"
]

for hamiltonian in hamiltonian_labels:
    for lattice in [[2,5],[1,10]]:
        rows = lattice[0]
        cols = lattice[1]
        print(f"\n=== Running with Hamiltonian = {hamiltonian} === on {rows}x{cols} lattice ===")
        args = ["python", "main.py"] + base_args + ["--hamiltonian_label", hamiltonian] + ["--rows", str(rows)] + ["--cols", str(cols)]
        subprocess.run(args)
