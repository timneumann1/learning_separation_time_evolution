# SWEEP B for fixed Hamiltonian and grid scruture

import subprocess

B_values = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
base_args = [
    "--hamiltonian_label", "heisenberg2d",
    "--n_qubits", "6",
    "--rows", "2",
    "--cols", "3"
]

for B in B_values:
    print(f"\n=== Running with B = {B} ===")
    args = ["python", "main.py"] + base_args + ["--B", str(B)]
    subprocess.run(args)

# use that B and SWEEP Hamiltonians over tow differnet grtid structires


