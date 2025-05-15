import subprocess

##### Data Generation #####

hamiltonian_labels = ['heisenberg', 'antiferro_XY', 'z', 'ising']
base_args = [
    "--n_qubits", "10",
    "--rows", "2",
    "--cols", "5",
    "--B", "1e-6" # optimal regularization parameter
]

for hamiltonian in hamiltonian_labels:
    print(f"\n=== Generating data for {hamiltonian} Hamiltonian ======")
    args = ["python", "data_generation.py"] + base_args + ["--hamiltonian_label", hamiltonian]
    print(args)
    subprocess.run(args)

    print(f"\n=== Predicting observables for {hamiltonian} Hamiltonian ======")
    args = ["python", "prediction.py"] + base_args + ["--hamiltonian_label", hamiltonian]
    print(args)
    subprocess.run(args)