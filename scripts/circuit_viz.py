from qiskit import QuantumCircuit
import sys

if __name__ == '__main__':
    file_name = sys.argv[1]

    circuit = QuantumCircuit.from_qasm_file(file_name)

    circuit.draw(output='mpl', filename=file_name + ".png")
