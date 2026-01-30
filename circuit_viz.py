import qiskit
import sys

assert(len(sys.argv) == 2)

qc = qiskit.QuantumCircuit.from_qasm_file(sys.argv[1])

qc.draw(output="mpl", filename=sys.argv[1] + ".png")
