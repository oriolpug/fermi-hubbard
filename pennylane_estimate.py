"""
Use Pennylane instead of Maestro to estimate the expectation value
"""
import time
import sys

from benchmark import _build_z_observables
from generate_qasm import generate_qasm_circuit
import pennylane as qml
" chi = 64:   Completed in 24452.86s -> from VM"
def build_pennylane_observables(n_qubits):
    """Build per-qubit Z observables: ['ZIII..', 'IZII..', 'IIZI..', ...]"""
    obs = []
    for i in range(n_qubits):
        pauli = ['I'] * n_qubits
        pauli[i] = 'Z'
        obs.append("".join(pauli))
    return obs

def expect_pennylane(config: dict = None, chi: int = 64) -> None:

    # Generate circuit
    qasm = generate_qasm_circuit()
    qml_circuit = qml.from_qasm(qasm)

    with qml.tape.QuantumTape() as qml_tape:
        qml_circuit()

    num_qubits = len(qml_tape.wires)

    # Build circuit with expectation values
    if "--gpu" in sys.argv:
        dev = qml.device("lightning.tensor", wires=num_qubits, method="mps", max_bond_dim=chi)
        unsupported = [op for op in qml_tape.operations if op.name not in dev.operations]
        if unsupported:
            from collections import Counter
            counts = Counter(op.name for op in unsupported)
            print(f"WARNING: {len(unsupported)} unsupported ops (will force computeState): {dict(counts)}")
    else:
        dev = qml.device("default.tensor", wires=num_qubits, method="mps", max_bond_dim=chi)

    if "--gpu" in sys.argv:
        @qml.qnode(dev)
        def circuit(qubit):
            qml_circuit()
            return qml.expval(qml.PauliZ(qubit))

        start = time.time()
        result = [circuit(i) for i in range(num_qubits)]
    else:
        @qml.qnode(dev)
        def circuit():
            qml_circuit()
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        start = time.time()
        result = circuit()

    elapsed = time.time() - start

    print(f" chi = {chi}:   Completed in {elapsed:.2f}s")
    print(f"    expectation value: {result}")

if __name__ == "__main__":
    try:
        expect_pennylane(chi=64)
        expect_pennylane(chi=256)
    except Exception as e:
        print(e)