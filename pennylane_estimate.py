"""
Use Pennylane instead of Maestro to estimate the expectation value
"""
import time
import sys

from benchmark import _build_z_observables
from generate_qasm import generate_qasm_circuit
import pennylane as qml

import maestro
from benchmark import _build_z_observables

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
    else:
        dev = qml.device("default.tensor", wires=num_qubits, method="mps", max_bond_dim=chi)

    observables = [qml.Z(i) for i in range(num_qubits)]
    grouped_obs = qml.pauli.group_observables(observables)

    @qml.qnode(dev)
    def circuit(group):
        qml_circuit()
        return [qml.expval(obs) for obs in group[0]]

    start = time.time()
    result = circuit(grouped_obs)
    elapsed = time.time() - start

    print(f" chi = {chi}:   Completed in {elapsed:.2f}s")
    print(f"    expectation value: {result}")

    return elapsed

def expect_maestro(config: dict = None, chi: int = 64) -> None:
    qasm = generate_qasm_circuit()
    parser = maestro.QasmToCirc()
    maestro_circuit = parser.parse_and_translate(qasm)

    obs = _build_z_observables(maestro_circuit.num_qubits)

    start = time.time()
    result = maestro_circuit.estimate(
        observables=obs,
        simulator_type=maestro.SimulatorType.Gpu if "--gpu" in sys.argv else maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.MatrixProductState,
        max_bond_dimension=chi
    )
    elapsed = time.time() - start

    print(f" chi = {chi}:   Completed in {elapsed:.2f}s")
    print(f"    expectation value: {result}")

    return elapsed
if __name__ == "__main__":
    try:
        if len(sys.argv) == 0 or (len(sys.argv) == 1 and sys.argv[0] in ["--gpu", "--pennylane"]) or (len(sys.argv) == 2 and "--pennylane" in sys.argv):
            print(f"Benchmarking Pennylane")
            expect_pennylane(chi=64)
            expect_pennylane(chi=256)

        if len(sys.argv) == 0 or (len(sys.argv) == 1 and sys.argv[0] in ["--gpu", "--maestro"]) or (len(sys.argv) == 2 and "--maestro" in sys.argv):
            print(f"Benchmarking Maestro")
            expect_maestro(chi=64)
            expect_maestro(chi=256)
    except Exception as e:
        print(e)