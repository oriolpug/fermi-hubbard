"""
Use Pennylane instead of Maestro to estimate the expectation value
"""
import time
import sys

import cupy as cp
import cuquantum
import traceback

from generate_qasm import generate_qasm_circuit
from qiskit import QuantumCircuit

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

    @qml.qnode(dev)
    def circuit():
        qml_circuit()
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    start = time.time()
    result = circuit()
    elapsed = time.time() - start

    print(f" chi = {chi}:   Completed in {elapsed:.2f}s")
    print(f"    expectation values: {result}")

    return

def expect_qibo(config: dict = None, chi: int = 64) -> None:
    qasm = generate_qasm_circuit()
    num_qubits = QuantumCircuit.from_qasm_str(qasm).num_qubits

    obs = _build_z_observables(num_qubits)
    obs = obs[num_qubits // 2]

    computation_settings = {
        "MPI_enabled": False,
        "NCCL_enabled": False,
        "expectation_enabled": False,
        "MPS_enabled": {
            "qr_method": False,
            "svd_method": {
                "abs_cutoff": 0.0,
                "max_extent": chi,
            }
        }
    }

    import qibotn.backends.cutensornet
    qibotn.backends.cutensornet.cuquantum = cuquantum
    qibotn.backends.cutensornet.cp = cp

    qibo.set_backend(
        backend="qibotn",
        platform="cutensornet",
        runcard=computation_settings
    )

    if "--gpu" in sys.argv:
        qibo.set_device("/GPU:0")
    else:
        qibo.set_device("/CPU:0")

    qibo_circuit = qibo.Circuit.from_qasm(qasm)
    #
    # computation_settings = {
    #     "MPS_enabled": {
    #         "cutoff": 0,  # Quimb uses 'cutoff', not 'abs_cutoff'
    #         "max_bond": chi  # Quimb uses 'max_bond'
    #     },
    #     # Leave this empty to populate it in your loop
    #     "expectation_enabled": {}
    # }
    from qibo.symbols import Z
    from qibo.hamiltonians import SymbolicHamiltonian

    qibo_obs = SymbolicHamiltonian(Z(num_qubits//2), backend=qibo.get_backend())

    start = time.time()
    state_result = qibo_circuit()
    elapsed = time.time() - start

    raw_state_vector = state_result.state()
    from qibo.backends import NumpyBackend
    cpu_backend = NumpyBackend()

    from qibo.symbols import Z
    from qibo.hamiltonians import SymbolicHamiltonian

    qibo_obs = [SymbolicHamiltonian(
        Z(i),
        nqubits=num_qubits,
        backend=cpu_backend
    ) for i in range(num_qubits)]

    # FIX: Cast the CuPy GPU array back to a host NumPy array for the CPU backend.
    cpu_state = cp.asnumpy(raw_state_vector)
    result = [float(qibo_obs[i].expectation(cpu_state, normalize=True).real) for i in range(num_qubits)]

    print(f" chi = {chi}:   Completed in {elapsed:.2f}s")
    print(f"    expectation values: {result}")

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

    print(f" chi = {chi}:   Completed in {elapsed:.5f}s")
    print(f"    expectation values: {result}")

    return elapsed
if __name__ == "__main__":
    try:
        if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--gpu", "--pennylane"]) or (len(sys.argv) > 2 and "--pennylane" in sys.argv):
            print(f"Benchmarking Pennylane")
            import pennylane as qml
            expect_pennylane(chi=64)
            expect_pennylane(chi=256)

        if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--gpu", "--maestro"]) or (len(sys.argv) > 2 and "--maestro" in sys.argv):
            print(f"Benchmarking Maestro")
            import maestro
            expect_maestro(chi=64)
            expect_maestro(chi=256)

        if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--gpu", "--qibo"]) or (len(sys.argv) > 2 and "--qibo" in sys.argv):
            print(f"Benchmarking Qibo")
            import qibo
            expect_qibo(chi=64)
            expect_qibo(chi=256)
    except Exception as e:
        traceback.print_exc()