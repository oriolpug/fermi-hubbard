import maestro


qasm_circuit = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
u3(pi/2,0,pi) q[9];
cx q[9],q[8];
cx q[8],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
"""

circuit_parser = maestro.QasmToCirc()
circuit = circuit_parser.parse_and_translate(qasm_circuit)

num_qubits = circuit.num_qubits

def _build_z_observables(n_qubits):
    """Build per-qubit Z observables: ['ZIII..', 'IZII..', 'IIZI..', ...]"""
    obs = []
    for i in range(n_qubits):
        pauli = ['I'] * n_qubits
        pauli[i] = 'Z'
        obs.append("".join(pauli))
    return obs

obs = _build_z_observables(num_qubits)

result = circuit.estimate(
    observables=obs,
    simulator_type=maestro.SimulatorType.QCSim,
    simulation_type=maestro.SimulationType.PauliPropagator
)

z_vals = result["expectation_values"]
print(z_vals)
