import numpy as np
from maestro.circuits import QuantumCircuit


class FermiHubbardModel:
    """
    1D Fermi-Hubbard model with Jordan-Wigner mapping.

    Qubits 0..N-1 encode spin-up, qubits N..2N-1 encode spin-down.
    The Hamiltonian is:
        H = -t Σ (c†_{i,σ} c_{i+1,σ} + h.c.) + U Σ n_{i,↑} n_{i,↓}

    Under Jordan-Wigner (nearest-neighbor, no Jordan-Wigner strings needed):
        Hopping: exp(-i dt t (XX + YY) / 2)
        Interaction: exp(-i dt U n_↑ n_↓)
                   = exp(-i dt U/4 (I - Z_↑ - Z_↓ + Z_↑Z_↓))
    """

    def __init__(self, n_sites, t=1.0, u=4.0):
        self.n_sites = n_sites
        self.t = t  # Hopping energy (Kinetic)
        self.u = u  # Interaction energy (Potential)
        self.n_qubits = 2 * n_sites

    def _add_hopping_term(self, circuit, q1, q2, dt):
        """
        Implements exp(-i * dt * t * (XX + YY) / 2) via:
            exp(-i θ XX/2) · exp(-i θ YY/2)

        where θ = t * dt.

        Decomposition of exp(-i θ XX/2):
            H(q1) H(q2) CX(q1,q2) Rz(q2, θ) CX(q1,q2) H(q1) H(q2)

        Decomposition of exp(-i θ YY/2):
            Sdg(q1) Sdg(q2) H(q1) H(q2) CX(q1,q2) Rz(q2, θ) CX(q1,q2) H(q1) H(q2) S(q1) S(q2)
        """
        theta = self.t * dt

        # --- exp(-i θ XX/2) ---
        circuit.h(q1)
        circuit.h(q2)
        circuit.cx(q1, q2)
        circuit.rz(q2, theta)
        circuit.cx(q1, q2)
        circuit.h(q1)
        circuit.h(q2)

        # --- exp(-i θ YY/2) ---
        # Conjugate by S†: maps Y basis → X basis
        circuit.sdg(q1)
        circuit.sdg(q2)
        circuit.h(q1)
        circuit.h(q2)
        circuit.cx(q1, q2)
        circuit.rz(q2, theta)
        circuit.cx(q1, q2)
        circuit.h(q1)
        circuit.h(q2)
        circuit.s(q1)
        circuit.s(q2)

    def _add_interaction_term(self, circuit, q_up, q_down, dt):
        """
        Implements exp(-i * dt * U * n_↑ n_↓) where n = (I - Z)/2.

        Expanding: n_↑ n_↓ = (I - Z_↑ - Z_↓ + Z_↑Z_↓) / 4

        So we need:
            exp(-i * dt * U/4 * (I - Z_↑ - Z_↓ + Z_↑Z_↓))
          = exp(-i dt U/4)                     [global phase, ignorable]
          * exp(+i dt U/4 Z_↑)                [Rz on qubit ↑]
          * exp(+i dt U/4 Z_↓)                [Rz on qubit ↓]
          * exp(-i dt U/4 Z_↑Z_↓)             [ZZ interaction]
        """
        angle = self.u * dt / 4.0

        # Single-qubit Z rotations: exp(+i angle Z) = Rz(-2*angle)
        circuit.rz(q_up, -2.0 * angle)
        circuit.rz(q_down, -2.0 * angle)

        # ZZ interaction: exp(-i angle ZZ) via CX-Rz-CX
        circuit.cx(q_up, q_down)
        circuit.rz(q_down, 2.0 * angle)
        circuit.cx(q_up, q_down)

    def _prepare_domain_wall(self, circuit):
        """Fill left half with UP and DOWN particles (domain wall state)."""
        half = self.n_sites // 2
        for i in range(half):
            circuit.x(i)                # Spin UP (Qubits 0 to N-1)
            circuit.x(i + self.n_sites)  # Spin DOWN (Qubits N to 2N-1)

    def trotterize(self, time, steps, prepare_initial_state=True):
        """
        Build a second-order Trotterized time-evolution circuit.

        Args:
            time (float): Total simulation time.
            steps (int): Number of Trotter steps.
            prepare_initial_state (bool): Whether to prepend the domain wall state.
        """
        circuit = QuantumCircuit()

        if prepare_initial_state:
            self._prepare_domain_wall(circuit)

        dt = time / steps

        up_offset = 0
        down_offset = self.n_sites

        for _ in range(steps):
            # Even bonds
            for i in range(0, self.n_sites - 1, 2):
                self._add_hopping_term(circuit, up_offset + i, up_offset + i + 1, dt)
                self._add_hopping_term(circuit, down_offset + i, down_offset + i + 1, dt)

            # Odd bonds
            for i in range(1, self.n_sites - 1, 2):
                self._add_hopping_term(circuit, up_offset + i, up_offset + i + 1, dt)
                self._add_hopping_term(circuit, down_offset + i, down_offset + i + 1, dt)

            # Interaction
            for i in range(self.n_sites):
                self._add_interaction_term(circuit, up_offset + i, down_offset + i, dt)

        return circuit