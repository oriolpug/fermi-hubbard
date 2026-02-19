from maestro.circuits import QuantumCircuit


class FermiHubbardModel:
    """
    1D Fermi-Hubbard model with Jordan-Wigner mapping.

    Qubit layout:
      Qubits [0, N)     → spin-up electrons
      Qubits [N, 2N)    → spin-down electrons

    Hamiltonian:
      H = -t Σ (c†_iσ c_{i+1,σ} + h.c.) + U Σ n_{i↑} n_{i↓}

    Gate decompositions:
      Hopping:     exp(-i dt t (XX + YY) / 2)  per bond
      Interaction: exp(-i dt U (I - Z↑ - Z↓ + Z↑Z↓) / 4)  per site
    """

    def __init__(self, n_sites, t=1.0, u=2.0):
        self.n_sites = n_sites
        self.t = t
        self.u = u

    def build_circuit(self, steps, dt, init_wall_idx, active_sites_range=None):
        """
        Build Trotterized time-evolution circuit.

        Args:
            steps: Number of Trotter steps.
            dt: Time per step (= T_EVOLUTION / steps).
            init_wall_idx: Global site index of domain wall.
            active_sites_range: Optional (start, end) for subsystem simulation.
        """
        circuit = QuantumCircuit()

        # ---- State Preparation: Domain Wall ----
        offset = 0
        if active_sites_range:
            start_site, end_site = active_sites_range
            offset = start_site
            n_active = end_site - start_site
            for local_i in range(n_active):
                if local_i + offset < init_wall_idx:
                    circuit.x(local_i)
                    circuit.x(local_i + n_active)
        else:
            for i in range(init_wall_idx):
                circuit.x(i)
                circuit.x(i + self.n_sites)

        # ---- Trotter Steps ----
        n_active = self.n_sites if not active_sites_range else (active_sites_range[1] - active_sites_range[0])
        up_offset = 0
        down_offset = n_active

        for _ in range(steps):
            # Even bonds
            for i in range(0, n_active - 1, 2):
                self._add_hopping(circuit, up_offset + i, up_offset + i + 1, dt)
                self._add_hopping(circuit, down_offset + i, down_offset + i + 1, dt)
            # Odd bonds
            for i in range(1, n_active - 1, 2):
                self._add_hopping(circuit, up_offset + i, up_offset + i + 1, dt)
                self._add_hopping(circuit, down_offset + i, down_offset + i + 1, dt)
            # On-site interaction
            for i in range(n_active):
                self._add_interaction(circuit, up_offset + i, down_offset + i, dt)

        return circuit

    def _add_hopping(self, qc, q1, q2, dt):
        """
        exp(-i θ (XX + YY) / 2) where θ = t * dt.
        Decomposed as exp(-iθ XX/2) · exp(-iθ YY/2).
        """
        theta = self.t * dt

        # exp(-iθ XX/2): H-CX-Rz-CX-H
        qc.h(q1); qc.h(q2)
        qc.cx(q1, q2); qc.rz(q2, theta); qc.cx(q1, q2)
        qc.h(q1); qc.h(q2)

        # exp(-iθ YY/2): S†-H-CX-Rz-CX-H-S
        qc.sdg(q1); qc.sdg(q2)
        qc.h(q1); qc.h(q2)
        qc.cx(q1, q2); qc.rz(q2, theta); qc.cx(q1, q2)
        qc.h(q1); qc.h(q2)
        qc.s(q1); qc.s(q2)

    def _add_interaction(self, qc, q_up, q_down, dt):
        """
        exp(-i dt U n↑ n↓) where n = (I-Z)/2.
        = exp(-i α) · exp(+iα Z↑) · exp(+iα Z↓) · exp(-iα Z↑Z↓)
        with α = U·dt/4.
        """
        angle = self.u * dt / 4.0
        qc.rz(q_up, -2.0 * angle)
        qc.rz(q_down, -2.0 * angle)
        qc.cx(q_up, q_down)
        qc.rz(q_down, 2.0 * angle)
        qc.cx(q_up, q_down)