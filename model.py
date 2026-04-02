import numpy as np
import pennylane as qml
from maestro.circuits import QuantumCircuit


def _jw_hopping_ops(q1, q2):
    """Build the two JW-mapped hopping Pauli strings for bond (q1, q2).

    Returns operators for:
        X_{q1} Z_{q1+1} ... Z_{q2-1} X_{q2}
        Y_{q1} Z_{q1+1} ... Z_{q2-1} Y_{q2}
    When q1 and q2 are adjacent, the Z string is empty.
    """
    z_string = [qml.PauliZ(q) for q in range(q1 + 1, q2)]
    xx = qml.prod(qml.PauliX(q1), *z_string, qml.PauliX(q2))
    yy = qml.prod(qml.PauliY(q1), *z_string, qml.PauliY(q2))
    return xx, yy


class FermiHubbardChainModel:
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

    def domain_wall_bitstring(self):
        """Return a bitstring for the domain-wall initial state.

        '1' = occupied (X gate applied), '0' = vacuum.
        Qubits 0..N-1 are spin-up, N..2N-1 are spin-down.
        """
        half = self.n_sites // 2
        sector = "1" * half + "0" * (self.n_sites - half)
        return sector + sector

    def build_observables(self, obs_type):
        """Build labelled observables for measurement.

        Args:
            obs_type: 'z' per-qubit Z, 'zz' nearest-neighbor ZZ, 'density' per-site occupation.
        """
        if obs_type == "z":
            return [(f"Z_{i}", qml.Z(i)) for i in range(self.n_qubits)]
        elif obs_type == "zz":
            result = []
            for i in range(self.n_sites - 1):
                result.append((f"ZZ_up_{i}_{i+1}", qml.Z(i) @ qml.Z(i + 1)))
            for i in range(self.n_sites - 1):
                j = self.n_sites + i
                result.append((f"ZZ_dn_{i}_{i+1}", qml.Z(j) @ qml.Z(j + 1)))
            return result
        elif obs_type == "density":
            result = []
            for i in range(self.n_sites):
                result.append((f"n_up_{i}", qml.Z(i)))
                result.append((f"n_dn_{i}", qml.Z(i + self.n_sites)))
            return result
        raise ValueError(f"Unknown obs_type: {obs_type}")

    def description(self):
        return f"1D chain, {self.n_sites} sites ({self.n_qubits} qubits), t={self.t}, U={self.u}"

    def hamiltonian(self):
        """
        Return the PennyLane Hamiltonian for the 1D Fermi-Hubbard model.

        H = -t Σ (X_i X_{i+1} + Y_i Y_{i+1}) / 2
            + U Σ (I - Z_↑ - Z_↓ + Z_↑Z_↓) / 4
        """
        coeffs = []
        ops = []
        up_offset = 0
        down_offset = self.n_sites

        # Hopping terms (nearest-neighbor, no JW string needed)
        for i in range(self.n_sites - 1):
            for offset in [up_offset, down_offset]:
                q1, q2 = offset + i, offset + i + 1
                xx, yy = _jw_hopping_ops(q1, q2)
                coeffs.extend([-self.t / 2, -self.t / 2])
                ops.extend([xx, yy])

        # Interaction terms: U/4 * (I - Z_↑ - Z_↓ + Z_↑Z_↓) per site
        for i in range(self.n_sites):
            q_up = up_offset + i
            q_down = down_offset + i
            coeffs.append(self.u / 4)
            ops.append(qml.Identity(q_up))
            coeffs.append(-self.u / 4)
            ops.append(qml.PauliZ(q_up))
            coeffs.append(-self.u / 4)
            ops.append(qml.PauliZ(q_down))
            coeffs.append(self.u / 4)
            ops.append(qml.prod(qml.PauliZ(q_up), qml.PauliZ(q_down)))

        return qml.Hamiltonian(coeffs, ops)

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


class FermiHubbardSquareModel:
    """
    2D Fermi-Hubbard model on a square lattice with Jordan-Wigner mapping.

    Sites are arranged on an Lx × Ly grid with row-major ordering:
        site (x, y) → linear index y * Lx + x

    Qubits 0..N-1 encode spin-up, qubits N..2N-1 encode spin-down,
    where N = Lx * Ly.

    Horizontal bonds connect adjacent qubits (no JW string needed).
    Vertical bonds span Lx qubits and require a JW string of length Lx-1.
    """

    def __init__(self, n_sites_x, n_sites_y, t=1.0, u=4.0):
        self.n_sites_x = n_sites_x
        self.n_sites_y = n_sites_y
        self.n_sites = n_sites_x * n_sites_y
        self.t = t
        self.u = u
        self.n_qubits = 2 * self.n_sites

    def _site_index(self, x, y):
        return y * self.n_sites_x + x

    def _get_bonds(self):
        """Return list of (site_i, site_j) with site_i < site_j."""
        bonds = []
        for y in range(self.n_sites_y):
            for x in range(self.n_sites_x):
                # Horizontal bond
                if x + 1 < self.n_sites_x:
                    bonds.append((self._site_index(x, y), self._site_index(x + 1, y)))
                # Vertical bond
                if y + 1 < self.n_sites_y:
                    bonds.append((self._site_index(x, y), self._site_index(x, y + 1)))
        return bonds

    def _add_hopping_term(self, circuit, q1, q2, dt):
        """
        Implements exp(-i * dt * t * (X Z...Z X + Y Z...Z Y) / 2)
        with a Jordan-Wigner string on all qubits between q1 and q2.

        When q1 and q2 are adjacent, this reduces to the standard XX+YY gate.

        Decomposition: rotate endpoints to Z basis, CNOT staircase to compute
        parity into q2, Rz, then undo.
        """
        theta = self.t * dt
        all_qubits = list(range(q1, q2 + 1))

        # --- exp(-i θ XZ...ZX / 2) ---
        circuit.h(q1)
        circuit.h(q2)
        for i in range(len(all_qubits) - 1):
            circuit.cx(all_qubits[i], all_qubits[i + 1])
        circuit.rz(q2, theta)
        for i in range(len(all_qubits) - 2, -1, -1):
            circuit.cx(all_qubits[i], all_qubits[i + 1])
        circuit.h(q1)
        circuit.h(q2)

        # --- exp(-i θ YZ...ZY / 2) ---
        circuit.sdg(q1)
        circuit.sdg(q2)
        circuit.h(q1)
        circuit.h(q2)
        for i in range(len(all_qubits) - 1):
            circuit.cx(all_qubits[i], all_qubits[i + 1])
        circuit.rz(q2, theta)
        for i in range(len(all_qubits) - 2, -1, -1):
            circuit.cx(all_qubits[i], all_qubits[i + 1])
        circuit.h(q1)
        circuit.h(q2)
        circuit.s(q1)
        circuit.s(q2)

    def _add_interaction_term(self, circuit, q_up, q_down, dt):
        """
        Implements exp(-i * dt * U * n_↑ n_↓).
        Same decomposition as the 1D model.
        """
        angle = self.u * dt / 4.0
        circuit.rz(q_up, -2.0 * angle)
        circuit.rz(q_down, -2.0 * angle)
        circuit.cx(q_up, q_down)
        circuit.rz(q_down, 2.0 * angle)
        circuit.cx(q_up, q_down)

    def _prepare_domain_wall(self, circuit):
        """Fill left half of the grid with UP and DOWN particles."""
        half = self.n_sites // 2
        for i in range(half):
            circuit.x(i)
            circuit.x(i + self.n_sites)

    def domain_wall_bitstring(self):
        half = self.n_sites // 2
        sector = "1" * half + "0" * (self.n_sites - half)
        return sector + sector

    def build_observables(self, obs_type):
        if obs_type == "z":
            return [(f"Z_{i}", qml.Z(i)) for i in range(self.n_qubits)]
        elif obs_type == "zz":
            result = []
            for s1, s2 in self._get_bonds():
                result.append((f"ZZ_up_{s1}_{s2}", qml.Z(s1) @ qml.Z(s2)))
            for s1, s2 in self._get_bonds():
                q1 = self.n_sites + s1
                q2 = self.n_sites + s2
                result.append((f"ZZ_dn_{s1}_{s2}", qml.Z(q1) @ qml.Z(q2)))
            return result
        elif obs_type == "density":
            result = []
            for i in range(self.n_sites):
                result.append((f"n_up_{i}", qml.Z(i)))
                result.append((f"n_dn_{i}", qml.Z(i + self.n_sites)))
            return result
        raise ValueError(f"Unknown obs_type: {obs_type}")

    def description(self):
        return (f"2D square {self.n_sites_x}x{self.n_sites_y}, "
                f"{self.n_sites} sites ({self.n_qubits} qubits), t={self.t}, U={self.u}")

    def hamiltonian(self):
        """
        Return the PennyLane Hamiltonian for the 2D square Fermi-Hubbard model.

        Hopping bonds include JW strings for non-adjacent qubits (vertical bonds).
        """
        coeffs = []
        ops = []
        up_offset = 0
        down_offset = self.n_sites

        for (s1, s2) in self._get_bonds():
            for offset in [up_offset, down_offset]:
                q1, q2 = offset + s1, offset + s2
                xx, yy = _jw_hopping_ops(q1, q2)
                coeffs.extend([-self.t / 2, -self.t / 2])
                ops.extend([xx, yy])

        for i in range(self.n_sites):
            q_up = up_offset + i
            q_down = down_offset + i
            coeffs.append(self.u / 4)
            ops.append(qml.Identity(q_up))
            coeffs.append(-self.u / 4)
            ops.append(qml.PauliZ(q_up))
            coeffs.append(-self.u / 4)
            ops.append(qml.PauliZ(q_down))
            coeffs.append(self.u / 4)
            ops.append(qml.prod(qml.PauliZ(q_up), qml.PauliZ(q_down)))

        return qml.Hamiltonian(coeffs, ops)

    def trotterize(self, time, steps, prepare_initial_state=True):
        """
        Build a second-order Trotterized time-evolution circuit.

        Bonds are split into two sets (even/odd indexed) for parallelism.
        """
        circuit = QuantumCircuit()

        if prepare_initial_state:
            self._prepare_domain_wall(circuit)

        dt = time / steps
        up_offset = 0
        down_offset = self.n_sites

        bonds = self._get_bonds()
        even_bonds = bonds[0::2]
        odd_bonds = bonds[1::2]

        for _ in range(steps):
            for bond_set in [even_bonds, odd_bonds]:
                for (s1, s2) in bond_set:
                    self._add_hopping_term(circuit, up_offset + s1, up_offset + s2, dt)
                    self._add_hopping_term(circuit, down_offset + s1, down_offset + s2, dt)

            for i in range(self.n_sites):
                self._add_interaction_term(circuit, up_offset + i, down_offset + i, dt)

        return circuit


class FermiHubbardHexModel:
    """
    2D Fermi-Hubbard model on a honeycomb (hexagonal) lattice
    with Jordan-Wigner mapping.

    Uses a brick-wall representation on an Lx × Ly grid:
        - All horizontal neighbor bonds: (x,y)-(x+1,y)
        - Vertical bonds only where (x + y) is even: (x,y)-(x,y+1)

    This gives honeycomb connectivity (coordination number 3 for interior sites).

    Sites use row-major ordering: site (x, y) → y * Lx + x.
    Qubits 0..N-1 encode spin-up, qubits N..2N-1 encode spin-down.
    """

    def __init__(self, n_sites_x, n_sites_y, t=1.0, u=4.0):
        self.n_sites_x = n_sites_x
        self.n_sites_y = n_sites_y
        self.n_sites = n_sites_x * n_sites_y
        self.t = t
        self.u = u
        self.n_qubits = 2 * self.n_sites

    def _site_index(self, x, y):
        return y * self.n_sites_x + x

    def _get_bonds(self):
        """Return list of (site_i, site_j) with site_i < site_j."""
        bonds = []
        for y in range(self.n_sites_y):
            for x in range(self.n_sites_x):
                # Horizontal bond
                if x + 1 < self.n_sites_x:
                    bonds.append((self._site_index(x, y), self._site_index(x + 1, y)))
                # Vertical bond (brick-wall: only where x + y is even)
                if y + 1 < self.n_sites_y and (x + y) % 2 == 0:
                    bonds.append((self._site_index(x, y), self._site_index(x, y + 1)))
        return bonds

    def _add_hopping_term(self, circuit, q1, q2, dt):
        """
        Implements exp(-i * dt * t * (X Z...Z X + Y Z...Z Y) / 2)
        with a Jordan-Wigner string on all qubits between q1 and q2.
        """
        theta = self.t * dt
        all_qubits = list(range(q1, q2 + 1))

        # --- exp(-i θ XZ...ZX / 2) ---
        circuit.h(q1)
        circuit.h(q2)
        for i in range(len(all_qubits) - 1):
            circuit.cx(all_qubits[i], all_qubits[i + 1])
        circuit.rz(q2, theta)
        for i in range(len(all_qubits) - 2, -1, -1):
            circuit.cx(all_qubits[i], all_qubits[i + 1])
        circuit.h(q1)
        circuit.h(q2)

        # --- exp(-i θ YZ...ZY / 2) ---
        circuit.sdg(q1)
        circuit.sdg(q2)
        circuit.h(q1)
        circuit.h(q2)
        for i in range(len(all_qubits) - 1):
            circuit.cx(all_qubits[i], all_qubits[i + 1])
        circuit.rz(q2, theta)
        for i in range(len(all_qubits) - 2, -1, -1):
            circuit.cx(all_qubits[i], all_qubits[i + 1])
        circuit.h(q1)
        circuit.h(q2)
        circuit.s(q1)
        circuit.s(q2)

    def _add_interaction_term(self, circuit, q_up, q_down, dt):
        """
        Implements exp(-i * dt * U * n_↑ n_↓).
        """
        angle = self.u * dt / 4.0
        circuit.rz(q_up, -2.0 * angle)
        circuit.rz(q_down, -2.0 * angle)
        circuit.cx(q_up, q_down)
        circuit.rz(q_down, 2.0 * angle)
        circuit.cx(q_up, q_down)

    def _prepare_domain_wall(self, circuit):
        """Fill left half of the grid with UP and DOWN particles."""
        half = self.n_sites // 2
        for i in range(half):
            circuit.x(i)
            circuit.x(i + self.n_sites)

    def domain_wall_bitstring(self):
        half = self.n_sites // 2
        sector = "1" * half + "0" * (self.n_sites - half)
        return sector + sector

    def build_observables(self, obs_type):
        if obs_type == "z":
            return [(f"Z_{i}", qml.Z(i)) for i in range(self.n_qubits)]
        elif obs_type == "zz":
            result = []
            for s1, s2 in self._get_bonds():
                result.append((f"ZZ_up_{s1}_{s2}", qml.Z(s1) @ qml.Z(s2)))
            for s1, s2 in self._get_bonds():
                q1 = self.n_sites + s1
                q2 = self.n_sites + s2
                result.append((f"ZZ_dn_{s1}_{s2}", qml.Z(q1) @ qml.Z(q2)))
            return result
        elif obs_type == "density":
            result = []
            for i in range(self.n_sites):
                result.append((f"n_up_{i}", qml.Z(i)))
                result.append((f"n_dn_{i}", qml.Z(i + self.n_sites)))
            return result
        raise ValueError(f"Unknown obs_type: {obs_type}")

    def description(self):
        return (f"2D honeycomb {self.n_sites_x}x{self.n_sites_y}, "
                f"{self.n_sites} sites ({self.n_qubits} qubits), t={self.t}, U={self.u}")

    def hamiltonian(self):
        """
        Return the PennyLane Hamiltonian for the honeycomb Fermi-Hubbard model.

        Hopping bonds include JW strings for non-adjacent qubits (vertical bonds).
        """
        coeffs = []
        ops = []
        up_offset = 0
        down_offset = self.n_sites

        for (s1, s2) in self._get_bonds():
            for offset in [up_offset, down_offset]:
                q1, q2 = offset + s1, offset + s2
                xx, yy = _jw_hopping_ops(q1, q2)
                coeffs.extend([-self.t / 2, -self.t / 2])
                ops.extend([xx, yy])

        for i in range(self.n_sites):
            q_up = up_offset + i
            q_down = down_offset + i
            coeffs.append(self.u / 4)
            ops.append(qml.Identity(q_up))
            coeffs.append(-self.u / 4)
            ops.append(qml.PauliZ(q_up))
            coeffs.append(-self.u / 4)
            ops.append(qml.PauliZ(q_down))
            coeffs.append(self.u / 4)
            ops.append(qml.prod(qml.PauliZ(q_up), qml.PauliZ(q_down)))

        return qml.Hamiltonian(coeffs, ops)

    def trotterize(self, time, steps, prepare_initial_state=True):
        """
        Build a second-order Trotterized time-evolution circuit.

        Bonds are split into two sets (even/odd indexed) for parallelism.
        """
        circuit = QuantumCircuit()

        if prepare_initial_state:
            self._prepare_domain_wall(circuit)

        dt = time / steps
        up_offset = 0
        down_offset = self.n_sites

        bonds = self._get_bonds()
        even_bonds = bonds[0::2]
        odd_bonds = bonds[1::2]

        for _ in range(steps):
            for bond_set in [even_bonds, odd_bonds]:
                for (s1, s2) in bond_set:
                    self._add_hopping_term(circuit, up_offset + s1, up_offset + s2, dt)
                    self._add_hopping_term(circuit, down_offset + s1, down_offset + s2, dt)

            for i in range(self.n_sites):
                self._add_interaction_term(circuit, up_offset + i, down_offset + i, dt)

        return circuit

class TFIMModel:
    """
    1D Transverse-Field Ising Model.

    Hamiltonian: H = -J Σ ZᵢZᵢ₊₁ - h Σ Xᵢ

    One qubit per site. Domain wall initial state: left half spin-up.
    """

    def __init__(self, n_sites, j=1.0, h=0.5):
        self.n_sites = n_sites
        self.n_qubits = n_sites
        self.j = j
        self.h = h

    def hamiltonian(self):
        """Build the 1D TFIM Hamiltonian: H = -J Σ ZᵢZᵢ₊₁ - h Σ Xᵢ."""
        coeffs = []
        ops = []
        for i in range(self.n_sites - 1):
            coeffs.append(-self.j)
            ops.append(qml.Z(i) @ qml.Z(i + 1))
        for i in range(self.n_sites):
            coeffs.append(-self.h)
            ops.append(qml.X(i))
        return qml.Hamiltonian(coeffs, ops)

    def domain_wall_bitstring(self):
        wall = self.n_sites // 2
        return "1" * wall + "0" * (self.n_sites - wall)

    def build_observables(self, obs_type):
        if obs_type == "z":
            return [(f"Z_{i}", qml.Z(i)) for i in range(self.n_sites)]
        elif obs_type == "zz":
            return [
                (f"Z_{i}Z_{i+1}", qml.Z(i) @ qml.Z(i + 1))
                for i in range(self.n_sites - 1)
            ]
        elif obs_type == "density":
            return [(f"Z_{i}", qml.Z(i)) for i in range(self.n_sites)]
        raise ValueError(f"Unknown obs_type: {obs_type}")

    def description(self):
        return f"1D TFIM, {self.n_sites} sites ({self.n_qubits} qubits), J={self.j}, h={self.h}"
