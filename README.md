# Fermi-Hubbard Simulation

Quantum simulation of the Fermi-Hubbard model using [Maestro](https://qoroquantum.de/) and [Divi](https://qoroquantum.de/) from Qoro Quantum, with support for multiple lattice topologies, classical/quantum backends, and QuEPP error mitigation.

## Physics

The Fermi-Hubbard Hamiltonian:

```
H = -t Σ (c†_{i,σ} c_{j,σ} + h.c.) + U Σ n_{i,↑} n_{i,↓}
```

where `t` is the hopping energy, `U` is the on-site interaction, and `⟨i,j⟩` are nearest-neighbor bonds defined by the lattice topology.

Qubits are mapped via Jordan-Wigner: qubits `0..N-1` encode spin-up, `N..2N-1` encode spin-down. Time evolution uses second-order Trotterization. The default initial state is a domain wall (left half filled, right half empty).

## Lattice topologies

| Topology | Class | CLI flag | Description |
|----------|-------|----------|-------------|
| 1D chain | `FermiHubbardChainModel` | `--topology chain` | Linear chain, nearest-neighbor hopping |
| 2D square | `FermiHubbardSquareModel` | `--topology square` | Square lattice (Lx x Ly grid) |
| 2D honeycomb | `FermiHubbardHexModel` | `--topology hex` | Honeycomb lattice (brick-wall encoding) |

All models are defined in `model.py` and provide:
- `hamiltonian()` -- PennyLane Hamiltonian
- `trotterize(time, steps)` -- Maestro `QuantumCircuit` for time evolution
- `build_observables(obs_type)` -- labelled observables for measurement (`z`, `zz`, `density`)
- `domain_wall_bitstring()` -- initial state bitstring

## Scripts

### `fermi_hubbard_quepp.py` -- QPU execution with error mitigation

Runs Fermi-Hubbard time evolution through a 3-phase Divi pipeline:

1. **Classical reference** -- exact simulation via `MaestroSimulator`
2. **Raw QPU/sim** -- direct execution on hardware or simulator cluster
3. **QuEPP** -- same backend with QuEPP error mitigation

```bash
# 1D chain (default), ZZ correlators, simulator
python fermi_hubbard_quepp.py

# 2D square lattice, density observables
python fermi_hubbard_quepp.py --topology square --n-sites-x 3 --n-sites-y 3 --obs density

# 2D honeycomb, on QPU hardware
python fermi_hubbard_quepp.py --topology hex --n-sites-x 3 --n-sites-y 2 --backend hardware

# Tune physics parameters
python fermi_hubbard_quepp.py --topology chain --n-sites 12 --hopping 1.0 --u-int 2.0 --n-steps 5 --dt 0.1

# Dry run (show circuit fan-out without executing)
python fermi_hubbard_quepp.py --dry-run
```

### `benchmark.py` -- Adaptive simulation pipeline

A 3-tier adaptive benchmark exploiting Lieb-Robinson light-cone locality:

1. **Scout** (Pauli Propagator) -- fast CPU scan of the full system to detect active sites
2. **Sniper** (MPS, CPU, chi=64) -- simulate only the active subregion
3. **Precision** (MPS, GPU, chi=256) -- high-accuracy re-run on GPU (optional)

```bash
python benchmark.py            # CPU tiers only
python benchmark.py --gpu      # Enable GPU precision tier
python benchmark.py --scaling  # Run system-size scaling sweep
```

### Other scripts

| Script | Purpose |
|--------|---------|
| `generate_qasm.py` | Export Fermi-Hubbard circuits as OpenQASM |
| `estimate.py` | Expectation values via PennyLane + cuQuantum MPS |
| `test_scout.py` | Quick validation of the Clifford Pauli Propagator scout |

## Installation

Requires Python 3.12.

```bash
poetry install
```

For GPU support (Linux only):
```bash
poetry install -E cuquantum
```

## Dependencies

- [qoro-maestro](https://qoroquantum.de/) -- classical simulation backends (MPS, Pauli Propagator, statevector)
- [divi](https://qoroquantum.de/) -- quantum execution pipelines, QuEPP error mitigation, QoroService QPU access
- [PennyLane](https://pennylane.ai/) -- Hamiltonian construction and quantum operator algebra
- [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), [Rich](https://rich.readthedocs.io/) -- numerics, plotting, terminal output

## License

Apache-2.0
