"""Quick small-scale test of the Clifford scout."""
import time
import maestro
from maestro.circuits import QuantumCircuit

# Import the model from benchmark
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from benchmark import FermiHubbardModel, _build_z_observables, _z_to_density

# Small system: 20 qubits (10 sites)
N_SITES = 100
TOTAL_QUBITS = 2 * N_SITES
WALL = N_SITES // 2  # site 5

print(f"=== Quick Scout Test: {TOTAL_QUBITS} qubits, {N_SITES} sites, wall at {WALL} ===\n")

# --- Tier 1: Clifford PP Scout ---
model = FermiHubbardModel(N_SITES, t=1.0, u=1.0)
scout_steps = 25
scout_circuit = model.build_clifford_scout_circuit(steps=scout_steps, init_wall_idx=WALL)

obs_list = _build_z_observables(TOTAL_QUBITS)

t0 = time.time()
result = scout_circuit.estimate(
    observables=obs_list,
    simulator_type=maestro.SimulatorType.QCSim,
    simulation_type=maestro.SimulationType.PauliPropagator,
)
t1 = time.time()

z_vals = result['expectation_values']
print(f"Scout (PP Clifford): {t1-t0:.3f}s")
print(f"  Z values: {[f'{v:.3f}' for v in z_vals]}")

# Detect active region
THRESHOLD = 0.001
active_start = WALL
active_end = WALL
for i in range(N_SITES):
    initial_z = -1.0 if i < WALL else 1.0
    if (abs(z_vals[i] - initial_z) > THRESHOLD or
            abs(z_vals[N_SITES + i] - initial_z) > THRESHOLD):
        active_start = min(active_start, i)
        active_end = max(active_end, i + 1)

MARGIN = 2
active_start = max(0, active_start - MARGIN)
active_end = min(N_SITES, active_end + MARGIN)
n_active = active_end - active_start
print(f"  Active region: [{active_start}, {active_end}) → {n_active} sites ({2*n_active} qubits)")

# --- Tier 2: MPS Low BD ---
print(f"\nSniper (MPS χ=16) on {2*n_active} qubits...")
sniper_model = FermiHubbardModel(n_active, t=1.0, u=1.0)
sniper_circuit = sniper_model.build_circuit(
    steps=20, dt=5.0/20, init_wall_idx=WALL,
    active_sites_range=(active_start, active_end)
)
sniper_obs = _build_z_observables(2 * n_active)
t0 = time.time()
result2 = sniper_circuit.estimate(
    observables=sniper_obs,
    simulator_type=maestro.SimulatorType.QCSim,
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=16
)
t1 = time.time()
print(f"  Completed in {t1-t0:.3f}s")
density = _z_to_density(result2['expectation_values'], n_active)
print(f"  Density: {[f'{d:.3f}' for d in density]}")

print("\n✓ All tiers completed successfully!")

# --- Visualization ---
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Scout Z values
up_z = z_vals[:N_SITES]
down_z = z_vals[N_SITES:]
sites = list(range(N_SITES))
ax1.bar([s - 0.15 for s in sites], up_z, width=0.3, color='#3498DB', alpha=0.8, label='Spin-up ⟨Z⟩')
ax1.bar([s + 0.15 for s in sites], down_z, width=0.3, color='#E74C3C', alpha=0.8, label='Spin-down ⟨Z⟩')
# Initial state reference
initial_z = [-1.0 if i < WALL else 1.0 for i in range(N_SITES)]
ax1.step([s + 0.5 for s in sites], initial_z, color='gray', linewidth=2, linestyle='--',
         where='mid', label='Initial ⟨Z⟩')
ax1.axvline(x=WALL - 0.5, color='red', linestyle=':', linewidth=2, label=f'Domain wall (site {WALL})')
ax1.set_xlabel('Site Index', fontsize=12)
ax1.set_ylabel('⟨Z⟩', fontsize=12)
ax1.set_title(f'Tier 1: Clifford Scout (PP)\n{TOTAL_QUBITS} qubits, {scout_steps} steps', fontsize=11)
ax1.legend(fontsize=9, loc='upper right')
ax1.set_ylim(-1.3, 1.3)
ax1.grid(axis='y', alpha=0.3)

# Right: Sniper density
sniper_sites = list(range(active_start, active_end))
ax2.bar(sniper_sites, density, color='#2F847C', width=1.0, alpha=0.8, label=f'MPS (χ=16)')
# Reference: initial density
init_density = [2.0 if i < WALL else 0.0 for i in range(active_start, active_end)]
ax2.step([s + 0.5 for s in sniper_sites], init_density, color='gray', linewidth=2,
         linestyle='--', where='mid', label='Initial density')
ax2.axvline(x=WALL - 0.5, color='red', linestyle=':', linewidth=2, label=f'Domain wall')
ax2.set_xlabel('Site Index', fontsize=12)
ax2.set_ylabel('Density ⟨n↑⟩ + ⟨n↓⟩', fontsize=12)
ax2.set_title(f'Tier 2: MPS Sniper\n{2*n_active} qubits, χ=16', fontsize=11)
ax2.legend(fontsize=9, loc='upper right')
ax2.set_ylim(-0.1, 2.3)
ax2.grid(axis='y', alpha=0.3)

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'test_scout_output.png')
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
