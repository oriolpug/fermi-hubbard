"""
Adaptive Fermi-Hubbard Benchmark
================================

Demonstrates a 3-tier adaptive simulation pipeline for the 1D Fermi-Hubbard
model using Maestro. The key insight: for local quench dynamics, information
propagates at a finite speed (Lieb-Robinson velocity), so most of the system
remains frozen. We exploit this to solve a nominally huge system by simulating
only the active light-cone region.

Pipeline:
  Tier 1 — "Scout" (Pauli Propagator, CPU):
    Runs a fast approximate simulation on the FULL system to identify which
    sites have non-trivial dynamics. Cost: seconds, scales linearly with
    system size.

  Tier 2 — "Sniper" (MPS, CPU, χ=64):
    Runs an MPS simulation ONLY on the active subregion detected by the
    scout. This gives a quick physics preview at moderate accuracy.

  Tier 3 — "Precision" (MPS, GPU, χ=256):
    Re-runs the active subregion with higher bond dimension on GPU for
    converged, publication-quality results. Optional — requires GPU.

Physics:
  - 1D Fermi-Hubbard chain, N sites (2N qubits: spin-up + spin-down)
  - Jordan-Wigner mapping (nearest-neighbor, no JW strings)
  - Domain-wall initial state: left half filled, right half empty
  - Time evolution via Trotterization

  We use U/t = 2.0 (moderate interaction) rather than U/t = 4.0 (strong Mott
  insulator) to ensure meaningful charge transport — in the Mott regime,
  particles barely move and the light cone stays trivially small.

Output:
  - adaptive_hubbard_density.png : Density profile (CPU vs GPU comparison)
  - adaptive_hubbard_scaling.png : Wall-clock time vs total system size

Usage:
  python benchmark.py
  python benchmark.py --gpu    # Enable GPU tier
"""

import sys
import time
import matplotlib.pyplot as plt
import maestro
import math
from model import FermiHubbardModel
# =============================================================================
# CONFIGURATION
# =============================================================================

# Physics parameters
T_HOP = 1.0          # Hopping energy t (kinetic term)
U_INT = 1.0          # On-site interaction U — using U/t=1 (metallic regime)
                     # for maximum charge spreading. Higher U/t suppresses
                     # transport (Mott insulator) and shrinks the light cone.
T_EVOLUTION = 5.0    # Evolution time. The light cone grows as v_LR * T ≈ 2t * T.
                     # T=5 gives a genuine ~20-site active region (40 qubits),
                     # past statevector limits. Increase to 20 for production.
N_STEPS = 50         # Trotter steps (dt = 0.1, good accuracy for t=1.0)

# MPS bond dimensions for each tier
CHI_CPU = 64         # Tier 2: CPU quick preview
CHI_GPU = 256        # Tier 3: GPU accuracy boost

# Safety margin (sites) added around the PP-detected active region.
# We use the Lieb-Robinson velocity bound to set this: v_LR ≈ 2t for 1D hopping,
# but the PP is approximate and may underestimate the true spreading.
# The margin accounts for dynamics the PP can't fully track.
SAFETY_MARGIN = 5

# Scaling sweep: total qubit counts to benchmark (each = 2 * N_SITES)
# NOTE: For production runs, use [200, 500, 1000]. Reduced here for testing.
SYSTEM_SIZES = [50]

# GPU tier: enable with --gpu flag or set to True
GPU_ENABLED = '--gpu' in sys.argv

# =============================================================================
# TIER 1: SCOUT (Pauli Propagator)
# =============================================================================

def run_scout(n_sites, total_qubits, init_wall_idx):
    """
    Tier 1: Fast light-cone detection using Pauli Propagator on the FULL system.

    The Pauli Propagator is an approximate simulation method that efficiently
    tracks how Pauli operators spread through a circuit. It can handle hundreds
    of qubits in seconds, making it ideal for scanning large systems.

    We measure per-site ⟨Z_i⟩ and check where it deviates from the initial
    domain-wall configuration. Sites with |⟨Z⟩ - Z_initial| > threshold are
    marked as active.

    Returns (active_start, active_end) site indices.
    """
    print(f"\n  Tier 1: Scout — Pauli Propagator on {total_qubits} qubits")

    scout_start_time = time.time()

    # Use half the Trotter steps for speed (still captures the light cone)
    scout_steps = max(4, N_STEPS // 2)
    scout_dt = T_EVOLUTION / scout_steps

    model = FermiHubbardModel(n_sites, t=T_HOP, u=U_INT)
    scout_circuit = model.build_circuit(
        steps=scout_steps, dt=scout_dt, init_wall_idx=init_wall_idx
    )

    # Build per-site Z observables
    obs_list = []
    for i in range(total_qubits):
        pauli = ['I'] * total_qubits
        pauli[i] = 'Z'
        obs_list.append("".join(pauli))

    result = scout_circuit.estimate(
        observables=obs_list,
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.PauliPropagator,
    )
    z_vals = result['expectation_values']
    scout_elapsed = time.time() - scout_start_time

    # Detect active sites: where ⟨Z⟩ differs from initial state
    THRESHOLD = 0.001
    active_start = init_wall_idx
    active_end = init_wall_idx

    for i in range(n_sites):
        initial_z = -1.0 if i < init_wall_idx else 1.0
        # Check both spin sectors
        if (abs(z_vals[i] - initial_z) > THRESHOLD or
                abs(z_vals[n_sites + i] - initial_z) > THRESHOLD):
            active_start = min(active_start, i)
            active_end = max(active_end, i + 1)

    # Small, honest safety margin
    active_start = max(0, active_start - SAFETY_MARGIN)
    active_end = min(n_sites, active_end + SAFETY_MARGIN)
    n_active = active_end - active_start

    print(f"    Active region: [{active_start}, {active_end}) "
          f"→ {n_active} sites ({2 * n_active} qubits)")
    print(f"    Completed in {scout_elapsed:.2f}s")

    return active_start, active_end, scout_elapsed


# =============================================================================
# TIER 2: SNIPER (MPS CPU)
# =============================================================================

def run_sniper_cpu(n_sites, start, end, init_wall_idx, chi=CHI_CPU):
    """
    Tier 2: MPS simulation on CPU with moderate bond dimension.

    MPS (Matrix Product State) is a tensor network method that represents
    quantum states as a chain of tensors. The bond dimension χ controls
    the maximum entanglement that can be captured:
      - χ=1:   product state (no entanglement)
      - χ=64:  good for moderate entanglement (quick preview)
      - χ=256: captures more entanglement (publication quality)

    For a 1D system after a local quench, entanglement grows linearly in time
    but is bounded by the area law in the bulk, making MPS an ideal method.
    """
    n_active = end - start
    n_qubits = 2 * n_active
    print(f"\n  Tier 2: Sniper — MPS CPU on {n_qubits} qubits (χ={chi})")

    model = FermiHubbardModel(n_active, t=T_HOP, u=U_INT)
    circuit = model.build_circuit(
        steps=N_STEPS, dt=T_EVOLUTION / N_STEPS,
        init_wall_idx=init_wall_idx, active_sites_range=(start, end)
    )

    obs_list = _build_z_observables(n_qubits)

    start_time = time.time()
    result = circuit.estimate(
        observables=obs_list,
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.MatrixProductState,
        max_bond_dimension=chi
    )
    elapsed = time.time() - start_time

    print(f"    Completed in {elapsed:.2f}s")
    return result['expectation_values'], elapsed


# =============================================================================
# TIER 3: PRECISION (MPS GPU)
# =============================================================================

def run_precision_gpu(n_sites, start, end, init_wall_idx, run_gpu, chi=CHI_GPU):
    """
    Tier 3: MPS simulation on GPU with high bond dimension.

    GPU acceleration is critical for large χ because the dominant cost
    in MPS simulation is tensor contraction (matrix multiplications of
    size χ × χ), which GPUs handle 10-100× faster than CPUs.

    This tier provides converged results for publication/validation.
    Requires a CUDA-capable GPU.
    """
    n_active = end - start
    n_qubits = 2 * n_active
    print(f"\n  Tier 3: Precision — MPS GPU on {n_qubits} qubits (χ={chi})")

    model = FermiHubbardModel(n_active, t=T_HOP, u=U_INT)
    circuit = model.build_circuit(
        steps=N_STEPS, dt=T_EVOLUTION / N_STEPS,
        init_wall_idx=init_wall_idx, active_sites_range=(start, end)
    )

    obs_list = _build_z_observables(n_qubits)

    start_time = time.time()
    match run_gpu:
        case True:
            result = circuit.estimate(
                observables=obs_list,
                simulator_type=maestro.SimulatorType.Gpu,
                simulation_type=maestro.SimulationType.MatrixProductState,
                max_bond_dimension=chi
            )
        case False:
            result = circuit.estimate(
                observables=obs_list,
                simulator_type=maestro.SimulatorType.QCSim,
                simulation_type=maestro.SimulationType.MatrixProductState,
                max_bond_dimension=chi
            )
    elapsed = time.time() - start_time

    print(f"    Completed in {elapsed:.2f}s")
    return result['expectation_values'], elapsed


# =============================================================================
# HELPERS
# =============================================================================

def _build_z_observables(n_qubits):
    """Build per-qubit Z observables: ['ZIII..', 'IZII..', 'IIZI..', ...]"""
    obs = []
    for i in range(n_qubits):
        pauli = ['I'] * n_qubits
        pauli[i] = 'Z'
        obs.append("".join(pauli))
    return obs


def _z_to_density(exp_vals, n_active_sites):
    """
    Convert per-qubit ⟨Z⟩ values to particle density per site.

    n_σ = (1 - ⟨Z_σ⟩) / 2       (per spin sector)
    n_total = n_↑ + n_↓           (total density, range [0, 2])
    """
    densities = []
    for i in range(n_active_sites):
        n_up = (1.0 - exp_vals[i]) / 2.0
        n_down = (1.0 - exp_vals[i + n_active_sites]) / 2.0
        densities.append(n_up + n_down)
    return densities


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_pipeline(total_qubits, run_gpu=False):
    """
    Run the full 3-tier adaptive simulation pipeline for a given system size.

    Returns a dict with all timing and physics data.
    """
    n_sites = total_qubits // 2
    init_wall_idx = n_sites // 2

    print(f"\n{'='*65}")
    print(f"  {total_qubits}-QUBIT FERMI-HUBBARD BENCHMARK")
    print(f"  {n_sites} sites, domain wall at site {init_wall_idx}")
    print(f"  t={T_HOP}, U={U_INT} (U/t={U_INT/T_HOP:.1f}), "
          f"T={T_EVOLUTION}, {N_STEPS} steps")
    print(f"{'='*65}")

    # Tier 1: Scout
    start, end, scout_time = run_scout(n_sites, total_qubits, init_wall_idx)
    n_active = end - start

    # Tier 2: MPS CPU
    cpu_vals, cpu_time = run_sniper_cpu(n_sites, start, end, init_wall_idx)
    cpu_density = _z_to_density(cpu_vals, n_active)

    # Tier 3: MPS GPU (optional)
    gpu_density = None
    gpu_time = None
    # if run_gpu:
    try:
        gpu_vals, gpu_time = run_precision_gpu(n_sites, start, end, init_wall_idx, run_gpu=run_gpu)
        gpu_density = _z_to_density(gpu_vals, n_active)
    except Exception as e:
        print(f"    GPU tier failed: {e}")
        print(f"    (Run with a CUDA GPU or omit --gpu flag)")

    return {
        'total_qubits': total_qubits,
        'n_sites': n_sites,
        'init_wall_idx': init_wall_idx,
        'active_start': start,
        'active_end': end,
        'active_qubits': 2 * n_active,
        'scout_time': scout_time,
        'cpu_time': cpu_time,
        'cpu_density': cpu_density,
        'gpu_time': gpu_time,
        'gpu_density': gpu_density,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_density_profile(data):
    """
    Plot particle density across the lattice.

    Shows the active simulation region vs frozen boundaries, and if GPU
    results are available, overlays both for a convergence comparison.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    n_sites = data['n_sites']
    start = data['active_start']
    end = data['active_end']
    init_wall = data['init_wall_idx']
    global_x = list(range(start, end))

    # CPU result
    ax.bar(global_x, data['cpu_density'], color='#2F847C', width=1.0, alpha=0.7,
           label=f'MPS CPU (χ={CHI_CPU}, {data["cpu_time"]:.1f}s)')

    # GPU result (overlay if available)
    if data['gpu_density'] is not None:
        ax.step(
            [x + 0.5 for x in global_x], data['gpu_density'],
            color='#E74C3C', linewidth=2, where='mid',
            label=f'MPS GPU (χ={CHI_GPU}, {data["gpu_time"]:.1f}s)'
        )

    # Frozen regions
    if start > 0:
        ax.bar(range(0, start), [2.0] * start, color='#B0B0B0', alpha=0.2,
               width=1.0, label='Frozen (filled)')
    if end < n_sites:
        ax.bar(range(end, n_sites), [0.0] * (n_sites - end), color='#D0D0D0',
               alpha=0.2, width=1.0, label='Frozen (empty)')

    # Domain wall
    ax.axvline(x=init_wall - 0.5, color='red', linestyle='--', linewidth=2,
               label='Initial domain wall')

    ax.set_xlabel('Lattice Site Index', fontsize=12)
    ax.set_ylabel('Total Density ⟨n↑⟩ + ⟨n↓⟩', fontsize=12)

    gpu_label = ""
    if data['gpu_density'] is not None:
        gpu_label = f" → GPU χ={CHI_GPU} ({data['gpu_time']:.1f}s)"

    ax.set_title(
        f'{data["total_qubits"]}-Qubit Fermi-Hubbard: '
        f'Scout ({data["scout_time"]:.1f}s) → '
        f'CPU χ={CHI_CPU} ({data["cpu_time"]:.1f}s)'
        f'{gpu_label}\n'
        f'Active subspace: {data["active_qubits"]} qubits out of {data["total_qubits"]} '
        f'({data["total_qubits"] // data["active_qubits"] if data["active_qubits"] > 0 else "∞"}× reduction) | '
        f'U/t={U_INT/T_HOP:.1f}, T={T_EVOLUTION}',
        fontsize=10
    )
    ax.set_ylim(-0.1, 2.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('adaptive_hubbard_density.png', dpi=150)
    print("\nSaved adaptive_hubbard_density.png")


def plot_scaling_sweep(results):
    """
    Plot wall-clock time vs total system size.

    Key message: the MPS time is flat (light cone is fixed), while only
    the scout time grows with system size.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sizes = [r['total_qubits'] for r in results]
    scout_times = [r['scout_time'] for r in results]
    cpu_times = [r['cpu_time'] for r in results]
    total_times = [s + c for s, c in zip(scout_times, cpu_times)]
    active_qubits = [r['active_qubits'] for r in results]

    # ---- Left: Time vs System Size ----
    ax1.plot(sizes, scout_times, 'o-', color='#3498DB', linewidth=2,
             markersize=8, label='Tier 1: Scout (PP)')
    ax1.plot(sizes, cpu_times, 's-', color='#E74C3C', linewidth=2,
             markersize=8, label=f'Tier 2: MPS CPU (χ={CHI_CPU})')
    ax1.plot(sizes, total_times, 'D-', color='#2C3E50', linewidth=2,
             markersize=8, label='Total')

    ax1.set_xlabel('Total System Size (qubits)', fontsize=12)
    ax1.set_ylabel('Wall-Clock Time (s)', fontsize=12)
    ax1.set_title('Scaling: Time vs System Size', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Annotate key insight
    if len(cpu_times) >= 2:
        ax1.annotate(
            'MPS time flat →\nLight cone fixed!',
            xy=(sizes[-1], cpu_times[-1]),
            xytext=(sizes[-1] * 0.55, max(cpu_times) * 1.3),
            fontsize=10, color='#E74C3C',
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5),
        )

    # ---- Right: Active Qubits vs System Size ----
    bars = ax2.bar(range(len(sizes)), active_qubits, color='#2F847C', alpha=0.8)
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_xlabel('Total System Size (qubits)', fontsize=12)
    ax2.set_ylabel('Active Subspace (qubits)', fontsize=12)
    ax2.set_title('Active Qubits Stay Constant', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)

    for i, (total, active) in enumerate(zip(sizes, active_qubits)):
        if active > 0:
            ratio = total / active
            ax2.text(i, active + 1, f'{ratio:.0f}× reduction',
                     ha='center', fontsize=9, fontweight='bold', color='#2C3E50')

    fig.tight_layout()
    fig.savefig('adaptive_hubbard_scaling.png', dpi=150)
    print("Saved adaptive_hubbard_scaling.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Adaptive Fermi-Hubbard Benchmark")
    print(f"GPU mode: {'ENABLED' if GPU_ENABLED else 'DISABLED (use --gpu to enable)'}")

    # ---- 1. Density Profile: Full pipeline on the largest system ----
    largest = max(SYSTEM_SIZES)
    density_data = run_pipeline(largest, run_gpu=GPU_ENABLED)
    plot_density_profile(density_data)

    # ---- 2. Scaling Sweep (CPU tiers only for speed) ----
    print(f"\n\n{'#'*65}")
    print(f"  SCALING SWEEP: {SYSTEM_SIZES}")
    print(f"{'#'*65}")

    sweep_results = []
    for total_qubits in SYSTEM_SIZES:
        data = run_pipeline(total_qubits, run_gpu=False)
        sweep_results.append(data)

    plot_scaling_sweep(sweep_results)

    # ---- 3. Summary Table ----
    print(f"\n\n{'='*75}")
    print(f"  BENCHMARK SUMMARY")
    print(f"  Physics: 1D Fermi-Hubbard, t={T_HOP}, U={U_INT} (U/t={U_INT/T_HOP:.1f}), T={T_EVOLUTION}")
    print(f"  Method:  Trotter ({N_STEPS} steps, dt={T_EVOLUTION/N_STEPS:.3f})")
    print(f"  Tiers:   PP (scout) → MPS CPU (χ={CHI_CPU}) → MPS GPU (χ={CHI_GPU})")
    print(f"{'='*75}")
    print(f"  {'System':>8s}  {'Active':>8s}  {'Ratio':>6s}  "
          f"{'Scout':>8s}  {'MPS CPU':>8s}  {'Total':>8s}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")
    for r in sweep_results:
        ratio = r['total_qubits'] / r['active_qubits'] if r['active_qubits'] > 0 else 0
        total = r['scout_time'] + r['cpu_time']
        print(f"  {r['total_qubits']:>6d}Q  {r['active_qubits']:>6d}Q  "
              f"{ratio:>5.0f}×  {r['scout_time']:>7.2f}s  "
              f"{r['cpu_time']:>7.2f}s  {total:>7.2f}s")
    print(f"{'='*75}")

    # GPU summary if available
    if density_data['gpu_time'] is not None:
        print(f"\n  GPU Tier (largest system, {largest}Q):")
        print(f"    χ={CHI_GPU}, Time: {density_data['gpu_time']:.2f}s")