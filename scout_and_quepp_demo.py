"""
2D Fermi-Hubbard: Scout + Divi Pipeline
========================================

Two-phase adaptive simulation for 2D Fermi-Hubbard lattices:

  Phase 1 -- Scout (Pauli Propagator, CPU):
      Runs a Clifford-only proxy on the FULL 2D lattice to detect which
      sites have non-trivial dynamics via the Lieb-Robinson light cone.
      Returns a bounding box of active sites.

  Phase 2 -- Time Evolution (Divi pipeline, user-chosen backend):
      Builds a sub-model on the active bounding box and runs Trotterized
      time evolution through the Divi CircuitPipeline, optionally with
      QuEPP error mitigation.

Supports square and honeycomb lattice topologies.

Usage:
    python scout_and_quepp_demo.py --topology square --n-sites-x 8 --n-sites-y 8
    python scout_and_quepp_demo.py --topology hex --n-sites-x 6 --n-sites-y 4
    python scout_and_quepp_demo.py --topology square --n-sites-x 10 --n-sites-y 10 --backend sim
    python scout_and_quepp_demo.py --backend maestro --no-quepp --obs density
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import maestro
import numpy as np
import pennylane as qml
from rich.console import Console
from rich.table import Table

from divi.backends import JobConfig, MaestroSimulator, QoroService, QiskitSimulator
from divi.circuits import MetaCircuit
from divi.circuits.quepp import QuEPP
from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import (
    CircuitSpecStage,
    MeasurementStage,
    PauliTwirlStage,
    QEMStage,
)
from divi.qprog import TimeEvolution
from divi.qprog.algorithms import CustomPerQubitState
from model import FermiHubbardSquareModel, FermiHubbardHexModel

console = Console()

SAFETY_MARGIN = 0
SCOUT_THRESHOLD = 0.001


# =============================================================================
# SCOUT (Phase 1)
# =============================================================================


def run_scout_2d(model, init_wall_idx, total_time):
    """Run the Clifford Pauli-Propagator scout on the full 2D lattice.

    Returns:
        active_sites: set of site indices where dynamics were detected
        bbox: (x_min, y_min, x_max, y_max) bounding box with safety margin
        elapsed: wall-clock time
    """
    n_sites = model.n_sites
    n_qubits = model.n_qubits
    nx, ny = model.n_sites_x, model.n_sites_y

    console.print(f"\n[bold cyan]Phase 1: Scout[/bold cyan] -- "
                  f"Pauli Propagator on {n_qubits} qubits ({nx}x{ny} grid)")

    # Calibrate steps to the Lieb-Robinson light cone for the actual
    # evolution time.  Each step expands the BFS window by 1, so
    # scout_steps ≈ light_cone_radius to avoid covering the whole grid.
    light_cone_radius = 2.0 * model.t * total_time
    scout_steps = max(2, int(light_cone_radius / 2) + 1)

    scout_circuit = model.build_clifford_scout_circuit(
        steps=scout_steps, init_wall_idx=init_wall_idx
    )

    # Per-qubit Z observables as Pauli strings
    obs_list = []
    for i in range(n_qubits):
        pauli = ['I'] * n_qubits
        pauli[i] = 'Z'
        obs_list.append("".join(pauli))

    t0 = time.time()
    result = scout_circuit.estimate(
        observables=obs_list,
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.PauliPropagator,
    )
    elapsed = time.time() - t0

    z_vals = result['expectation_values']

    # Detect active sites
    active_sites = set()
    for i in range(n_sites):
        initial_z = -1.0 if i < init_wall_idx else 1.0
        if (abs(z_vals[i] - initial_z) > SCOUT_THRESHOLD or
                abs(z_vals[n_sites + i] - initial_z) > SCOUT_THRESHOLD):
            active_sites.add(i)

    if not active_sites:
        console.print("  [yellow]No active sites detected -- using sites around wall[/yellow]")
        active_sites = {max(0, init_wall_idx - 1), min(n_sites - 1, init_wall_idx)}

    # Convert to (x, y) coordinates and compute bounding box
    def site_to_xy(s):
        return s % nx, s // nx

    coords = [site_to_xy(s) for s in active_sites]
    x_min = max(0, min(c[0] for c in coords) - SAFETY_MARGIN)
    y_min = max(0, min(c[1] for c in coords) - SAFETY_MARGIN)
    x_max = min(nx, max(c[0] for c in coords) + 1 + SAFETY_MARGIN)
    y_max = min(ny, max(c[1] for c in coords) + 1 + SAFETY_MARGIN)

    sub_nx = x_max - x_min
    sub_ny = y_max - y_min

    console.print(f"  Active sites: {len(active_sites)} / {n_sites}")
    console.print(f"  Bounding box: x=[{x_min},{x_max}) y=[{y_min},{y_max}) "
                  f"-> {sub_nx}x{sub_ny} = {sub_nx * sub_ny} sites "
                  f"({2 * sub_nx * sub_ny} qubits)")

    # ASCII grid diagram
    # Legend: # = active, . = frozen (filled half), o = frozen (empty half)
    #         [ ] = bounding box border, --- = domain wall
    console.print(f"\n  [bold]Lattice ({nx}x{ny}):[/bold]  "
                  "[dim]# active  . frozen-filled  o frozen-empty  "
                  "[ ] bbox  --- domain wall[/dim]")

    # The domain wall sits between the last filled row and first empty row.
    # wall_row = first row that contains any site >= init_wall_idx.
    wall_row = init_wall_idx // nx

    for y in range(ny):
        # Domain wall separator line between rows
        if y == wall_row:
            sep = "  "
            for x in range(nx):
                sep += "[bold red]---[/bold red]"
            console.print(sep)

        row = "  "
        for x in range(nx):
            s = y * nx + x
            in_bbox = x_min <= x < x_max and y_min <= y < y_max
            if s in active_sites:
                ch = "[bold green]#[/bold green]"
            elif s < init_wall_idx:
                ch = "[dim].[/dim]"
            else:
                ch = "[dim]o[/dim]"
            if in_bbox and (x == x_min or x == x_max - 1
                            or y == y_min or y == y_max - 1):
                ch = f"[cyan]\\[[/cyan]{ch}[cyan]][/cyan]"
            else:
                ch = f" {ch} "
            row += ch
        console.print(row)

    console.print(f"\n  [dim]Scout completed in {elapsed:.2f}s[/dim]")

    return active_sites, (x_min, y_min, x_max, y_max), elapsed


# =============================================================================
# SUB-MODEL CONSTRUCTION
# =============================================================================


def build_sub_model(topology, bbox, full_model):
    """Build a model on the active bounding-box sub-grid."""
    x_min, y_min, x_max, y_max = bbox
    sub_nx = x_max - x_min
    sub_ny = y_max - y_min

    if topology == "square":
        return FermiHubbardSquareModel(sub_nx, sub_ny, t=full_model.t, u=full_model.u)
    else:
        return FermiHubbardHexModel(sub_nx, sub_ny, t=full_model.t, u=full_model.u)


def sub_grid_domain_wall_bitstring(bbox, full_nx, full_n_sites):
    """Compute domain-wall bitstring for the sub-grid.

    A sub-grid site (sx, sy) maps to full-grid linear index
    (y_min + sy) * full_nx + (x_min + sx). It is filled ('1') if
    that index < full_n_sites // 2.
    """
    x_min, y_min, x_max, y_max = bbox
    sub_nx = x_max - x_min
    sub_ny = y_max - y_min
    init_wall = full_n_sites // 2

    sector = []
    for sy in range(sub_ny):
        for sx in range(sub_nx):
            full_idx = (y_min + sy) * full_nx + (x_min + sx)
            sector.append("1" if full_idx < init_wall else "0")
    sector_str = "".join(sector)
    return sector_str + sector_str  # spin-up + spin-down


# =============================================================================
# BACKEND FACTORY
# =============================================================================


def make_backend(backend_mode, shots):
    match backend_mode:
        case "hardware":
            return QoroService(job_config=JobConfig(
                qpu_system="superconducting_qpus", shots=shots, use_circuit_packing=False
            ))
        case "sim":
            return QoroService(job_config=JobConfig(
                simulator_cluster="qoro_maestro", shots=shots
            ))
        case "maestro":
            return MaestroSimulator(shots=shots)

        case "qiskit":
            return QiskitSimulator(shots=shots, force_sampling=True)


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_results(observables, exact_vals, noisy_vals, quepp_vals,
                 obs_type, model_desc, backend_label):
    """Plot exact vs raw vs QuEPP expectation values with error annotation."""
    labels = [lbl for lbl, _ in observables]
    n = len(labels)
    x = np.arange(n)

    exact_arr = np.array(exact_vals)
    noisy_arr = np.array(noisy_vals)
    noisy_mae = float(np.mean(np.abs(noisy_arr - exact_arr)))

    has_quepp = quepp_vals is not None
    n_panels = 2 if has_quepp else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)

    # --- Panel 1: expectation values ---
    ax = axes[0, 0]
    ax.plot(x, exact_vals, 'o-', color='#2C3E50', markersize=5, linewidth=1.5,
            label='Exact (statevector)', zorder=3)
    ax.plot(x, noisy_vals, 's--', color='#E74C3C', markersize=4, linewidth=1.2,
            alpha=0.8, label=f'{backend_label} (raw)')
    if has_quepp:
        ax.plot(x, quepp_vals, 'D--', color='#27AE60', markersize=4, linewidth=1.2,
                alpha=0.8, label=f'{backend_label} + QuEPP')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel('Expectation value')
    ax.set_title(f'{model_desc}\nObservable: {obs_type}')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # --- Panel 2: per-observable error bars ---
    if has_quepp:
        quepp_arr = np.array(quepp_vals)
        quepp_mae = float(np.mean(np.abs(quepp_arr - exact_arr)))

        raw_err = np.abs(noisy_arr - exact_arr)
        quepp_err = np.abs(quepp_arr - exact_arr)

        ax2 = axes[0, 1]
        w = 0.35
        ax2.bar(x - w / 2, raw_err, width=w, color='#E74C3C', alpha=0.7, label='Raw error')
        ax2.bar(x + w / 2, quepp_err, width=w, color='#27AE60', alpha=0.7, label='QuEPP error')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=90, fontsize=7)
        ax2.set_ylabel('|error|')
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.3)

        mae_improvement = (1 - quepp_mae / noisy_mae) * 100 if noisy_mae > 1e-12 else 0
        color = '#27AE60' if mae_improvement > 0 else '#E74C3C'
        ax2.set_title(
            f'Per-observable error\n'
            f'{mae_improvement:+.0f}% MAE improvement from QuEPP',
            color=color, fontweight='bold',
        )

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__) or '.', 'scout_quepp_results.png')
    fig.savefig(out_path, dpi=150)
    console.print(f"\nSaved plot to {out_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="2D Fermi-Hubbard: Scout + Divi Pipeline"
    )
    parser.add_argument(
        "--topology", choices=["square", "hex"], default="square",
        help="Lattice topology (default: square)"
    )
    parser.add_argument("--n-sites-x", type=int, default=8, help="Grid width")
    parser.add_argument("--n-sites-y", type=int, default=8, help="Grid height")
    parser.add_argument("--n-steps", type=int, default=1, help="Trotter steps")
    parser.add_argument("--dt", type=float, default=0.1, help="Trotter step size")
    parser.add_argument("--hopping", type=float, default=1.0, help="Hopping t")
    parser.add_argument("--u-int", type=float, default=1.0, help="Interaction U")
    parser.add_argument("--shots", type=int, default=20_000, help="Shots per circuit")
    parser.add_argument(
        "--obs", choices=["z", "zz", "density"], default="density",
        help="Observable type"
    )
    parser.add_argument(
        "--backend", choices=["hardware", "sim", "maestro", "qiskit"], default="maestro",
        help="Backend for time evolution"
    )
    parser.add_argument("--no-quepp", action="store_true", help="Skip QuEPP phase")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run scout + show circuit fan-out analysis without executing"
    )
    args = parser.parse_args()

    n_steps = args.n_steps
    dt = args.dt
    total_time = n_steps * dt
    backend_labels = {
        "hardware": "QPU", "sim": "Qoro Sim Cluster",
        "maestro": "MaestroSimulator (local)",
    }
    backend_label = backend_labels[args.backend]

    # ---- Build full model ----
    if args.topology == "square":
        full_model = FermiHubbardSquareModel(
            args.n_sites_x, args.n_sites_y, t=args.hopping, u=args.u_int
        )
    else:
        full_model = FermiHubbardHexModel(
            args.n_sites_x, args.n_sites_y, t=args.hopping, u=args.u_int
        )

    init_wall_idx = full_model.n_sites // 2

    console.print(f"\n[bold cyan]{'=' * 65}[/bold cyan]")
    console.print("[bold cyan]  2D Fermi-Hubbard: Scout + Divi Pipeline[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 65}[/bold cyan]")
    console.print(
        f"  Model: {full_model.description()}\n"
        f"  Evolution: T={total_time:.2f} ({n_steps} steps, dt={dt})\n"
        f"  Backend: {backend_label}"
    )

    # =================================================================
    # Phase 1: Scout
    # =================================================================
    active_sites, bbox, scout_time = run_scout_2d(full_model, init_wall_idx, total_time)

    # =================================================================
    # Phase 2: Time evolution on active sub-lattice via Divi
    # =================================================================
    sub_model = build_sub_model(args.topology, bbox, full_model)
    bitstring = sub_grid_domain_wall_bitstring(
        bbox, full_model.n_sites_x, full_model.n_sites
    )

    hamiltonian = sub_model.hamiltonian()
    initial_state = CustomPerQubitState(bitstring)
    observables = sub_model.build_observables(args.obs)
    n_obs = len(observables)

    console.print(f"\n[bold cyan]Phase 2: Time Evolution[/bold cyan] -- "
                  f"{sub_model.description()}")
    console.print(f"  Observables: {n_obs} x {args.obs}")

    common_kwargs = dict(
        hamiltonian=hamiltonian,
        time=total_time,
        n_steps=n_steps,
        initial_state=initial_state,
    )

    # -----------------------------------------------------------------
    # Dry run: scout + circuit fan-out, no execution
    # -----------------------------------------------------------------
    if args.dry_run:
        console.print(f"\n[bold]Dry run -- Raw (no error mitigation):[/bold]")
        te_raw = TimeEvolution(
            **common_kwargs,
            observable=observables[0][1],
            backend=MaestroSimulator(shots=args.shots),
        )
        fan_raw = te_raw.dry_run()
        raw_per_obs = fan_raw if isinstance(fan_raw, int) else 1
        console.print(
            f"\n[bold]{raw_per_obs} circuits/observable x "
            f"{n_obs} observables = {raw_per_obs * n_obs} total circuits (raw)[/bold]"
        )

        console.print(f"\n[bold]Dry run -- With QuEPP error mitigation:[/bold]")
        te_quepp = TimeEvolution(
            **common_kwargs,
            observable=observables[0][1],
            backend=MaestroSimulator(shots=args.shots),
            qem_protocol=QuEPP(
                sampling="exhaustive", truncation_order=3, n_twirls=10
            ),
        )
        fan_quepp = te_quepp.dry_run()
        quepp_per_obs = fan_quepp if isinstance(fan_quepp, int) else 110
        console.print(
            f"\n[bold]{quepp_per_obs} circuits/observable x "
            f"{n_obs} observables = {quepp_per_obs * n_obs} total circuits (QuEPP)[/bold]"
        )
        return

    def build_metas():
        metas = []
        for label, obs in observables:
            te = TimeEvolution(
                **common_kwargs,
                observable=obs,
                backend=MaestroSimulator(shots=args.shots),
            )
            processed_ham = te.trotterization_strategy.process_hamiltonian(
                te._hamiltonian
            )
            ops = te._build_ops(processed_ham)
            ops = [qml.Identity(w) for w in te._circuit_wires] + ops
            tape = qml.tape.QuantumScript(
                ops=ops, measurements=[qml.expval(obs)]
            )
            metas.append(MetaCircuit(
                source_circuit=tape, symbols=np.array([], dtype=object)
            ))
        return metas

    metas = build_metas()

    base_pipeline = CircuitPipeline(
        stages=[CircuitSpecStage(), MeasurementStage()]
    )
    quepp_pipeline = CircuitPipeline(
        stages=[
            CircuitSpecStage(),
            MeasurementStage(),
            QEMStage(
                protocol=QuEPP(
                    sampling="exhaustive", truncation_order=3, n_twirls=10
                )
            ),
            PauliTwirlStage(n_twirls=10),
        ]
    )

    def run_batch(pipeline, backend, label):
        t0 = time.time()
        console.print(f"\n[bold]{label}...[/bold]")
        result = pipeline.run(metas, PipelineEnv(backend=backend))
        elapsed = time.time() - t0
        raw = next(iter(result.values()))
        vals = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        for i, (lbl, _) in enumerate(observables):
            console.print(f"  {lbl:>12s}: {float(vals[i]):+.4f}")
        console.print(f"  [dim]Completed in {elapsed:.1f}s[/dim]")
        return [float(v) for v in vals], elapsed

    # Classical reference
    exact_vals, exact_time = run_batch(
        base_pipeline,
        MaestroSimulator(shots=args.shots),
        "Classical reference (MaestroSimulator)",
    )

    # Backend (raw)
    noisy_vals, noisy_time = run_batch(
        base_pipeline,
        make_backend(args.backend, args.shots),
        f"{backend_label} (raw)",
    )

    # Backend + QuEPP
    quepp_vals, quepp_time = None, None
    if not args.no_quepp:
        quepp_vals, quepp_time = run_batch(
            quepp_pipeline,
            make_backend(args.backend, args.shots),
            f"{backend_label} + QuEPP",
        )

    # =================================================================
    # Summary
    # =================================================================
    exact_arr = np.array(exact_vals)
    noisy_arr = np.array(noisy_vals)
    noisy_mse = float(np.mean((noisy_arr - exact_arr) ** 2))
    noisy_mae = float(np.mean(np.abs(noisy_arr - exact_arr)))

    # Per-observable profile
    console.print(f"\n[bold]Observable Profile ({args.obs}):[/bold]")
    profile = Table(show_header=True)
    profile.add_column("Observable", justify="center")
    profile.add_column("Exact", justify="right")
    profile.add_column("Raw", justify="right")
    profile.add_column("Raw Err", justify="right")
    if quepp_vals is not None:
        profile.add_column("QuEPP", justify="right")
        profile.add_column("QuEPP Err", justify="right")

    for i, (lbl, _) in enumerate(observables):
        n_err = abs(noisy_vals[i] - exact_vals[i])
        row = [lbl, f"{exact_vals[i]:+.4f}", f"{noisy_vals[i]:+.4f}", f"{n_err:.4f}"]
        if quepp_vals is not None:
            q_err = abs(quepp_vals[i] - exact_vals[i])
            style = "green" if q_err < n_err else ("yellow" if q_err == n_err else "red")
            row.extend([f"{quepp_vals[i]:+.4f}", f"[{style}]{q_err:.4f}[/{style}]"])
        profile.add_row(*row)
    console.print(profile)

    # Aggregate metrics
    summary = Table(title=f"{sub_model.description()} -- {backend_label}")
    summary.add_column("Metric", style="bold")
    summary.add_column("Raw", justify="right")

    if quepp_vals is not None:
        quepp_arr = np.array(quepp_vals)
        quepp_mse = float(np.mean((quepp_arr - exact_arr) ** 2))
        quepp_mae = float(np.mean(np.abs(quepp_arr - exact_arr)))
        mse_pct = (1 - quepp_mse / noisy_mse) * 100 if noisy_mse > 1e-12 else 0
        mae_pct = (1 - quepp_mae / noisy_mae) * 100 if noisy_mae > 1e-12 else 0

        summary.add_column("QuEPP", justify="right")
        summary.add_column("Improvement", justify="right")

        mse_s = "green" if quepp_mse < noisy_mse else "red"
        mae_s = "green" if quepp_mae < noisy_mae else "red"
        summary.add_row(
            "MSE", f"{noisy_mse:.6f}",
            f"[{mse_s}]{quepp_mse:.6f}[/{mse_s}]",
            f"[{mse_s}]{mse_pct:.0f}%[/{mse_s}]",
        )
        summary.add_row(
            "MAE", f"{noisy_mae:.6f}",
            f"[{mae_s}]{quepp_mae:.6f}[/{mae_s}]",
            f"[{mae_s}]{mae_pct:.0f}%[/{mae_s}]",
        )
    else:
        summary.add_row("MSE", f"{noisy_mse:.6f}")
        summary.add_row("MAE", f"{noisy_mae:.6f}")
    console.print(summary)

    timing = (
        f"\n[dim]Scout: {scout_time:.1f}s | "
        f"Observables: {n_obs} | Shots: {args.shots:,} | "
        f"Timing: exact {exact_time:.0f}s, raw {noisy_time:.0f}s"
    )
    if quepp_time is not None:
        timing += f", QuEPP {quepp_time:.0f}s"
    timing += "[/dim]"
    console.print(timing)

    # =================================================================
    # Plot
    # =================================================================
    plot_results(
        observables, exact_vals, noisy_vals, quepp_vals,
        args.obs, sub_model.description(), backend_label,
    )


if __name__ == "__main__":
    main()
