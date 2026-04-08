# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
Fermi-Hubbard Time Evolution with QuEPP Error Mitigation
=========================================================

Demonstrates Divi's QuEPP error mitigation on Trotterized Fermi-Hubbard
time evolution, run on a real QPU via QoroService.

Supports 1D chain, 2D square, and 2D honeycomb lattice topologies.

Pipeline:
    1. Classical reference — ``TimeEvolution`` with ``MaestroSimulator``
       provides exact expectation values.

    2. QPU (raw) — ``CircuitPipeline`` with ``QoroService`` sends the
       Trotter circuits to the target backend.

    3. QPU + QuEPP — Same backend, but pipeline includes ``QEMStage``
       with ``QuEPP`` for error mitigation.

Usage:
    python fermi_hubbard_quepp.py                                                 # 1D chain, default
    python fermi_hubbard_quepp.py --topology chain --n-sites 4 --obs z            # Chain, per-site Z
    python fermi_hubbard_quepp.py --topology square --n-sites-x 3 --n-sites-y 3   # 2D square lattice
    python fermi_hubbard_quepp.py --topology hex --n-sites-x 3 --n-sites-y 2      # 2D honeycomb
    python fermi_hubbard_quepp.py --obs density                                   # Per-site occupation
    python fermi_hubbard_quepp.py --backend sim                                   # Qoro simulator cluster
    python fermi_hubbard_quepp.py --dry-run                                       # Show circuit fan-out
"""

import argparse
import time

import numpy as np
import pennylane as qml
from rich.console import Console
from rich.table import Table

from divi.backends import JobConfig, MaestroSimulator, QoroService
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
from model import (
    FermiHubbardChainModel, FermiHubbardSquareModel, FermiHubbardHexModel,
    FermiHubbardChainProxyModel, FermiHubbardSquareProxyModel, FermiHubbardHexProxyModel,
    TFIMModel,
)

console = Console()


# =============================================================================
# MODEL FACTORY
# =============================================================================


def create_model(args):
    """Instantiate a Fermi-Hubbard model from CLI arguments."""
    match args.topology:
        case "chain":
            return FermiHubbardChainModel(n_sites=args.n_sites, t=args.hopping, u=args.u_int)
        case "square":
            return FermiHubbardSquareModel(args.n_sites_x, args.n_sites_y, t=args.hopping, u=args.u_int)
        case "hex":
            return FermiHubbardHexModel(args.n_sites_x, args.n_sites_y, t=args.hopping, u=args.u_int)
        case "chain-proxy":
            return FermiHubbardChainProxyModel(n_sites=args.n_sites, t=args.hopping, u=args.u_int)
        case "square-proxy":
            return FermiHubbardSquareProxyModel(args.n_sites_x, args.n_sites_y, t=args.hopping, u=args.u_int)
        case "hex-proxy":
            return FermiHubbardHexProxyModel(args.n_sites_x, args.n_sites_y, t=args.hopping, u=args.u_int)
        case "tfim":
            return TFIMModel(n_sites=args.n_sites, j=args.hopping, h=args.u_int)


# =============================================================================
# BACKEND FACTORY
# =============================================================================


def make_backend(backend_mode: str, shots: int):
    """Create a backend for the chosen target."""
    match backend_mode:
        case "hardware":
            return QoroService(job_config=JobConfig(qpu_system="superconducting_qpus", shots=shots, use_circuit_packing=False))
        case "sim":
            return QoroService(job_config=JobConfig(simulator_cluster="qoro_maestro", shots=shots))
        case "maestro":
            return MaestroSimulator(shots=shots)


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Fermi-Hubbard Time Evolution + QuEPP Error Mitigation"
    )
    parser.add_argument(
        "--topology", choices=[
            "chain", "square", "hex",
            "chain-proxy", "square-proxy", "hex-proxy",
            "tfim",
        ], default="chain",
        help="Lattice topology (default: chain)"
    )
    parser.add_argument("--n-sites", type=int, default=8, help="Number of sites for 1D chain")
    parser.add_argument("--n-sites-x", type=int, default=3, help="Grid width for 2D topologies")
    parser.add_argument("--n-sites-y", type=int, default=3, help="Grid height for 2D topologies")
    parser.add_argument("--n-steps", type=int, default=1, help="Number of Trotter steps")
    parser.add_argument("--dt", type=float, default=0.1, help="Trotter step size")
    parser.add_argument("--hopping", type=float, default=1.0, help="Hopping parameter t")
    parser.add_argument("--u-int", type=float, default=4.0, help="On-site interaction U")
    parser.add_argument("--shots", type=int, default=20_000, help="Shots per circuit")
    parser.add_argument(
        "--obs", choices=["z", "zz", "density"], default="zz",
        help="Observable type: 'z' per-qubit, 'zz' nearest-neighbor, 'density' per-site occupation"
    )
    parser.add_argument(
        "--backend", choices=["hardware", "sim", "maestro"], default="sim",
        help="'hardware' for QPU, 'sim' for Qoro simulator cluster, 'maestro' for local MaestroSimulator"
    )
    parser.add_argument(
        "--no-quepp", action="store_true",
        help="Skip QuEPP phase (run only classical reference + raw execution)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show circuit fan-out analysis without executing"
    )
    parser.add_argument(
        "--no-stim-normalization", action="store_true",
        help="Skip stim gate normalization in QuEPP (disable decomposition of non-stim-compatible gates)"
    )
    args = parser.parse_args()

    n_steps = args.n_steps
    dt = args.dt
    total_time = n_steps * dt
    backend_labels = {"hardware": "QPU", "sim": "Qoro Sim Cluster", "maestro": "MaestroSimulator (local)"}
    backend_label = backend_labels[args.backend]
    obs_labels = {"z": "⟨Zᵢ⟩", "zz": "⟨ZᵢZⱼ⟩", "density": "⟨nᵢ⟩"}
    obs_label = obs_labels[args.obs]

    # Build model, Hamiltonian, initial state, and observables
    model = create_model(args)
    hamiltonian = model.hamiltonian()
    initial_state = CustomPerQubitState(model.domain_wall_bitstring())
    observables = model.build_observables(args.obs)
    n_obs = len(observables)

    console.print(f"\n[bold cyan]{'=' * 65}[/bold cyan]")
    console.print("[bold cyan]  Fermi-Hubbard Time Evolution + QuEPP Error Mitigation[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 65}[/bold cyan]")
    console.print(
        f"  Model: {model.description()}\n"
        f"  Hamiltonian: H = -t hopping + U interaction  (t={args.hopping}, U={args.u_int})\n"
        f"  Evolution: T={total_time:.2f} ({n_steps} Trotter steps, dt={dt})\n"
        f"  Initial state: domain wall\n"
        f"  Observables: {n_obs} x {obs_label}\n"
        f"  Backend: {backend_label}"
    )

    # Common TimeEvolution kwargs
    common_kwargs = dict(
        hamiltonian=hamiltonian,
        time=total_time,
        n_steps=n_steps,
        initial_state=initial_state,
    )

    # -----------------------------------------------------------------
    # Dry run: show fan-out analysis and exit
    # -----------------------------------------------------------------
    if args.dry_run:
        console.print(f"\n[bold]Dry run — circuit fan-out for one observable:[/bold]")
        te_dry = TimeEvolution(
            **common_kwargs,
            observable=observables[0][1],
            backend=MaestroSimulator(shots=args.shots),
            qem_protocol=QuEPP(
                sampling="exhaustive", truncation_order=3, n_twirls=10,
                normalize_for_stim=not args.no_stim_normalization,
            ),
        )
        te_dry.dry_run()
        console.print(
            f"\n[dim]× {n_obs} observables = "
            f"{n_obs} × (circuits per observable) total QPU circuits[/dim]"
        )
        return

    # -----------------------------------------------------------------
    # Build MetaCircuits for all observables (shared across phases)
    # -----------------------------------------------------------------
    def build_metas() -> list[MetaCircuit]:
        """Build one MetaCircuit per observable using TimeEvolution internals."""
        metas = []
        for label, obs in observables:
            te = TimeEvolution(
                **common_kwargs,
                observable=obs,
                backend=MaestroSimulator(shots=args.shots),  # dummy
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

    # Pipeline definitions
    base_pipeline = CircuitPipeline(
        stages=[CircuitSpecStage(), MeasurementStage()]
    )
    quepp_pipeline = CircuitPipeline(
        stages=[
            CircuitSpecStage(),
            MeasurementStage(),
            QEMStage(
                protocol=QuEPP(
                    sampling="exhaustive", truncation_order=3, n_twirls=10,
                    normalize_for_stim=not args.no_stim_normalization,
                )
            ),
            PauliTwirlStage(n_twirls=10),
        ]
    )

    def run_batch(pipeline, backend, label):
        """Run all observables as a single batched pipeline call."""
        t0 = time.time()
        console.print(f"\n[bold]{label}...[/bold]")
        result = pipeline.run(metas, PipelineEnv(backend=backend))
        elapsed = time.time() - t0
        # CircuitSpecStage reduces a sequence to {(): [v0, v1, ...]}
        raw = next(iter(result.values()))
        vals = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        for i, (lbl, _) in enumerate(observables):
            console.print(f"  {lbl:>8s}: {float(vals[i]):+.4f}")
        console.print(f"  [dim]Completed in {elapsed:.1f}s[/dim]")
        return [float(v) for v in vals], elapsed

    # -----------------------------------------------------------------
    # Phase 1: Classical reference (MaestroSimulator) — batched
    # -----------------------------------------------------------------
    exact_vals, exact_time = run_batch(
        base_pipeline,
        MaestroSimulator(shots=args.shots),
        "Phase 1: Classical reference (MaestroSimulator)",
    )

    # -----------------------------------------------------------------
    # Phase 2: QPU / Sim (raw, no error mitigation) — batched
    # -----------------------------------------------------------------
    noisy_vals, noisy_time = run_batch(
        base_pipeline,
        make_backend(args.backend, args.shots),
        f"Phase 2: {backend_label} (raw)",
    )

    # -----------------------------------------------------------------
    # Phase 3: QPU / Sim + QuEPP error mitigation — batched
    # -----------------------------------------------------------------
    if not args.no_quepp:
        quepp_vals, quepp_time = run_batch(
            quepp_pipeline,
            make_backend(args.backend, args.shots),
            f"Phase 3: {backend_label} + QuEPP",
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    exact_arr = np.array(exact_vals)
    noisy_arr = np.array(noisy_vals)

    noisy_mse = float(np.mean((noisy_arr - exact_arr) ** 2))
    noisy_mae = float(np.mean(np.abs(noisy_arr - exact_arr)))

    # Per-observable profile
    console.print(f"\n[bold]Observable Profile {obs_label}:[/bold]")
    profile = Table(show_header=True)
    profile.add_column("Observable", justify="center")
    profile.add_column("Exact", justify="right")
    profile.add_column("Raw", justify="right")
    profile.add_column("Raw Err", justify="right")
    if not args.no_quepp:
        profile.add_column("QuEPP", justify="right")
        profile.add_column("QuEPP Err", justify="right")

    for i, (lbl, _) in enumerate(observables):
        n_err = abs(noisy_vals[i] - exact_vals[i])
        row = [lbl, f"{exact_vals[i]:+.4f}", f"{noisy_vals[i]:+.4f}", f"{n_err:.4f}"]
        if not args.no_quepp:
            q_err = abs(quepp_vals[i] - exact_vals[i])
            style = "green" if q_err < n_err else ("yellow" if q_err == n_err else "red")
            row.extend([f"{quepp_vals[i]:+.4f}", f"[{style}]{q_err:.4f}[/{style}]"])
        profile.add_row(*row)
    console.print(profile)

    # Aggregate metrics
    summary = Table(title=f"{model.description()} {obs_label} — {backend_label}")
    summary.add_column("Metric", style="bold")
    summary.add_column("Raw", justify="right")

    if not args.no_quepp:
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
        f"\n[dim]Observables: {n_obs} | "
        f"Shots: {args.shots:,} | "
        f"Timing: exact {exact_time:.0f}s, raw {noisy_time:.0f}s"
    )
    if not args.no_quepp:
        timing += f", QuEPP {quepp_time:.0f}s"
    timing += "[/dim]"
    console.print(timing)


if __name__ == "__main__":
    main()
