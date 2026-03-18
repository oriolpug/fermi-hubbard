# GPU-Accelerated Quantum Simulation: Benchmarking Maestro, PennyLane, Qibo, and Qiskit on Highly-Entangled Circuits

## Introduction

Quantum computing is promising a revolution in fields from material simulation to drug discovery.
But, before we get there, classical simulation tools will remain vital. 

The kinds of problems where quantum computing will have the strongest advantage are on systems with high entanglement.
These also pose the biggest challenge for classical simulation methods, due to scaling constraints. 
Results today require a strong software-hardware integration stack, 
which unlocks the power of modern computing tools such as GPUs and orchestrates resources efficiently.

That is why at Qoro Quantum we have developed Maestro. 
Maestro is an automated simulation engine that allows seamless switching between backends,
allowing users without any background in GPU programming to run large-scale quantum workflows
like the ones real-world problems call for.

We have compared Maestro to alternative, open-sourced frameworks for quantum simulation. 
We have found none matches the combination of versatility and high performance Maestro brings to the table.

- Why classical simulation of highly-entangled circuits is hard
- The five backends compared: Maestro GPU, Maestro CPU, PennyLane, Qibo, Qiskit
- Summary of findings (1–2 sentences)

---

## The Benchmark: Fermi-Hubbard

The Fermi-Hubbard model describes electrons on a lattice. 
Each site can hold zero, one, or two electrons (spin-up and spin-down), and two competing processes determine the physics: 
electrons can hop to neighboring sites, and two electrons on the same site repel each other.
It's the simplest model that captures both kinetic and correlation effects simultaneously, but that does not make it trivial. 
It remains unsolved for two dimensions, and its wide applicability makes it a prime target for quantum computers.

A solution of the Fermi-Hubbard model would allow us to understand processes from high-temperature superconductivity to Mott transistors,
which could go as far as triggering a new industrial revolution.

A key parameter in quantum simulation is the number of qubits we simulate. 
In the Fermi-Hubbard model, each lattice site can hold an up-spin and a down-spin, 
so we'll have twice as many qubits as lattice sites. 
As we will see, qubit count plays a key role in determining simulation feasibility and performance:
due to algorithmic exponential scaling in memory, adding one qubit to the system doubles memory requirements.

We use Matrix Product State simulators to time-evolve the Trotterized Hamiltonian of the Fermi-Hubbard model, which, 
with enough layers, yields a deep, highly-entangled state characterizing the equilibrium dynamics of the lattice.

## Experimental Setup

- Hardware: Intel Cascade Lake 12 vCPU (85 GB Memory), NVIDIA A100 (40 GB Memory)
- 10 Trotter layers for the Fermi-Hubbard model Hamiltonian
- Metric: wall-clock time for full ⟨Z_i⟩ expectation values
- MPS parameters: $\chi = 256$, no absolute cutoff 
---

## Results

Since qubit count plays such an important role in simulator performance, we have ran the experiment under a "manageable", 26-qubit scenario, 
as well as in a "stress" 46-qubit scenario.

### Small systems (26 qubits)

We compare the performance of different quantum simulation platforms. 
We have chosen Pennylane and Qibo for the diversity of backends they implement, which support GPU as well as CPU simulation with MPS,
as well as Qiskit, which supports MPS simulation exclusively on CPUs.

We compare the time undertook by each simulator compared to Maestro's GPU mode:

![26 qubits](scaling_26%20qubits.png)

Adding GPUs makes simulation five times faster than the CPU-exclusive mode on Maestro, and more than seven times faster than any other platform. 


### Large systems (46 qubits) 

Trying to scale to bigger systems poses a problem in many platforms. 
To obtain meaningful results, such as expectation values or quantum state bitstring measurement samples, they often first contract the MPS tensor network into a state vector,
and then implement a sampler or estimator that is not specific to Matrix Product State simulation.

Such an approach, however, breaks down at large qubit counts. Due to exponential scaling, the memory requirements to store that state vector become more and more onerous,
reaching the terabytes at ~35 qubits. This issue makes many open-source simulation platforms unable to handle large-scale, high-entanglement problems.

The open-source fallback in this case remains Qiskit's MPS simulator. However, due to the fact that it does not support GPU simulation, 
throughput remains suboptimal compared to the 

## The Adaptive Maestro Pipeline

- 3-tier strategy: Scout (Pauli Propagator) → Sniper (MPS CPU) → Precision (MPS GPU)
- Lieb-Robinson light-cone exploitation
- Scaling plot: MPS time stays flat as total system grows

*[INSERT: adaptive pipeline diagram + scaling plot]*

---

## Discussion

- When to use which backend (summary table)
- MPS bond dimension and convergence
- Limitations: volume-law entanglement, memory walls

---

## Conclusion

---

## Appendix

### A. Gate counts table
### B. Reproducing the benchmarks (code snippets)
### C. Raw timing data table
