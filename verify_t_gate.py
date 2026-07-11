"""Verification loop for the d=3 T-cultivation circuit using gate-by-gate sim.

Mirrors cuttingDecomp/run.py:123-127:
    1. post-select on every DETECTOR XOR == 0 (clean syndrome),
    2. of kept shots, compare OBSERVABLE_INCLUDE(0) against noiseless_raw == 0.
"""
import re
import sys
import time
from pathlib import Path

import tsim
import stim
import gen

import gate_by_gate as gbg

# # injection + cultivation (degenerate)
circuit_source_injection_T = """
 QUBIT_COORDS(0, 0) 0
 QUBIT_COORDS(0, 1) 1
 QUBIT_COORDS(1, 0) 2
 QUBIT_COORDS(1, 1) 3
 QUBIT_COORDS(1, 2) 4
 QUBIT_COORDS(2, 0) 5
 QUBIT_COORDS(2, 1) 6
 QUBIT_COORDS(2, 2) 7
 QUBIT_COORDS(2, 3) 8
 QUBIT_COORDS(3, 0) 9
 QUBIT_COORDS(3, 1) 10
 QUBIT_COORDS(3, 2) 11
 QUBIT_COORDS(3, 3) 12
 QUBIT_COORDS(3, 4) 13
 QUBIT_COORDS(4, 0) 14
 QUBIT_COORDS(4, 1) 15
 QUBIT_COORDS(4, 2) 16
 QUBIT_COORDS(4, 3) 17
 R 1 14
 RX 4 2 10 12
 R 0 9 16 11 6 3 13 8 7 5 17 15
 TICK
 CX 2 5 4 7 10 15 12 17
 TICK
 CX 2 3 5 6 7 8 10 11 12 13 15 16
 TICK
 CX 2 0 5 9 7 11 10 6 12 8
 TICK
 CX 4 3 7 6 10 9 12 11 17 16
 TICK
 CX 2 5 4 7 10 15 12 17
 TICK
 M 7 5 17 15
 MX 4 2 10 12
 DETECTOR(2, 2, 0) rec[-8]
 DETECTOR(2, 0, 0) rec[-7]
 DETECTOR(4, 3, 0) rec[-6]
 DETECTOR(4, 1, 0) rec[-5]
 TICK
 RX 4 2 10 12 15 14
 R 7 5 17
 TICK
 CX 2 5 4 7 10 6 12 17 15 16
 TICK
 CX 4 3 7 6 10 9 12 11 14 15 17 16
 TICK
 CX 2 3 5 6 7 8 10 11 12 13 16 15
 TICK
 CX 2 0 5 9 7 11 10 15 12 8
 TICK
 CX 2 5 4 7 12 17 15 14
 TICK
 T_DAG 13
 TICK
 M 17 5 7
 MX 15 10 16 13 2 4 12
 DETECTOR(4, 3, 1) rec[-10]
 DETECTOR(2, 0, 1) rec[-9]
 DETECTOR(2, 2, 1) rec[-8]
 DETECTOR(4, 1, 1) rec[-7]
 DETECTOR(4, 2, 1) rec[-5]
 DETECTOR(3, 1, 1) rec[-5] rec[-6] rec[-12]
 DETECTOR(1, 0, 1) rec[-3] rec[-13]
 DETECTOR(1, 2, 1) rec[-2] rec[-14]
 DETECTOR(3, 3, 1) rec[-1] rec[-11]
 TICK
 RX 2 4 10
 R 5 7 15
 TICK
 CX 2 5 4 7 10 15
 TICK
 CX 2 0 5 9 7 11 10 6
 TICK
 CX 2 3 5 6 7 8 10 11
 TICK
 CX 4 3 7 6 10 9 15 14
 TICK
 CX 0 2 6 10 9 5 11 7
 TICK
 CX 3 2 6 5 8 7 11 10
 TICK
 CX 3 4 6 7 9 10 14 15
 TICK
 CX 2 5 4 7 10 15
 TICK
 MX 2 4 10
 M 5 7 15
 DETECTOR(1.25, 0.25, 2, -1, -9) rec[-9] rec[-6]
 DETECTOR(1.5, 1.875, 2, -1, -9) rec[-8] rec[-5]
 DETECTOR(1.75, 0.25, 2, -1, -9) rec[-3]
 DETECTOR(2, 1.875, 2, -1, -9) rec[-2]
 DETECTOR(3, 0.875, 2, -1, -9) rec[-13] rec[-12] rec[-4]
 DETECTOR(3.5, 0.875, 2, -1, -9) rec[-1]
 TICK
 RX 15 10 5 2 7 1
 TICK
 T_DAG 0 3 6 8 9 11 14
 TICK
 CX 1 0 2 3 5 6 7 8 10 9 15 14
 TICK
 CX 3 1 6 7 10 15
 TICK
 CX 6 3 10 11
 TICK
 CX 6 10
 TICK
 MX 6
 TICK
 RX 6
 TICK
 CX 6 10
 TICK
 CX 6 3 10 11
 TICK
 CX 3 1 6 7 10 15
 TICK
 CX 1 0 2 3 5 6 7 8 10 9 15 14
 TICK
 T 0 3 6 8 9 11 14
 TICK
 MX 15 10 5 2 7 1
 DETECTOR(1.60714, 0.75, 3, -1, -9) rec[-29] rec[-28] rec[-26] rec[-24] rec[-23] rec[-21] rec[-20] rec[-18] rec[-17] rec[-12] rec[-11] rec[-7]
 DETECTOR(4, 1, 4) rec[-6]
 DETECTOR(3, 1, 4) rec[-5]
 DETECTOR(2, 1, 4) rec[-4] rec[-7]
 DETECTOR(1, 0, 4) rec[-3]
 DETECTOR(2, 2, 4) rec[-2]
 DETECTOR(0, 1, 4) rec[-1]
 TICK
 """

# # noiseless projection
circuit_source_projection_proj = """
    TICK
    CX 8 3 11 6 0 9 8 14 11 9 0 3 11 14 8 9 0 6 6 14 6 3
    T 6
    TICK
    MX 0 11 8
    M 9 3 14
    MX 6
    DETECTOR(0.625, 0.125, 0, -1, -9) rec[-20] rec[-19] rec[-14] rec[-7]
    DETECTOR(0.875, 0.125, 0, -1, -9) rec[-17] rec[-4]
    DETECTOR(1.25, 1.4375, 0, -1, -9) rec[-20] rec[-14] rec[-6] rec[-5]
    DETECTOR(1.5, 1.4375, 0, -1, -9) rec[-16] rec[-3]
    DETECTOR(2.5, 0.9375, 0, -1, -9) rec[-14] rec[-6]
    DETECTOR(2.75, 0.9375, 0, -1, -9) rec[-15] rec[-2]
    OBSERVABLE_INCLUDE(0) rec[-34] rec[-1]
"""


def replace_t_with_s(s):
    s = re.sub(r'^(\s*)T_DAG(\s)', r'\1S_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)T(\s)', r'\1S\2', s, flags=re.MULTILINE)


def replace_s_with_t(c):
    p = str(c)
    p = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', p, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', p, flags=re.MULTILINE)


def add_noise(circuit, noise_strength):
    if noise_strength <= 0:
        return circuit
    nm = gen.NoiseModel.uniform_depolarizing(noise_strength)
    return nm.noisy_circuit_skipping_mpp_boundaries(circuit)


def build_noisy_circuit(noise_strength):
    """T -> S substitution lets the Clifford noise model run, then S -> T back."""
    clifford_inj = stim.Circuit(replace_t_with_s(circuit_source_injection_T))
    noisy_inj = add_noise(clifford_inj, noise_strength)
    return tsim.Circuit(replace_s_with_t(noisy_inj)) + tsim.Circuit(circuit_source_projection_proj)


def verify(noise_strength: float, n_shots: int, noiseless_raw: int = 0,
           chunk_size: int = 30000):
    """Post-select on a clean syndrome, then measure logical fidelity.

    Shots are processed in batches of ``chunk_size`` to keep the batched
    amplitude eval within GPU memory (a single huge batch exhausts it). Equal
    batch sizes let JAX reuse the JIT-compiled kernels across chunks, so only
    the first chunk pays compilation. ``kept`` and ``errors`` accumulate across
    chunks; ``psr`` and ``fidelity`` are derived from the totals.
    """
    circ = build_noisy_circuit(noise_strength)
    splits, noise_ops = gbg.preprocessing(circ)
    n_det = sum(1 for g in circ if g.name == "DETECTOR")
    ref = [0] * n_det                       # noiseless detectors are deterministically 0

    n_kept = 0
    n_errors = 0
    remaining = n_shots
    print("starting batching")
    t_start = time.perf_counter()
    batch_idx = 0
    while remaining > 0:
        batch = min(chunk_size, remaining)
        t_batch = time.perf_counter()
        passed, _y, _det, obs = gbg.gate_by_gate(circ, splits, noise_ops, ref, shots=batch)
        n_kept += sum(1 for s in range(batch) if passed[s])
        n_errors += sum(
            1
            for s in range(batch)
            if passed[s] and obs[s] and obs[s][0] != noiseless_raw
        )                                   # undetected logical errors among kept shots
        remaining -= batch
        batch_idx += 1
        # print(f"  batch {batch_idx} ({batch} shots): "
        #       f"{time.perf_counter() - t_batch:.2f}s this batch, "
        #       f"{time.perf_counter() - t_start:.2f}s elapsed")

    psr = n_kept / n_shots
    fidelity = 1 - n_errors / n_kept if n_kept else 0.0
    return {
        "p": noise_strength, "shots": n_shots,
        "kept": n_kept, "errors": n_errors,
        "psr": psr, "fidelity": fidelity,
    }


def main():
    NOISE_STRENGTHS = [0.001, 0.002, 0.005]
    SHOTS = 1200000                             # gate-by-gate is slow; crank as time allows

    print(f"{'p':<8} {'kept':>6} {'errors':>7} {'PSR':>8} {'fidelity':>10}")
    print(f"{'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*10}")
    for p in NOISE_STRENGTHS:
        r = verify(p, SHOTS)
        print(f"{'p':<8} {'kept':>6} {'errors':>7} {'PSR':>8} {'fidelity':>10}")
        print(f"{'-' * 8} {'-' * 6} {'-' * 7} {'-' * 8} {'-' * 10}")
        print(f"{r['p']:<8} {r['kept']:>6} {r['errors']:>7} "
              f"{r['psr']:>8.3f} {r['fidelity']:>10.4f}")


if __name__ == "__main__":
    main()
