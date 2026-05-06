"""Verification loop for the d=3 T-cultivation circuit using gate-by-gate sim.

Mirrors cuttingDecomp/run.py:123-127:
    1. post-select on every DETECTOR XOR == 0 (clean syndrome),
    2. of kept shots, compare OBSERVABLE_INCLUDE(0) against noiseless_raw == 0.
"""
import re
import sys
from pathlib import Path

_CUTTING = Path(__file__).resolve().parent.parent / "cuttingDecomp"
sys.path.insert(0, str(_CUTTING))

import tsim
import stim

from d_3_circuit_definitions import (
    circuit_source_injection_T,
    circuit_source_projection_proj,
)
import gate_by_gate as gbg


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


def verify(noise_strength: float, n_shots: int, noiseless_raw: int = 0):
    circ = build_noisy_circuit(noise_strength)
    splits = gbg.preprocessing(circ)
    n_det = sum(1 for g in circ if g.name == "DETECTOR")
    ref = [0] * n_det                       # noiseless detectors are deterministically 0

    n_kept = 0
    n_errors = 0
    for _ in range(n_shots):
        ok, _y, _det, obs = gbg.gate_by_gate(circ, splits, ref)
        if not ok:
            continue                        # post-selection rejected
        n_kept += 1
        if obs and obs[0] != noiseless_raw:
            n_errors += 1                   # undetected logical error

    psr = n_kept / n_shots
    fidelity = 1 - n_errors / n_kept if n_kept else 0.0
    return {
        "p": noise_strength, "shots": n_shots,
        "kept": n_kept, "errors": n_errors,
        "psr": psr, "fidelity": fidelity,
    }


def main():
    NOISE_STRENGTHS = [0.001, 0.002, 0.005]
    SHOTS = 200                             # gate-by-gate is slow; crank as time allows

    print(f"{'p':<8} {'kept':>6} {'errors':>7} {'PSR':>8} {'fidelity':>10}")
    print(f"{'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*10}")
    for p in NOISE_STRENGTHS:
        r = verify(p, SHOTS)
        print(f"{r['p']:<8} {r['kept']:>6} {r['errors']:>7} "
              f"{r['psr']:>8.3f} {r['fidelity']:>10.4f}")


if __name__ == "__main__":
    main()
