"""Benchmark harness: weak-sim techniques vs. the gate-by-gate sampler.

Two independent comparisons, because pyzx's BSS weak-sim and the gate-by-gate
method cannot run on the same graph:

1. ``head_to_head`` -- the real thesis comparison. Runs the gate-by-gate sampler
   and tsim's BSS stabilizer-rank detector sampler on the SAME d=3 T-cultivation
   circuit and reports steady-state throughput (shots/sec) for each, plus a
   PSR/fidelity cross-check that both agree within Monte-Carlo noise.

2. ``weak_sim_sweep`` -- the pyzx pure-python BSS weak-sim scaling curve on plain
   Clifford+T circuits (no detectors). ``pyzx.simulation.simulate`` cannot consume
   tsim's closed ``pyzx_param`` cultivation graph (0 open boundaries + a
   ``DyadicNumber`` scalar it rejects), so this is a separate curve. It motivates
   why the pure-python technique does not reach cultivation scale.

Run directly for a demo:  python weak_sim_bench.py
"""
import random
import time

import numpy as np
import pyzx as zx

import gate_by_gate as gbg
import verify_t_gate as vtg      # build_noisy_circuit lives here


def _time(fn):
    """Return (result, seconds) for a zero-arg callable."""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# 1. pyzx pure-python BSS weak-sim on plain Clifford+T circuits (scaling curve)
# --------------------------------------------------------------------------- #
def weak_sim_sample(g, rng=random):
    """Autoregressively sample the output bits of U|0...0> via pyzx BSS.

    ``g`` must be a native pyzx circuit graph (e.g. from ``zx.generate.cliffordT``)
    where every qubit line has an open input and output boundary. One ``simulate``
    call per qubit computes the conditional P(bit_k = 0 | earlier bits); the doubled
    diagram already yields a real probability, so there is no ``abs(.)**2``.

    Returns (bitstring, seconds).
    """
    n = g.qubit_count()
    t0 = time.perf_counter()
    bits, p_prefix = "", 1.0
    for k in range(n):
        h = g.copy()
        h.apply_state("0" * n)
        h.apply_effect(bits + "0" + "/" * (n - k - 1))   # fix 0..k-1, test bit k = 0, trace the rest
        h.compose(h.adjoint())                            # doubled -> real probability P(prefix, bit_k=0)
        zx.simplify.full_reduce(h)
        p0 = float(np.real(zx.simulation.simulate(zx.simulation.Strategy.BSS, h)))
        cond0 = p0 / p_prefix if p_prefix > 1e-12 else 1.0
        cond0 = min(max(cond0, 0.0), 1.0)
        if rng.random() < cond0:
            bits += "0"
            p_prefix = p0
        else:
            bits += "1"
            p_prefix = max(p_prefix - p0, 0.0)
    return bits, time.perf_counter() - t0


def weak_sim_sweep(qubits=range(2, 7), depth=20, seed=0, samples=1, verbose=True):
    """Time pyzx weak-sim per sample across circuit sizes. Returns list of dict rows."""
    rows = []
    if verbose:
        print(f"{'qubits':>6} {'depth':>6} {'tcount':>6} {'s/sample':>10}")
    for q in qubits:
        g = zx.generate.cliffordT(q, depth, seed=seed)
        tcount = zx.tcount(g)
        secs = sum(weak_sim_sample(g)[1] for _ in range(samples)) / samples
        rows.append({"qubits": q, "depth": depth, "tcount": tcount, "sec_per_sample": secs})
        if verbose:
            print(f"{q:>6} {depth:>6} {tcount:>6} {secs:>10.3f}")
    return rows


# --------------------------------------------------------------------------- #
# 2. gate-by-gate vs tsim BSS sampler on the SAME cultivation circuit
# --------------------------------------------------------------------------- #
def _psr_fidelity(kept, errors, shots):
    psr = kept / shots if shots else 0.0
    fidelity = 1 - errors / kept if kept else 0.0
    return psr, fidelity


def _gbg_counts(passed, obs, shots, noiseless_raw=0):
    kept = sum(1 for s in range(shots) if passed[s])
    errors = sum(
        1 for s in range(shots)
        if passed[s] and obs[s] and obs[s][0] != noiseless_raw
    )
    return kept, errors


def _tsim_counts(dets, obss, noiseless_raw=0):
    dets = np.asarray(dets)
    obss = np.asarray(obss)
    kept_mask = ~dets.any(axis=1)                     # clean syndrome: every detector == 0
    obs0 = obss[:, 0] if obss.ndim == 2 and obss.shape[1] else np.zeros(len(dets), dtype=int)
    kept = int(kept_mask.sum())
    errors = int(((obs0 != noiseless_raw) & kept_mask).sum())
    return kept, errors


def head_to_head(noise_strength=0.001, shots=30000, seed=0, verbose=True):
    """Benchmark gate-by-gate vs tsim's BSS detector sampler on one cultivation circuit.

    Each method is run twice at the SAME batch size: the first call is JIT/compile
    bound, the second is steady state (JAX caches the kernel for a fixed batch size).
    We report both so compile cost and throughput are separated. PSR/fidelity from
    each method are returned as an independent-Monte-Carlo correctness cross-check.
    """
    circ = vtg.build_noisy_circuit(noise_strength)
    n_det = sum(1 for g in circ if g.name == "DETECTOR")
    ref = [0] * n_det

    # ---- gate-by-gate ----
    (splits, noise_ops), gbg_setup = _time(lambda: gbg.preprocessing(circ))
    _, gbg_compile = _time(lambda: gbg.gate_by_gate(circ, splits, noise_ops, ref, shots=shots))
    gbg_out, gbg_run = _time(lambda: gbg.gate_by_gate(circ, splits, noise_ops, ref, shots=shots))
    passed, _y, _det, obs = gbg_out
    gbg_kept, gbg_err = _gbg_counts(passed, obs, shots)
    gbg_psr, gbg_fid = _psr_fidelity(gbg_kept, gbg_err, shots)

    # ---- tsim BSS stabilizer-rank sampler ----
    sampler, tsim_setup = _time(
        lambda: circ.compile_detector_sampler(strategy="bss", seed=seed)
    )
    _, tsim_compile = _time(lambda: sampler.sample(shots, separate_observables=True))
    (dets, obss), tsim_run = _time(lambda: sampler.sample(shots, separate_observables=True))
    tsim_kept, tsim_err = _tsim_counts(dets, obss)
    tsim_psr, tsim_fid = _psr_fidelity(tsim_kept, tsim_err, shots)

    result = {
        "noise_strength": noise_strength,
        "shots": shots,
        "n_detectors": n_det,
        "gate_by_gate": {
            "setup_s": gbg_setup, "compile_s": gbg_compile, "steady_s": gbg_run,
            "shots_per_s": shots / gbg_run if gbg_run else float("inf"),
            "kept": gbg_kept, "errors": gbg_err, "psr": gbg_psr, "fidelity": gbg_fid,
        },
        "tsim_bss": {
            "setup_s": tsim_setup, "compile_s": tsim_compile, "steady_s": tsim_run,
            "shots_per_s": shots / tsim_run if tsim_run else float("inf"),
            "kept": tsim_kept, "errors": tsim_err, "psr": tsim_psr, "fidelity": tsim_fid,
        },
    }
    if verbose:
        _print_head_to_head(result)
    return result


def _print_head_to_head(r):
    g, t = r["gate_by_gate"], r["tsim_bss"]
    print(f"\ncultivation circuit  p={r['noise_strength']}  shots={r['shots']}  "
          f"detectors={r['n_detectors']}")
    print(f"{'':20} {'gate_by_gate':>16} {'tsim_bss':>16}")
    print("-" * 54)
    rows = [
        ("setup / compile s", f"{g['setup_s']:.3f} / {g['compile_s']:.3f}",
                              f"{t['setup_s']:.3f} / {t['compile_s']:.3f}"),
        ("steady-state s", f"{g['steady_s']:.3f}", f"{t['steady_s']:.3f}"),
        ("shots/sec", f"{g['shots_per_s']:,.0f}", f"{t['shots_per_s']:,.0f}"),
        ("kept (PSR)", f"{g['kept']} ({g['psr']:.3f})", f"{t['kept']} ({t['psr']:.3f})"),
        ("fidelity", f"{g['fidelity']:.4f}", f"{t['fidelity']:.4f}"),
    ]
    for label, a, b in rows:
        print(f"{label:20} {a:>16} {b:>16}")
    speedup = t["steady_s"] / g["steady_s"] if g["steady_s"] else float("inf")
    faster = "gate_by_gate" if speedup > 1 else "tsim_bss"
    print(f"\nsteady-state: {faster} faster by {max(speedup, 1 / speedup):.1f}x")


if __name__ == "__main__":
    print("=== pyzx weak-sim scaling curve (plain Clifford+T) ===")
    weak_sim_sweep(qubits=range(2, 6), depth=20)
    print("\n=== gate-by-gate vs tsim BSS (cultivation circuit) ===")
    head_to_head(noise_strength=0.001, shots=5000)
