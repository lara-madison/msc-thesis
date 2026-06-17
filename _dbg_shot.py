"""Run a single shot of test_big2 with instrumentation to log p-values from each perform_had."""
import random
from fractions import Fraction

import tsim
import gate_by_gate as gbg


# Patch perform_had to log p values.
ORIGINAL_PERFORM_HAD = gbg.perform_had
LOG = []


def patched_perform_had(dics, split_circ, num_qubits, y, q):
    rec = dict(zip(dics.rec_list, dics.measurement_rec))
    m = dict(zip(dics.reset_list, dics.reset_vals))
    val0 = gbg.create_y(0, y, q, dics.strings)
    val1 = gbg.create_y(1, y, q, dics.strings)
    z0 = gbg.comp_amplitude(val0 | rec | m, split_circ, num_qubits)
    z1 = gbg.comp_amplitude(val1 | rec | m, split_circ, num_qubits)
    denom = abs(z0) ** 2 + abs(z1) ** 2
    if denom == 0:
        denom = 1
    p = abs(z0) ** 2 / denom
    LOG.append((q, p, abs(z0), abs(z1)))
    y[q] = 0 if random.random() < p else 1
    return y


gbg.perform_had = patched_perform_had


CIRCUIT_TEXT = open(__file__).__class__  # placeholder, we read from test_gbg below


def main():
    import test_gbg as tg
    # Recreate the test circuit
    circuit_src = '''
        RX 1 2 3
        R 4 5 6
        CX 1 4 2 5 3 6
        CX 1 5 2 6
        RX 7 8
        CX 2 4 3 5 7 6 8 1
        RX 9 10 11
        CX 9 4 10 5 11 7 1 8
        CX 11 9 5 10 6 7
        RX 12
        CX 4 9 7 11
        CX 9 11
        T_DAG 11
        CX 9 11
        CX 7 11
        RX 13 14
        CX 13 9 14 7
        CX 9 13 7 14
        RX 7
        R 6 15
        RX 5
        R 1
        RX 9
        R 4 0
        CX 14 15 7 6 5 1 9 4 8 0
        CX 3 6 14 7 10 5 8 1 13 9 2 4
        CX 15 14 2 5 11 9 0 8
        CX 11 7 2 6 3 5 0 4
        RX 8
        CX 7 11 6 2 5 3 4 0
        CX 6 3 5 2 9 13 8 0
        CX 6 15 5 10 1 8 9 11 4 2
        CX 7 6 5 1 9 4 0 8
        MX 7

        DETECTOR[POST-SELECTION] rec[-1]
        M 6
        DETECTOR[POST-SELECTION] rec[-1]
        M 14
        DETECTOR[POST-SELECTION] rec[-1]
        MX 5
        DETECTOR[POST-SELECTION] rec[-1]
        M 1
        DETECTOR[POST-SELECTION] rec[-1]
        MX 9
        DETECTOR[POST-SELECTION] rec[-1]
        M 4
        DETECTOR[POST-SELECTION] rec[-1]
        MX 0
        DETECTOR[POST-SELECTION] rec[-1]
        RX 9 4 1 5 7 6
        T_DAG 2 15 8 10 11 3 13
        CX 9 13 4 2 1 8 5 10 7 11 6 15
        CX 11 9 5 1 2 6
        CX 2 11 5 3
        CX 2 5
        MX 2
        DETECTOR[POST-SELECTION] rec[-1]
        RX 2
        CX 2 5
        CX 2 11 5 3
        CX 11 9 5 1 2 6
        CX 9 13 4 2 1 8 5 10 7 11 6 15
        T 2 15 8 10 11 3 13
        MX 9
        DETECTOR[POST-SELECTION] rec[-1]
        MX 4
        DETECTOR[POST-SELECTION] rec[-1]
        MX 1
        DETECTOR[POST-SELECTION] rec[-1]
        MX 5
        DETECTOR[POST-SELECTION] rec[-1]
        MX 7
        DETECTOR[POST-SELECTION] rec[-1]
        MX 6
        DETECTOR[POST-SELECTION] rec[-1]
    '''
    circuit = tsim.Circuit(circuit_src)
    splits = gbg.preprocessing(circuit)
    print(f"got {len(splits)} splits")
    random.seed(0)
    detects_ref = None
    for shot in range(20):
        LOG.clear()
        passed, result, detects, obs = gbg.gate_by_gate(circuit, splits, detects_ref)
        if shot == 0:
            detects_ref = detects
            print(f"shot 0 detects={detects}")
            for i, (q, p, z0a, z1a) in enumerate(LOG):
                marker = "  <- nondet" if not (p == 0 or p == 1) else ""
                print(f"  ph[{i}] q={q}  p(0)={p}  |z0|={z0a:.4g}  |z1|={z1a:.4g}{marker}")
        else:
            nondet = [(i, q, p, z0a, z1a) for i, (q, p, z0a, z1a) in enumerate(LOG) if not (p == 0 or p == 1)]
            if not passed or nondet:
                print(f"\nshot {shot}: passed={passed}  ref={detects_ref}")
                for i, (q, p, z0a, z1a) in enumerate(LOG):
                    marker = "  <- nondet" if not (p == 0 or p == 1) else ""
                    print(f"  ph[{i}] q={q}  p(0)={p}  |z0|={z0a:.4g}  |z1|={z1a:.4g}{marker}")


if __name__ == "__main__":
    main()
