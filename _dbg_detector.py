"""Diagnostic script: reproduce the detector mismatch on the big circuit."""
import sys
sys.path.insert(0, "/Users/laramadison/Desktop/MastersProgram/Thesis/thesisNotebooks")

import random
import numpy as np
import tsim
import gate_by_gate as gbg

big = tsim.Circuit("""
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

 RX 15 10 5 2 7 1
 T_DAG 0 3 6 8 9 11 14
 CX 1 0 2 3 5 6 7 8 10 9 15 14
 CX 3 1 6 7 10 15
 CX 6 3 10 11
 CX 6 10
 MX 6
 RX 6
 CX 6 10
 CX 6 3 10 11
 CX 3 1 6 7 10 15
 CX 1 0 2 3 5 6 7 8 10 9 15 14
 T 0 3 6 8 9 11 14
 MX 15 10 5 2 7 1
 DETECTOR(1.60714, 0.75, 3, -1, -9) rec[-29] rec[-28] rec[-26] rec[-24] rec[-23] rec[-21] rec[-20] rec[-18] rec[-17] rec[-12] rec[-11] rec[-7]
 DETECTOR(4, 1, 4) rec[-6]
 DETECTOR(3, 1, 4) rec[-5]
 DETECTOR(2, 1, 4) rec[-4] rec[-7]
 DETECTOR(1, 0, 4) rec[-3]
 DETECTOR(2, 2, 4) rec[-2]
 DETECTOR(0, 1, 4) rec[-1]
 """)

circ = big.copy()
circ_splits = gbg.preprocessing(circ)

# Patch gate_by_gate to log per-detector values
orig = gbg.gate_by_gate

def patched(circuit, split_circs, detectors=None):
    circ_until_now = tsim.Circuit()
    num_qubits = circuit.num_qubits
    y = [0] * num_qubits
    all_dicts = gbg.AllDictionaries(num_qubits)
    is_initialized = [False] * num_qubits
    new_detectors = []
    num_Had = 0
    det_idx = 0

    for gate in circuit:
        targets = gate.targets_copy()
        for i in range(0, len(targets)):
            if gate.name in ("CX", "CNOT", "ZCX") and i % 2 == 0:
                a = targets[i].qubit_value
                b = targets[i + 1].qubit_value
                y[b] ^= y[a]
            if gate.name == "H":
                y = gbg.perform_had(all_dicts, split_circs[num_Had], num_qubits, y, targets[i].qubit_value)
                num_Had += 1
            if gate.name == "X":
                q = targets[i].qubit_value
                y[q] ^= 1
            if gate.name == "R":
                q = targets[i].qubit_value
                if is_initialized[q]:
                    all_dicts.reset_vals.append(y[q])
                y[q] = 0
                is_initialized[q] = True
            if gate.name == "RX":
                q = targets[i].qubit_value
                if not is_initialized[q]:
                    y[q] = 0 if random.random() < 0.5 else 1
                    is_initialized[q] = True
                else:
                    all_dicts.reset_vals.append(y[q])
                    y[q] = 0 if random.random() < 0.5 else 1
            if gate.name == "M":
                q = targets[i].qubit_value
                all_dicts.measurement_rec.append(y[q])
            if gate.name == "MX":
                q = targets[i].qubit_value
                y = gbg.perform_had(all_dicts, split_circs[num_Had], num_qubits, y, q)
                num_Had += 1
                all_dicts.measurement_rec.append(y[q])

        if gate.name == "DETECTOR":
            offsets = [t.value for t in targets]
            mvals = [all_dicts.measurement_rec[t.value] for t in targets]
            mlen = len(all_dicts.measurement_rec)
            absolute_idx = [mlen + t.value for t in targets]
            result = 0
            for v in mvals:
                result ^= v
            new_detectors.append(result)
            yield (det_idx, offsets, absolute_idx, mvals, result)
            det_idx += 1

# Run multiple seeded trials and find a failure
random.seed(1337)
np.random.seed(424)

records = []
N = 30
detects_ref = None
for run in range(N):
    detector_log = list(patched(circ, circ_splits, detects_ref))
    detector_results = [r[4] for r in detector_log]
    if detects_ref is None:
        detects_ref = detector_results
        records.append((run, "REF", detector_log))
        print(f"Run {run}: REF detectors = {detector_results}")
    else:
        for di, dlog in enumerate(detector_log):
            if dlog[4] != detects_ref[di]:
                print(f"Run {run}: DETECTOR {di} MISMATCH (got {dlog[4]}, expected {detects_ref[di]})")
                print(f"  offsets: {dlog[1]} -> abs idx: {dlog[2]}")
                print(f"  measurement_rec values used: {dlog[3]}")
                # also dump full measurement_rec at this point
                break
        else:
            print(f"Run {run}: PASSED detectors = {detector_results}")
