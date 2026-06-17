"""Compare the paramSafe path vs full-reduce-with-substituted-values for split[4] of test_big2.

Walks the test_big2 circuit gate-by-gate, builds circ_until_now up to the H of MX 2,
materializes the raw pyzx diagram, then evaluates two ways:

  A) paramSafe path: zx.full_reduce(g, paramSafe=True), find_stabilizer_decomp, evaluate_scalar
  B) substitute-first path: substitute concrete values for all m[k]/rec[j]/y vars,
     zx.full_reduce(g)  (NO paramSafe), find_stabilizer_decomp, evaluate_scalar

For deterministic m[k]=0, rec[j]=0 inputs and y=0, it should produce |z0| >> |z1| if vec_sim
is right (MX 2 deterministic 0). The current paramSafe path produces a √2 ratio.
"""
import math
import random
from fractions import Fraction

import tsim
import pyzx_param as zx
import gate_by_gate as gbg


CIRCUIT_TEXT = """
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
"""


def build_circ_until_split4():
    """Build circ_until_now exactly as preprocessing would, up through the H of the 5th MX."""
    circuit = tsim.Circuit(CIRCUIT_TEXT)
    circ_until_now = tsim.Circuit()
    num_Had = 0
    for gate in circuit:
        targets = gate.targets_copy()
        for i in range(len(targets)):
            if gate.name in ("CX", "CNOT", "ZCX") and i % 2 == 0:
                a = targets[i].qubit_value
                b = targets[i + 1].qubit_value
                circ_until_now.append_from_stim_program_text(f"CX {a} {b}")
            elif gate.name == "H":
                q = targets[i].qubit_value
                circ_until_now.append_from_stim_program_text(f"H {q}")
                num_Had += 1
                if num_Had == 5:
                    return circ_until_now, circuit.num_qubits
            elif gate.name == "Z":
                circ_until_now.append_from_stim_program_text(f"Z {targets[i].qubit_value}")
            elif gate.name == "S":
                q = targets[i].qubit_value
                if gate.tag == "T":
                    circ_until_now.append_from_stim_program_text(f"T {q}")
                else:
                    circ_until_now.append_from_stim_program_text(f"S {q}")
            elif gate.name == "S_DAG":
                q = targets[i].qubit_value
                if gate.tag == "T":
                    circ_until_now.append_from_stim_program_text(f"T_DAG {q}")
                else:
                    circ_until_now.append_from_stim_program_text(f"S_DAG {q}")
            elif gate.name == "X":
                circ_until_now.append_from_stim_program_text(f"X {targets[i].qubit_value}")
            elif gate.name == "R":
                circ_until_now.append_from_stim_program_text(f"R {targets[i].qubit_value}")
            elif gate.name == "RX":
                circ_until_now.append_from_stim_program_text(f"RX {targets[i].qubit_value}")
            elif gate.name == "M":
                circ_until_now.append_from_stim_program_text(f"M {targets[i].qubit_value}")
            elif gate.name == "MX":
                q = targets[i].qubit_value
                circ_until_now.append_from_stim_program_text(f"H {q}")
                num_Had += 1
                if num_Had == 5:
                    return circ_until_now, circuit.num_qubits
                circ_until_now.append_from_stim_program_text(f"M {q}")
    raise RuntimeError("never reached split 5")


def all_params(g):
    seen = set()
    for v in g.vertices():
        seen |= set(g.get_params(v))
    return seen


def substitute_values(g, vals, default=0):
    """Inline boolean variable values into vertex phases. Unknown vars get `default`."""
    for v in list(g.vertices()):
        params = set(g.get_params(v))
        if not params:
            continue
        added = Fraction(0)
        for p in params:
            v_val = vals.get(p, default)
            added += Fraction(int(v_val))
        if added != 0:
            g.add_to_phase(v, added)
        g.set_params(v, set())


def evaluate_with_substitution(circ_until_now, num_qubits, y_full, m_vals, rec_vals, c_value, q_split=2):
    """Substitute all vars, then full_reduce (no paramSafe) + find_stabilizer_decomp + evaluate."""
    g = circ_until_now.diagram("pyzx")
    # Mirror the existing post-select logic in gate_by_gate.split_circuit_reduce.
    last_vertices = {}
    for v in g.vertices():
        q = g.qubit(v)
        if q not in last_vertices or g.row(v) > g.row(last_vertices[q]):
            last_vertices[q] = v
    strings = gbg.generate_labels(num_qubits)
    y_alt = list(y_full)
    y_alt[q_split] = c_value
    for qubit, bit in enumerate(y_alt):
        if qubit in last_vertices:
            out_vertex = last_vertices[qubit]
            g.set_type(out_vertex, zx.VertexType.X)
            g.add_params(out_vertex, strings[qubit])
    # Build values dict.
    vals = {}
    vals.update({strings[q]: int(y_alt[q]) for q in range(num_qubits)})
    vals.update({f"m[{k}]": int(v) for k, v in enumerate(m_vals)})
    vals.update({f"rec[{j}]": int(v) for j, v in enumerate(rec_vals)})
    params = all_params(g)
    missing = sorted(p for p in params if p not in vals)
    if missing:
        # default to 0 for unknown silent-meas vars (m[k]) introduced by RX-after-M;
        # likewise any unbound rec[j] (shouldn't happen at this split).
        for p in missing:
            vals[p] = 0
    substitute_values(g, vals)
    zx.full_reduce(g)
    gs = zx.simulate.find_stabilizer_decomp(g)
    amp = 0
    for h in gs:
        amp += h.scalar.evaluate_scalar({})
    return amp / (math.sqrt(2) ** num_qubits)


def evaluate_paramsafe(circ_until_now, num_qubits, y_full, m_vals, rec_vals, c_value, q_split=2):
    """Reproduce the existing paramSafe path."""
    g = circ_until_now.diagram("pyzx")
    reset_list = [{f"m[{i}]"} for i in range(40)]
    last_vertices = {}
    for v in g.vertices():
        q = g.qubit(v)
        param_val = g.get_params(v)
        if param_val in reset_list:
            for neighbor in g.neighbors(v):
                if "rec" in str(g.get_params(neighbor)):
                    new_sym = str(g.get_params(neighbor)).replace("{", "").replace("}", "").replace("'", "")
                    g.set_phase(v, new_sym)
        if q not in last_vertices or g.row(v) > g.row(last_vertices[q]):
            last_vertices[q] = v
    strings = gbg.generate_labels(num_qubits)
    y_alt = list(y_full)
    y_alt[q_split] = c_value
    for qubit, bit in enumerate(y_alt):
        if qubit in last_vertices:
            out_vertex = last_vertices[qubit]
            g.set_type(out_vertex, zx.VertexType.X)
            g.add_params(out_vertex, strings[qubit])
    zx.full_reduce(g, paramSafe=True)
    gs = zx.simulate.find_stabilizer_decomp(g)
    vals = {}
    vals.update({strings[q]: Fraction(int(y_alt[q])) for q in range(num_qubits)})
    vals.update({f"m[{k}]": Fraction(int(v)) for k, v in enumerate(m_vals)})
    vals.update({f"rec[{j}]": Fraction(int(v)) for j, v in enumerate(rec_vals)})
    amp = 0
    for h in gs:
        amp += h.scalar.evaluate_scalar(vals)
    return amp / (math.sqrt(2) ** num_qubits)


def main():
    random.seed(0)
    circ_until_now, num_qubits = build_circ_until_split4()
    print(f"num_qubits={num_qubits}")
    # We need a plausible y vector at this point in the simulation. Easiest: run a real shot.
    # But for an isolated comparison we can pick zeros and small variations.
    # Try a few configurations and compare A vs B.
    configs = [
        ("all-zero  m=0 rec=0 y=0", [0]*16, [0]*7, [0]*8),
        ("y=1 elsewhere",          [1]*16, [0]*7, [0]*8),
        ("m[0]=1",                 [0]*16, [1,0,0,0,0,0,0], [0]*8),
        ("m[3]=1",                 [0]*16, [0,0,0,1,0,0,0], [0]*8),
        ("rec[0]=1",               [0]*16, [0]*7, [1,0,0,0,0,0,0,0]),
        ("rec[7]=1",               [0]*16, [0]*7, [0,0,0,0,0,0,0,1]),
    ]
    for label, y_full, m_vals, rec_vals in configs:
        z0_A = evaluate_paramsafe(circ_until_now, num_qubits, y_full, m_vals, rec_vals, c_value=0)
        z1_A = evaluate_paramsafe(circ_until_now, num_qubits, y_full, m_vals, rec_vals, c_value=1)
        z0_B = evaluate_with_substitution(circ_until_now, num_qubits, y_full, m_vals, rec_vals, c_value=0)
        z1_B = evaluate_with_substitution(circ_until_now, num_qubits, y_full, m_vals, rec_vals, c_value=1)
        denom_A = abs(z0_A) ** 2 + abs(z1_A) ** 2
        denom_B = abs(z0_B) ** 2 + abs(z1_B) ** 2
        pA = abs(z0_A) ** 2 / denom_A if denom_A else float('nan')
        pB = abs(z0_B) ** 2 / denom_B if denom_B else float('nan')
        print(f"\n[{label}]")
        print(f"  paramSafe : z0={z0_A:.6g}  z1={z1_A:.6g}  P(0)={pA:.4f}")
        print(f"  substitute: z0={z0_B:.6g}  z1={z1_B:.6g}  P(0)={pB:.4f}")


if __name__ == "__main__":
    main()
