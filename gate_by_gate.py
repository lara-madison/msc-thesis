"""Gate-by-gate autoregressive sampler for low-magic circuits, on tsim + pyzx_param.

Numerical / correctness notes, 2026-09-16:

1. power2 rebase (``split_circuit_reduce``). ``zx.full_reduce`` accumulates an
   unbounded sqrt(2) exponent -- pivot and lcomp each add O(k**2)
   (pyzx_param/rewrite_rules/rules.py:662, :800) -- and nothing rebased it, so
   deep circuits drove ``prefactor.power2`` toward the float32 cliff where
   tsim's ``jnp.pow(2.0, power)`` (tsim/core/exact_scalar.py:178) flushes to 0
   and BOTH Hadamard branch amplitudes vanish. Measured -15..-40 on big2 and
   -59..-9 on the 18q test circuit; now pinned to ~0. tsim does the same thing
   for its own sampler in tsim/compile/pipeline.py:157-170.

2. float64 Born rule (``perform_had``). Amplitudes arrive as complex64, and
   squaring halves the exponent headroom -- |z| ~ 2**-75 is enough to flush
   |z|**2 to zero in float32. Now widened to float64 and rescaled per shot by
   the larger magnitude, so ``denom == 0`` means a genuine zero and gets a
   warning rather than a silent discard.

3. ``strip_inert_noise``. Removes noise channels that act only on a qubit
   between its MX and its next reset. See that function's docstring.

KNOWN LIMITATION -- "MX-then-use". ``preprocessing`` emits MX as ``H q; M q``,
dropping the trailing h of tsim's ``h; m; h``
(tsim/core/instructions.py:1054-1058). That is fine when the qubit is reset
before being touched again, and wrong otherwise: ``y`` carries one Z-basis
label per wire, and a true MX leaves an X-eigenstate that no Z-label can
represent. In a 25-circuit randomised sweep against stim (comparing measurement
records), every one of the 6 distribution mismatches had an MX-then-use site and
none of the correct circuits did. Three cheap rewrites were tried and all scored
worse than leaving it alone; a real fix needs a per-wire basis frame beside y.

NOT A BUG -- live-lane RX. Binding ``m[i]`` to the Z-basis label ``y[:, q]`` is
correct, despite tsim's rx docstring saying it measures in the X basis
(tsim/core/instructions.py:1098-1107). Sampling it as an X measurement instead
was implemented and measured: big2 kept fell 12.55% -> 6.70% and a GHZ circuit
fell 100% -> 50%. Reverted; do not retry.

STILL OPEN: on the d=3 cultivation circuit at p=0.001, ~25% of shots are still
discarded on zero denominators for reasons not yet identified, and the sampler's
output has never been validated against a reference sampler.
"""

import numpy as np
import tsim
from tsim.core.graph import evaluate_graph
from tsim.noise.channels import (
    error_probs,
    pauli_channel_1_probs,
    pauli_channel_2_probs,
)

import pyzx_param as zx
from fractions import Fraction
import functools
import random
import string
import itertools
import warnings

from pyzx_param.graph.graph_s import GraphS


# ---------------------------------------------------------------------------
# Circuit preconditioning
# ---------------------------------------------------------------------------

_NOISE_1Q = frozenset({"DEPOLARIZE1", "X_ERROR", "Z_ERROR", "PAULI_CHANNEL_1"})
_NOISE_2Q = frozenset({"DEPOLARIZE2", "PAULI_CHANNEL_2"})
_NON_QUBIT_OPS = frozenset({
    "DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "QUBIT_COORDS", "SHIFT_COORDS",
})


def strip_inert_noise(circuit: tsim.Circuit) -> tsim.Circuit:
    """Drop noise acting only on qubits between their MX and their next reset.

    Such a channel is a physical no-op: it cannot change the already-recorded
    measurement, and the reset discards the state it touches. Verified against
    stim on the d=3 cultivation circuit -- removing 107 such targets moved the
    detector rates by 8e-4 against a sampling sigma of 7.9e-4.

    gate_by_gate is *not* neutral to them. It emits MX as ``H q; M q``, dropping
    the trailing h of tsim's ``h; m; h`` (tsim/core/instructions.py:1054-1058),
    so between an MX and its reset the wire is left in the wrong basis frame and
    a Pauli there acts as the wrong Pauli. Removing the dead channels keeps the
    physics and stops that divergence being exercised.

    NOTE: idempotent, and called from both preprocessing and gate_by_gate. They
    must walk the identical gate stream or the split and m[]/rec[] indices
    desynchronise, so neither may skip it.
    """
    out = tsim.Circuit()
    dead: set[int] = set()
    for gate in circuit:
        name = gate.name
        if name in _NON_QUBIT_OPS:
            out.append(gate)
            continue
        qubits = [t.qubit_value for t in gate.targets_copy()]
        if name in _NOISE_1Q:
            live = [q for q in qubits if q not in dead]
            if live:
                out.append(name=name, targets=live,
                           arg=gate.gate_args_copy(), tag=gate.tag)
            continue
        if name in _NOISE_2Q:
            live: list[int] = []
            for a, b in zip(qubits[0::2], qubits[1::2]):
                if a not in dead or b not in dead:
                    live += [a, b]
            if live:
                out.append(name=name, targets=live,
                           arg=gate.gate_args_copy(), tag=gate.tag)
            continue
        for q in qubits:
            dead.discard(q)
        if name == "MX":
            dead.update(qubits)
        out.append(gate)
    return out


# ---------------------------------------------------------------------------
# Utilities
#  ---------------------------------------------------------------------------

def generate_labels(n, start_label = None):
    labels = []
    length = 1
    while len(labels) < n:
        for combo in itertools.product(string.ascii_lowercase, repeat=length):
            m = ''.join(combo)
            if start_label is not None:
                m = start_label + m
            labels.append(m)
            if len(labels) == n:
                break
        length += 1
    return labels


# ---------------------------------------------------------------------------
# Computing amplitudes
#  ---------------------------------------------------------------------------


def comp_amplitude(pVals: list, compiled_gs: list[GraphS], n_qubits: int) -> complex:
    # g = paramsafe_graph.copy()

    # for v in list(g.vertices()):
    #     params = set(g.get_params(v))
    #     if not params:
    #         continue
    #     added = Fraction(0)
    #     for p in params:
    #         added += Fraction(int(val_param.get(p, 0)))
    #     if added != 0:
    #         g.add_to_phase(v, added)
    #     g.set_params(v, set())
    # zx.full_reduce(g)
    # gs = zx.simulate.find_stabilizer_decomp(g)
    #
    # gs = zx.simulate.find_stabilizer_decomp(g)
    # NOTE: the absolute amplitude scale is meaningless here - split_circuit_reduce
    # rebases each decomposition's power2 and every consumer uses only the ratio
    # p0:p1. Dividing by sqrt(2)**n_qubits would only push us toward underflow.
    return tsim.compile.evaluate.evaluate(compiled_gs, pVals)


################ IN comp_amp test running the evaluate in a loop for shots/batch times

# ---------------------------------------------------------------------------
# Preprocessing
#  ---------------------------------------------------------------------------
def split_circuit_reduce(circ_until_now: tsim.Circuit, y: list[int], m_len: int, rec_len: int, n_e: int, num_qubits: int) -> list[GraphS]:
    """Build the diagram for the partial circuit, attach symbolic y-output post-selections,
    then do paramSafe full_reduce once. The interior clifford / pivot-gadget passes are
    amortised across all shots; per-shot work is just substitute + the param-unsafe
    closing simps + stabilizer decomp (cheap when post-substitution tcount==0).

    ``n_e`` is the number of error bits parameterising noise channels seen so far;
    their ``e{i}`` columns must be declared here so substituted values resolve them."""
    g = circ_until_now.get_graph() #.diagram("pyzx")
    #zx.draw(g,labels=True)

    reset_list = [f"m[{i}]" for i in range(m_len)]
    rec_list = [f"rec[{i}]" for i in range(rec_len)]
    e_list = [f"e{i}" for i in range(n_e)]

    paramList = y + reset_list + rec_list + e_list

    last_vertices = {}
    for v in g.vertices():
        q = g.qubit(v)
        if q not in last_vertices or g.row(v) > g.row(last_vertices[q]):
            last_vertices[q] = v

    for qubit in range(num_qubits):
        if qubit in last_vertices:
            out_vertex = last_vertices[qubit]
            g.set_type(out_vertex, zx.VertexType.X)
            g.add_params(out_vertex, y[qubit])

    zx.full_reduce(g, paramSafe=True)
    gs = tsim.compile.stabrank.find_stab(g, "cat5")
    # gs = zx.simulate.find_stabilizer_decomp(g)

    # NOTE: full_reduce accumulates an unbounded sqrt(2) exponent (pivot and lcomp
    # each add O(k**2)), which underflows float32 in tsim's evaluator. All terms of
    # the decomposition are summed, so shifting them by one shared offset is a
    # uniform rescale that leaves p0:p1 untouched. Mirrors the power2 balancing in
    # tsim/compile/pipeline.py. Zero terms are skipped so they cannot skew the base.
    live = [x for x in gs if not x.scalar.is_zero]
    if live:
        power2_base = max(x.scalar.power2 for x in live)
        for x in live:
            x.scalar.add_power(-power2_base)

    compiled_gs = tsim.compile.compile.compile_scalar_graphs(gs, paramList)
    return compiled_gs

def preprocessing(circuit: tsim.Circuit) :
    circuit = strip_inert_noise(circuit)
    circ_until_now = tsim.Circuit()
    num_qubits = circuit.num_qubits
    y = generate_labels(num_qubits)
    is_initialized = [False] * num_qubits
    all_sub_circuits = []
    noise_ops: list[dict] = []
    n_e = 0  # mirrors tsim's b.num_error_bits counter
    rec_len = 0
    m_len = 0

    for gate in circuit:

        targets = gate.targets_copy()
        for i in range(0, len(targets)):

            if gate.name in ("CX", "CNOT", "ZCX") and i % 2 == 0:
                a = targets[i].qubit_value
                b = targets[i + 1].qubit_value
                circ_until_now.append_from_stim_program_text(f"CX {a} {b}")

            elif gate.name == "H":
                q = targets[i].qubit_value
                circ_until_now.append_from_stim_program_text(f"H {q}")

                all_sub_circuits.append((split_circuit_reduce(circ_until_now, y, m_len, rec_len, n_e, num_qubits), n_e))

            elif gate.name == "Z":
                q = targets[i].qubit_value
                circ_until_now.append_from_stim_program_text(f"Z {q}")

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

            elif gate.name == "X":  # Dont split
                q = targets[i].qubit_value
                circ_until_now.append_from_stim_program_text(f"X {q}")

            elif gate.name == "R":
                q = targets[i].qubit_value
                circ_until_now.append_from_stim_program_text(f"R {q}")
                if(is_initialized[q]):
                    m_len += 1
                else: is_initialized[q] = True

            elif gate.name == "RX":
                q = targets[i].qubit_value
                circ_until_now.append_from_stim_program_text(f"RX {q}")
                if (is_initialized[q]):
                    m_len += 1
                else: is_initialized[q] = True

            elif gate.name == "M":
                q = targets[i].qubit_value
                rec_len += 1
                circ_until_now.append_from_stim_program_text(f"M {q}")

            elif gate.name == "MX":
                q = targets[i].qubit_value

                # HADAMARD
                circ_until_now.append_from_stim_program_text(f"H {q}")
                all_sub_circuits.append((split_circuit_reduce(circ_until_now, y, m_len, rec_len, n_e, num_qubits), n_e))

                # Measure
                rec_len += 1
                circ_until_now.append_from_stim_program_text(f"M {q}")

            elif gate.name == "X_ERROR":
                q = targets[i].qubit_value
                p = gate.gate_args_copy()[0]
                circ_until_now.append_from_stim_program_text(f"X_ERROR({p}) {q}")
                noise_ops.append({
                    "name": "X_ERROR", "qubits": (q,),
                    "probs": error_probs(p),
                    "e_start": n_e, "n_bits": 1,
                })
                n_e += 1

            elif gate.name == "Z_ERROR":
                q = targets[i].qubit_value
                p = gate.gate_args_copy()[0]
                circ_until_now.append_from_stim_program_text(f"Z_ERROR({p}) {q}")
                noise_ops.append({
                    "name": "Z_ERROR", "qubits": (q,),
                    "probs": error_probs(p),
                    "e_start": n_e, "n_bits": 1,
                })
                n_e += 1

            elif gate.name == "DEPOLARIZE1":
                q = targets[i].qubit_value
                p = gate.gate_args_copy()[0]
                circ_until_now.append_from_stim_program_text(f"DEPOLARIZE1({p}) {q}")
                noise_ops.append({
                    "name": "DEPOLARIZE1", "qubits": (q,),
                    "probs": pauli_channel_1_probs(p / 3, p / 3, p / 3),
                    "e_start": n_e, "n_bits": 2,
                })
                n_e += 2

            elif gate.name == "DEPOLARIZE2" and i % 2 == 0:
                q1 = targets[i].qubit_value
                q2 = targets[i + 1].qubit_value
                p = gate.gate_args_copy()[0]
                circ_until_now.append_from_stim_program_text(f"DEPOLARIZE2({p}) {q1} {q2}")
                noise_ops.append({
                    "name": "DEPOLARIZE2", "qubits": (q1, q2),
                    "probs": pauli_channel_2_probs(*([p / 15] * 15)),
                    "e_start": n_e, "n_bits": 4,
                })
                n_e += 4

            elif gate.name == "PAULI_CHANNEL_1":
                q = targets[i].qubit_value
                args = gate.gate_args_copy()
                px, py, pz = args[0], args[1], args[2]
                circ_until_now.append_from_stim_program_text(
                    f"PAULI_CHANNEL_1({px}, {py}, {pz}) {q}"
                )
                noise_ops.append({
                    "name": "PAULI_CHANNEL_1", "qubits": (q,),
                    "probs": pauli_channel_1_probs(px, py, pz),
                    "e_start": n_e, "n_bits": 2,
                })
                n_e += 2

            elif gate.name == "PAULI_CHANNEL_2" and i % 2 == 0:
                q1 = targets[i].qubit_value
                q2 = targets[i + 1].qubit_value
                args = list(gate.gate_args_copy())
                arg_str = ",".join(str(a) for a in args)
                circ_until_now.append_from_stim_program_text(
                    f"PAULI_CHANNEL_2({arg_str}) {q1} {q2}"
                )
                noise_ops.append({
                    "name": "PAULI_CHANNEL_2", "qubits": (q1, q2),
                    "probs": pauli_channel_2_probs(*args),
                    "e_start": n_e, "n_bits": 4,
                })
                n_e += 4
        #
        # if gate.name == "DETECTOR":
        #     targets = gate.targets_copy()
        #     targ_s = ""
        #     if gate.name == "DETECTOR":
        #         arr = [" rec"] * len(targets)
        #         for i, val in enumerate(targets):
        #             targ_s += arr[i] + f"[{val.value}]"
        #
        #     circ_until_now.append_from_stim_program_text(f"DETECTOR{targ_s}")

    return all_sub_circuits, noise_ops


# ---------------------------------------------------------------------------
# Algorithm for gate by gate
# ---------------------------------------------------------------------------

def _sample_e_bits(noise_ops: list[dict], shots: int) -> np.ndarray:
    """Draw one independent error realization per shot.

    Returns a (shots, n_e_total) uint8 array. Column ``e_start + i`` holds bit
    ``i`` of the sampled outcome for the channel that starts at ``e_start``.
    """
    n_e = sum(op["n_bits"] for op in noise_ops)
    e = np.zeros((shots, n_e), dtype=np.uint8)
    for op in noise_ops:
        ks = np.random.choice(len(op["probs"]), size=shots, p=op["probs"])
        for i in range(op["n_bits"]):
            e[:, op["e_start"] + i] = (ks >> i) & 1
    return e


def gate_by_gate(circuit: tsim.Circuit, split_circs: list[GraphS],
                 noise_ops: list[dict], detectors: list = None, shots: int = 1):
    circuit = strip_inert_noise(circuit)
    num_qubits = circuit.num_qubits
    y = np.zeros((shots, num_qubits), dtype=np.uint8)
    all_dicts = AllDictionaries(num_qubits, shots)
    is_initialized = [False] * num_qubits
    e_sample = _sample_e_bits(noise_ops, shots)
    passed = np.ones(shots, dtype=bool)
    new_detectors = []      # one (shots,) uint8 column per DETECTOR gate
    new_observables = []    # one (shots,) uint8 column per OBSERVABLE_INCLUDE gate
    num_Had = 0
    for gate in circuit:

        targets = gate.targets_copy()
        for i in range(0, len(targets)):

            if gate.name in ("CX", "CNOT", "ZCX") and i % 2 == 0:
                a = targets[i].qubit_value
                b = targets[i + 1].qubit_value
                y[:, b] ^= y[:, a]

            if gate.name == "H":
                compiled_gs, e_len = split_circs[num_Had]
                y = perform_had(all_dicts, compiled_gs, num_qubits, y,
                                targets[i].qubit_value, passed, e_sample, e_len, shots)
                num_Had += 1

            if gate.name == "X":   # Dont split
                q = targets[i].qubit_value
                y[:, q] ^= 1

            if gate.name == "R":
                q = targets[i].qubit_value
                # Only subsequent resets get an m[i] parameter in tsim's parametrized
                # graph (see tsim/core/instructions.py:_r). A fresh reset adds a new
                # X-spider lane with no parameter, so appending here would shift the
                # m[i] indexing. The is_initialized gate mirrors preprocessing's m_len.
                if is_initialized[q]:
                    all_dicts.reset_vals.append(y[:, q].copy())
                y[:, q] = 0
                is_initialized[q] = True

            if gate.name == "RX":
                q = targets[i].qubit_value
                if is_initialized[q]:
                    all_dicts.reset_vals.append(y[:, q].copy())
                y[:, q] = (np.random.random(shots) >= 0.5).astype(np.uint8)
                is_initialized[q] = True

            if gate.name == "M":
                q = targets[i].qubit_value
                all_dicts.measurement_rec.append(y[:, q].copy())

            if gate.name == "MX":
                q = targets[i].qubit_value

                # HADAMARD (must run before recording this MX's own measurement)
                compiled_gs, e_len = split_circs[num_Had]
                y = perform_had(all_dicts, compiled_gs, num_qubits, y,
                                targets[i].qubit_value, passed, e_sample, e_len, shots)
                num_Had += 1

                # Measure
                all_dicts.measurement_rec.append(y[:, q].copy())

            # X_ERROR / Z_ERROR / DEPOLARIZE1 / DEPOLARIZE2 / PAULI_CHANNEL_*
            # are driven symbolically: their parameter spiders sit in split_circs
            # and are substituted via e_bits in perform_had.

        if gate.name == "DETECTOR":
            # t.value stays the raw (negative) stim offset; measurement_rec is a
            # measurement-ordered column list, so negative indexing resolves the
            # right measurement. The zeros seed avoids in-place xor on a stored column.
            col = functools.reduce(
                np.bitwise_xor,
                (all_dicts.measurement_rec[t.value] for t in targets),
                np.zeros(shots, dtype=np.uint8),
            )
            new_detectors.append(col)
            k = len(new_detectors) - 1
            ref = detectors[k] if detectors is not None else int(col[0])
            passed &= (col == ref)

        if gate.name == "OBSERVABLE_INCLUDE":
            col = functools.reduce(
                np.bitwise_xor,
                (all_dicts.measurement_rec[t.value] for t in targets),
                np.zeros(shots, dtype=np.uint8),
            )
            new_observables.append(col)

    if detectors is None:
        detectors = [int(c[0]) for c in new_detectors]

    observables = [[int(c[s]) for c in new_observables] for s in range(shots)]
    return passed.tolist(), y.tolist(), detectors, observables


def perform_had(dics, split_graph: GraphS, num_qubits, y, q, passed, e_sample: np.ndarray, e_len: int = 0, shots: int = 1):
    reset_mat = (np.stack(dics.reset_vals, axis=1) if dics.reset_vals
                 else np.zeros((shots, 0), dtype=np.uint8))
    meas_mat = (np.stack(dics.measurement_rec, axis=1) if dics.measurement_rec
                else np.zeros((shots, 0), dtype=np.uint8))
    # Column order [y, reset, rec, e] matches split_circuit_reduce's paramList.
    precompute = np.concatenate([reset_mat, meas_mat, e_sample[:, :e_len]], axis=1)

    val0 = y.copy()
    val0[:, q] = 0
    val1 = y.copy()
    val1[:, q] = 1

    paramList0 = np.concatenate([val0, precompute], axis=1)
    paramList1 = np.concatenate([val1, precompute], axis=1)

    z0 = comp_amplitude(paramList0, split_graph, num_qubits)
    z1 = comp_amplitude(paramList1, split_graph, num_qubits)

    # NOTE: z0/z1 arrive as complex64. Squaring halves the exponent headroom, so
    # |z| ~ 2**-75 is enough to flush |z|**2 to zero in float32. Widen to float64
    # and divide out the larger magnitude before squaring - only the ratio matters,
    # and this pins denom into [1, 2] for every live shot.
    a0 = np.abs(np.asarray(z0, dtype=np.complex128))  # weight of outcome 0
    a1 = np.abs(np.asarray(z1, dtype=np.complex128))  # weight of outcome 1
    scale = np.maximum(a0, a1)
    scale = np.where(scale > 0, scale, 1.0)
    p0 = (a0 / scale) ** 2
    p1 = (a1 / scale) ** 2
    denom = p0 + p1
    safe_denom = np.where(denom > 0, denom, 1.0)
    # Outcome 1 when u*denom >= |z0|^2 (matches the old scalar sampling rule).
    bit1 = np.random.random(shots) * safe_denom >= p0

    # Already-failed shots stay failed and keep an arbitrary 0; live shots take
    # the sampled bit. `passed &= ...` is in place so the caller sees the update.
    live = passed & (denom > 0)
    y[:, q] = 0
    y[live, q] = bit1[live].astype(np.uint8)

    # NOTE: denom is now scale-free, so a zero can no longer come from underflow -
    # it means both branch amplitudes are exactly zero. Count only shots that were
    # still live, otherwise every later Hadamard re-reports the same dead shots.
    newly_dead = int((passed & (denom == 0)).sum())
    if newly_dead:
        warnings.warn(
            f"Both Hadamard branch amplitudes are exactly zero on qubit {q} "
            f"for {newly_dead} live shot(s); discarding them.",
            stacklevel=2,
        )

    passed &= (denom > 0)

    return y


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
#

class AllDictionaries:
    def __init__(self, num_qubits, shots):
        self.strings = generate_labels(num_qubits)
        self.rec_list = [f"rec[{i}]" for i in range(40)]
        self.reset_list = [f"m[{i}]" for i in range(40)]
        # reset_vals / measurement_rec hold one (shots,) uint8 column per reset /
        # measurement, appended in circuit order (was a per-shot list of lists).
        self.reset_vals = []
        self.new_detectors = []
        self.measurement_rec = []

def create_y(value: int, y: list, q: int, strings: list) -> list:
    y_alt = y.copy()
    y_alt[q] = value

    fractions = [Fraction(n) for n in y_alt]
    val = dict(zip(strings, fractions))

    return fractions
