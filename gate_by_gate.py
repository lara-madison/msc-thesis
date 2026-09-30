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
   between its MX, or its M, and its next reset. See that function's docstring.

Numerical / correctness notes, 2026-09-30:

4. Error labels (``_apply_error_flips``). Noise channels only ever lived in the
   diagram; the X component of a sampled error was never XOR'd into ``y``. The
   stale label contradicts the diagram, so both branch amplitudes of the next
   Hadamard on any *other* qubit are exactly zero. This was the ~25% of shots
   the d=3 cultivation circuit lost to zero denominators at p=0.001 (26.4% ->
   0%). It also made M record the pre-error bit.

5. Readout noise (``_record_measurement``). ``M(p)`` / ``MX(p)`` flip
   probabilities were silently dropped. They are now a classical flip on the
   reported bit only, never a diagram element. With both fixes, the cultivation
   circuit's kept rate matches tsim's own detector sampler.

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

KNOWN LIMITATION -- live-lane R of a superposed qubit. A live-lane reset caps
the old wire with an ``m[i]`` spider that binds to ``y``. If the wire was left in
superposition by an H with no measurement in between, that binding is
consistent only half the time: ``R 0 1; H 0; R 0; H 1`` keeps ~50% of shots.
Probably the source of big2's noiseless ~15% loss; not yet verified.
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


_RESETS = frozenset({"R", "RX"})
_PAULIS = "IXYZ"


def _next_op_is_reset(gates: list) -> list[set[int]]:
    """For each gate, the targets whose next non-noise operation is a reset."""
    marks: list[set[int]] = [set() for _ in gates]
    reset_next: set[int] = set()
    for idx in reversed(range(len(gates))):
        name = gates[idx].name
        if name in _NON_QUBIT_OPS or name in _NOISE_1Q or name in _NOISE_2Q:
            continue
        qubits = [t.qubit_value for t in gates[idx].targets_copy()]
        marks[idx] = {q for q in qubits if q in reset_next}
        if name in _RESETS:
            reset_next.update(qubits)
        else:
            reset_next.difference_update(qubits)
    return marks


def _live_leg_marginal(name: str, args: list[float], live_is_first: bool) -> tuple[str, list[float]]:
    """The exact 1-qubit channel a 2-qubit channel induces on its live leg.

    Exact only because the other leg is inert: its Pauli component is discarded,
    so the joint distribution collapses to the live leg's marginal.
    """
    if name == "DEPOLARIZE2":
        return "DEPOLARIZE1", [4 * args[0] / 5]
    pairs = [(a, b) for a in _PAULIS for b in _PAULIS if (a, b) != ("I", "I")]
    marginal = dict.fromkeys("XYZ", 0.0)
    for (a, b), prob in zip(pairs, args):
        pauli = a if live_is_first else b
        if pauli != "I":
            marginal[pauli] += prob
    return "PAULI_CHANNEL_1", [marginal["X"], marginal["Y"], marginal["Z"]]


def strip_inert_noise(circuit: tsim.Circuit) -> tsim.Circuit:
    """Drop noise acting only on qubits between a measurement and their next reset.

    Such a channel is a physical no-op: it cannot change the already-recorded
    measurement, and the reset discards the state it touches. Verified against
    stim on the d=3 cultivation circuit -- removing 107 such targets moved the
    detector rates by 8e-4 against a sampling sigma of 7.9e-4.

    gate_by_gate is *not* neutral to them. It emits MX as ``H q; M q``, dropping
    the trailing h of tsim's ``h; m; h`` (tsim/core/instructions.py:1054-1058),
    so between an MX and its reset the wire is left in the wrong basis frame and
    a Pauli there acts as the wrong Pauli. Between an M and a reset, any X or Z
    component makes the live-lane reset's ``m[i]`` binding contradict the
    diagram, so every later Hadamard sees two zero amplitudes. Removing the dead
    channels keeps the physics and stops both divergences being exercised.

    After an MX the window runs to the next operation on the qubit, whatever it
    is. After an M it opens only if that next operation is a reset: M-then-use
    is simulated correctly, so noise before a later gate is physical and kept.
    A 2-qubit channel with one dead leg becomes the exact 1-qubit marginal on
    its live leg, so no error component is left inside the window.

    NOTE: idempotent, and called from both preprocessing and gate_by_gate. They
    must walk the identical gate stream or the split and m[]/rec[] indices
    desynchronise, so neither may skip it.
    """
    gates = list(circuit)
    reset_next = _next_op_is_reset(gates)
    out = tsim.Circuit()
    dead: set[int] = set()
    for idx, gate in enumerate(gates):
        name = gate.name
        if name in _NON_QUBIT_OPS:
            out.append(gate)
            continue
        qubits = [t.qubit_value for t in gate.targets_copy()]
        args = gate.gate_args_copy()
        if name in _NOISE_1Q:
            live = [q for q in qubits if q not in dead]
            if live:
                out.append(name=name, targets=live, arg=args, tag=gate.tag)
            continue
        if name in _NOISE_2Q:
            live: list[int] = []
            for a, b in zip(qubits[0::2], qubits[1::2]):
                if a not in dead and b not in dead:
                    live += [a, b]
                elif a not in dead or b not in dead:
                    marginal_name, marginal_args = _live_leg_marginal(
                        name, args, live_is_first=a not in dead)
                    out.append(name=marginal_name, targets=[a if a not in dead else b],
                               arg=marginal_args, tag=gate.tag)
            if live:
                out.append(name=name, targets=live, arg=args, tag=gate.tag)
            continue
        for q in qubits:
            dead.discard(q)
        if name == "MX":
            dead.update(qubits)
        elif name == "M":
            dead.update(reset_next[idx])
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

# Which error bits of a channel carry an X component, as offsets from e_start.
# tsim splits every Pauli channel into one Z spider and one X spider per qubit
# (tsim/core/instructions.py:636-671, :706, :722); only the X ones flip a
# Z-basis label. Offset // 2 indexes into the op's qubit tuple.
_X_BIT_OFFSETS = {
    "X_ERROR": (0,),
    "Z_ERROR": (),
    "DEPOLARIZE1": (1,),
    "PAULI_CHANNEL_1": (1,),
    "DEPOLARIZE2": (1, 3),
    "PAULI_CHANNEL_2": (1, 3),
}


def _apply_error_flips(y: np.ndarray, op: dict, e_sample: np.ndarray) -> None:
    """XOR the sampled X components of one noise channel into the y labels.

    The channel already sits in the diagram as parameterized spiders, so its
    amplitude contribution is handled. But y is the Z-basis label that
    split_circuit_reduce post-selects the output wires on, and an X or Y error
    moves the wire to a different basis state. Leaving y stale makes the
    post-selection contradict the diagram: every amplitude vanishes, so the next
    Hadamard on any *other* qubit sees p0 == p1 == 0 and the shot is discarded.
    Any measurement of the qubit in between also records the pre-error bit.
    """
    for offset in _X_BIT_OFFSETS[op["name"]]:
        q = op["qubits"][offset // 2]
        y[:, q] ^= e_sample[:, op["e_start"] + offset]


def _record_measurement(dics, outcome: np.ndarray, gate, shots: int) -> None:
    """Append one measurement, applying M(p) / MX(p) readout noise to the report only.

    tsim builds M(p) as ``X^e; M; X^e`` (tsim/core/instructions.py:818-839): the
    recorded bit flips and the post-measurement state does not. That puts an error
    spider between the measurement and the next reset, the window strip_inert_noise
    exists to keep empty, so preprocessing leaves it out of the diagram and the flip
    is applied here as a classical XOR on the reported bit. rec[i] keeps the true
    outcome so the diagram's post-selection stays consistent with y.
    """
    dics.measurement_rec.append(outcome.copy())
    args = gate.gate_args_copy()
    flip = (np.random.random(shots) < args[0]).astype(np.uint8) if args else 0
    dics.reported_rec.append(outcome ^ flip)


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
    num_noise = 0
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
                _record_measurement(all_dicts, y[:, q], gate, shots)

            if gate.name == "MX":
                q = targets[i].qubit_value

                # HADAMARD (must run before recording this MX's own measurement)
                compiled_gs, e_len = split_circs[num_Had]
                y = perform_had(all_dicts, compiled_gs, num_qubits, y,
                                targets[i].qubit_value, passed, e_sample, e_len, shots)
                num_Had += 1

                # Measure
                _record_measurement(all_dicts, y[:, q], gate, shots)

            # X_ERROR / Z_ERROR / DEPOLARIZE1 / DEPOLARIZE2 / PAULI_CHANNEL_*
            # are driven symbolically: their parameter spiders sit in split_circs
            # and are substituted via e_bits in perform_had. Their X components
            # still have to be XOR'd into y here -- see _apply_error_flips.
            # NOTE: the i % 2 gate and the name sets mirror preprocessing exactly,
            # so num_noise stays aligned with the noise_ops it built.
            if gate.name in _NOISE_1Q or (gate.name in _NOISE_2Q and i % 2 == 0):
                op = noise_ops[num_noise]
                num_noise += 1
                if op["name"] != gate.name:
                    raise RuntimeError(
                        f"noise_ops desynchronised: expected {gate.name}, "
                        f"got {op['name']} at index {num_noise - 1}"
                    )
                _apply_error_flips(y, op, e_sample)

        if gate.name == "DETECTOR":
            # t.value stays the raw (negative) stim offset; reported_rec is a
            # measurement-ordered column list, so negative indexing resolves the
            # right measurement. The zeros seed avoids in-place xor on a stored column.
            col = functools.reduce(
                np.bitwise_xor,
                (all_dicts.reported_rec[t.value] for t in targets),
                np.zeros(shots, dtype=np.uint8),
            )
            new_detectors.append(col)
            k = len(new_detectors) - 1
            ref = detectors[k] if detectors is not None else int(col[0])
            passed &= (col == ref)

        if gate.name == "OBSERVABLE_INCLUDE":
            col = functools.reduce(
                np.bitwise_xor,
                (all_dicts.reported_rec[t.value] for t in targets),
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
        # measurement_rec is the true outcome, bound to rec[i] in the diagram;
        # reported_rec is what DETECTOR / OBSERVABLE_INCLUDE see, after readout noise.
        self.reset_vals = []
        self.new_detectors = []
        self.measurement_rec = []
        self.reported_rec = []

def create_y(value: int, y: list, q: int, strings: list) -> list:
    y_alt = y.copy()
    y_alt[q] = value

    fractions = [Fraction(n) for n in y_alt]
    val = dict(zip(strings, fractions))

    return fractions
