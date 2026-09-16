"""Differential test: gate_by_gate against stim, on random Clifford circuits.

    python -m unittest test_vs_stim

This is the harness that found the MX-then-use defect (and that refuted the
theory that live-lane RX needed its own split -- see gate_by_gate's module
header). It is kept separate from test_gbg.py because it is slower and because
it needs stim as a reference.

WHAT IT COMPARES, AND WHY IT MATTERS
------------------------------------
It compares **measurement records**, not the final ``y`` register.

Comparing ``y`` against a trailing ``M`` is invalid: ``y`` is gate_by_gate's
internal Z-basis label, and after an ``MX`` the wire is left in an X-eigenstate,
so the two are not the same observable. An earlier version of this harness did
exactly that and produced confident, entirely spurious mismatches. Measurement
records are apples-to-apples -- same order, same semantics, no frame ambiguity.

Records are pulled out of ``AllDictionaries.measurement_rec`` via a subclass,
since ``gate_by_gate`` does not return them.

Circuits are Clifford only, so stim is an exact reference. Raise ``SHOTS`` for a
tighter comparison; the statistical floor is ~sqrt(0.25 / SHOTS) per outcome.
"""
import random
import unittest
import warnings

import numpy as np
import stim
import tsim

import gate_by_gate as gbg

SHOTS = 40_000
# ~sqrt(0.25/40000) = 0.0025 per outcome; 0.02 is a comfortable ~8 sigma even
# after taking a max over every outcome in the record.
TOL = 0.02
SEED = 11

_MEASURE = ("M", "MX", "MZ")
_RESET = ("R", "RX")
_NON_QUBIT = ("DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "QUBIT_COORDS", "SHIFT_COORDS")


def has_mx_then_use(circuit_text: str) -> bool:
    """True if any qubit is acted on after an MX without an intervening reset.

    This is gate_by_gate's known-wrong pattern: preprocessing emits MX as
    ``H q; M q``, dropping the trailing h of tsim's ``h; m; h``, so the wire is
    left in the wrong basis frame. Harmless if the qubit is reset before reuse.
    """
    open_mx: set[int] = set()
    for line in circuit_text.splitlines():
        parts = line.split()
        if not parts or parts[0] in _NON_QUBIT:
            continue
        name, qubits = parts[0], [int(x) for x in parts[1:]]
        for q in qubits:
            if q in open_mx and name not in _RESET:
                return True
            if name in _RESET or name in _MEASURE:
                open_mx.discard(q)
        if name == "MX":
            open_mx.update(qubits)
    return False


def random_clifford(rng: random.Random, num_qubits: int, depth: int) -> str:
    """A random Clifford circuit that ends with at least one measurement."""
    lines = [f"{rng.choice(_RESET)} {q}" for q in range(num_qubits)]
    for _ in range(depth):
        op = rng.choices(
            ["CX", "H", "S", "R", "RX", "M", "MX"], weights=[5, 3, 2, 1, 2, 2, 2]
        )[0]
        if op == "CX":
            a, b = rng.sample(range(num_qubits), 2)
            lines.append(f"CX {a} {b}")
        else:
            lines.append(f"{op} {rng.randrange(num_qubits)}")
    if not any(ln.startswith(("M ", "MX ")) for ln in lines):
        lines.append("M 0")
    return "\n".join(lines)


def stim_record_dist(circuit_text: str) -> tuple[dict[str, float], int]:
    """Distribution over stim's full measurement record."""
    sample = stim.Circuit(circuit_text).compile_sampler().sample(SHOTS)
    keys = ["".join(map(str, row.astype(int))) for row in sample]
    values, counts = np.unique(keys, return_counts=True)
    return dict(zip(values, counts / SHOTS)), sample.shape[1]


def gbg_record_dist(circuit_text: str) -> tuple[dict[str, float], float, int]:
    """Distribution over gate_by_gate's measurement record, among kept shots."""
    circuit = tsim.Circuit(circuit_text)
    splits, noise_ops = gbg.preprocessing(circuit)

    captured = {}
    original = gbg.AllDictionaries

    class _Spy(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["dicts"] = self

    gbg.AllDictionaries = _Spy
    try:
        passed, _y, _det, _obs = gbg.gate_by_gate(
            circuit, splits, noise_ops, shots=SHOTS
        )
    finally:
        gbg.AllDictionaries = original

    record = captured["dicts"].measurement_rec
    if not record:
        return {}, 0.0, 0
    columns = np.stack(record, axis=1)
    keys = ["".join(map(str, columns[s])) for s in range(SHOTS) if passed[s]]
    if not keys:
        return {}, 0.0, columns.shape[1]
    values, counts = np.unique(keys, return_counts=True)
    return dict(zip(values, counts / len(keys))), sum(passed) / SHOTS, columns.shape[1]


def max_deviation(circuit_text: str) -> tuple[float, float]:
    """Return (max per-outcome probability difference vs stim, kept fraction)."""
    reference, n_ref = stim_record_dist(circuit_text)
    measured, kept, n_got = gbg_record_dist(circuit_text)
    if n_ref != n_got:
        raise AssertionError(
            f"record length mismatch: stim={n_ref} gate_by_gate={n_got}"
        )
    outcomes = set(reference) | set(measured)
    if not outcomes:
        return 1.0, kept
    return max(abs(measured.get(k, 0.0) - reference.get(k, 0.0)) for k in outcomes), kept


# A circuit the sweep found, reduced to the essentials: qubit 1 is measured with
# MX and then used by CX without an intervening reset.
MX_THEN_USE_CIRCUIT = """R 0
R 1
RX 2
M 2
MX 1
H 2
R 2
CX 1 2
M 0
M 1"""


class TestAgainstStim(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_matches_stim(self):
        """Random Clifford circuits, excluding the known-wrong MX-then-use pattern."""
        rng = random.Random(SEED)
        checked = 0
        while checked < 12:
            num_qubits = rng.choice([2, 3])
            text = random_clifford(rng, num_qubits, rng.choice([6, 8, 10]))
            if has_mx_then_use(text):
                continue
            np.random.seed(SEED + checked)
            with self.subTest(circuit=text):
                deviation, kept = max_deviation(text)
                self.assertLess(
                    deviation,
                    TOL,
                    f"distribution differs from stim by {deviation:.4f} "
                    f"(kept {kept:.3f})\n{text}",
                )
            checked += 1
        self.assertEqual(checked, 12)

    def test_discards_do_not_bias(self):
        """Shots lost to zero denominators must not skew the survivors.

        Some circuits legitimately discard a large fraction of shots. That is
        rejection sampling, so the kept shots must still match stim.
        """
        rng = random.Random(SEED)
        worst_kept = 1.0
        for i in range(12):
            num_qubits = rng.choice([2, 3])
            text = random_clifford(rng, num_qubits, rng.choice([6, 8, 10]))
            if has_mx_then_use(text):
                continue
            np.random.seed(SEED + 100 + i)
            deviation, kept = max_deviation(text)
            worst_kept = min(worst_kept, kept)
            if kept < 0.99:
                self.assertLess(
                    deviation, TOL,
                    f"circuit discarding {1 - kept:.1%} of shots also biased "
                    f"by {deviation:.4f}\n{text}",
                )
        self.assertLessEqual(worst_kept, 1.0)

    @unittest.expectedFailure
    def test_known_limitation_mx_then_use(self):
        """Documents the open MX-then-use defect.

        preprocessing emits MX as ``H q; M q``, dropping the trailing h of
        tsim's ``h; m; h`` (tsim/core/instructions.py:1054-1058). ``y`` holds
        one Z-basis label per wire and a true MX leaves an X-eigenstate, which
        no Z-label represents, so using the qubit before resetting it diverges.

        When this starts passing, MX has been fixed -- drop the decorator and
        fold the pattern back into test_matches_stim.
        """
        self.assertTrue(has_mx_then_use(MX_THEN_USE_CIRCUIT))
        np.random.seed(SEED)
        deviation, _kept = max_deviation(MX_THEN_USE_CIRCUIT)
        self.assertLess(deviation, TOL, f"differs from stim by {deviation:.4f}")


if __name__ == "__main__":
    unittest.main()
