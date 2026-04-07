"""
Gate-by-gate weak simulation by calculating amplitudes (Algorithm 3).

Simulates a quantum circuit C = U_k ... U_1 consisting of CNOT, Hadamard,
and Z(alpha) gates, and returns a sample from P(x) = |<x|C|0>|^2.
"""

import numpy as np
from typing import Union


# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------

def hadamard_matrix() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def cnot_matrix() -> np.ndarray:
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)


def z_rotation_matrix(alpha: float) -> np.ndarray:
    return np.array([[1, 0],
                     [0, np.exp(1j * alpha)]], dtype=complex)


# ---------------------------------------------------------------------------
# Gate dataclasses
# ---------------------------------------------------------------------------

class HadamardGate:
    def __init__(self, target: int):
        self.target = target  # qubit index

    def __repr__(self):
        return f"H(q={self.target})"


class CNOTGate:
    def __init__(self, control: int, target: int):
        self.control = control
        self.target = target

    def __repr__(self):
        return f"CNOT(ctrl={self.control}, tgt={self.target})"


class ZRotationGate:
    def __init__(self, target: int, alpha: float):
        self.target = target
        self.alpha = alpha

    def __repr__(self):
        return f"Z({self.alpha:.4f}, q={self.target})"


Gate = Union[HadamardGate, CNOTGate, ZRotationGate]


# ---------------------------------------------------------------------------
# Amplitude calculator
# ---------------------------------------------------------------------------

def compute_amplitude(
    x: list[int],
    circuit_so_far: list[Gate],
    n_qubits: int,
) -> complex:
    """
    Compute <x | C | 0> where C is the product of gates in circuit_so_far
    (applied left-to-right, i.e. circuit_so_far[-1] is the outermost/latest gate).

    We build the full 2^n state vector starting from |0...0> and apply each gate.
    """
    dim = 2 ** n_qubits

    # Initialise |0...0>
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    # Apply gates in order U_1, U_2, ..., U_t
    for gate in circuit_so_far:
        state = _apply_gate(state, gate, n_qubits)

    # Return amplitude <x|state>
    x_index = _bitstring_to_index(x)
    return complex(state[x_index])


def _apply_gate(state: np.ndarray, gate: Gate, n_qubits: int) -> np.ndarray:
    """Apply a single gate to the state vector."""
    if isinstance(gate, HadamardGate):
        return _apply_single_qubit(state, hadamard_matrix(), gate.target, n_qubits)
    elif isinstance(gate, ZRotationGate):
        return _apply_single_qubit(state, z_rotation_matrix(gate.alpha), gate.target, n_qubits)
    elif isinstance(gate, CNOTGate):
        return _apply_cnot(state, gate.control, gate.target, n_qubits)
    else:
        raise ValueError(f"Unknown gate type: {type(gate)}")


def _apply_single_qubit(
    state: np.ndarray, mat: np.ndarray, target: int, n_qubits: int
) -> np.ndarray:
    """Apply a 2x2 unitary to qubit `target` of the state vector."""
    # Reshape into tensor, contract, reshape back
    state = state.reshape([2] * n_qubits)
    # Move target axis to front for easy contraction
    state = np.moveaxis(state, target, 0)
    state = np.einsum("ij,j...->i...", mat, state)
    state = np.moveaxis(state, 0, target)
    return state.reshape(-1)


def _apply_cnot(
    state: np.ndarray, control: int, target: int, n_qubits: int
) -> np.ndarray:
    """Apply CNOT with given control and target qubits."""
    state = state.reshape([2] * n_qubits)
    # Flip target qubit wherever control qubit == 1
    slices_ctrl1 = [slice(None)] * n_qubits
    slices_ctrl1[control] = 1
    sub = state[tuple(slices_ctrl1)]
    # Swap |0> and |1> along target axis within the control=1 subspace
    sub = np.flip(sub, axis=target if target < control else target - 1)
    state[tuple(slices_ctrl1)] = sub
    return state.reshape(-1)


def _bitstring_to_index(bits: list[int]) -> int:
    """Convert a bitstring [b_0, b_1, ..., b_{n-1}] to an integer index."""
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    return idx


# ---------------------------------------------------------------------------
# CNOT classical update
# ---------------------------------------------------------------------------

def cnot_update_sample(y: list[int], gate: CNOTGate) -> list[int]:
    """
    Apply CNOT to the classical sample vector y in-place:
    target bit is XORed with control bit.
    """
    y = y.copy()
    y[gate.target] ^= y[gate.control]
    return y


# ---------------------------------------------------------------------------
# Algorithm 3: SAMPLE
# ---------------------------------------------------------------------------

def sample(gates: list[Gate], n_qubits: int, rng: np.random.Generator | None = None) -> list[int]:
    """
    Gate-by-gate weak simulation (Algorithm 3).

    Parameters
    ----------
    gates     : Ordered list of gates [U_1, ..., U_k].
    n_qubits  : Number of qubits.
    rng       : Optional numpy random generator (for reproducibility).

    Returns
    -------
    y : A bitstring sampled from P(x) = |<x|C|0>|^2.
    """
    if rng is None:
        rng = np.random.default_rng()

    # C tracks gates applied so far (for amplitude queries); start with identity
    circuit_so_far: list[Gate] = []

    # y is the running sample, initialised to |0...0>
    y = [0] * n_qubits

    for gate in gates:  # forward pass: t = 1 to k
        circuit_so_far.append(gate)           # C := U_t * C

        if isinstance(gate, CNOTGate):
            # Update sample classically
            y = cnot_update_sample(y, gate)

        elif isinstance(gate, HadamardGate):
            q = gate.target

            # Build bitstrings y_0 and y_1 (differ only at qubit q)
            y0 = y.copy(); y0[q] = 0
            y1 = y.copy(); y1[q] = 1

            # Compute amplitudes z0, z1
            z0 = compute_amplitude(y0, circuit_so_far, n_qubits)
            z1 = compute_amplitude(y1, circuit_so_far, n_qubits)

            # Compute probability p = |z0|^2 / (|z0|^2 + |z1|^2)
            denom = abs(z0) ** 2 + abs(z1) ** 2
            if denom < 1e-15:
                # Degenerate case: assign uniform
                p = 0.5
            else:
                p = abs(z0) ** 2 / denom

            # Sample y_q
            y[q] = 0 if rng.random() < p else 1

        # Z(alpha): no update needed — it only contributes a phase

    return y


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Bell-state circuit ===")
    # Produces |Φ+> = (|00> + |11>) / sqrt(2)
    # Expected: samples should be 00 or 11 with equal probability
    bell_circuit: list[Gate] = [
        HadamardGate(target=0),
        CNOTGate(control=0, target=1),
    ]

    rng = np.random.default_rng(42)
    counts = {"00": 0, "11": 0, "01": 0, "10": 0}
    N = 1000
    for _ in range(N):
        result = sample(bell_circuit, n_qubits=2, rng=rng)
        key = "".join(map(str, result))
        counts[key] = counts.get(key, 0) + 1

    print(f"Samples from {N} runs:")
    for k, v in sorted(counts.items()):
        if v > 0:
            print(f"  |{k}>: {v} ({100*v/N:.1f}%)")

    print()
    print("=== GHZ-state circuit (3 qubits) ===")
    # (|000> + |111>) / sqrt(2) — should only see 000 or 111
    ghz_circuit: list[Gate] = [
        HadamardGate(target=0),
        CNOTGate(control=0, target=1),
        CNOTGate(control=0, target=2),
    ]

    rng = np.random.default_rng(0)
    counts_ghz: dict[str, int] = {}
    for _ in range(1000):
        result = sample(ghz_circuit, n_qubits=3, rng=rng)
        key = "".join(map(str, result))
        counts_ghz[key] = counts_ghz.get(key, 0) + 1

    print(f"Samples from {N} runs:")
    for k, v in sorted(counts_ghz.items()):
        if v > 0:
            print(f"  |{k}>: {v} ({100*v/N:.1f}%)")

    print()
    print("=== Single-qubit Z-rotation (phase gate) ===")
    # H then Z(pi) = H then Z gives |-> state, should always measure 1
    z_circuit: list[Gate] = [
        HadamardGate(target=0),
        ZRotationGate(target=0, alpha=np.pi),
    ]
    rng = np.random.default_rng(7)
    z_counts: dict[str, int] = {}
    for _ in range(200):
        result = sample(z_circuit, n_qubits=1, rng=rng)
        key = str(result[0])
        z_counts[key] = z_counts.get(key, 0) + 1

    print("H then Z(π) — expected |-> so output=1 always:")
    for k, v in sorted(z_counts.items()):
        print(f"  |{k}>: {v} ({100*v/200:.1f}%)")
