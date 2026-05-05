import numpy as np
from tsim.core.graph import evaluate_graph
import tsim
import pyzx_param as zx
from fractions import Fraction
import random
import string
import itertools

from pyzx_param.graph.graph_s import GraphS


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


def comp_amplitude(val_param: dict[str, Fraction], circs: list[GraphS], n_qubits: int, ) -> complex:
    """
    Compute <x|C|0> by contracting the ZX graph of the circuit postselected on x.

    The trick: to get <x|C|0>, we append the bra <x| to the circuit as Z-phase
    gates on the outputs, then contract the whole diagram to a scalar.
    """
    amplitude = 0
    for circ in circs:
        # Extract the scalar using the parameters
        amplitude += circ.scalar.evaluate_scalar(val_param)

    return amplitude / (np.sqrt(2) ** n_qubits)


# ---------------------------------------------------------------------------
# Preprocessing
#  ---------------------------------------------------------------------------

def split_circuit_reduce(circ_until_now: tsim.Circuit, y: list[int], num_qubits: int, reset_list: list):
    """
    Compute <x|C|0> by contracting the ZX graph of the circuit postselected on x.

    The trick: to get <x|C|0>, we append the bra <x| to the circuit as Z-phase
    gates on the outputs, then contract the whole diagram to a scalar.
    """
    g = circ_until_now.diagram("pyzx")

    last_vertices  = {}
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

    # Post-select outputs on the bitstring x by plugging in Z[0] or Z[pi] spiders
    # A Z-spider with phase 0 at an output wire selects |0>, phase pi selects |1>
    for qubit, bit in enumerate(y):
        if qubit in last_vertices:
            out_vertex = last_vertices[qubit]
            # phase = Fraction(0) if bit == 0 else Fraction(1)  # 0 = |0>, pi = |1>
            # Insert an X-spider with the parameter phase as the output
            g.set_type(out_vertex, zx.VertexType.X)
            g.add_params(out_vertex, y[qubit])

    # Simplify with ParamZX's parameterised reduction and stab decomp
    zx.full_reduce(g, paramSafe=True)
    gs = zx.simulate.find_stabilizer_decomp(g)
    return gs

def preprocessing(circuit: tsim.Circuit) :
    circ_until_now = tsim.Circuit()
    num_qubits = circuit.num_qubits
    y = generate_labels(num_qubits)
    reset_list = [{f"m[{i}]"} for i in range(40)]
    is_initialized = [False] * num_qubits
    all_sub_circuits = []

    num_gates = len(circuit)
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

                all_sub_circuits.append(split_circuit_reduce(circ_until_now, y, num_qubits, reset_list))

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
                is_initialized[q] = True

            elif gate.name == "RX":
                q = targets[i].qubit_value
                if not is_initialized[q]:
                    circ_until_now.append_from_stim_program_text(f"RX {q}")
                    is_initialized[q] = True
                else:
                #     circ_until_now.append_from_stim_program_text(f"H {q}")
                #     all_sub_circuits.append(split_circuit_reduce(circ_until_now, y, num_qubits, reset_list))
                    circ_until_now.append_from_stim_program_text(f"RX {q}")
                    # circ_until_now.append_from_stim_program_text(f"X {q}")

            elif gate.name == "M":
                q = targets[i].qubit_value
                circ_until_now.append_from_stim_program_text(f"M {q}")

            elif gate.name == "MX":
                q = targets[i].qubit_value

                # HADAMARD
                circ_until_now.append_from_stim_program_text(f"H {q}")
                all_sub_circuits.append(split_circuit_reduce(circ_until_now, y, num_qubits, reset_list))

                # Measure
                circ_until_now.append_from_stim_program_text(f"M {q}")

                # # HADAMARD
                # circ_until_now.append_from_stim_program_text(f"H {q}")
                # all_sub_circuits.append(split_circuit_reduce(circ_until_now, y, num_qubits, reset_list))

            elif gate.name == "X_ERROR":
                q = targets[i].qubit_value
                prob = gate.gate_args_copy()
                circ_until_now.append_from_stim_program_text(f"X_ERROR({prob[0]}) {q}")

            elif gate.name == "Z_ERROR":
                q = targets[i].qubit_value
                prob = gate.gate_args_copy()
                circ_until_now.append_from_stim_program_text(f"Z_ERROR({prob[0]}) {q}")

        # if gate.name == "DETECTOR":
        #     targets = gate.targets_copy()
        #     targ_s = ""
        #     if gate.name == "DETECTOR":
        #         arr = [" rec"] * len(targets)
        #         for i, val in enumerate(targets):
        #             targ_s += arr[i] + f"[{val.value}]"
        #
        #     circ_until_now.append_from_stim_program_text(f"DETECTOR{targ_s}")

    return all_sub_circuits


# ---------------------------------------------------------------------------
# Algorithm for gate by gate
# ---------------------------------------------------------------------------

def gate_by_gate(circuit: tsim.Circuit, split_circs: list[list[GraphS]], detectors: list = None):
    circ_until_now = tsim.Circuit()
    num_qubits = circuit.num_qubits
    y = [0] * num_qubits
    all_dicts = AllDictionaries(num_qubits)
    strings = generate_labels(num_qubits)
    is_initialized = [False] * num_qubits
    # dictionary = dict(zip(strings, y))
    # circ_until_now.append_from_stim_program_text(f"R {' '.join(str(q) for q in range(num_qubits))}")
    new_detectors = []
    num_Had = 0
    num_gates = len(circuit)
    for gate in circuit:

        targets = gate.targets_copy()
        for i in range(0, len(targets)):

            if gate.name in ("CX", "CNOT", "ZCX") and i%2==0:
                a = targets[i].qubit_value
                b = targets[i + 1].qubit_value
                y[b] ^= y[a]
                circ_until_now.append_from_stim_program_text(f"CX {a} {b}")

            if gate.name == "H":
                y = perform_had(all_dicts, split_circs[num_Had], num_qubits, y, targets[i].qubit_value)
                num_Had += 1

            if gate.name == "X":   #Dont split
                q = targets[i].qubit_value
                y[q] ^= 1

            if gate.name == "R":
                q = targets[i].qubit_value
                y[q] = 0
                is_initialized[q] = True

            if gate.name == "RX":

                q = targets[i].qubit_value
                # HADAMARD
                if not is_initialized[q]:
                    y[q] = 0 if random.random() < 0.5 else 1
                    is_initialized[q] = True
                else:
                    # y = perform_had(all_dicts, split_circs[num_Had], num_qubits, y, targets[i].qubit_value)
                    # num_Had += 1

                    # # --- Initialize into |+> --- # always 50/50 so just sample directly
                    y[q] = 0 if random.random() < 0.5 else 1


            if gate.name == "M":
                q = targets[i].qubit_value
                all_dicts.measurement_rec.append(y[q])

            if gate.name == "MX":
                # print("num had:" + str(num_Had))
                # print(y)

                # HADAMARD
                y = perform_had(all_dicts, split_circs[num_Had], num_qubits, y, targets[i].qubit_value)
                num_Had += 1

                # Measure
                all_dicts.measurement_rec.append(y[q])

                # # HADAMARD
                # y = perform_had(all_dicts, split_circs[num_Had], num_qubits, y, targets[i].qubit_value)
                # num_Had += 1

            if gate.name == "X_ERROR":
                q = targets[i].qubit_value
                prob = gate.gate_args_copy()

                if random.random() < prob[0]:
                    y[q] ^= 1

            if gate.name == "Z_ERROR":
                q = targets[i].qubit_value
                prob = gate.gate_args_copy()

        if gate.name == "DETECTOR":
            result = 0
            for i in targets:
                result ^= all_dicts.measurement_rec[i.value]

            new_detectors.append(result)
            if detectors is not None:
                if detectors[len(new_detectors) - 1] != new_detectors[-1]:
                    print("not passed")
                    return False, None, detectors


    if detectors is None:
        detectors = new_detectors

    return True, y, detectors


def perform_had(dics, split_circ: list[GraphS], num_qubits, y, q):
    rec = dict(zip(dics.rec_list, dics.measurement_rec))
    m = dict(zip(dics.reset_list, dics.reset_vals))

    val0 = create_y(0, y, q, dics.strings)
    val1 = create_y(1, y, q, dics.strings)

    z0 = comp_amplitude(val0 | rec | m, split_circ, num_qubits)
    z1 = comp_amplitude(val1 | rec | m, split_circ, num_qubits)

    denom = abs(z0) ** 2 + abs(z1) ** 2
    if denom == 0:
        denom = 1
    p = abs(z0) ** 2 / denom

    y[q] = 0 if random.random() < p else 1

    return y




# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
#

class AllDictionaries:
    def __init__(self, num_qubits):
        self.strings = generate_labels(num_qubits)
        self.rec_list = [f"rec[{i}]" for i in range(40)]
        self.reset_list = [f"m[{i}]" for i in range(40)]
        self.reset_vals = [0] * len(self.reset_list)
        self.new_detectors = []
        self.measurement_rec = []

def create_y(value: int, y: list, q: int, strings: list) -> dict[str, Fraction]:
    y_alt = y.copy()
    y_alt[q] = value

    fractions = [Fraction(n) for n in y_alt]
    val = dict(zip(strings, fractions))

    return val


# ---------------------------------------------------------------------------
# Amplitude calculator PREV
# ---------------------------------------------------------------------------
#
# def compute_amplitude(
#     x: list[int],
#     circuit_so_far: tsim.Circuit,
#     n_qubits: int
# ) -> complex:
#     """
#     Compute <x | C | 0> where C is the product of gates in circuit_so_far
#     (applied left-to-right, i.e. circuit_so_far[-1] is the outermost/latest gate).
#
#     We build the full 2^n state vector starting from |0...0> and apply each gate.
#     """
#     dim = 2 ** n_qubits
#
#     state = np.zeros(dim, dtype=complex)
#     state[0] = 1.0
#
#     # Initialise |0...0> if there is no prior saved matrix
#     # if prevMatrix is None:
#     #     state = np.zeros(dim, dtype=complex)
#     #     state[0] = 1.0
#     # else:
#     #     state = prevMatrix
#
#     # Apply gates in order U_1, U_2, ..., U_t
#     for gate in circuit_so_far:
#         state = _apply_gate(state, gate, n_qubits)
#
#     # prevMatrix = state
#     # Return amplitude <x|state>
#     x_index = _bitstring_to_index(x)
#     return (complex(state[x_index]), state)
#
#
#
# def compute_amplitude(x: list[int], circuit_so_far: tsim.Circuit, n_qubits: int) -> complex:
#     """
#     Compute <x|C|0> by contracting the ZX graph of the circuit postselected on x.
#
#     The trick: to get <x|C|0>, we append the bra <x| to the circuit as Z-phase
#     gates on the outputs, then contract the whole diagram to a scalar.
#     """
#     # Build PyZX circuit from QASM
#     g = circuit_so_far.diagram("pyzx")
#
#     last_vertices = {}
#     for v in g.vertices():
#         q = g.qubit(v)
#         if q not in last_vertices or g.row(v) > g.row(last_vertices[q]):
#             last_vertices[q] = v
#
#     # Post-select outputs on the bitstring x by plugging in Z[0] or Z[pi] spiders
#     # A Z-spider with phase 0 at an output wire selects |0>, phase pi selects |1>
#     for qubit, bit in enumerate(x):
#         out_vertex = last_vertices[qubit]
#         phase = Fraction(0) if bit == 0 else Fraction(1)  # 0 = |0>, pi = |1>
#         # Insert an X-spider with the right phase before the output
#         g.set_type(out_vertex, zx.VertexType.X)
#         # g.set_phase(out_vertex, phase)
#         g.add_params(out_vertex, 'a')
#         # else: g.add_params(out_vertex, 'b')
#
#     zx.full_reduce(g, paramSafe=True)
#
#     # Extract the scalar — PyZX stores it as g.scalar
#     amplitude = complex(g.scalar.to_number())
#     return amplitude / (np.sqrt(2) ** n_qubits)

# def _apply_gate(state: np.ndarray, gate: Gate, n_qubits: int) -> np.ndarray:
#     """Apply a single gate to the state vector."""
#     if isinstance(gate, HadamardGate):
#         return _apply_single_qubit(state, hadamard_matrix(), gate.target, n_qubits)
#     elif isinstance(gate, ZRotationGate):
#         return _apply_single_qubit(state, z_rotation_matrix(gate.alpha), gate.target, n_qubits)
#     elif isinstance(gate, CNOTGate):
#         return _apply_cnot(state, gate.control, gate.target, n_qubits)
#     elif isinstance(gate, ResetGate):
#         return _apply_reset(state, gate.target, n_qubits)
#     else:
#         raise ValueError(f"Unknown gate type: {type(gate)}")
#
# def _apply_single_qubit(
#     state: np.ndarray, mat: np.ndarray, target: int, n_qubits: int
# ) -> np.ndarray:
#     """Apply a 2x2 unitary to qubit `target` of the state vector."""
#     # Reshape into tensor, contract, reshape back
#     state = state.reshape([2] * n_qubits)
#     # Move target axis to front for easy contraction
#     state = np.moveaxis(state, target, 0)
#     state = np.einsum("ij,j...->i...", mat, state)
#     state = np.moveaxis(state, 0, target)
#     return state.reshape(-1)
#
#
# def _apply_cnot(
#     state: np.ndarray, control: int, target: int, n_qubits: int
# ) -> np.ndarray:
#     """Apply CNOT with given control and target qubits."""
#     state = state.reshape(2, n_qubits)
#     # Flip target qubit wherever control qubit == 1
#     slices_ctrl1 = [slice(None)] * n_qubits
#     slices_ctrl1[control] = 1
#     sub = state[tuple(slices_ctrl1)]
#     # Swap |0> and |1> along target axis within the control=1 subspace
#     sub = np.flip(sub, axis=target if target < control else target - 1)
#     state[tuple(slices_ctrl1)] = sub
#     return state.reshape(-1)
#
# def _apply_reset(state: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
#     state = state.reshape([2] * n_qubits)
#     # Zero out all amplitudes where target qubit is |1⟩
#     idx = [slice(None)] * n_qubits
#     idx[target] = 1
#     state[tuple(idx)] = 0
#     state = state.reshape(-1)
#     # Renormalise
#     norm = np.linalg.norm(state)
#     if norm > 1e-15:
#         state /= norm
#     return state
