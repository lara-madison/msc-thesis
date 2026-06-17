import gate_by_gate as gbg
import tsim

circuit = tsim.Circuit("""
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
""")

splits, noise_ops = gbg.preprocessing(circuit)

runs = 100
passes = fails = crashes = 0
for i in range(runs):
    try:
        passed, results, detects, obs = gbg.gate_by_gate(circuit, splits, noise_ops, shots=1)
        if passed[0]:
            passes += 1
        else:
            fails += 1
    except AssertionError:
        crashes += 1

print(f"shots=1 x {runs}: passes={passes} fails={fails} denom0_crashes={crashes}")
