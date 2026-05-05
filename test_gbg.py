import unittest
import gate_by_gate as gbg
import tsim

class MyTestCase(unittest.TestCase):
    def test_x_gate_flips_qubit(self):
        # X on qubit 0 should always produce |1>
        circuit = tsim.Circuit("R 0\nX 0")
        circ_splits = gbg.preprocessing(circuit)
        counts = {}
        N, num, j = 100, 0, 0
        detects = None
        for _ in range(N):
            passed, result, detects = gbg.gate_by_gate(circuit, circ_splits, detects)
            j += 1
            if passed:
                num += 1
                key = "".join(map(str, result))
                counts[key] = counts.get(key, 0) + 1
        for k, v in sorted(counts.items()):
            self.assertTrue(k == "1" and v == N)

    def test_h_h(self):
        # X on qubit 0 should always produce |1>
        circuit = tsim.Circuit("R 0\nH 0\nH 0")
        circ_splits = gbg.preprocessing(circuit)
        counts = {}
        N, num, j = 100, 0, 0
        detects = None
        for _ in range(N):
            passed, result, detects = gbg.gate_by_gate(circuit, circ_splits, detects)
            j += 1
            if passed:
                num += 1
                key = "".join(map(str, result))
                counts[key] = counts.get(key, 0) + 1
        for k, v in sorted(counts.items()):
            self.assertTrue(k == "0" and v == N)


    def test_bell_state(self):
        # H then CNOT produces Bell state — only |00> or |11> should appear
        circuit = tsim.Circuit("R 0 1\nH 0\nCX 0 1")
        circ_splits = gbg.preprocessing(circuit)

        circ_splits = gbg.preprocessing(circuit)
        counts = {}
        N, num, j = 100, 0, 0
        detects = None
        for _ in range(N):
            passed, result, detects = gbg.gate_by_gate(circuit, circ_splits, detects)
            j += 1
            if passed:
                num += 1
                key = "".join(map(str, result))
                counts[key] = counts.get(key, 0) + 1
        for k, v in sorted(counts.items()):
            self.assertTrue(k == "11" or k == "00")

    def test_tricky_dect(self):
        # X on qubit 0 should always produce |1>
        circuit = tsim.Circuit("""
            RX 0
            R 1
            RX 2
            CX 0 1
            T_DAG 0
            H 1
            CX 2 1 2 0
            MX 2
            RX 2
            CX 2 1 2 0
            MX 2
            DETECTOR rec[-1] rec[-2]
            """)
        circ_splits = gbg.preprocessing(circuit)
        counts = {}
        N, num, j = 500, 0, 0
        detects = None
        for _ in range(N):
            passed, result, detects = gbg.gate_by_gate(circuit, circ_splits, detects)
            j += 1
            if passed:
                num += 1
                key = "".join(map(str, result))
                counts[key] = counts.get(key, 0) + 1
        self.assertTrue(num == N)


    def test_t(self):
        # H then CNOT produces Bell state — only |00> or |11> should appear
        circuit = tsim.Circuit("""
        RX 0
        T 0
        MX 0""")
        circ_splits = gbg.preprocessing(circuit)
        counts = {}
        N, num, j = 5000, 0, 0
        detects = None
        for _ in range(N):
            passed, result, detects = gbg.gate_by_gate(circuit, circ_splits, detects)
            j += 1
            if passed:
                num += 1
                key = "".join(map(str, result))
                counts[key] = counts.get(key, 0) + 1
        for k, v in sorted(counts.items()):
            if k == 0:
                print(100 * v / N)
                self.assertTrue(83 <= (100 * v / N) <= 88)
            elif k == 1:
                print(100 * v / N)
                self.assertTrue(12 <= (100 * v / N) <= 16)

if __name__ == '__main__':
    unittest.main()
