import unittest
import gate_by_gate as gbg
import pyzx_param as param
import tsim
import numpy as np

# <+|T|+> gives P(0) = cos^2(pi/8), P(1) = sin^2(pi/8).
P_T_ZERO = np.cos(np.pi / 8) ** 2   # 0.853553...
P_T_ONE = np.sin(np.pi / 8) ** 2    # 0.146447...


BIG2_CIRCUIT = """
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
        """


class MyTestCase(unittest.TestCase):
    def test_x_gate_flips_qubit(self):
        # X on qubit 0 should always produce |1>
        circuit = tsim.Circuit("R 0\nX 0")
        splits, noise_ops = gbg.preprocessing(circuit)
        counts = {}
        N, num = 10000, 0
        passed, results, detects, obs = gbg.gate_by_gate(circuit, splits, noise_ops, shots=N)
        for s in range(N):
            if passed[s]:
                num += 1
                key = "".join(map(str, results[s]))
                counts[key] = counts.get(key, 0) + 1
        # Assert outside the loop: an empty counts dict would pass vacuously.
        self.assertEqual(num, N)
        self.assertEqual(counts, {"1": N})

    def test_h_h(self):
        # X on qubit 0 should always produce |1>
        circuit = tsim.Circuit("R 0\nH 0\nH 0")
        splits, noise_ops = gbg.preprocessing(circuit)
        counts = {}
        N, num = 1000, 0
        passed, results, detects, obs = gbg.gate_by_gate(circuit, splits, noise_ops, shots=N)
        for s in range(N):
            if passed[s]:
                num += 1
                key = "".join(map(str, results[s]))
                counts[key] = counts.get(key, 0) + 1
        # H H = identity, so every shot must come back to |0>.
        self.assertEqual(num, N)
        self.assertEqual(counts, {"0": N})


    def test_bell_state(self):
        # H then CNOT produces Bell state — only |00> or |11> should appear
        circuit = tsim.Circuit("R 0 1\nH 0\nCX 0 1")
        splits, noise_ops = gbg.preprocessing(circuit)
        counts = {}
        N, num = 1000, 0
        passed, results, detects, obs = gbg.gate_by_gate(circuit, splits, noise_ops, shots=N)
        for s in range(N):
            if passed[s]:
                num += 1
                key = "".join(map(str, results[s]))
                counts[key] = counts.get(key, 0) + 1
        self.assertEqual(num, N)
        self.assertEqual(set(counts), {"00", "11"})     # both branches must occur
        for k, v in counts.items():
            self.assertTrue(0.45 <= v / N <= 0.55, f"{k} at {v / N:.3f}, expected ~0.5")

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
        splits, noise_ops = gbg.preprocessing(circuit)
        counts = {}
        N, num = 50000, 0
        passed, results, detects, obs = gbg.gate_by_gate(circuit, splits, noise_ops, shots=N)
        for s in range(N):
            if passed[s]:
                num += 1
                key = "".join(map(str, results[s]))
                counts[key] = counts.get(key, 0) + 1
        self.assertTrue(num == N)


    def test_t(self):
        # H then CNOT produces Bell state — only |00> or |11> should appear
        circuit = tsim.Circuit("""
        RX 0
        T 0
        MX 0""")
        splits, noise_ops = gbg.preprocessing(circuit)
        counts = {}
        N, num = 500000, 0
        passed, results, detects, obs = gbg.gate_by_gate(circuit, splits, noise_ops, shots=N)
        for s in range(N):
            if passed[s]:
                num += 1
                key = "".join(map(str, results[s]))
                counts[key] = counts.get(key, 0) + 1
        # NOTE: keys are strings, so the old `if k == 0` never matched and this
        # test asserted nothing. N=500000 puts the binomial sigma near 0.05%,
        # so a 1% band is ~20 sigma and will not flake.
        self.assertEqual(num, N)
        self.assertEqual(set(counts), {"0", "1"})
        self.assertAlmostEqual(counts["0"] / N, P_T_ZERO, delta=0.01)
        self.assertAlmostEqual(counts["1"] / N, P_T_ONE, delta=0.01)


    def test_correctness(self):
        # H then CNOT produces Bell state — only |00> or |11> should appear
        circuit = tsim.Circuit("""
        RX 0
        T 0
        MX 0""")
        splits, noise_ops = gbg.preprocessing(circuit)
        counts = {}
        N, num = 500000, 0
        passed, results, detects, obs = gbg.gate_by_gate(circuit, splits, noise_ops, shots=N)
        for s in range(N):
            if passed[s]:
                num += 1
                key = "".join(map(str, results[s]))
                counts[key] = counts.get(key, 0) + 1
        # NOTE: keys are strings, so the old `if k == 0` never matched and this
        # test asserted nothing. N=500000 puts the binomial sigma near 0.05%,
        # so a 1% band is ~20 sigma and will not flake.
        self.assertEqual(num, N)
        self.assertEqual(set(counts), {"0", "1"})
        self.assertAlmostEqual(counts["0"] / N, P_T_ZERO, delta=0.01)
        self.assertAlmostEqual(counts["1"] / N, P_T_ONE, delta=0.01)

    def test_big(self):
        # X on qubit 0 should always produce |1>
        circuit = tsim.Circuit("""
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
         TICK
         RX 15 10 5 2 7 1
         TICK
         T_DAG 0 3 6 8 9 11 14
         TICK
         CX 1 0 2 3 5 6 7 8 10 9 15 14
         TICK
         CX 3 1 6 7 10 15
         TICK
         CX 6 3 10 11
         TICK
         CX 6 10
         TICK
         MX 6
         TICK
         RX 6
         TICK
         CX 6 10
         TICK
         CX 6 3 10 11
         TICK
         CX 3 1 6 7 10 15
         TICK
         CX 1 0 2 3 5 6 7 8 10 9 15 14
         TICK
         T 0 3 6 8 9 11 14
         TICK
         MX 15 10 5 2 7 1
         DETECTOR(1.60714, 0.75, 3, -1, -9) rec[-29] rec[-28] rec[-26] rec[-24] rec[-23] rec[-21] rec[-20] rec[-18] rec[-17] rec[-12] rec[-11] rec[-7]
         DETECTOR(4, 1, 4) rec[-6]
         DETECTOR(3, 1, 4) rec[-5]
         DETECTOR(2, 1, 4) rec[-4] rec[-7]
         DETECTOR(1, 0, 4) rec[-3]
         DETECTOR(2, 2, 4) rec[-2]
         DETECTOR(0, 1, 4) rec[-1]
         TICK
        CX 8 3 11 6 0 9 8 14 11 9 0 3 11 14 8 9 0 6 6 14 6 3
        T 6
        TICK
        MX 0 11 8
        M 9 3 14
        MX 6
        DETECTOR(0.625, 0.125, 0, -1, -9) rec[-20] rec[-19] rec[-14] rec[-7]
        DETECTOR(0.875, 0.125, 0, -1, -9) rec[-17] rec[-4]
        DETECTOR(1.25, 1.4375, 0, -1, -9) rec[-20] rec[-14] rec[-6] rec[-5]
        DETECTOR(1.5, 1.4375, 0, -1, -9) rec[-16] rec[-3]
        DETECTOR(2.5, 0.9375, 0, -1, -9) rec[-14] rec[-6]
        DETECTOR(2.75, 0.9375, 0, -1, -9) rec[-15] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-34] rec[-1]
         """)
        splits, noise_ops = gbg.preprocessing(circuit)
        counts = {}
        N, num = 1000, 0
        passed, results, detects, obs = gbg.gate_by_gate(circuit, splits, noise_ops, shots=N)
        for s in range(N):
            if passed[s]:
                num += 1
        self.assertTrue(num == N)


    def test_big2(self):
        # X on qubit 0 should always produce |1>
        circuit = tsim.Circuit(BIG2_CIRCUIT)
        splits, noise_ops = gbg.preprocessing(circuit)
        N, num = 100, 0
        passed, results, detects, obs = gbg.gate_by_gate(circuit, splits, noise_ops, shots=N)
        for s in range(N):
            if passed[s]:
                num += 1
        # Impossible (zero-amplitude) RX trajectories are rejected as failed
        # post-selection rather than crashing, so not every shot passes here.
        # `0 <= num <= N` was true by construction; require real survivors so a
        # total collapse to zero amplitudes actually fails the test. The observed
        # rate is ~15%, and 0.85**100 makes a spurious num==0 a 1-in-10-million event.
        self.assertEqual(len(passed), N)
        self.assertGreater(num, 0)


    def test_power2_is_rebased(self):
        """Regression guard for the float32 underflow fix.

        full_reduce accumulates an unbounded sqrt(2) exponent (pivot and lcomp
        each add O(k**2)). split_circuit_reduce now rebases each decomposition by
        a shared offset. Without it this circuit compiles to power2 in -15..-40
        and deeper circuits walk toward the float32 cliff, where tsim's
        `jnp.pow(2.0, power)` (core/exact_scalar.py) flushes to exactly 0 and
        both Hadamard branch amplitudes vanish.
        """
        circuit = tsim.Circuit(BIG2_CIRCUIT)
        splits, _noise_ops = gbg.preprocessing(circuit)
        self.assertGreater(len(splits), 0)

        for k, (compiled, _e_len) in enumerate(splits):
            power2 = np.asarray(compiled.prefactor.power2)
            if power2.size == 0:
                continue
            top = int(power2.max())
            # The rebase pins the dominant term at 2**0. Measured: exactly 0 on
            # every split of this circuit and of the d=3 cultivation circuit;
            # without it they run -38..-15 and -49..-9 respectively. The -4 floor
            # leaves room for compile.py shifting things down when the
            # decomposition's DyadicNumber carries a nonzero k, while staying
            # tight enough to actually catch the rebase being removed.
            self.assertLessEqual(top, 0, f"split {k} overflows: power2 max {top}")
            self.assertGreaterEqual(
                top, -4,
                f"split {k} power2 max {top}: the dominant term is not rebased to "
                f"~2**0, so the exponent drifts with circuit depth toward the "
                f"float32 underflow cliff at -150",
            )

    # def test_many_t(self):
    #     # H then CNOT produces Bell state — only |00> or |11> should appear
    #     g = param.generate.cliffordT(20, 200)
    #     param.full_reduce(g, paramSafe=True)
    #     gs = param.simulate.find_stabilizer_decomp(g)
    #
    #     amplitude = 0
    #     for h in gs:
    #         amplitude += h.scalar.evaluate_scalar(dict(val_param))
    #     print(amplitude / (np.sqrt(2) ** n_qubits))

if __name__ == '__main__':
    unittest.main()
