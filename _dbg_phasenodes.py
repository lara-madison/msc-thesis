"""Dump scalar structure for each split in test_big2 to localize the structural sqrt(2)."""
import tsim
import gate_by_gate as gbg


def dump_split(idx, gs):
    print(f"\n=== split_circs[{idx}] : {len(gs)} stabilizer term(s) ===")
    for i, h in enumerate(gs):
        s = h.scalar
        print(f"  term {i}: power2={s.power2}, phase={s.phase}, is_zero={s.is_zero}")
        nodes = list(zip(s.phasenodes, s.phasenodevars))
        if nodes:
            print(f"    phasenodes ({len(nodes)}):")
            for const, vars_ in nodes:
                print(f"      const={const}  vars={vars_}")
        if s.phasepairs:
            print(f"    phasepairs ({len(s.phasepairs)}):")
            for pp in s.phasepairs:
                print(f"      alpha={pp.alpha}  paramsA={pp.paramsA}  beta={pp.beta}  paramsB={pp.paramsB}")
        if s.phasevars_halfpi:
            print(f"    halfpi: {dict(s.phasevars_halfpi)}")
        if getattr(s, "phasevars_pi", None):
            print(f"    phasevars_pi: {s.phasevars_pi}")
        if getattr(s, "phasevars_pi_pair", None):
            print(f"    phasevars_pi_pair: {s.phasevars_pi_pair}")
        ff = getattr(s, "floatfactor", None)
        if ff is not None:
            try:
                ffc = ff.to_complex()
            except Exception:
                ffc = ff
            print(f"    floatfactor={ffc}  approx={getattr(s, 'approximate_floatfactor', None)}")


def main():
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
    """)

    splits = gbg.preprocessing(circuit)
    print(f"got {len(splits)} splits")
    # Focus on split 4 (MX 2's H) but also print 0..4 for context.
    for idx in range(min(len(splits), 5)):
        dump_split(idx, splits[idx])


if __name__ == "__main__":
    main()
