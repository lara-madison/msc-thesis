"""Compare diagrams: gate_by_gate's H+M+RX vs tsim's MX+RX."""
import tsim

# Reference: tsim's MX + RX
c1 = tsim.Circuit("R 0\nMX 0\nRX 0")
g1 = c1.diagram("pyzx")
print("=== tsim MX + RX ===")
for v in g1.vertices():
    print(f"  v{v}: type={g1.type(v)} phase={g1.phase(v)} params={g1.get_params(v)} row={g1.row(v)} q={g1.qubit(v)}")
print("  edges:")
for e in g1.edges():
    print(f"    {e} type={g1.edge_type(e)}")

# What gate_by_gate does (just H+M instead of MX): H + M then RX
c2 = tsim.Circuit("R 0\nH 0\nM 0\nRX 0")
g2 = c2.diagram("pyzx")
print()
print("=== gate_by_gate's H + M + RX ===")
for v in g2.vertices():
    print(f"  v{v}: type={g2.type(v)} phase={g2.phase(v)} params={g2.get_params(v)} row={g2.row(v)} q={g2.qubit(v)}")
print("  edges:")
for e in g2.edges():
    print(f"    {e} type={g2.edge_type(e)}")

# Properly: H + M + H + RX
c3 = tsim.Circuit("R 0\nH 0\nM 0\nH 0\nRX 0")
g3 = c3.diagram("pyzx")
print()
print("=== H + M + H + RX (proposed fix) ===")
for v in g3.vertices():
    print(f"  v{v}: type={g3.type(v)} phase={g3.phase(v)} params={g3.get_params(v)} row={g3.row(v)} q={g3.qubit(v)}")
print("  edges:")
for e in g3.edges():
    print(f"    {e} type={g3.edge_type(e)}")
