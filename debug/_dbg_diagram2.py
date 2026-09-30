"""Compare diagrams for all (M/MX) followed by (R/RX) combos."""
import tsim

def show(title, prog):
    c = tsim.Circuit(prog)
    g = c.diagram("pyzx")
    print(f"=== {title} ===")
    for v in g.vertices():
        print(f"  v{v}: type={g.type(v)} phase={g.phase(v)} params={g.get_params(v)} row={g.row(v)} q={g.qubit(v)}")
    print("  edges:")
    for e in g.edges():
        print(f"    {e} type={g.edge_type(e)}")
    print()

# tsim canonical
show("tsim M + R", "R 0\nM 0\nR 0")
show("tsim M + RX", "R 0\nM 0\nRX 0")
show("tsim MX + R", "R 0\nMX 0\nR 0")
show("tsim MX + RX", "R 0\nMX 0\nRX 0")

# What gate_by_gate's preprocessing currently produces (after fix)
show("preproc MX(=H+M+H) + RX", "R 0\nH 0\nM 0\nH 0\nRX 0")
show("preproc MX(=H+M+H) + R", "R 0\nH 0\nM 0\nH 0\nR 0")

# Without the fix
show("preproc OLD MX(=H+M) + RX", "R 0\nH 0\nM 0\nRX 0")
show("preproc OLD MX(=H+M) + R", "R 0\nH 0\nM 0\nR 0")
