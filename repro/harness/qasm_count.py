"""Single gate counter used for every input and output circuit in the campaign.

Counts OpenQASM 2 gate statements. Declarations (qreg/creg), gate definitions,
barrier/measure/reset are not gates. Every single-qubit gate counts as 1;
the only two-qubit gate allowed is cx, anything else raises ValueError so a
gate-set mismatch can never be silently miscounted.
"""
import re
import sys

ONE_Q = {"rz", "h", "x", "t", "tdg", "s", "sdg", "z", "y", "rx", "ry",
         "u1", "u2", "u3", "u", "p", "sx", "sxdg", "id"}
TWO_Q = {"cx"}
NOT_GATES = {"openqasm", "include", "qreg", "creg", "barrier", "measure",
             "reset", "opaque"}

_COMMENT = re.compile(r"//[^\n]*")
_GATE_DEF = re.compile(r"\bgate\b[^{]*\{[^}]*\}", re.S)
_QREG = re.compile(r"qreg\s+\w+\s*\[\s*(\d+)\s*\]")


def count_text(text):
    """Return dict(total, cx, n_qubits) for OpenQASM 2 source text."""
    text = _COMMENT.sub("", text)
    text = _GATE_DEF.sub("", text)
    n_qubits = sum(int(m) for m in _QREG.findall(text))
    total = cx = 0
    for stmt in text.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        name = re.match(r"[A-Za-z_][\w]*", stmt)
        if name is None:
            raise ValueError(f"unparseable statement: {stmt[:80]!r}")
        op = name.group(0).lower()
        if op in NOT_GATES:
            continue
        if op in TWO_Q:
            cx += 1
        elif op not in ONE_Q:
            raise ValueError(f"unexpected gate {op!r} in statement {stmt[:80]!r}")
        total += 1
    return {"total": total, "cx": cx, "n_qubits": n_qubits}


def count_file(path):
    with open(path) as f:
        return count_text(f.read())


if __name__ == "__main__":
    for p in sys.argv[1:]:
        c = count_file(p)
        print(f"{p}\t{c['total']}\t{c['cx']}\t{c['n_qubits']}")
