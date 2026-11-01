#!/usr/bin/env bash
# Download GUOQ's benchmarks/nam_rz (250 circuits) into benchmarks/guoq_nam_rz/ and check the 248
# circuits of the suite (all but qft_10 and qft_16) against repro/inputs/S248.csv.
# Usage (from anywhere): repro/inputs/fetch_guoq_benchmarks.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
COMMIT=8c4c3a5a6dfc9f7fc375ec16c2180139f0a8cb1a   # github.com/qqq-wisc/guoq
DEST=$REPO/benchmarks/guoq_nam_rz
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

curl -fsSL "https://github.com/qqq-wisc/guoq/archive/$COMMIT.tar.gz" -o "$TMP/guoq.tar.gz"
tar -xzf "$TMP/guoq.tar.gz" -C "$TMP" "guoq-$COMMIT/benchmarks/nam_rz"
mkdir -p "$DEST"
python3 - "$REPO" "$TMP/guoq-$COMMIT/benchmarks/nam_rz" "$DEST" <<'EOF'
import csv, hashlib, os, shutil, sys
repo, src, dest = sys.argv[1:]
rows = list(csv.DictReader(open(os.path.join(repo, "repro/inputs/S248.csv"))))
for name in os.listdir(src):
    if name.endswith(".qasm"):
        shutil.copyfile(os.path.join(src, name), os.path.join(dest, name))
bad = []
for r in rows:
    name = os.path.basename(r["path"])
    if hashlib.sha256(open(os.path.join(dest, name), "rb").read()).hexdigest() != r["sha256"]:
        bad.append(name)
if bad:
    sys.exit(f"sha256 mismatch for {len(bad)} circuits: {bad[:5]}")
print(f"{len(os.listdir(dest))} circuits in {dest}; the {len(rows)} of the suite match repro/inputs/S248.csv")
EOF
