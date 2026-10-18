"""Rebuild the notebook with interpretation cells and re-attach the outputs of an executed run.

Code cells are matched by exact source text, so outputs can only attach to the
identical code that produced them; the script fails if any code cell is unmatched.
Usage: python scripts/merge_outputs.py EXECUTED.ipynb
"""
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
executed = nbformat.read(sys.argv[1], as_version=4)
by_source = {}
for c in executed.cells:
    if c.cell_type == "code":
        by_source.setdefault(c.source, []).append(c)
subprocess.run([sys.executable, str(ROOT / "scripts" / "build_notebook.py")], check=True)
target_path = ROOT / "notebooks" / "diabetes_classification.ipynb"
nb = nbformat.read(target_path, as_version=4)
unmatched = 0
for c in nb.cells:
    if c.cell_type != "code":
        continue
    match = by_source.get(c.source)
    if not match:
        unmatched += 1
        print("UNMATCHED:", c.source[:80])
        continue
    src = match.pop(0)
    c.outputs, c.execution_count = src.outputs, src.execution_count
if unmatched:
    sys.exit(f"{unmatched} code cells have no executed counterpart")
nb.metadata = executed.metadata | {"colab": {"provenance": []}}
nbformat.write(nb, target_path)
print("merged outputs into", target_path, "| markdown cells:", sum(c.cell_type == "markdown" for c in nb.cells))
