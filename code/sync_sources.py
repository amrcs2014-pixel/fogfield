"""Keep the three copies of the simulator in step after editing fogfield_src_fullrun.py (canonical):

  fogfield_src.py      the same source with the quick Colab configuration (QUICK = True)
  parts/01..10_*.py    its ten sections (re-split at the existing section boundaries)
  ../FogField_Q1.ipynb the notebook, compiled from fogfield_src.py

    python sync_sources.py
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
full = open("fogfield_src_fullrun.py", encoding="utf-8").read()
q = full.replace("QUICK = False   # REAL FULL RUN",
                 "QUICK = True                 # <<< set False for the full paper-scale grid")
q = q.replace('root = Path("./fogfield_out_v2").resolve()', 'root = Path("./fogfield_out").resolve()')
assert q != full
open("fogfield_src.py", "w", encoding="utf-8").write(q)

parts = sorted(glob.glob("parts/*.py"))
marks = ["\n".join(open(p, encoding="utf-8").read().split("\n")[:3]) for p in parts]
idx = [q.find(m) for m in marks]
assert all(k >= 0 for k in idx) and idx == sorted(idx) and idx[0] == 0, idx
for i, p in enumerate(parts):
    end = idx[i + 1] if i + 1 < len(idx) else len(q)
    open(p, "w", encoding="utf-8").write(q[idx[i]:end])
assert "".join(open(p, encoding="utf-8").read() for p in parts) == q
print("fogfield_src.py and parts/ in step with fogfield_src_fullrun.py")
sys.exit(subprocess.call([sys.executable, "build_notebook.py", "fogfield_src.py",
                          os.path.join("..", "FogField_Q1.ipynb")]))
