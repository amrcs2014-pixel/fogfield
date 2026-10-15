"""Unpack the released experiment ledger and derived dataset caches into code/fogfield_out_v2/,
which is where fogfield_src_fullrun.py (and driver.py) look for them.

    cd code
    python restore_ledger.py
    python driver.py --mode analysis     # tables and figures from the ledger, no re-computation
"""
import os, shutil, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..'))
OUT = os.path.join(HERE, 'fogfield_out_v2')

for d in ['cache', 'ckpt', 'results', 'figs', 'tables', 'logs']:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
with zipfile.ZipFile(os.path.join(REPO, 'results', 'ledger.zip')) as z:
    z.extractall(OUT)
    n = len(z.namelist())
cache = os.path.join(REPO, 'data', 'cache')
for f in os.listdir(cache):
    shutil.copy(os.path.join(cache, f), os.path.join(OUT, 'cache', f))
print('restored %d ledger units and %d cache files into %s' % (n, len(os.listdir(cache)), OUT))
