"""Assemble the public code repository for FogField in ../github/fogfield (code, results, derived data).

The repository's hand-written files (README.md, LICENSE, CITATION.cff, .gitignore, requirements.txt)
live in the repo folder and are left alone; this script refreshes code/, results/, data/ and the
notebook from the build folder."""
import os, shutil, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
REPO = os.path.join(ROOT, 'github', 'fogfield')
OUT = os.path.join(HERE, 'fogfield_out_full')

CODE = ['fogfield_src_fullrun.py', 'fogfield_src.py', 'driver.py', 'build_notebook.py', 'smoke.py',
        'make_paper_figs2.py', 'restore_ledger.py']
TOOLS = ['priority_milp_eq2.py', 'split_source.py', 'package.py', 'make_docx.py', 'make_repo.py']
RESULT_FILES = ['run_manifest.json', 'tables/potential_check.json', 'tables/resume_selftest.json']
CACHE = ['eua_melbcbd_816.npz', 'geolife_u60_g400_s30_c500.npz']


def main():
    for d in ['code', 'results', 'data']:
        p = os.path.join(REPO, d)
        if os.path.exists(p):
            shutil.rmtree(p)
    for d in ['code/parts', 'code/tools', 'results/tables', 'results/figures', 'data/cache']:
        os.makedirs(os.path.join(REPO, d), exist_ok=True)

    for f in CODE:
        shutil.copy(os.path.join(HERE, f), os.path.join(REPO, 'code', f))
    for f in sorted(os.listdir(os.path.join(HERE, 'parts'))):
        shutil.copy(os.path.join(HERE, 'parts', f), os.path.join(REPO, 'code', 'parts', f))
    for f in TOOLS:
        shutil.copy(os.path.join(HERE, f), os.path.join(REPO, 'code', 'tools', f))
    shutil.copy(os.path.join(ROOT, 'FogField_Q1.ipynb'), os.path.join(REPO, 'FogField_Q1.ipynb'))

    for f in RESULT_FILES:
        shutil.copy(os.path.join(OUT, f), os.path.join(REPO, 'results', os.path.basename(f)))
    for f in sorted(os.listdir(os.path.join(OUT, 'tables'))):
        if f.startswith('table'):
            shutil.copy(os.path.join(OUT, 'tables', f), os.path.join(REPO, 'results', 'tables', f))
    for f in sorted(os.listdir(os.path.join(HERE, 'paper_figs'))):
        if f.startswith('fig_'):
            shutil.copy(os.path.join(HERE, 'paper_figs', f), os.path.join(REPO, 'results', 'figures', f))
    for f in sorted(os.listdir(os.path.join(OUT, 'figs'))):          # notebook-rendered diagnostics
        shutil.copy(os.path.join(OUT, 'figs', f), os.path.join(REPO, 'results', 'figures', f))

    # the checkpointed experiment ledger: one JSON per work unit
    n = 0
    with zipfile.ZipFile(os.path.join(REPO, 'results', 'ledger.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for study in ['main', 'scalability', 'milp', 'equilibrium']:
            d = os.path.join(OUT, 'results', study)
            for f in sorted(os.listdir(d)):
                z.write(os.path.join(d, f), 'results/%s/%s' % (study, f))
                n += 1
    for f in CACHE:
        shutil.copy(os.path.join(OUT, 'cache', f), os.path.join(REPO, 'data', 'cache', f))

    # the run logs, released as an asset rather than committed
    with zipfile.ZipFile(os.path.join(ROOT, 'github', 'fogfield_run_logs.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(HERE)):
            if f.endswith('.log'):
                z.write(os.path.join(HERE, f), f)
    print('repo assembled at', REPO, '| ledger units:', n)


if __name__ == '__main__':
    main()
