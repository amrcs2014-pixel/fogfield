"""Assemble the public code repository (v1.1: v2 ledger) for FogField in ../github/fogfield (code, results, derived data).

The repository's hand-written files (README.md, LICENSE, CITATION.cff, .gitignore, requirements.txt)
live in the repo folder and are left alone; this script refreshes code/, results/, data/ and the
notebook from the build folder."""
import os, shutil, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
REPO = os.path.join(ROOT, 'github', 'fogfield')
OUT = os.path.join(HERE, 'fogfield_out_v2')          # v1.1: the v2 ledger (common random numbers)

CODE = ['fogfield_src_fullrun.py', 'fogfield_src.py', 'driver.py', 'build_notebook.py', 'smoke.py',
        'make_paper_figs2.py', 'restore_ledger.py', 'paper_numbers.py', 'grid_status.py', 'run_grid.ps1',
        'test_crn.py', 'sync_sources.py']
TOOLS = ['split_source.py', 'package.py', 'make_docx.py', 'make_repo.py',
         'revise_v2_patch.py', 'revise_v2_patch2.py', 'revise_v2_patch3.py', 'revise_v2_patch4.py',
         'revise_v2_patch5.py', 'revise_v2_patch6.py', 'revise_v2_patch7.py']
RESULT_FILES = ['run_manifest.json', 'tables/potential_check.json', 'tables/resume_selftest.json']
CACHE = ['eua_melbcbd_816.npz', 'geolife_uall_g400_s30_c500.npz']


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
    for f in sorted(os.listdir(os.path.join(HERE, 'paper_figs_v2'))):     # paper figures (paper_numbers.py)
        if f.startswith('fig_'):
            shutil.copy(os.path.join(HERE, 'paper_figs_v2', f), os.path.join(REPO, 'results', 'figures', f))
    shutil.copy(os.path.join(HERE, 'paper_numbers.json'), os.path.join(REPO, 'results', 'paper_numbers.json'))
    for f in sorted(os.listdir(os.path.join(ROOT, 'paper'))):              # the paper's tables as CSV/JSON
        if f.startswith('table') and f.endswith(('.csv', '.json')):
            shutil.copy(os.path.join(ROOT, 'paper', f), os.path.join(REPO, 'results', 'tables', 'paper_' + f))
    for f in sorted(os.listdir(os.path.join(OUT, 'figs'))):          # notebook-rendered diagnostics
        shutil.copy(os.path.join(OUT, 'figs', f), os.path.join(REPO, 'results', 'figures', f))

    # the checkpointed experiment ledger: one JSON per work unit
    n = 0
    with zipfile.ZipFile(os.path.join(REPO, 'results', 'ledger.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for study in ['main', 'scalability', 'milp', 'equilibrium', 'lll_schedule', 'viewbound']:
            d = os.path.join(OUT, 'results', study)
            for f in sorted(os.listdir(d)):
                z.write(os.path.join(d, f), 'results/%s/%s' % (study, f))
                n += 1
    for f in CACHE:
        shutil.copy(os.path.join(OUT, 'cache', f), os.path.join(REPO, 'data', 'cache', f))

    # the run logs, released as an asset rather than committed
    with zipfile.ZipFile(os.path.join(ROOT, 'github', 'fogfield_run_logs_v1.1.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(HERE)):                                   # v1.0 run (23 Sep 2026)
            if f.endswith('.log'):
                z.write(os.path.join(HERE, f), 'v1.0/' + f)
        for f in sorted(os.listdir(os.path.join(HERE, 'logs_v2'))):          # v1.1 run (15 Oct 2026)
            if f.endswith(('.log', '.err')) or f == 'GRID_DONE':
                z.write(os.path.join(HERE, 'logs_v2', f), 'v1.1/' + f)
    print('repo assembled at', REPO, '| ledger units:', n)


if __name__ == '__main__':
    main()
