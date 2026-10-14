"""Compile paper/latex_source, package the deliverables into Final/Paper_FogField, verify that the
zipped LaTeX source compiles on its own, and build the Word copy.

    python package.py            # everything
    python package.py --no-docx  # skip the Word conversion
"""
import os, re, shutil, subprocess, zipfile, tempfile, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
PAPER = os.path.join(ROOT, 'paper')
SRC = os.path.join(PAPER, 'latex_source')
FINAL = os.path.join(ROOT, 'Final')
OUT = os.path.join(FINAL, 'Paper_FogField')
NAME = 'FogField'


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace')


def compile_tex(cwd):
    for cmd in (['pdflatex', '-interaction=nonstopmode', 'main.tex'], ['bibtex', 'main'],
                ['pdflatex', '-interaction=nonstopmode', 'main.tex'],
                ['pdflatex', '-interaction=nonstopmode', 'main.tex']):
        run(cmd, cwd)
    log = open(os.path.join(cwd, 'main.log'), encoding='utf-8', errors='replace').read()
    errs = re.findall(r'^! (.*)$', log, re.M)
    undef = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)' on page", log)))
    pages = run(['pdfinfo', os.path.join(cwd, 'main.pdf')], cwd).stdout
    pages = next((l.split()[-1] for l in pages.splitlines() if l.startswith('Pages')), '?')
    return errs, undef, pages


if __name__ == '__main__':
    errs, undef, pages = compile_tex(SRC)
    if errs or undef:
        print('COMPILE PROBLEMS', errs[:3], undef[:5]); sys.exit(1)
    print('compiled main.pdf:', pages, 'pages')

    # ---- latex_source (self-contained: tex, bib, bbl, figures)
    dst = os.path.join(OUT, 'latex_source')
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, 'figs'))
    for f in os.listdir(SRC):
        if f.endswith(('.tex', '.bib', '.bbl')):
            shutil.copy(os.path.join(SRC, f), dst)
    for f in os.listdir(os.path.join(SRC, 'figs')):
        shutil.copy(os.path.join(SRC, 'figs', f), os.path.join(dst, 'figs', f))
    shutil.copy(os.path.join(SRC, 'main.pdf'), os.path.join(OUT, NAME + '.pdf'))

    # ---- result tables as CSV (the LaTeX tables in latex_source/tab_*.tex are typeset from these)
    tdir = os.path.join(OUT, 'tables')
    os.makedirs(tdir, exist_ok=True)
    for f in sorted(os.listdir(PAPER)):
        if f.startswith('table') and f.endswith('.csv'):
            shutil.copy(os.path.join(PAPER, f), tdir)

    # ---- zip + standalone compile check
    zp = os.path.join(OUT, NAME + '_latex_source.zip')
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(dst):
            for f in files:
                z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), dst))
    tmp = tempfile.mkdtemp()
    zipfile.ZipFile(zp).extractall(tmp)
    errs, undef, pages2 = compile_tex(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    print('standalone compile of the zip:', 'OK' if not errs and not undef else 'PROBLEMS %s %s' % (errs[:3], undef[:5]),
          '(%s pages)' % pages2)

    # ---- Word copy
    if '--no-docx' not in sys.argv:
        sys.path.insert(0, HERE)
        import make_docx
        make_docx.build(os.path.join(OUT, NAME + '.docx'))

    print('packaged to', OUT)
