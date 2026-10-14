"""One-off: turn the single-file draft paper/fogfield.tex into the clean multi-file source in
paper/latex_source/ (main.tex + one file per section + tab_*.tex + refs.bib + figs/).

After this has run once, paper/latex_source/ is the canonical, hand-editable source and
build/package.py compiles and packages it. Re-running this script overwrites the section
files from the draft, so do not re-run it after editing the split source."""
import os, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.normpath(os.path.join(HERE, '..', 'paper'))
OUT = os.path.join(PAPER, 'latex_source')
os.makedirs(os.path.join(OUT, 'figs'), exist_ok=True)

src = open(os.path.join(PAPER, 'fogfield.tex'), encoding='utf-8').read()

# ---- small fixes to the draft text ------------------------------------------------------
src = src.replace('\\end{figure} Onboard computation adds', '\\end{figure}\n\nOnboard computation adds')

# ---- split off the body ------------------------------------------------------------------
body = src[src.index('\\begin{abstract}'):src.index('\\begin{thebibliography}')]

abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', body, re.S).group(1).strip()
keywords = re.search(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}', body, re.S).group(1).strip()
rest = body[body.index('\\end{IEEEkeywords}') + len('\\end{IEEEkeywords}'):]

SECTIONS = [
    ('Introduction', 'intro'),
    ('Related Work', 'related'),
    ('System Model', 'model'),
    ('FogField as an Exact Potential Game', 'game'),
    ('Decentralized Dynamics and the Gossip Layer', 'dynamics'),
    ('Theoretical Analysis', 'theory'),
    ('Evaluation Methodology', 'setup'),
    ('Results', 'results'),
    ('Limitations and Threats to Validity', 'limitations'),
    ('Conclusion', 'conclusion'),
]
TABLES = {'tab:gap': 'tab_gap', 'tab:bej': 'tab_bej', 'tab:scale': 'tab_scale',
          'tab:eq': 'tab_eq', 'tab:rec': 'tab_rec'}

# cut at every \section{...} (not \subsection); drop the %% ===== rulers first
rest = re.sub(r'(?m)^%%\s*=+\s*$\n?', '', rest)
parts = re.split(r'(?m)^(?=\\section\{)', rest)
parts = [p for p in parts if p.strip()]
assert len(parts) == len(SECTIONS), [p[:40] for p in parts]


def pull_tables(text):
    def repl(m):
        env = m.group(0)
        lab = re.search(r'\\label\{(tab:[^}]+)\}', env).group(1)
        name = TABLES[lab]
        open(os.path.join(OUT, name + '.tex'), 'w', encoding='utf-8').write(env.strip() + '\n')
        return '\\input{%s}' % name
    return re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', repl, text, flags=re.S)


for (title, name), text in zip(SECTIONS, parts):
    assert text.startswith('\\section{' + title), (title, text[:60])
    text = re.sub(r'(?m)^%%\s*=+\s*$\n?', '', text)       # drop the %% ===== rulers
    text = pull_tables(text)
    open(os.path.join(OUT, name + '.tex'), 'w', encoding='utf-8').write(text.strip() + '\n')

open(os.path.join(OUT, 'abstract.tex'), 'w', encoding='utf-8').write(abstract + '\n')

# ---- main.tex ----------------------------------------------------------------------------
main = r'''%% FogField --- manuscript for IEEE Transactions on Network and Service Management
%% Compile: pdflatex main && bibtex main && pdflatex main && pdflatex main
\documentclass[journal,10pt]{IEEEtran}

\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{url}
\usepackage{cite}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage[hidelinks]{hyperref}

\graphicspath{{figs/}}

\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{definition}{Definition}

\begin{document}

\title{FogField: Fully Decentralized Repositioning of \\ UAV-Mounted Fog Nodes via Potential Games and Gossip}

\author{%
  Author~Name~1, Author~Name~2, and Author~Name~3%
  \thanks{Manuscript received [date]. (Corresponding author: [name].)}%
  \thanks{The authors are with [department, institution, city, country]
  (e-mail: [address]).}%
  \thanks{Replication package: all code, the checkpointed experiment ledger, and the
  scripts that regenerate every figure and table in this paper are released with the
  manuscript.}%
}

\markboth{IEEE Transactions on Network and Service Management}%
{FogField: Fully Decentralized Fog-Node Repositioning via Potential Games and Gossip}

\maketitle

\begin{abstract}
\input{abstract}
\end{abstract}

\begin{IEEEkeywords}
''' + keywords + r'''
\end{IEEEkeywords}

\input{intro}
\input{related}
\input{model}
\input{game}
\input{dynamics}
\input{theory}
\input{setup}
\input{results}
\input{limitations}
\input{conclusion}

\bibliographystyle{IEEEtran}
\bibliography{refs}

\end{document}
'''
open(os.path.join(OUT, 'main.tex'), 'w', encoding='utf-8').write(main)

# ---- figures actually used ---------------------------------------------------------------
used = set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{figs/([^}]+)\}', src))
for u in sorted(used):
    shutil.copy(os.path.join(PAPER, 'figs', u), os.path.join(OUT, 'figs', u))
# the split sources reference figures via \graphicspath, so strip the figs/ prefix
for f in os.listdir(OUT):
    if f.endswith('.tex'):
        p = os.path.join(OUT, f)
        t = open(p, encoding='utf-8').read().replace('{figs/fig_', '{fig_')
        open(p, 'w', encoding='utf-8').write(t)

print('sections:', [n for _, n in SECTIONS])
print('tables:', sorted(TABLES.values()))
print('figures:', sorted(used))
