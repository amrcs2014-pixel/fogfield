"""Word (.docx) copy of the FogField manuscript (IEEEtran source in paper/latex_source).

The LaTeX is flattened and simplified for pandoc: section, figure, table, equation and citation
numbers are taken from the compiled main.aux so they match the PDF; the reference list comes from
the BibTeX-generated main.bbl (IEEEtran order); figures are rasterised to PNG; equations stay
native Word equations. python-docx then applies the formatting (fonts, table rules, page numbers).
Run after main.tex has been compiled (package.py does both)."""
import os, re, sys
import pypandoc, pymupdf
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, '..', 'paper', 'latex_source'))
BUILD = os.path.join(HERE, 'docx_build')
os.makedirs(BUILD, exist_ok=True)

AUTHORS = 'Amr H. Abdelhaliem'
AFFIL = ('Department of Cyber Security, Faculty of Science and Information Technology, '
         'Irbid National University, Irbid, Jordan. E-mail: amrcs2014@gmail.com')


def read(name):
    return open(os.path.join(SRC, name), encoding='utf-8').read()


# ------------------------------------------------------------------ brace-aware helpers
def group(s, i):
    """s[i] == '{' -> (content, index after the closing brace)."""
    assert s[i] == '{', s[i:i + 30]
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == '\\':
            j += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError('unbalanced group at %d' % i)


def sub_cmd(s, name, nargs, fn, opt=False):
    """Replace \\name[opt]{a1}..{an} (optionally starred) by fn(args, optarg)."""
    out, i = [], 0
    pat = re.compile(r'\\' + name + r'(?![A-Za-z])\*?')
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            return ''.join(out)
        out.append(s[i:m.start()])
        j = m.end()
        o = None
        if opt:
            while j < len(s) and s[j] in ' \n':
                j += 1
            if j < len(s) and s[j] == '[':
                k = s.index(']', j)
                o, j = s[j + 1:k], k + 1
        args = []
        for _ in range(nargs):
            while s[j] in ' \n':
                j += 1
            a, j = group(s, j)
            args.append(a)
        out.append(fn(args, o))
        i = j


def strip_comments(s):
    return re.sub(r'(?<!\\)%.*', '', s)


def roman(n):
    out = ''
    for v, r in [(10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]:
        while n >= v:
            out += r
            n -= v
    return out


# ------------------------------------------------------------------ numbers from the compiled document
def aux():
    t = open(os.path.join(SRC, 'main.aux'), encoding='utf-8', errors='replace').read()
    labels = {}
    for m in re.finditer(r'\\newlabel\{([^}]+)\}', t):
        content, _ = group(t, m.end())           # {{num}{page}{title}{anchor}{}}
        num, _ = group(content, 0)
        num = re.sub(r'\\mbox\s*\{([^}]*)\}', r'\1', num).strip()
        labels[m.group(1)] = num
    cites = {m.group(1): int(m.group(2)) for m in re.finditer(r'\\bibcite\{([^}]+)\}\{(\d+)\}', t)}
    return labels, cites


def ieee_cite(nums):
    """IEEE style: [1], [3]--[5]."""
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append('[%d]--[%d]' % (nums[i], nums[j]))
        else:
            parts.extend('[%d]' % n for n in nums[i:j + 1])
        i = j + 1
    return ', '.join(parts)


def bibliography(cites):
    t = read('main.bbl')
    t = t[t.index('\\bibitem'):]
    t = t.split('\\end{thebibliography}')[0]
    items = re.split(r'\\bibitem\{([^}]*)\}', t)[1:]
    out = ['\\section*{References}\n']
    for key, body in zip(items[0::2], items[1::2]):
        body = ' '.join(body.split())
        body = re.sub(r'\\BIBentry[A-Za-z]+', '', body)
        out.append('[%d] %s\n\n' % (cites[key], body))
    return ''.join(out)


# ------------------------------------------------------------------ algorithm -> numbered lines
def algorithm_lines(body):
    body = re.sub(r'\\begin\{algorithmic\}(\[\d+\])?', '', body).replace('\\end{algorithmic}', '')
    lines = [l.strip() for l in re.split(r'\\State(?![A-Za-z])', body) if l.strip()]
    out = []
    for n, l in enumerate(lines, 1):
        l = l.replace('\\quad', '~~~~').replace('w.p.\\ ', 'w.p. ')
        out.append('%d:~%s' % (n, ' '.join(l.split())))
    return '\n\n'.join(out) + '\n\n'


# ------------------------------------------------------------------ figures
def rasterise(name):
    src = os.path.join(SRC, 'figs', name if name.endswith('.pdf') else name + '.pdf')
    dst = os.path.join(BUILD, os.path.basename(src)[:-4] + '.png')
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        page = pymupdf.open(src)[0]
        page.get_pixmap(dpi=300).save(dst)
    return dst


# ------------------------------------------------------------------ flatten
def flatten():
    labels, cites = aux()
    s = strip_comments(read('main.tex'))

    def inp(a, _):
        f = a[0] if a[0].endswith('.tex') else a[0] + '.tex'
        return strip_comments(read(f))
    for _ in range(3):
        s = sub_cmd(s, 'input', 1, inp)

    title = re.search(r'\\title\{(.*?)\}\s*\n', s, re.S).group(1).replace('\\\\', ' ')
    title = ' '.join(title.split())
    body = s[s.index('\\begin{document}') + len('\\begin{document}'):s.index('\\end{document}')]

    # front matter that we rebuild ourselves
    body = sub_cmd(body, 'title', 1, lambda a, o: '')
    body = sub_cmd(body, 'author', 1, lambda a, o: '')
    body = sub_cmd(body, 'markboth', 2, lambda a, o: '')
    body = body.replace('\\maketitle', '')
    body = body.replace('\\begin{abstract}', '\\section*{Abstract}\n').replace('\\end{abstract}', '')
    body = re.sub(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}',
                  lambda m: '\\textbf{Index Terms---}' + ' '.join(m.group(1).split()) + '\n',
                  body, flags=re.S)
    body = sub_cmd(body, 'IEEEPARstart', 2, lambda a, o: a[0] + a[1])

    # cross references and citations, numbered as in the PDF
    body = sub_cmd(body, 'cite', 1, lambda a, o: ieee_cite(cites[k.strip()] for k in a[0].split(',')))
    body = sub_cmd(body, 'eqref', 1, lambda a, o: '(%s)' % labels[a[0]])
    body = sub_cmd(body, 'ref', 1, lambda a, o: labels[a[0]])

    # section numbering in the IEEE style (I., A.)
    counters = {'sec': 0, 'sub': 0}

    def section(m):
        star, name = m.group(1), m.group(2)
        if star:
            return '\\section*{%s}' % name
        counters['sec'] += 1
        counters['sub'] = 0
        return '\\section*{%s. %s}' % (roman(counters['sec']), name)

    def subsection(m):
        star, name = m.group(1), m.group(2)
        if star:
            return '\\subsection*{%s}' % name
        counters['sub'] += 1
        return '\\subsection*{%s. %s}' % (chr(64 + counters['sub']), name)
    # process in document order so the counters are right
    pat = re.compile(r'\\(section|subsection)(\*?)\{([^}]*)\}')
    out, i = [], 0
    for m in pat.finditer(body):
        out.append(body[i:m.start()])
        mm = re.match(r'\\(?:section|subsection)(\*?)\{([^}]*)\}', m.group(0))
        out.append(section(mm) if m.group(1) == 'section' else subsection(mm))
        i = m.end()
    out.append(body[i:])
    body = ''.join(out)

    # theorem-like environments
    thm = {'theorem': 0, 'proposition': 0, 'definition': 0}

    def thm_env(m):
        kind, name, content = m.group(1), m.group(2), m.group(3)
        thm[kind] += 1
        label = '%s %d' % (kind.capitalize(), thm[kind])
        if name:
            label += ' (%s)' % name
        content = re.sub(r'\\label\{[^}]*\}', '', content).strip()
        return '\\textbf{%s.} \\emph{%s}\n' % (label, content)
    body = re.sub(r'\\begin\{(theorem|proposition|definition)\}(?:\[([^\]]*)\])?(.*?)\\end\{\1\}',
                  thm_env, body, flags=re.S)
    body = re.sub(r'\\begin\{proof\}(?:\[([^\]]*)\])?',
                  lambda m: '\\emph{%s.} ' % (m.group(1) or 'Proof'), body)
    body = re.sub(r'\s*\\end\{proof\}', '~$\\\\square$', body)

    # equations, numbered as in the PDF
    def eq_env(m):
        e = m.group(1)
        lm = re.search(r'\\label\{([^}]+)\}', e)
        e = re.sub(r'\\label\{[^}]+\}', '', e).strip()
        return '\\[ %s \\qquad (%s) \\]' % (e, labels[lm.group(1)])
    body = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', eq_env, body, flags=re.S)

    def align_env(m):
        e = m.group(1)
        lm = re.search(r'\\label\{([^}]+)\}', e)
        e = re.sub(r'\\label\{[^}]+\}', '', e).replace('\\nonumber', '').strip()
        return '\\[ \\begin{aligned} %s \\end{aligned} \\qquad (%s) \\]' % (e, labels[lm.group(1)])
    body = re.sub(r'\\begin\{align\}(.*?)\\end\{align\}', align_env, body, flags=re.S)

    # tables (caption above in IEEE; Word puts it where the template says)
    def table_env(m):
        env = m.group(0)
        cap, _ = group(env, re.search(r'\\caption\{', env).end() - 1)
        num = labels[re.search(r'\\label\{([^}]+)\}', env).group(1)]
        tab = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', env, re.S).group(0)
        tab = re.sub(r'\\cmidrule(\([a-z]*\))?\{[^}]*\}', '', tab)
        tab = tab.replace('@{}', '').replace('\\addlinespace', '')
        tab = re.sub(r'\\(scriptsize|footnotesize|small)\b', '', tab)
        return '\\begin{table}\n\\caption{\\textbf{TABLE %s.} %s}\n%s\n\\end{table}\n' % (num, cap, tab)
    body = re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', table_env, body, flags=re.S)

    def figure_env(m):
        env = m.group(0)
        gm = re.search(r'\\includegraphics(\[[^\]]*\])?\{([^}]+)\}', env)
        opt = gm.group(1) or ''
        w = re.search(r'width=([0-9.]*)\\(columnwidth|textwidth|linewidth)', opt)
        frac = float(w.group(1) or 1) if w else 1.0
        if w and w.group(2) == 'columnwidth':
            frac *= 0.62                      # one IEEE column on a single-column Word page
        png = rasterise(gm.group(2))
        cap, _ = group(env, re.search(r'\\caption\{', env).end() - 1)
        num = labels[re.search(r'\\label\{([^}]+)\}', env).group(1)]
        return ('\\begin{figure}\n\\includegraphics[width=%.2f\\textwidth]{%s}\n\\caption{\\textbf{Fig. %s.} %s}\n'
                '\\end{figure}\n' % (min(frac, 1.0), png.replace('\\', '/'), num, cap))
    body = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', figure_env, body, flags=re.S)

    def algo_env(m):
        env = m.group(0)
        cap, _ = group(env, re.search(r'\\caption\{', env).end() - 1)
        num = labels[re.search(r'\\label\{([^}]+)\}', env).group(1)]
        alg = re.search(r'\\begin\{algorithmic\}.*?\\end\{algorithmic\}', env, re.S).group(0)
        return '\\textbf{Algorithm %s.} %s\n\n%s' % (num, cap, algorithm_lines(alg))
    body = re.sub(r'\\begin\{algorithm\}.*?\\end\{algorithm\}', algo_env, body, flags=re.S)

    body = re.sub(r'\\bibliographystyle\{[^}]*\}', '', body)
    body = re.sub(r'\\bibliography\{[^}]*\}', lambda m: bibliography(cites), body)

    for c in ['centering', 'noindent', 'clearpage', 'newpage', 'sloppy']:
        body = re.sub(r'\\' + c + r'(?![A-Za-z])', '', body)
    body = re.sub(r'\\label\{[^}]*\}', '', body)
    body = re.sub(r'\\(scriptsize|footnotesize|small)(?![A-Za-z])', '', body)

    head = ('\\documentclass{article}\n'
            '\\title{%s}\n\\author{%s}\n\\date{}\n\\begin{document}\n\\maketitle\n\n%s\n\n' % (title, AUTHORS, AFFIL))
    out = head + body + '\n\\end{document}\n'
    path = os.path.join(BUILD, 'main_flat.tex')
    open(path, 'w', encoding='utf-8').write(out)
    return path


# ------------------------------------------------------------------ Word formatting
SERIF, SANS = 'Times New Roman', 'Arial'


def set_font(style, name, size=None, bold=None):
    f = style.font
    f.name = name
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn('w:rFonts'))
    if fonts is None:
        fonts = OxmlElement('w:rFonts'); rpr.append(fonts)
    for a in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
        fonts.set(qn(a), name)
    for a in ('w:asciiTheme', 'w:hAnsiTheme', 'w:cstheme', 'w:eastAsiaTheme'):
        if fonts.get(qn(a)) is not None:
            del fonts.attrib[qn(a)]
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    f.color.rgb = RGBColor(0, 0, 0)


def insert_ordered(parent, child, after):
    """Insert child after the last existing element whose tag is in `after` (schema order), else first."""
    pos = 0
    for i, el in enumerate(parent):
        if el.tag in after:
            pos = i + 1
    parent.insert(pos, child)


TBLPR_BEFORE_BORDERS = {qn('w:' + t) for t in ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize',
                                                'tblStyleColBandSize', 'tblW', 'jc', 'tblCellSpacing', 'tblInd']}
TCPR_BEFORE_BORDERS = {qn('w:' + t) for t in ['cnfStyle', 'tcW', 'gridSpan', 'hMerge', 'vMerge']}
RPR_BEFORE_SZ = {qn('w:' + t) for t in ['rStyle', 'rFonts', 'b', 'bCs', 'i', 'iCs', 'caps', 'smallCaps', 'strike',
                                         'dstrike', 'outline', 'shadow', 'emboss', 'imprint', 'noProof', 'snapToGrid',
                                         'vanish', 'webHidden', 'color', 'spacing', 'w', 'kern', 'position']}


def border(parent_tag, edges, sz):
    el = OxmlElement(parent_tag)
    for e in edges:
        b = OxmlElement('w:' + e)
        b.set(qn('w:val'), 'single'); b.set(qn('w:sz'), str(sz)); b.set(qn('w:space'), '0'); b.set(qn('w:color'), '000000')
        el.append(b)
    return el


def format_docx(path):
    doc = Document(path)
    for st in doc.styles:
        if st.type != 1:
            continue
        n = st.name
        if n.startswith('Heading'):
            lvl = int(re.sub(r'\D', '', n) or 1)
            set_font(st, SANS, {1: 13, 2: 11.5}.get(lvl, 11), True)
            st.paragraph_format.space_before = Pt(12 if lvl == 1 else 8)
            st.paragraph_format.space_after = Pt(4)
        elif n == 'Title':
            set_font(st, SANS, 16, True)
            st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif n in ('Author', 'Date', 'Subtitle'):
            set_font(st, SANS, 11, False)
            st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif n in ('Image Caption', 'Table Caption', 'Caption'):
            set_font(st, SERIF, 10, False)
            st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            st.paragraph_format.line_spacing = 1.1
            st.font.italic = False
        elif n == 'Compact':
            set_font(st, SERIF, 10)
        else:
            set_font(st, SERIF, 11)
            if n in ('Normal', 'Body Text', 'First Paragraph'):
                st.paragraph_format.line_spacing = 1.3
                st.paragraph_format.space_after = Pt(6)
                st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    # affiliation line under the author; reference entries with a hanging indent
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt.startswith('Department of Cyber Security'):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = SANS; r.font.size = Pt(10)
        elif re.match(r'\[\d+\] ', txt):
            p.paragraph_format.left_indent = Cm(1.0)
            p.paragraph_format.first_line_indent = Cm(-1.0)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.1
            for r in p.runs:
                r.font.size = Pt(10)
        if p._p.xpath('.//w:drawing'):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
    # tables: booktabs-like rules, 9 pt, full width, kept on one page with their caption
    for t in doc.tables:
        prev = t._tbl.getprevious()
        if prev is not None and prev.tag == qn('w:p'):
            Paragraph(prev, t._parent).paragraph_format.keep_with_next = True
        for row in t.rows[:-1]:
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.keep_with_next = True
        tblPr = t._tbl.tblPr
        for old in tblPr.findall(qn('w:tblBorders')):
            tblPr.remove(old)
        w = tblPr.find(qn('w:tblW'))
        if w is None:
            w = OxmlElement('w:tblW'); insert_ordered(tblPr, w, {qn('w:tblStyle'), qn('w:tblpPr'), qn('w:tblOverlap'),
                                                                 qn('w:bidiVisual'), qn('w:tblStyleRowBandSize'),
                                                                 qn('w:tblStyleColBandSize')})
        w.set(qn('w:type'), 'pct'); w.set(qn('w:w'), '5000')
        insert_ordered(tblPr, border('w:tblBorders', ['top', 'bottom'], 12), TBLPR_BEFORE_BORDERS)
        rows = t.rows
        if rows:
            for c in rows[0].cells:
                tcPr = c._tc.get_or_add_tcPr()
                for old in tcPr.findall(qn('w:tcBorders')):
                    tcPr.remove(old)
                insert_ordered(tcPr, border('w:tcBorders', ['bottom'], 6), TCPR_BEFORE_BORDERS)
        for row in rows:
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.line_spacing = 1.0
                    for r in p.runs:
                        r.font.size = Pt(9)
                for mr in c._tc.iter(qn('m:r')):
                    rpr = mr.find(qn('w:rPr'))
                    if rpr is None:
                        rpr = OxmlElement('w:rPr'); insert_ordered(mr, rpr, {qn('m:rPr')})
                    for old in rpr.findall(qn('w:sz')):
                        rpr.remove(old)
                    sz = OxmlElement('w:sz'); sz.set(qn('w:val'), '18')
                    insert_ordered(rpr, sz, RPR_BEFORE_SZ)
    # page set-up and page numbers
    for sec in doc.sections:
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        sec.left_margin = sec.right_margin = Cm(2.5)
        sec.top_margin = sec.bottom_margin = Cm(2.5)
        fp = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = fp.add_run()
        for tag, txt in [('begin', None), (None, 'PAGE'), ('end', None)]:
            if tag:
                fc = OxmlElement('w:fldChar'); fc.set(qn('w:fldCharType'), tag); run._r.append(fc)
            else:
                it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = txt; run._r.append(it)
    doc.save(path)


def build(out_path):
    flat = flatten()
    pypandoc.convert_file(flat, 'docx', format='latex', outputfile=out_path,
                          extra_args=['--resource-path=' + BUILD])
    format_docx(out_path)
    print('wrote', out_path)
    return out_path


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BUILD, 'FogField.docx'))
