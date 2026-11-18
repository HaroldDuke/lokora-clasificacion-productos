"""Convierte los informes en Markdown de esta carpeta a LaTeX.

Uso: python3 md_a_latex.py avance_i.md [avance_ii.md ...]
Genera un .tex por cada .md. Para el PDF: tectonic avance_i.tex
"""
import re
import sys
from pathlib import Path

ESC = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
       '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
SIMB = {'→': r'$\rightarrow$', '≥': r'$\geq$', '×': r'$\times$', '–': '--', '—': '---'}


def esc(t):
    t = ''.join(ESC.get(c, c) for c in t)
    for a, b in SIMB.items():
        t = t.replace(a, b)
    return t


def inline(t):
    partes = re.split(r'(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*|https?://[^\s)]+)', t)
    out = []
    for p in partes:
        if not p:
            continue
        if p.startswith('`'):
            out.append(r'\texttt{' + esc(p[1:-1]) + '}')
        elif p.startswith('**'):
            out.append(r'\textbf{' + inline(p[2:-2]) + '}')
        elif p.startswith('*') and len(p) > 2:
            out.append(r'\emph{' + inline(p[1:-1]) + '}')
        elif p.startswith('http'):
            url = p.rstrip('.,;')
            out.append(r'\url{' + url + '}' + esc(p[len(url):]))
        else:
            out.append(esc(p))
    return ''.join(out)


def tabla(filas, alin):
    if alin[0] != 'r' and all(a == 'r' for a in alin[1:]) and len(alin) > 2:
        resto = 0.58 / (len(alin) - 1)
        cols = r'>{\raggedright\arraybackslash}p{0.28\textwidth}' + ''.join(
            r'>{\raggedleft\arraybackslash}p{%.3f\textwidth}' % resto for _ in alin[1:])
    else:
        cols = ''.join('r' if a == 'r' else 'X' for a in alin)
    s = [r'\begin{center}\small', r'\begin{tabularx}{\textwidth}{' + cols + '}', r'\toprule']
    for i, f in enumerate(filas):
        celdas = [inline(c) for c in f]
        if i == 0:
            celdas = [r'\textbf{' + c + '}' for c in celdas]
        s.append(' & '.join(celdas) + r' \\')
        if i == 0:
            s.append(r'\midrule')
    s += [r'\bottomrule', r'\end{tabularx}', r'\end{center}']
    return '\n'.join(s)


PREAMBULO = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[spanish]{babel}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs,tabularx,graphicx,hyperref,enumitem,parskip}
\hypersetup{hidelinks}
\setlength{\parskip}{0.6em}
\begin{document}
"""


def convertir(md):
    lineas = md.read_text().splitlines()
    out = [PREAMBULO]
    h1 = False
    titulo = True
    i = 0
    while i < len(lineas):
        l = lineas[i]
        if not l.strip():
            i += 1
            continue
        if l.startswith('#'):
            n = len(l) - len(l.lstrip('#'))
            t = inline(l[n:].strip())
            if titulo:
                out.append(r'{\LARGE\bfseries ' + t + r'\par}\vspace{0.4em}')
                titulo = False
            elif n == 1:
                h1 = True
                out.append(r'\section*{' + t + '}')
            else:
                out.append((r'\subsection*{' if h1 else r'\section*{') + t + '}')
        elif l.startswith('!['):
            m = re.match(r'!\[(.*?)\]\((.*?)\)', l)
            img = 'figuras/' + Path(m.group(2)).name
            out.append(r'\begin{figure}[htbp]\centering' + '\n' +
                       r'\includegraphics[width=0.92\textwidth]{' + img + '}\n' +
                       r'{\small ' + inline(m.group(1)) + r'\par}' + '\n' + r'\end{figure}')
        elif l.startswith('|'):
            filas, alin = [], None
            while i < len(lineas) and lineas[i].startswith('|'):
                c = [x.strip() for x in lineas[i].strip().strip('|').split('|')]
                if all(re.fullmatch(r':?-+:?', x) for x in c):
                    alin = ['r' if x.endswith(':') else 'l' for x in c]
                else:
                    filas.append(c)
                i += 1
            out.append(tabla(filas, alin or ['l'] * len(filas[0])))
            continue
        elif re.match(r'\d+\. ', l):
            out.append(r'\begin{enumerate}[nosep]')
            while i < len(lineas) and re.match(r'\d+\. ', lineas[i]):
                out.append(r'\item ' + inline(re.sub(r'^\d+\. ', '', lineas[i])))
                i += 1
            out.append(r'\end{enumerate}')
            continue
        else:
            out.append(r'\noindent ' + inline(l) + r'\par')
        i += 1
    out.append(r'\end{document}')
    destino = md.with_suffix('.tex')
    destino.write_text('\n'.join(out) + '\n')
    print('escrito', destino)


for a in sys.argv[1:]:
    convertir(Path(a))
