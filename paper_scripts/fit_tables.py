"""Keep every supplement table inside the text width (2026-10-28).
Wraps each top-level tabular in adjustbox{max width=\linewidth} (shrinks only when too wide), and shortens the
widest tables first so they do not shrink to an unreadable size: "median (lo--hi)" cells go on two lines, long row
names and "a / b / c" spacing are shortened. Safe to run more than once.
Run from paper_scripts/ (env RESULTS, OUT): python3 fit_tables.py   (make_results3.py also calls it at the end)"""
import os
from _paths import RES, PLACE, STACK, OUT, TABLES, FIGURES
import re

FILES = [os.path.join(os.environ.get("PAPER_DIR", "."), "appendix.tex")] + [
    os.path.join(TABLES, f) for f in ("res3_appendix.tex", "res3_datasets.tex", "res3_tools.tex", "res3_heldout.tex",
                                       "res3_landmarks.tex", "stack_results_ari.tex")]
OPEN, CLOSE = "\\begin{adjustbox}{max width=\\linewidth}", "\\end{adjustbox}"


def wrap(txt):
    """wrap top-level tabular environments that are not wrapped yet"""
    out, depth, i = [], 0, 0
    tok = re.compile(r"\\begin\{tabular\}|\\end\{tabular\}")
    for m in tok.finditer(txt):
        out.append(txt[i:m.start()]); i = m.end()
        if m.group(0).startswith("\\begin"):
            if depth == 0 and not "".join(out).rstrip().endswith(OPEN):
                out.append(OPEN)
            out.append(m.group(0)); depth += 1
        else:
            depth -= 1; out.append(m.group(0))
            if depth == 0 and not txt[i:].lstrip().startswith(CLOSE):
                out.append(CLOSE)
    out.append(txt[i:])
    return "".join(out)


def two_line(txt):
    """'0.151 (0.086--0.282)' -> two-line cell, inside the gap table only"""
    a = txt.find("\\label{tab:gaps}")
    if a < 0: return txt
    s = txt.rfind("\\begin{table}", 0, a)
    body = re.sub(r"(?<![{\w.])(\d+\.\d+) \((\d+\.\d+--\d+\.\d+)\)", r"\\makecell[r]{\1\\\\(\2)}", txt[s:a])
    body = body.replace("margin $\\eps$, first zone", "\\makecell[r]{margin $\\eps$\\\\first zone}")
    return txt[:s] + body + txt[a:]


SHORT = [(", flip searched &", ", searched &"), (", flip known &", ", known &"),
         ("First held-out set &", "First set &"), ("Second held-out set &", "Second set &"),
         ("Allen MERFISH, whole sections &", "Allen MERFISH, whole &")]


def shorten(txt):
    for a, b in SHORT: txt = txt.replace(a, b)
    return re.sub(r"(\d+) / (\d+)(?: / (\d+))?(?= &| \\\\)", lambda m: "/".join(g for g in m.groups() if g), txt)


def main():
    for f in FILES:
        if not os.path.exists(f):
            print("skip (not found)", f); continue
        t = open(f).read()
        n = t
        if os.path.basename(f) == "appendix.tex":
            n = re.sub(r"\\resizebox\{\\linewidth\}\{!\}\{(\\input\{[^}]+\})\}", r"\1", n)
        else:
            n = shorten(two_line(n))
        n = wrap(n)
        if n != t: open(f, "w").write(n); print("fixed", f)


if __name__ == "__main__":
    main()
