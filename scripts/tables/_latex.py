"""Number formats and the table writer shared by the scripts that write LaTeX.

Scores and differences carry three decimals, and a difference gets an explicit sign unless it rounds to zero.
Posterior probabilities carry two decimals and saturate at '<0.01' and '>0.99'; p-values carry three and saturate at
'<0.001'. A missing value prints as '--' in tables, and the numbers file passes missing='n/a'.
"""
import math
import os

from wheatrust.stats import verdict as stats_verdict

MINUS = "$-$"


def _missing(x):
    if x is None:
        return True
    try:
        return not math.isfinite(x)
    except TypeError:
        return False


def f3(x, missing="--"):
    return missing if _missing(x) else f"{x:.3f}"


def f2(x, missing="--"):
    return missing if _missing(x) else f"{x:.2f}"


def signed(x, d=3, missing="--"):
    if _missing(x):
        return missing
    s = f"{abs(x):.{d}f}"
    if float(s) == 0:
        return s
    return f"+{s}" if x >= 0 else f"{MINUS}{s}"


def prob(x, missing="--"):
    if _missing(x):
        return missing
    if x < 0.005:
        return "$<$0.01"
    if x > 0.995:
        return "$>$0.99"
    return f"{x:.2f}"


def prob_fine(x, missing="--"):
    """Three decimals, or four when three would round onto a label threshold (0.5, 0.75, 0.95)."""
    if _missing(x):
        return missing
    if x > 0.9995:
        return "$>$0.999"
    if x < 0.0005:
        return "$<$0.001"
    return f"{x:.4f}" if any(abs(x - t) < 0.0005 for t in (0.5, 0.75, 0.95)) else f"{x:.3f}"


def pval(x, missing="--"):
    if _missing(x):
        return missing
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def pm(m, s):
    """Mean plus or minus standard deviation."""
    return "--" if _missing(m) or _missing(s) else f"{m:.3f} $\\pm$ {s:.3f}"


def verdict(pb, pe, pw):
    """Verbal label of the three posteriors (wheatrust.stats.verdict, where the thresholds are defined)."""
    return stats_verdict(pb, pe, pw)


def write_table(path, caption, colspec, header, rows, label, note=None, size="\\small", source=None, placement="H"):
    """A full-width booktabs table in the MDPI layout. Rows are LaTeX row bodies without the closing '\\\\'; rule
    lines (\\midrule, \\cmidrule, \\addlinespace) pass through unchanged. Tables stay where they are cited
    ([H]) unless another float placement is given."""
    lines = [f"% Written by {source} from results/. Rebuild instead of editing." if source else "%",
             f"\\begin{{table}}[{placement}]", f"\\caption{{{caption}\\label{{{label}}}}}",
             "\\begin{adjustwidth}{-\\extralength}{0cm}", "\\centering", size,
             f"\\begin{{tabularx}}{{\\fulllength}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    for r in rows:
        rule = r.strip() in ("\\midrule", "\\toprule", "\\bottomrule") or r.startswith(("\\cmidrule", "\\addlinespace"))
        lines.append(r if rule else r + " \\\\")
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{adjustwidth}"]
    if note:
        lines.append(f"\\noindent{{\\footnotesize {note}}}")
    lines.append("\\end{table}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
