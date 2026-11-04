#!/usr/bin/env bash
# Rebuild every table and figure of the paper from the saved results.
#   RESULTS=/path/to/results OUT=/path/to/out bash rebuild_all.sh
# Defaults: RESULTS = ../results (this repo), OUT = ./out (tables in out/tables, figures in out/figures).
# Needs numpy, scipy, matplotlib (see ../requirements.txt). No raw data and no GPU. Takes a few minutes.
# A step that fails is reported at the end and does not stop the others. Exit code 1 if any step failed.
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python3}
export RESULTS=${RESULTS:-$(cd .. && pwd)/results}
export OUT=${OUT:-$(pwd)/out}
mkdir -p "$OUT/tables" "$OUT/figures"
echo "results: $RESULTS"; echo "output:  $OUT"
fail=()
run() {
  echo; echo "=== $* ==="
  if ! "$PY" "$@" > "$OUT/log_$1.txt" 2>&1; then echo "FAILED: $1 (see $OUT/log_$1.txt)"; fail+=("$1"); else echo "ok: $1"; fi
}
# tables and macros (make_results3 and make_review_tables import earlier steps)
run make_results.py          # res_*.tex, res_numbers.tex
run make_results2.py         # res2_*.tex (both modes: flip searched and flip known), needed by make_results3
run make_results3.py         # res2_*.tex, res3_*.tex, res3_numbers.tex (also runs fit_tables.py when the paper source is present)
run boot_sections.py         # section-level bootstrap, writes out/boot_sections.json (shipped copy: results/boot_sections.json)
run make_stack_results.py    # stack table (Best, Cap 1 columns), stack_numbers.tex, figures/stack.pdf
run make_stack_extras.py     # Table 4 (stackv2), stackplace_numbers.tex
run make_review_tables.py    # rev_*.tex, review_numbers.tex
run make_tables.py           # older gt_accuracy / landmarks / cross_platform tables
run make_new.py              # new_numbers.tex, figures/refusal.pdf
run fit_tables.py            # keep supplement tables inside the text width (wraps in adjustbox; skips files not present)
# figures
run make_figures.py          # figures/benchmark.pdf, certificate.pdf
run make_fig_results.py      # figures/results.pdf
run make_fig_landmarks.py    # figures/landmarks.pdf
run make_fig_overview.py     # figures/overview2.pdf, stack_example.pdf (needs results/stack/fig_data.npz)
echo
if [ ${#fail[@]} -gt 0 ]; then echo "Steps that failed: ${fail[*]}"; exit 1; fi
echo "All steps finished. Tables: $OUT/tables  Figures: $OUT/figures"
