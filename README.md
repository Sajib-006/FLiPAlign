# FLiPAlign

Code for the paper *Placing and labelling tissue sections across spatial platforms without shared labels*.

Two serial tissue sections, often measured on different spatial platforms, are each clustered on their own, so
their region labels share no names. FLiPAlign places one section on the other by searching every rotation, mirror
flip and shift. Then it either proves a margin for its answer (a score certificate) or refuses to answer. A second
part labels a whole stack of sections with one set of labels, with an optional split of one label into two, and
gives a proven upper bound for each stack.

## Install

Python 3.10 or later.

```
pip install -r requirements.txt
```

`requirements.txt` pins the versions used on the cluster for the paper (numpy 2.2.6, scipy 1.18.1, scikit-learn 1.9.1,
pandas 3.0.6, h5py 3.16.0, matplotlib 3.11.2).

This covers FLiPAlign, the stack labelling and the proof checks. The published tools used as baselines need their
own packages (paste-bio, POT, anndata, scSLAT, spateo, CAST, STalign); we ran them in separate environments.

## Quick example (about one minute, no download)

```
python3 example.py
```

The script builds a synthetic pair. Section A is 70% of section B, turned by 37 degrees, mirrored and shifted, and
its labels are renamed at random. Expected output:

```
best score 0.491, proven search gap 0.060
answer: turn 38.5 deg, flip 1, shift (-16.0, 2.0) units
certified, margin eps = 0.015
median error of the placed cells: 2.21 units
```

To use your own data, pass two coordinate arrays (in units of the cell or spot spacing) and two integer label
arrays to `SmoothAlign(A, la, B, lb, h=2.0, sigma=20.0, tpad=300.0, contrast=True)`, then call `.solve()` and
`far_upper_bound(...)` as in `example.py`. `smoothpar.py` has the parallel versions (Linux).

## Proof checks (no data needed)

The folder `proof_checks/` holds the brute-force checks of the complexity results, including the finite
enumeration used in the split-labelling proof:

```
cd proof_checks
python3 check_split.py          # split labelling: value lemma, reduction from 3DM, stack bound (about 25 min, needs SciPy)
python3 check_reduction.py      # cycle hardness reduction (about 1 min)
python3 check_approx.py         # forest cover ratio (about 2 min)
python3 e2_backward_checks.py   # backward direction of the reductions, tight family (a few min)
python3 check_zero_marginal.py   # split hardness on chance-corrected (zero-sum) tables (about 1 hour)
```

All checks pass with zero failures; the logs of our run are `proof_checks/check_split.log`,
`proof_checks/check_reduction.log` and `proof_checks/check_zero_marginal.log`.

`test_pairbound.py` checks the pair upper bound used by the stack certificate. `check_fft.py` and
`lambda_check.py` check the Fourier scores against direct sums and the movement bound.

## Files

| Files | What they do |
| --- | --- |
| `smoothalign.py`, `smoothpar.py`, `finecert.py` | FLiPAlign: label maps, search, bound, decision, fine stage |
| `bench_gt.py`, `bench_gt2.py`, `bench_domains.py` | known-answer cases and region labels |
| `run_compare.py`, `run_colon.py`, `real_fine.py`, `lung_load.py` | real cross-platform pairs |
| `bench_baselines.py`, `baselines.py`, `bench_landmark_tools.py` | published tools (PASTE, PASTE2, SLAT, CPD, spateo, CAST, STalign) |
| `landmark_common.py`, `landmark_eval_flip.py`, `landmarks/` | hand-placed landmarks and landmark error |
| `stack_jlp.py` | one-to-one stack labelling (chain, forest, star) |
| `stack_cap.py` | split labelling (Algorithm 1 of the paper) |
| `stack_cap_bound.py` | stack certificate from proven pair bounds (integer programs) |
| `stack_v2.py` | disjoint gene panels, class-matched control, gene baseline |
| `stack_place_load.py`, `real_fine_stack.py`, `stack_from_placed.py` | placement and labelling together |
| `stack_ctrl.py` | cap 1 against cap 2 under the same score |
| `stack_refuse.py` | the two steps together, cutting the stack at refused links |
| `stack_sweep.py` | caps 1 to 4 and class counts 8 and 12 |
| `stack_brain2.py`, `stack_brain2_agg.py`, `stack_consistency.py` | second brain (Zhuang MERFISH) and the atlas-free agreement score |
| `stack_staligner.py` | STAligner baseline on the stacks (GPU) |
| `gala_prep.py`, `gala_run.py`, `gala_summary.py` | GALA baseline on the seven real pairs (GPU) |
| `paper_scripts/` | tables and figures of the paper, from `results/` (`rebuild_all.sh`) |
| `results/` | saved results behind every table and figure |

## Paper settings

The paper used `SmoothAlign(A, la, B, lb, h=2.0, sigma=20.0, tpad=300.0, contrast=True)`. The defaults in
`smoothalign.py` are different (h=4.0, sigma=10.0, tpad=30.0, contrast=False), so pass the paper values as shown in
the example above. The benchmark scripts set their values in the code, and `example.py` uses its own values for the small
synthetic pair.

## Data

All data are public. Put them under `data/` as the loaders expect (the folder is not part of this repository):

- Allen Brain Cell Atlas MERFISH, mouse C57BL6J-638850 (`data/abc/`): expression matrices, cell metadata and
  reconstructed coordinates. Download from the Allen Brain Cell Atlas bucket
  <https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/> (directory `MERFISH-C57BL6J-638850`, with the
  metadata and `-CCF` coordinate directories).
- 10x Genomics Xenium datasets: human breast (Rep1, Rep2), mouse brain, kidney, lymph node, liver, heart, tonsil,
  pancreas, lung, skin, mouse pup, human brain (`data/xenium/`, `data/xen3/`, `data/xen4/`). Download from the 10x
  Genomics datasets page <https://www.10xgenomics.com/datasets>.
- Human breast Xenium and Visium (CytAssist) from the same block (`data/cytassist/`): Janesick et al. 2023 (the
  10x Genomics breast cancer Xenium and Visium dataset, listed on the page above).
- Human colon Visium and Visium HD (`data/crc/`): Oliveira et al. 2025 (10x Genomics Visium HD colorectal cancer
  datasets, listed on the page above).
- Human DLPFC Visium, 12 sections (`data/dlpfc/`): spatialLIBD, <http://spatial.libd.org/spatialLIBD/>.
- Human lung Visium HD and Xenium (lung pairs 6 and 7): 10x Genomics datasets page.

The appendix of the paper lists each dataset, its role and the exact cases.

## Reproducing the results

Each script prints its usage in its first lines. The placement runs used 16 cores per case (AMD EPYC, no GPU) and up
to three hours per search; the stack runs take minutes.

All tables, number macros and figures of the paper are rebuilt from the saved results, with no raw data and no GPU:

```
cd paper_scripts
bash rebuild_all.sh
```

This takes a few minutes and writes `paper_scripts/out/tables/` and `paper_scripts/out/figures/`. Set `RESULTS=/path`
to read results from another folder (default: `results/`) and `OUT=/path` to write elsewhere. The layout of
`results/` is: `placement/` (one file per case, tool and start), `stack/` (one file per stack) and the top-level
`results_*.json`, `gt_manifest.json` and `boot_sections.json`.

| Script | Output |
| --- | --- |
| `make_results.py`, `make_results2.py`, `make_results3.py` | placement tables and numbers (`res_*`, `res2_*`, `res3_*`) |
| `boot_sections.py` | section-level bootstrap of the held-out counts |
| `make_stack_results.py` | main stack table (columns Best and Cap 1), `stack_numbers.tex`, stack figure |
| `make_stack_extras.py` | Table 4 (disjoint gene panels) and the end-to-end numbers (`stackv2*`, `stackplace_numbers.tex`) |
| `make_review_tables.py` | supplement tables and numbers for cap 1, the cap sweep, the end-to-end test (`rev_*`, `review_numbers.tex`) |
| `make_tables.py`, `make_new.py` | older tables and the refusal figure |
| `make_figures.py`, `make_fig_results.py`, `make_fig_landmarks.py`, `make_fig_overview.py` | figures |
| `fit_tables.py` | wraps wide supplement tables to the text width (needs the paper source for `appendix.tex`, else skipped) |

Two steps need the LaTeX source of the paper and are skipped with a message when it is absent: the appendix
reference list (`res3_appbib.tex`) and the `appendix.tex` width fix. The rebuilt tables match the ones in the paper.

## License

MIT
