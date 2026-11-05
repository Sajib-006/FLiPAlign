# FLiPAlign

Code for the paper *Placing and labelling tissue sections across spatial platforms without shared labels*.

Serial tissue sections are often measured on different spatial platforms, and each section is clustered on its
own, so their region labels share no names. This repository has two methods:

- **FLiPAlign** places one section on another. It searches every rotation, mirror flip and shift. Then it either
  proves that no placement far from its answer scores as well (a score certificate), or it refuses to answer.
- **Split labelling** gives a whole stack of sections one set of labels. It lets one label split into two, and it
  gives a proven upper bound for each stack.

The repository also has the brute-force checks of the complexity proofs, the saved results behind every table and
figure, and one script that rebuilds all tables and figures from those results.

## Start here

| Goal | Command | Time | Needs |
| --- | --- | --- | --- |
| See the method work | `python3 example.py` | 1 min | nothing |
| Rebuild every table and figure of the paper | `bash paper_scripts/rebuild_all.sh` | 5 min | nothing (uses saved results) |
| Check the complexity proofs | `cd proof_checks && python3 check_reduction.py` (others below) | 1 min to hours | nothing |
| Rerun an experiment | see [Where each result comes from](#where-each-result-comes-from) | minutes to hours | public data |

## Repository layout

```
example.py            small synthetic example (start here)
repo_paths.py         puts the code folders on the import path (every script imports it)
requirements.txt      pinned package versions

flipalign/            THE PLACEMENT METHOD
  smoothalign.py        label maps, branch-and-bound search, the bound, certificate or refusal
  smoothpar.py          parallel versions of the search and the bound (Linux)
  finecert.py           fine stage of the certificate
  certalign.py, alignfast.py   earlier versions, still used by some data loaders

stacklabel/           THE STACK LABELLING METHOD
  stack_jlp.py          overlap tables, one-to-one labelling (chain, forest, star), scores
  stack_cap.py          split labelling (Algorithm 1 of the paper)
  stack_cap_bound.py    proven upper bound for each stack (integer programs)

runs_placement/       placement experiments (known-answer sets, real pairs, landmarks)
runs_stacks/          stack experiments (controls, sweeps, second brain, the two steps together)
runs_baselines/       the published tools we compare with (PASTE, PASTE2, SLAT, CPD, spateo, CAST, STalign,
                      GALA, STAligner)
checks/               numerical checks (Fourier scores, movement bound, pair bound)
proof_checks/         brute-force checks of the complexity proofs

landmarks/            hand-placed landmarks of the seven real pairs
results/              saved results behind every table and figure
paper_scripts/        build the tables and figures of the paper from results/
```

## Install

Python 3.10 or later.

```
pip install -r requirements.txt
```

`requirements.txt` gives lower bounds; the paper's runs used numpy 2.2.6, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6,
h5py 3.16.0 and matplotlib 3.11.2. This is enough for FLiPAlign, split labelling, the proof checks and the rebuild of
all tables and figures. The published tools need their own packages (paste-bio, POT, anndata, scSLAT, spateo, CAST,
STalign, GALA, STAligner). We ran each of them in its own environment.

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

To use your own data, pass two coordinate arrays (in units of the cell or spot spacing) and two integer label arrays
to `SmoothAlign(A, la, B, lb, h=2.0, sigma=20.0, tpad=300.0, contrast=True)`, then call `.solve()` and
`far_upper_bound(...)` as in `example.py`.

## Rebuild every table and figure (no data, no GPU)

```
bash paper_scripts/rebuild_all.sh
```

This reads `results/` and writes `paper_scripts/out/tables/` and `paper_scripts/out/figures/`. It takes a few
minutes, and the rebuilt tables are identical to the ones in the paper. Set `RESULTS=/path` or `OUT=/path` to read
or write elsewhere. Two small steps need the LaTeX source of the paper (the appendix reference list and a table width
fix); they are skipped with a message when it is absent.

## Where each result comes from

Run every experiment script from the repository root, for example `python3 runs_stacks/stack_ctrl.py 1`. Each
script prints its usage in its first lines. Experiment scripts write new results to `results/`. The shipped results
are sorted into `results/placement/` (one file per case, tool and start) and `results/stack/` (one file per stack).
The table and figure scripts in the last column are all in `paper_scripts/`.

**Main text**

| Paper item | Experiment scripts | Table or figure script |
| --- | --- | --- |
| Fig. 1, the method on real data | `runs_stacks/fig_data.py`, `flipalign/smoothalign.py` | `make_fig_method.py` |
| Table 1, complexity | proofs in the supplement, checked by `proof_checks/` | written by hand |
| Fig. 2, placement of held-out sections and the certificate | `runs_placement/bench_gt.py`, `runs_placement/bench_gt2.py`, `runs_baselines/bench_baselines.py`; example case: `runs_placement/fig_case_data.py` | `make_fig_placement.py` |
| Fig. 3, seven real pairs scored on landmarks | `runs_placement/real_fine.py`, `runs_placement/landmark_eval_flip.py`, `runs_baselines/bench_landmark_tools.py`, `runs_baselines/gala_*.py`; maps: `runs_placement/fig_realpairs_data.py` | `make_fig_realpairs.py` |
| Fig. 4, stack labelling | `stacklabel/`, `runs_stacks/stack_ctrl.py`, `runs_stacks/stack_sweep.py`, `runs_stacks/stack_brain2.py`, `runs_stacks/stack_consistency.py`, `runs_stacks/stack_refuse.py` | `make_fig_stacks.py` |
| Section-level bootstrap | saved placement results | `boot_sections.py` |

**Supplement**

| Supplement item | Experiment scripts | Table or figure script |
| --- | --- | --- |
| Placement counts per tool and start, certificate precision | the placement runs above | `make_results3.py`, `make_review_tables.py` |
| Datasets, tool settings, held-out sets, landmark table | the placement runs above | `make_results3.py` |
| Results by start and tool (figure) | the placement runs above | `make_fig_results.py` |
| Stack table in numbers, full one-to-one table, ARI table, stack figure | `stacklabel/`, `runs_stacks/stack_ctrl.py` | `make_stack_results.py` |
| Stack example figure | `runs_stacks/fig_data.py` | `make_fig_overview.py` |
| Disjoint gene panels, class-matched control, STAligner | `runs_stacks/stack_v2.py`, `runs_baselines/stack_staligner.py` | `make_stack_extras.py` |
| Cap-1 control | `runs_stacks/stack_ctrl.py` | `make_review_tables.py` |
| Cap and class-count sweep | `runs_stacks/stack_sweep.py` | `make_review_tables.py` |
| Second brain and atlas-free agreement (kappa) | `runs_stacks/stack_brain2.py`, `runs_stacks/stack_brain2_agg.py`, `runs_stacks/stack_consistency.py` | `make_review_tables.py` |
| The two steps together (refused links cut the stack) | `runs_stacks/stack_place_load.py`, `runs_stacks/real_fine_stack.py`, `runs_stacks/stack_from_placed.py`, `runs_stacks/stack_refuse.py` | `make_stack_extras.py`, `make_review_tables.py` |
| GALA on the seven real pairs | `runs_baselines/gala_prep.py`, `runs_baselines/gala_run.py` | `runs_baselines/gala_summary.py` |
| Refusal figure, older tables | the placement runs above | `make_new.py`, `make_tables.py`, `make_figures.py` |
| Run time of the stack methods | `runs_stacks/time_stack.py` | numbers in the text |
| Fourier score against direct sums, movement bound | `checks/check_fft.py`, `checks/lambda_check.py` | numbers in the text |
| Pair bound used by the stack certificate | `checks/test_pairbound.py` | numbers in the text |

## Result file names

`results/placement/` holds one JSON file per case, tool and start. The prefix says which test it belongs to:

| Prefix | Meaning |
| --- | --- |
| `gt1`, `gt2` | development cases (breast; colon, mouse brain and DLPFC) |
| `gt3`, `gt4` | first and second (clean) held-out sets |
| `gtNh2` | FLiPAlign, flip searched (`h=2`); `fk_` flip known; `fc_` mirror self-check; `gtNfine`, `fkfine_` fine stage |
| `gtNbase`, `gtNspateo`, `gtNcast`, `gtNstalign`, `gtNslat` | published tools as given (`base` holds PASTE, PASTE2 and CPD) |
| `gtNfo_*`, `rs_*`, `rp_*` | tools handed the true flip, also a rough angle, also a rough position |
| `grid_`, `gtNlocal`, `abl_`, `rl_`, `w7_` | 1 degree grid, local search, score ablations, re-clustering, wider window |
| `abc`, `abc_ov70`, `gt_ovNN`, `fk_ovNN` | Allen MERFISH neighbours and the overlap sweep |
| `lm_*_pairK` | the seven real pairs scored on landmarks (`lm_flip` is FLiPAlign; `lm_given`, `lm_rough`, `lm_pose` the tools from three starts) |

`results/stack/` holds one file per stack and experiment (`stack_`, `stackcap_`, `stackbound2_`, `stackctrl_`,
`stacksweep_`, `stackv2_`, `stackb2_` for brain 2, `stackcons_`, `stackstal_`, `stackplace*`). The script that reads
each prefix is listed in the table above.

## Proof checks (no data needed)

The folder `proof_checks/` holds brute-force checks of the complexity results, including the finite enumeration used
in the split-labelling proof.

```
cd proof_checks
python3 check_reduction.py           # cycle hardness reduction (about 1 min)
python3 check_approx.py              # forest cover ratio (about 2 min)
python3 e2_backward_checks.py        # backward direction of the reductions, tight family (a few min)
python3 check_split.py               # split labelling: value lemma, reduction from 3DM, stack bound (about 25 min)
python3 check_zero_marginal.py       # split hardness on chance-corrected (zero-sum) tables (about 1 hour)
python3 check_zero_marginal_3dm.py   # the same on the 3DM graph of the proof (several hours; cap 3 is slow)
python3 check_no_cap.py             # split hardness on zero-sum tables with no cap (about 3 min)
python3 check_odd_cap.py            # split hardness for every finite cap, non-negative tables (about 8 min)
```

All checks pass with zero failures. The logs of our runs are the `.log` files in the same folder.

## Data

All data are public. Put them under `data/` in the repository root (this folder is not part of the repository):

- Allen Brain Cell Atlas MERFISH, mouse C57BL6J-638850 (`data/abc/`): expression matrices, cell metadata and
  reconstructed coordinates. Download from the Allen Brain Cell Atlas bucket
  <https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/> (directory `MERFISH-C57BL6J-638850`, with the
  metadata and `-CCF` coordinate directories).
- Second mouse brain, Zhuang-ABCA-1 MERFISH (`data/zhuang/`), from the same bucket.
- 10x Genomics Xenium datasets: human breast (Rep1, Rep2), mouse brain, kidney, lymph node, liver, heart, tonsil,
  pancreas, lung, skin, mouse pup, human brain (`data/xenium/`, `data/xen3/`, `data/xen4/`). Download from the 10x
  Genomics datasets page <https://www.10xgenomics.com/datasets>.
- Human breast Xenium and Visium (CytAssist) from the same block (`data/cytassist/`): Janesick et al. 2023 (the
  10x Genomics breast cancer Xenium and Visium dataset, listed on the page above).
- Human colon Visium and Visium HD (`data/crc/`): Oliveira et al. 2025 (10x Genomics Visium HD colorectal cancer
  datasets, listed on the page above).
- Human DLPFC Visium, 12 sections (`data/dlpfc/`): spatialLIBD, <http://spatial.libd.org/spatialLIBD/>.
- Human lung Visium HD and Xenium, lung pairs 6 and 7 (`data/lung/`, or set `LUNG_DIR`): 10x Genomics datasets page.

The appendix of the paper lists each dataset, its role and the exact cases.

## Compute

The placement runs used 16 CPU cores per case (AMD EPYC, no GPU) and up to three hours per search. The stack runs
take minutes on a laptop. Only the GALA and STAligner baselines need a GPU.

## Paper settings

The paper used `SmoothAlign(A, la, B, lb, h=2.0, sigma=20.0, tpad=300.0, contrast=True)`, also available as
`paper_defaults(A, la, B, lb)` in `flipalign/smoothalign.py`. The defaults in
`flipalign/smoothalign.py` are different (h=4.0, sigma=10.0, tpad=30.0, contrast=False), so pass the paper values as
shown above. The experiment scripts set their values in the code, and `example.py` uses its own values for the small
synthetic pair.

## License

MIT
