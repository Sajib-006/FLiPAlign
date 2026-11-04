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
```

All checks pass with zero failures; the logs of our run are `proof_checks/check_split.log` and
`proof_checks/check_reduction.log`.

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
| `paper_scripts/` | tables and figures of the paper, from `results/` |
| `results/` | saved results behind every table and figure |

## Data

All data are public. Put them under `data/` as the loaders expect:

- Allen Brain Cell Atlas MERFISH, mouse C57BL6J-638850 (`data/abc/`): expression matrices, cell metadata and
  reconstructed coordinates from the Allen Brain Cell Atlas.
- 10x Genomics Xenium datasets: human breast (Rep1, Rep2), mouse brain, kidney, lymph node, liver, heart, tonsil,
  pancreas, lung, skin, mouse pup, human brain (`data/xenium/`, `data/xen3/`, `data/xen4/`).
- Human breast Visium (CytAssist) from the same block as the Xenium breast sections (`data/cytassist/`).
- Human colon Visium and Visium HD (`data/crc/`).
- Human DLPFC Visium, 12 sections (`data/dlpfc/`).
- Human lung Visium HD and Xenium (lung pairs 6 and 7).

The appendix of the paper lists each dataset, its role and the exact cases.

## Reproducing the results

Each script prints its usage in its first lines. The placement runs used 16 cores per case (AMD EPYC, no GPU) and up
to three hours per search; the stack runs take minutes. `paper_scripts/make_results3.py` and
`paper_scripts/make_stack_results.py` rebuild the tables from `results/`.

## License

MIT
