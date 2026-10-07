# PlatePrior

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23154571.svg)](https://doi.org/10.5281/zenodo.23154571)

Plate-wise batch Bayesian optimization of culture media with a pretrained tabular model.

Built with PriorLabs-TabPFN.

PlatePrior chooses the conditions of a 96-well plate with TabPFN, a pretrained tabular
model used without retraining, and the Kriging believer rule: it selects four conditions,
treats the model's predictions for them as provisional measurements, updates its
predictions and repeats until the plate is full. In the code this pipeline is the method
`tabpfn_kb` (`strategies.py`).

This repository contains the implementation, the simulation benchmark, the run-level
results of all 8,920 simulated campaigns, and the scripts that regenerate every table and
figure of the accompanying manuscript:

> Jun Sese. A tabular foundation model finds higher-biomass media with limited experimental
> feedback in simulated culture-medium optimization. Manuscript submitted.

All results are from simulation; no laboratory data are included.

## Repository layout

| Path | Contents |
|---|---|
| `simulator.py` | Kinetic growth model (Haldane kinetics; essential components combined by Liebig's law of the minimum), 96-well plate, measurement noise and edge effect |
| `embed.py` | Embedding controls: an eight-component growth model presented as more variables (padded or mixed) |
| `surrogates.py` | TabPFN and scikit-learn Gaussian-process surrogates behind a common interface |
| `surrogate_botorch.py` | BoTorch Gaussian process with the dimension-scaled length-scale prior (DS GP) |
| `strategies.py` | Plate-filling rules (Kriging believer, local penalization) and the method registry |
| `botorch_arm.py` | GP qLogNEI |
| `highdim_arms.py` | Simplified, plate-adapted versions of TuRBO, SAASBO and GIT-BO (Fixed TR TS, SAAS qLogNEI, FD subspace KB); not faithful reproductions |
| `driver.py`, `driver_v2.py` | One campaign: an initial Latin hypercube plate and four adaptive plates. The results use protocol v2 (`driver_v2.py`) |
| `modal_app.py` | Runs the benchmark on the Modal cloud platform |
| `results/` | Run-level results, one CSV per phase (CC BY 4.0) |
| `analysis/` | Scripts that regenerate the summary tables (`analysis/tables/`) and figures (`analysis/figures/`) |

Some code comments are in Japanese and refer to internal design notes that are not part of
this repository.

## Reproducing the tables and figures

This needs only the CSV files in `results/` and takes a few minutes on a laptop; no GPU or
cloud account is required.

```sh
uv sync
uv run python analysis/reanalyze.py         # main evaluation, embedding controls, curves
uv run python analysis/grid_table.py        # factorial simulation
uv run python analysis/structural_table.py  # comparison with structural baselines
uv run python analysis/make_figures.py      # Figs 1-3 (PNG and TIFF)
uv run python analysis/parameter_associations.py  # post hoc associations (S1 Fig, S3 Table)
```

The outputs in `analysis/tables/` and `analysis/figures/` are committed, so they can be
compared with a fresh run. `analysis/tables/manifest.json` and
`analysis/tables/parameter_associations/manifest.json` record the SHA-256 of each input. The figures use Arial; on systems without it, matplotlib substitutes another font.

| Manuscript item | Source files (in `analysis/tables/` unless stated) |
|---|---|
| Table 2 | `main_contrasts.csv`, `raw_sensitivity.csv` |
| Table 3, S1 Table | `scores.csv` |
| Table 4 | `embedding_contrasts.csv` |
| Table 5 | `grid_contrasts.csv`, `grid_regression.csv` |
| Table 6 | `structural_contrasts.csv` |
| S2 Table | `raw_sensitivity.csv`, `main_contrasts.csv` |
| Fig 1 | `raw_sensitivity.csv` (observed benefit panel) |
| Fig 2 | `raw_sensitivity.csv`, `embedding_contrasts.csv`; scenario-level points from `results/` (written to `analysis/figures/Fig2_scenario_effects.csv`) |
| Fig 3 | `curves.csv` |
| S1 Fig, S3 Table | `parameter_associations/associations.csv`, `scenario_parameters.csv`, `sensitivity.csv` |

## Data

Each file `results/v2a_runs_<phase>.csv` has one row per round (0-4) of each campaign.

| Phase | Experiment in the manuscript |
|---|---|
| `P1A` | Main evaluation with 8, 20 and 40 components (SAAS qLogNEI at 40) |
| `P3` | Main evaluation with 80 and 120 components |
| `B1` | Embedding controls |
| `G1` | Factorial simulation (essential probability x weight concentration) |
| `B0`, `P2`, `S`, `SM` | Pilot and smoke-test runs; excluded from all estimates |

Main columns: `method`, `scenario`, `seed`, `round`, `n_sub` (number of components),
`best_true` (noise-free biomass of the best condition tested so far), `best_obs` (best
noisy measurement so far), `config`, `embed`, `ess_p` and `w_conc` (factorial levels),
`phase`, `device_actual`, `wall_s` (seconds to fit the model and propose the plate in that
round), `tabpfn_model` and `ckpt_sha`
(TabPFN checkpoint).

## Rerunning the simulations

Rerunning requires a [Modal](https://modal.com) account; the TabPFN methods run on GPUs.

```sh
uv sync --extra tabpfn --extra botorch --extra modal
uv run modal setup
uv run modal run modal_app.py --phase P1A --out results/rerun.csv   # writes results/rerun_P1A.csv
```

Use `--dry-run` to print the number of runs without starting them. The cloud runs used
Python 3.12, PyTorch 2.14, the `tabpfn` package 9.0.0 with the TabPFN v2 regression
checkpoint, and BoTorch 0.18.1. The TabPFN model weights are not included in this
repository; the `tabpfn` package downloads them, and they are subject to Prior Labs'
license terms.

## License

- Code: MIT License (`LICENSE`)
- Data, summary tables and figures (`results/`, `analysis/tables/`, `analysis/figures/`):
  CC BY 4.0 (`results/LICENSE.md`)
- TabPFN is a third-party dependency and is not included here. The code uses the TabPFN v2
  regression checkpoint by default, which is distributed under the Prior Labs License 1.1
  (Apache 2.0 with an attribution requirement). Later TabPFN generations, selectable through
  the `PBO_TABPFN_MODEL` environment variable, are distributed under separate Prior Labs
  licenses, some of which restrict commercial use. Check the license of the weights you use.

## Citation

Please cite the manuscript above. To cite this software and data, use the Zenodo record:
version 1.0.1 (the version cited in the manuscript), doi:[10.5281/zenodo.23200480](https://doi.org/10.5281/zenodo.23200480);
all versions,
doi:[10.5281/zenodo.23154571](https://doi.org/10.5281/zenodo.23154571) (see also `CITATION.cff`).

The code was developed with the help of AI coding assistants and reviewed by the author.
