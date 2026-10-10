# Spectral sufficiency of UAV wheat rust classification

Code, per-fold results, and holdout predictions for the paper

> **Assessing the Spectral Sufficiency of UAV Wheat Rust Classification with Views and Band Budgets Derived from One
> Hyperspectral Acquisition.** Saurav K. Aryal and co-authors. Submitted to *Sensors* (MDPI), 2026.

The study asks how much spectral information patch-level classification of wheat rust needs around grain filling.
It compares the hyperspectral cube of a snapshot camera with the RGB and five-band multispectral views derived from
that same cube, and with band budgets of 1 to 32 bands selected inside the training folds, under nested, repeated
cross-validation (10 repeats of 5 folds) with two learners, a FLAML AutoML search and TabPFN v2. Paired Bayesian and
frequentist tests compare every configuration on the same folds.

Everything the paper reports can be rebuilt in about two minutes from the results shipped here, without the image
data. Rebuilding those results from the images is also scripted, and takes roughly 85 CPU-hours.

## Data

The imagery is Task 1 of the **ICPR 2026 competition Beyond Visible Spectrum: AI for Agriculture**. It is not
redistributed here. To obtain it:

1. Open the competition data page at
   <https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026/data>, sign in, and join the
   competition (late entry is open) so that the data can be downloaded.
2. Download and unpack the Task 1 files so that the folder looks like this.

```
data/
  train/HS/   Health_hyper_1.tif ... (600 files, 125-band hyperspectral cubes, 32 x 32 pixels)
  train/MS/   Health_hyper_1.tif ... (600 files, five-band multispectral images, 64 x 64 pixels)
  train/RGB/  Health_hyper_1.png ... (600 files, 8-bit RGB images, 64 x 64 pixels)
  val/HS/     val_000a83c1.tif ...   (300 unlabelled holdout samples)
  val/MS/     val_000a83c1.tif ...
  val/RGB/    val_000a83c1.png ...
```

The class of a training sample (Health, Rust, or Other) is the prefix of its file name. If the data live elsewhere,
point the environment variable `WHEATRUST_DATA` to the folder that contains `train/` and `val/`. Nothing in the data
folder is ever written.

If you use the data, please cite the competition, and follow the terms its organisers set.

```bibtex
@misc{icpr2026agvis,
  author       = {Han, Liangxiu and Huang, Wenjiang and Zhang, Xin and Sobeih, Tam and Shi, Yue and Dong, Yingying},
  title        = {{ICPR} 2026 Competition on Beyond Visible Spectrum: {AI} for Agriculture},
  howpublished = {Competition of the 28th International Conference on Pattern Recognition (ICPR 2026), Lyon, France},
  url          = {https://han-research.gitlab.io/Agvision/},
  year         = {2026}
}
```

When this repository was released the organisers had not yet published their report on the 2026 competition. Please
cite that report as well once it appears.

## Installation

The code was run with Python 3.12 on CPUs only, one thread per process.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

TabPFN needs its v2 classifier checkpoint, which is the package default. Download it once into `models/` (or set
`WHEATRUST_MODELS` to another folder).

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Prior-Labs/TabPFN-v2-clf', 'tabpfn-v2-classifier-finetuned-zk73skhh.ckpt', local_dir='models')"
```

Its SHA-256 is `cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49`. The checkpoint is only needed to
retrain TabPFN, not to rebuild the paper from the shipped results.

## Reproducing the paper

### From the shipped results (about two minutes, no image data needed)

```bash
bash reproduce.sh
```

This runs `scripts/analysis/`, `scripts/tables/`, and `scripts/plots/` in order. It writes the analysis files to
`results/analysis/` and the paper's outputs to `outputs/`.

| Output | What it holds | Script |
|---|---|---|
| `outputs/numbers.tex` | every number quoted in the text, as LaTeX macros | `scripts/tables/01_numbers.py` |
| `outputs/tables/features.tex` | Table 1, the engineered features | `scripts/tables/02_tables.py` |
| `outputs/robustness/headline.tex` | Table 2, the contrasts behind the main statements at three ROPEs and after adjustment over all contrasts | `scripts/tables/04_robustness.py` |
| `outputs/tables/views.tex`, `hr.tex`, `sensors.tex`, `holdout.tex` | Tables 3 to 6 | `scripts/tables/02_tables.py` |
| `outputs/tables/ablation.tex` | Table A1 | `scripts/tables/02_tables.py` |
| `outputs/robustness/contrasts_f1_ci.tex`, `contrasts_hr_ci.tex`, `contrasts_learner_ci.tex` | Tables A2 to A4, every contrast with its credible interval | `scripts/tables/04_robustness.py` |
| `outputs/robustness/ndvi_threshold.tex` | Table A5, the NDVI threshold of the vegetated subset and the NDVI-balanced comparison | `scripts/tables/04_robustness.py` |
| `outputs/robustness/holdout_all.tex`, `label_shift.tex`, `crossbatch.tex` | Tables A6 to A8, all ten submissions, holdout class priors, and transfer between file formats | `scripts/tables/04_robustness.py` |
| `outputs/robustness/robustness.tex` | the numbers of the robustness analyses, as LaTeX macros read after `numbers.tex` | `scripts/tables/04_robustness.py` |
| `outputs/figures/fig_data.pdf` | Figure 1, spectra and band noise | `scripts/plots/01_fig_data.py` |
| `outputs/figures/fig_ndvi.pdf` | Figure 2, patch NDVI by class | `scripts/plots/02_fig_ndvi.py` |
| `outputs/figures/fig_views.pdf` | Figure 3, views and representations | `scripts/plots/03_fig_views.py` |
| `outputs/figures/fig_sensor.pdf` | Figure 4, band budgets and selected bands | `scripts/plots/04_fig_sensor.py` |
| `outputs/figures/fig_importance.pdf` | Figure 5, wavelength importance for the Health versus Rust decision (all and vegetated patches) and for three-class macro-F1 | `scripts/plots/05_fig_importance.py` |
| `outputs/figures/fig_band_separability.pdf` | Figure A1, model-free separability of each band | `scripts/plots/06_fig_band_separability.py` |

`scripts/tables/02_tables.py` also writes `outputs/tables/contrasts_*.tex`, the same contrasts with unadjusted
p-values instead of credible intervals.

`scripts/tables/03_check_statements.py` then recomputes every verbal label the paper uses (for example "probably
better" or "practically equivalent") and every other statement that depends on a number, and reports any that no
longer fit. Pass `--manuscript DIR` to also check that the manuscript cites the corresponding macros.

### Robustness analyses

Seven further checks, run by `bash reproduce.sh` after the main analyses, ask how far the conclusions depend on choices
a reader might question. Only the per-band check needs the images, and its result is shipped. The numbers are in
`outputs/robustness/robustness.tex`, as macros named `\Rob...` and keyed values such as
`\res{rob:planned:hs_vs_ms}{lo}` that are read after `outputs/numbers.tex`, and the tables are in
`outputs/robustness/`. The analysis files go to `results/analysis/`.

| Question | Script | Writes |
|---|---|---|
| How wide are the credible intervals, do the verbal labels hold for a ROPE of 0.01 or 0.03, which contrasts were planned, and which differences survive Holm or Benjamini-Hochberg over all contrasts? | `scripts/analysis/09_contrast_robustness.py` | `contrast_robustness.csv`, `multiplicity_summary.csv`, `rope_summary.csv`; tables `headline.tex` (Table 2) and `contrasts_*_ci.tex` (Tables A2 to A4 with credible intervals) |
| Does the advantage of the cube on vegetated patches depend on the NDVI threshold of 0.5, or on the greener canopy of the Health patches that a threshold leaves? | `scripts/analysis/10_ndvi_threshold.py` | `ndvi_threshold.csv`, `ndvi_threshold_auc.csv`, `ndvi_imbalance.csv`, `ndvi_balance.csv`; table `ndvi_threshold.tex` (Table A5) |
| How did all ten submissions score, does the class balance of the holdout differ from training, and does the rank agreement with CV hold without the RGB submission? | `scripts/analysis/11_holdout_checks.py` | `holdout_all.csv`, `label_shift.csv`, `holdout_tau.csv`; tables `holdout_all.tex` and `label_shift.tex` |
| Could the two HS file formats or the file numbers stand in for acquisition batches or plots? | `scripts/analysis/12_file_structure.py` | `file_structure.json`, `overlap_pairs.csv`; table `crossbatch.tex` with the transfer between formats per view and learner |
| How stable is the band selection across outer folds? | `scripts/analysis/13_selection_stability.py` | `selection_stability.csv` |
| How many of the engineered MS features does each MS band enter, which a permutation ranking of the bands partly reflects? | `scripts/analysis/14_ms_feature_use.py` (runs the feature code on synthetic patches, no data) | `ms_feature_use.csv` |
| How well does each band separate Rust from Health patches without a model? | `scripts/eda/04_band_separability.py` (images) and `scripts/plots/06_fig_band_separability.py` | `results/audit/band_separability.csv` and `outputs/figures/fig_band_separability.pdf` |

The class-balance check reads only the submitted predictions, never a holdout label, and no robustness analysis
changes a model, a selection, or a submission. To rerun the per-band check, point `WHEATRUST_DATA` to the images and
run `python scripts/eda/04_band_separability.py` (under a minute).

### From the images (roughly 85 CPU-hours)

```bash
WORKERS=8 bash reproduce.sh --all
```

The steps, in order, are listed below with their run times on one 2.8 GHz core. `--workers N` (or `WORKERS` in
`reproduce.sh`) runs independent units in parallel, and every long step can be stopped and restarted because finished
units are kept on disk.

| Step | Script | Time |
|---|---|---|
| Data audit (sample table, provenance of the derived views, noise, empty patches, overlapping tiles) | `scripts/eda/01_audit.py`, `02_overlap_offsets.py` | 1 min |
| Outer folds, grouped by overlapping tiles | `scripts/training/01_make_folds.py` | seconds |
| Feature cache and the stretch control | `scripts/training/02_build_features.py`, `03_stretch_control.py` | 1 min |
| Adversarial validation and the file-format check | `scripts/eda/03_adversarial_validation.py` | 3 min |
| Per-band separability of Rust and Health patches | `scripts/eda/04_band_separability.py` | 1 min |
| Band selection inside every outer training fold | `scripts/training/04_band_selection.py` | 35 min |
| Experiment units, FLAML and TabPFN | `scripts/training/05_run_units.py --exp E1 E2 E3f E3t E4 E4t E5 E9 E9t` | 40 h |
| Wavelength importance by window permutation | `scripts/training/06_window_permutation.py` | 36 h |
| Transfer between the two HS file formats | `scripts/training/07_crossbatch.py` | 8 h |
| Holdout refits and submission files | `scripts/training/08_submissions.py` | 12 min |

The platform reruns that the paper uses to bound platform effects are the same units written to their own folders,
`python scripts/training/05_run_units.py --exp REPRO REPROX`, meant to run on a second machine.

### Determinism across machines

On one platform every unit is deterministic. Across processor architectures and operating systems the numerical
libraries round differently, so FLAML's search can settle on a different but nearly equivalent configuration in a few
folds, and TabPFN's probabilities move slightly. The paper reports the size of this effect (its Software and Hardware
section), and the platform reruns in `results/REPRO` and `results/REPROX` document it.

## Repository layout

```
wheatrust/           shared code: paths and data access, features, band selection, learners, experiment
                     registry, unit runner, and the paired tests
scripts/eda/         data audit, overlap offsets, adversarial validation, per-band separability
scripts/training/    folds, features, band selection, experiment units, window permutation, cross-batch
                     transfer, holdout refits
scripts/analysis/    summaries, band-selection frequencies, Health versus Rust analysis, contrasts with Holm
                     correction, window statistics, platform check, leaderboard analysis, and the robustness
                     analyses (09 to 13)
scripts/tables/      numbers.tex, the tables, the statement checker, and the robustness numbers and tables
scripts/plots/       the five figures of the paper and the per-band separability figure
results/             per-fold results and out-of-fold predictions behind every number (see results/README.md)
submissions/         the seven holdout submission files the paper reports
outputs/             numbers.tex, tables, and figures as rebuilt by reproduce.sh, and outputs/robustness/
reproduce.sh         runs the pipeline
```

## Notes on the design

- The RGB and multispectral images distributed with the data are views derived from the hyperspectral cube, not
  separate cameras. `scripts/eda/01_audit.py` establishes this, and the comparisons in the paper therefore isolate
  band placement and the derivation pipeline.
- Patches that share image content (tiles offset by half a tile) are kept in the same outer fold, and every
  preprocessing, selection, and tuning step runs inside the training folds only.
- The holdout was used only to produce the submissions, for the duplicate check, and for adversarial validation.
  Nothing was tuned or selected on the leaderboard scores, which are recorded in `results/leaderboard/`. The study
  made ten submissions. The seven this paper describes are in `submissions/`, and the scores of the other three,
  which use models outside its scope, are in `results/leaderboard/other_submissions.csv`, so that every submission
  is reported.

## Citation

If this code or these results help your work, please cite the paper. The entry will be completed on publication.

```bibtex
@article{aryal2026spectral,
  author  = {Aryal, Saurav K. and Uprety, Aashma and Adhikari, Ashish},
  title   = {Assessing the Spectral Sufficiency of {UAV} Wheat Rust Classification with Views and Band Budgets
             Derived from One Hyperspectral Acquisition},
  journal = {Sensors},
  year    = {2026},
  note    = {Submitted}
}
```

## License and acknowledgements

The code and the shipped results are released under the MIT License (see `LICENSE`). The competition data are not
part of this repository and remain under the terms set by the competition organisers, whom we thank for releasing
them. 
