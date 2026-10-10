# Shipped results

These files hold every per-fold result behind the paper, so that `bash reproduce.sh` can rebuild all numbers, tables,
and figures without the image data. A key such as `E4:ms_eng` names experiment `E4` and variant `ms_eng`.

| Folder | Contents |
|---|---|
| `E1` | FLAML baseline on all views |
| `E2` | TabPFN on all views (the procedure of submission S05) |
| `E3f` | FLAML on three stratified draws of three quarters of each outer training fold, used for the gain from the last 120 training patches |
| `E3t` | TabPFN with eight ensemble members on the RGB, MS, cleaned HS, and all-views features |
| `E4` | FLAML on single views and representations (engineered, raw, PCA), the single-source views, and the stretch control |
| `E4t` | TabPFN on the single-source views and the stretch control |
| `E5` | FLAML feature-family and dimensionality ablations |
| `E9`, `E9t` | band budgets selected inside each outer training fold and the fixed RGB and MS placements, with FLAML and TabPFN; `E9/selections` holds the selected bands of every fold |
| `E8` | wavelength importance by window permutation for the Health versus Rust decision, on all and on vegetated patches |
| `E10` | transfer between the two HS file formats |
| `E12` | adversarial validation between training and holdout inputs and between the two HS file formats |
| `REPRO`, `REPROX` | the same units rerun on a second machine, to bound platform effects |
| `audit` | the data audit (sample table, provenance of the derived views, band noise, empty patches, overlapping tiles, NDVI composition) and the per-band separability of Rust and Health patches (`band_separability.csv`, from `scripts/eda/04_band_separability.py`) |
| `folds` | the outer folds of all 10 repeats |
| `features` | the feature dictionary |
| `holdout/selections` | the five bands selected on all training patches for submission S07 |
| `leaderboard` | the seven submissions the paper reports, their public and private scores, and the split sizes inferred from them; `other_submissions.csv` holds the scores, predicted class counts, and CV accuracy with its range at the private size of the study's three other submissions, whose models lie outside this paper and whose code is not part of this repository |

Each experiment folder holds `fold_metrics.csv` (one row per outer fold, with all metrics and the configuration
FLAML chose) and `oof.csv.gz` (the class probabilities each outer fold's model gave its validation patches). The
`platform` column names the machine type a unit ran on. `results/analysis/` is written by `scripts/analysis/` and is
not part of the repository.
