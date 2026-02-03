# The Autocorrelation Mirage: Pseudoreplication in Panel Forecasting

This repository contains the code and experiments for the paper "The Autocorrelation Mirage: Pseudoreplication Dominates Label Leakage in Panel Forecasting."

## Key Finding

**Episode memorization** is a failure mode where models learn to identify which episode an observation belongs to rather than predicting event risk. Standard K-Fold CV—the default in Scikit-Learn—enables this by allowing the same episode to appear in both training and test sets.

**Result**: AUC inflates from 0.56 to 0.86 (+54%) with *no data leakage whatsoever*. This exceeds even explicit time-to-event leakage (+39%).

## The ΔCV Diagnostic

```
ΔCV = AUC_random - AUC_grouped
```

In our simulation, **ΔCV = 0.30**. A gap > 0.05 indicates the model exploits autocorrelation structure rather than learning generalizable signal.

## Quick Start

### Setup

```bash
cd /project/shakeri-lab/b/leakage
bash setup.sh
source venv/bin/activate
```

### Run Quick Test (Local)

```bash
python experiments/run_simulation.py --quick
```

### Run Full Experiment (SLURM)

```bash
sbatch slurm/submit_simulation.sh
squeue -u $USER
tail -f slurm_logs/simulation_*.out
```

## Project Structure

```
leakage/
├── src/
│   ├── data_generation.py      # Leak-free synthetic data (AR(1) process)
│   ├── leakage_injection.py    # Add various types of leakage
│   ├── evaluation.py           # Grouped CV vs Random CV
│   ├── diagnostics.py          # Leakage detection tools
│   └── plotting.py             # Generate figures
├── experiments/
│   └── run_simulation.py       # Main simulation (4 conditions × N replicates)
├── slurm/
│   ├── submit_simulation.sh    # Full experiment (100 replicates)
│   └── submit_quick_test.sh    # Quick test (10 replicates)
├── results/
│   ├── tables/                 # Generated tables (CSV)
│   └── figures/                # Generated figures (PNG)
├── manuscript/
│   ├── main2.tex               # Paper manuscript
│   └── figs/                   # Manuscript figures
├── requirements.txt
├── setup.sh
├── README.md                   # This file
└── CLAUDE.md                   # Development documentation
```

## Results (100 Replicates)

| Condition | AUC | Brier | Inflation | ΔCV |
|-----------|-----|-------|-----------|-----|
| Leak-Free + Grouped CV | 0.56 ± 0.07 | 0.29 ± 0.05 | baseline | — |
| **Leak-Free + Random CV** | **0.86 ± 0.04** | 0.15 ± 0.02 | **+54%** | **0.30** |
| Norm. Leak + Grouped | 0.56 ± 0.07 | 0.29 ± 0.05 | +0%* | — |
| Explicit Leak + Grouped | 0.77 ± 0.04 | 0.16 ± 0.03 | +39% | — |

*Normalization leak shows +0% because synthetic features lack pre-event trends. Real-world features (deteriorating vitals, accumulating errors) would activate this trap.

**Key insight**: Pseudoreplication (+54%) exceeds explicit leakage (+39%). The standard tool (`KFold`) is more dangerous than the worst feature-engineering error.

## Usage Examples

### Generate Leak-Free Data

```python
from src.data_generation import generate_panel_data, prepare_modeling_data

df = generate_panel_data(n_episodes=30, T_max=60, seed=42)
X, y, groups = prepare_modeling_data(df)
print(f"Data shape: {X.shape}, Event rate: {y.mean():.1%}")
```

### Compute the ΔCV Diagnostic

```python
from src.evaluation import evaluate_grouped_cv, evaluate_random_cv

results_grouped = evaluate_grouped_cv(X, y, groups)
results_random = evaluate_random_cv(X, y)

delta_cv = results_random['auc'].mean() - results_grouped['auc'].mean()
print(f"ΔCV = {delta_cv:.2f}")
if delta_cv > 0.05:
    print("WARNING: Model likely exploits episode memorization!")
```

### Correct Evaluation with Grouped CV

```python
from src.evaluation import evaluate_grouped_cv

results = evaluate_grouped_cv(X, y, groups)
print(f"AUC: {results['auc'].mean():.3f}")
```

## Reproducing Paper Results

1. **Run Full Simulation**:
   ```bash
   sbatch slurm/submit_simulation.sh
   ```

2. **View Results**:
   ```bash
   cat results/tables/table3.csv
   ```

3. **Regenerate Figures**:
   ```python
   from src.plotting import generate_all_figures
   import pandas as pd

   df = pd.read_csv('results/simulation_results_latest.csv')
   generate_all_figures(df, 'results/figures')
   ```

## Why This Matters

Pseudoreplication requires only default `KFold`—a single line of code that every practitioner uses. Leakage requires specific conditions (explicit T_e - t features, or trending features with global normalization). This asymmetry makes pseudoreplication the greater threat:

- **Invisible**: The pipeline looks correct
- **Ubiquitous**: Default in all ML libraries
- **Activated by default**: No special code needed

The fix is simple: use `GroupKFold` with `groups=episode_id`.

## Dependencies

- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0
- tqdm >= 4.62.0

## Citation

```bibtex
@article{autocorrelation_mirage2026,
  title={The Autocorrelation Mirage: Pseudoreplication Dominates Label Leakage
         in Panel Forecasting},
  author={...},
  journal={...},
  year={2026}
}
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
