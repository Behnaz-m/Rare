#!/usr/bin/env python
"""
Main simulation experiment for the temporal leakage paper.

This script:
1. Generates leak-free panel data
2. Creates 4 experimental conditions
3. Evaluates each condition with grouped and random CV
4. Produces Table 3 for the paper
5. Saves results for figure generation

Run with: python experiments/run_simulation.py [--n_replicates N] [--output_dir DIR]
"""

import numpy as np
import pandas as pd
import sys
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import (
    generate_panel_data,
    prepare_modeling_data,
    get_feature_columns
)
from src.leakage_injection import (
    add_explicit_leak,
    apply_global_normalization
)
from src.evaluation import (
    evaluate_grouped_cv,
    evaluate_random_cv,
    compute_effective_sample_size
)


def run_single_replicate(seed: int, verbose: bool = False) -> dict:
    """
    Run one complete replicate of the experiment.

    Parameters
    ----------
    seed : int
        Random seed for this replicate
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Results for all 4 conditions
    """
    # Generate data
    df = generate_panel_data(
        n_episodes=30,
        T_max=60,
        ar_coef=0.7,
        noise_std=0.3,
        hazard_coef=0.15,
        base_hazard=-3.0,
        alpha_std=0.5,
        horizon=14,
        seed=seed
    )

    feature_cols = get_feature_columns(df)

    results = {}

    # ====== CONDITION 1: Leak-Free + Grouped CV (Baseline) ======
    X, y, groups = prepare_modeling_data(df, feature_cols)

    # Grouped CV (correct)
    grouped_res = evaluate_grouped_cv(X, y, groups)
    auc_grouped = grouped_res['auc'].dropna().mean()
    brier_grouped = grouped_res['brier'].mean()

    results['leak_free_grouped'] = {
        'auc': auc_grouped,
        'brier': brier_grouped,
        'method': 'grouped'
    }

    # ====== CONDITION 2: Leak-Free + Random CV (Pseudoreplication) ======
    random_res = evaluate_random_cv(X, y, normalize_before=True, seed=seed)
    auc_random = random_res['auc'].mean()
    brier_random = random_res['brier'].mean()

    results['leak_free_random'] = {
        'auc': auc_random,
        'brier': brier_random,
        'method': 'random'
    }

    # ====== CONDITION 3: Normalization Leak + Grouped CV ======
    df_norm, _ = apply_global_normalization(df, feature_cols)
    norm_cols = [f'{col}_norm' for col in feature_cols]
    X_norm, y_norm, groups_norm = prepare_modeling_data(df_norm, norm_cols)

    # Use grouped CV but with leaked features
    grouped_res_norm = evaluate_grouped_cv(X_norm, y_norm, groups_norm, normalize_per_fold=False)
    auc_norm = grouped_res_norm['auc'].dropna().mean()
    brier_norm = grouped_res_norm['brier'].mean()

    results['norm_leak_grouped'] = {
        'auc': auc_norm,
        'brier': brier_norm,
        'method': 'grouped'
    }

    # ====== CONDITION 4: Explicit Leak + Grouped CV ======
    df_leak = add_explicit_leak(df, seed=seed)
    leak_cols = feature_cols + ['X_leak']
    X_leak, y_leak, groups_leak = prepare_modeling_data(df_leak, leak_cols)

    grouped_res_leak = evaluate_grouped_cv(X_leak, y_leak, groups_leak)
    auc_leak = grouped_res_leak['auc'].dropna().mean()
    brier_leak = grouped_res_leak['brier'].mean()

    results['explicit_leak_grouped'] = {
        'auc': auc_leak,
        'brier': brier_leak,
        'method': 'grouped'
    }

    # Store effective sample size (same for all conditions from same data)
    n, m, rho, n_eff = compute_effective_sample_size(groups, y)
    results['effective_n'] = {
        'n': n,
        'm': m,
        'rho': rho,
        'n_eff': n_eff
    }

    results['seed'] = seed
    results['n_episodes'] = df['episode_id'].nunique()
    results['event_rate'] = df[df['at_risk'] == 1]['Y'].mean()

    return results


def run_full_experiment(
    n_replicates: int = 100,
    start_seed: int = 0,
    output_dir: str = "results"
) -> pd.DataFrame:
    """
    Run the complete simulation experiment with multiple replicates.

    Parameters
    ----------
    n_replicates : int
        Number of random replicates
    start_seed : int
        Starting seed
    output_dir : str
        Directory to save results

    Returns
    -------
    pd.DataFrame
        All results
    """
    print(f"Running {n_replicates} replicates...")

    all_results = []

    for i in tqdm(range(n_replicates), desc="Replicates"):
        seed = start_seed + i
        try:
            results = run_single_replicate(seed, verbose=False)
            all_results.append(results)
        except Exception as e:
            print(f"Error in replicate {i} (seed={seed}): {e}")
            continue

    # Convert to DataFrame
    rows = []
    for r in all_results:
        for condition in ['leak_free_grouped', 'leak_free_random', 'norm_leak_grouped', 'explicit_leak_grouped']:
            rows.append({
                'seed': r['seed'],
                'condition': condition,
                'auc': r[condition]['auc'],
                'brier': r[condition]['brier'],
                'n_episodes': r['n_episodes'],
                'event_rate': r['event_rate'],
                'n_eff': r['effective_n']['n_eff'],
                'rho': r['effective_n']['rho']
            })

    df_results = pd.DataFrame(rows)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/simulation_results_{timestamp}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Also save latest
    df_results.to_csv(f"{output_dir}/simulation_results_latest.csv", index=False)

    return df_results


def generate_table_3(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 3 for the paper: Summary of simulation results.

    Parameters
    ----------
    df_results : pd.DataFrame
        Raw results from run_full_experiment

    Returns
    -------
    pd.DataFrame
        Formatted table for paper
    """
    # Compute statistics by condition
    summary = df_results.groupby('condition').agg({
        'auc': ['mean', 'std'],
        'brier': ['mean', 'std']
    }).round(3)

    # Flatten column names
    summary.columns = ['auc_mean', 'auc_std', 'brier_mean', 'brier_std']
    summary = summary.reset_index()

    # Compute inflation relative to baseline
    baseline_auc = summary[summary['condition'] == 'leak_free_grouped']['auc_mean'].values[0]
    baseline_brier = summary[summary['condition'] == 'leak_free_grouped']['brier_mean'].values[0]

    summary['auc_inflation'] = ((summary['auc_mean'] - baseline_auc) / baseline_auc * 100).round(1)
    summary['brier_improvement'] = ((baseline_brier - summary['brier_mean']) / baseline_brier * 100).round(1)

    # Rename conditions for paper
    condition_names = {
        'leak_free_grouped': 'Leak-Free + Grouped CV',
        'leak_free_random': 'Leak-Free + Random CV',
        'norm_leak_grouped': 'Normalization Leak + Grouped CV',
        'explicit_leak_grouped': 'Explicit Leak + Grouped CV'
    }
    summary['Condition'] = summary['condition'].map(condition_names)

    # Format for paper
    summary['Brier'] = summary.apply(lambda x: f"{x['brier_mean']:.3f} +/- {x['brier_std']:.3f}", axis=1)
    summary['AUC'] = summary.apply(lambda x: f"{x['auc_mean']:.3f} +/- {x['auc_std']:.3f}", axis=1)
    summary['Inflation'] = summary.apply(
        lambda x: 'baseline' if x['condition'] == 'leak_free_grouped' else f"+{x['auc_inflation']:.0f}%",
        axis=1
    )

    # Select columns for paper
    table = summary[['Condition', 'AUC', 'Brier', 'Inflation']]

    # Reorder rows
    order = ['Leak-Free + Grouped CV', 'Leak-Free + Random CV',
             'Normalization Leak + Grouped CV', 'Explicit Leak + Grouped CV']
    table = table.set_index('Condition').loc[order].reset_index()

    return table


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run temporal leakage simulation experiment')
    parser.add_argument('--n_replicates', type=int, default=100,
                        help='Number of simulation replicates (default: 100)')
    parser.add_argument('--start_seed', type=int, default=42,
                        help='Starting random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run with 10 replicates')

    args = parser.parse_args()

    if args.quick:
        args.n_replicates = 10

    print("=" * 60)
    print("TEMPORAL LEAKAGE SIMULATION EXPERIMENT")
    print("=" * 60)

    # Run experiment
    print(f"\nConfiguration:")
    print(f"  Replicates: {args.n_replicates}")
    print(f"  Start seed: {args.start_seed}")
    print(f"  Output dir: {args.output_dir}")

    df_results = run_full_experiment(
        n_replicates=args.n_replicates,
        start_seed=args.start_seed,
        output_dir=args.output_dir
    )

    # Generate Table 3
    print("\n" + "=" * 60)
    print("TABLE 3: Simulation Results")
    print("=" * 60)
    table3 = generate_table_3(df_results)
    print(table3.to_string(index=False))

    # Save table
    os.makedirs(f"{args.output_dir}/tables", exist_ok=True)
    table3.to_csv(f"{args.output_dir}/tables/table3.csv", index=False)
    print(f"\nTable saved to {args.output_dir}/tables/table3.csv")

    # Print additional statistics
    print("\n" + "=" * 60)
    print("ADDITIONAL STATISTICS")
    print("=" * 60)

    baseline = df_results[df_results['condition'] == 'leak_free_grouped']
    print(f"Average effective n: {baseline['n_eff'].mean():.1f}")
    print(f"Average ICC (rho): {baseline['rho'].mean():.3f}")
    print(f"Average event rate: {baseline['event_rate'].mean():.1%}")

    # Effect sizes
    leak_free_auc = df_results[df_results['condition'] == 'leak_free_grouped']['auc'].mean()
    explicit_leak_auc = df_results[df_results['condition'] == 'explicit_leak_grouped']['auc'].mean()
    print(f"\nAUC gap (explicit leak vs baseline): {explicit_leak_auc - leak_free_auc:.3f}")

    # Generate figures
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    try:
        from src.plotting import generate_all_figures
        figures = generate_all_figures(df_results, output_dir=f"{args.output_dir}/figures")
        print("Figures generated successfully!")
    except Exception as e:
        print(f"Warning: Could not generate figures: {e}")
        print("You can generate figures later by running: python src/plotting.py")

    return df_results


if __name__ == "__main__":
    df_results = main()
