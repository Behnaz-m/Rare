"""
Generate all figures for the temporal leakage paper.

Figures:
1. Figure 1: 4-panel simulation results
2. Figure 2: ROC comparison (standard vs corrected)
3. Figure 3: Leakage signature diagnostic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from typing import Dict, List, Optional, Tuple
import warnings

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Figure settings for publication
FIGSIZE_SINGLE = (5, 4)
FIGSIZE_DOUBLE = (10, 4)
FIGSIZE_QUAD = (10, 8)
DPI = 300
FONTSIZE = 10


def setup_figure_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': FONTSIZE,
        'axes.labelsize': FONTSIZE,
        'axes.titlesize': FONTSIZE + 1,
        'xtick.labelsize': FONTSIZE - 1,
        'ytick.labelsize': FONTSIZE - 1,
        'legend.fontsize': FONTSIZE - 1,
        'figure.titlesize': FONTSIZE + 2,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })


def plot_auc_comparison_bars(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Create bar chart comparing AUC across conditions.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with columns: condition, auc_mean, auc_std
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        The figure object
    """
    setup_figure_style()

    # Prepare data
    condition_order = [
        'leak_free_grouped',
        'leak_free_random',
        'norm_leak_grouped',
        'explicit_leak_grouped'
    ]

    condition_labels = {
        'leak_free_grouped': 'Leak-Free\n+ Grouped CV',
        'leak_free_random': 'Leak-Free\n+ Random CV',
        'norm_leak_grouped': 'Normalization\nLeak',
        'explicit_leak_grouped': 'Explicit\nLeak'
    }

    colors = {
        'leak_free_grouped': '#2ecc71',  # Green (correct)
        'leak_free_random': '#e74c3c',   # Red (wrong)
        'norm_leak_grouped': '#e67e22',  # Orange (implicit leak)
        'explicit_leak_grouped': '#9b59b6'  # Purple (explicit leak)
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Aggregate by condition
    agg = results_df.groupby('condition')['auc'].agg(['mean', 'std']).reset_index()

    x_pos = np.arange(len(condition_order))
    bars = []

    for i, cond in enumerate(condition_order):
        row = agg[agg['condition'] == cond]
        if len(row) > 0:
            mean_val = row['mean'].values[0]
            std_val = row['std'].values[0]
            bar = ax.bar(i, mean_val, yerr=std_val, capsize=5,
                        color=colors[cond], edgecolor='black', linewidth=1)
            bars.append(bar)

            # Add value label on bar
            ax.text(i, mean_val + std_val + 0.02, f'{mean_val:.3f}',
                   ha='center', va='bottom', fontsize=FONTSIZE-1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([condition_labels[c] for c in condition_order])
    ax.set_ylabel('AUC-ROC')
    ax.set_ylim(0.4, 1.05)
    ax.set_title('Impact of Leakage and CV Method on Reported AUC')

    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random classifier')
    ax.axhline(y=0.97, color='red', linestyle=':', alpha=0.5, label='Typical "high" reported AUC')

    ax.legend(loc='lower right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


def plot_four_panel_simulation(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create the main 4-panel figure for the paper (Figure 1).

    Panels:
    A. AUC distribution by condition
    B. Brier score distribution
    C. Inflation magnitude
    D. Effective sample size impact

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from simulation experiment
    output_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
        The figure object
    """
    setup_figure_style()

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_QUAD)

    condition_labels = {
        'leak_free_grouped': 'Baseline\n(Correct)',
        'leak_free_random': 'Random CV\n(Pseudorep.)',
        'norm_leak_grouped': 'Norm. Leak',
        'explicit_leak_grouped': 'Explicit Leak'
    }

    colors = ['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6']
    conditions = ['leak_free_grouped', 'leak_free_random', 'norm_leak_grouped', 'explicit_leak_grouped']

    # Panel A: AUC boxplot
    ax = axes[0, 0]
    data_auc = [results_df[results_df['condition'] == c]['auc'].values for c in conditions]
    bp = ax.boxplot(data_auc, patch_artist=True, labels=[condition_labels[c] for c in conditions])
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('A. AUC Distribution by Condition')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Panel B: Brier score boxplot
    ax = axes[0, 1]
    data_brier = [results_df[results_df['condition'] == c]['brier'].values for c in conditions]
    bp = ax.boxplot(data_brier, patch_artist=True, labels=[condition_labels[c] for c in conditions])
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Brier Score')
    ax.set_title('B. Calibration (Brier Score)')

    # Panel C: Inflation comparison
    ax = axes[1, 0]
    baseline_mean = results_df[results_df['condition'] == 'leak_free_grouped']['auc'].mean()

    inflation_data = []
    for c in conditions:
        c_mean = results_df[results_df['condition'] == c]['auc'].mean()
        inflation = (c_mean - baseline_mean) / baseline_mean * 100
        inflation_data.append(inflation)

    bars = ax.bar(range(len(conditions)), inflation_data, color=colors, edgecolor='black')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([condition_labels[c] for c in conditions], rotation=0)
    ax.set_ylabel('AUC Inflation (%)')
    ax.set_title('C. Performance Inflation Relative to Baseline')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Add labels
    for i, v in enumerate(inflation_data):
        ax.text(i, v + 1, f'{v:+.1f}%', ha='center', va='bottom', fontsize=FONTSIZE-2)

    # Panel D: Scatter of n_eff vs AUC
    ax = axes[1, 1]

    for c, color in zip(conditions, colors):
        subset = results_df[results_df['condition'] == c]
        ax.scatter(subset['n_eff'], subset['auc'], c=color, alpha=0.3,
                  label=condition_labels[c].replace('\n', ' '), s=20)

    ax.set_xlabel('Effective Sample Size')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('D. Effective Sample Size vs Performance')
    ax.legend(loc='lower right', fontsize=FONTSIZE-2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


def plot_roc_comparison(
    y_true_correct: np.ndarray,
    y_prob_correct: np.ndarray,
    y_true_wrong: np.ndarray,
    y_prob_wrong: np.ndarray,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves comparing correct vs incorrect evaluation (Figure 2).

    Parameters
    ----------
    y_true_correct : np.ndarray
        True labels from correct (grouped) evaluation
    y_prob_correct : np.ndarray
        Predicted probabilities from correct evaluation
    y_true_wrong : np.ndarray
        True labels from wrong (random) evaluation
    y_prob_wrong : np.ndarray
        Predicted probabilities from wrong evaluation
    output_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
        The figure object
    """
    setup_figure_style()

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Compute ROC curves
    fpr_correct, tpr_correct, _ = roc_curve(y_true_correct, y_prob_correct)
    auc_correct = roc_auc_score(y_true_correct, y_prob_correct)

    fpr_wrong, tpr_wrong, _ = roc_curve(y_true_wrong, y_prob_wrong)
    auc_wrong = roc_auc_score(y_true_wrong, y_prob_wrong)

    # Plot
    ax.plot(fpr_correct, tpr_correct, color='#2ecc71', lw=2,
            label=f'Grouped CV (AUC = {auc_correct:.3f})')
    ax.plot(fpr_wrong, tpr_wrong, color='#e74c3c', lw=2,
            label=f'Random CV (AUC = {auc_wrong:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves: Correct vs Incorrect Evaluation')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


def plot_leakage_signature(
    signatures: Dict,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot leakage signatures for multiple features (Figure 3).

    A leak-free feature has a flat signature; a leaked feature
    shows a monotonic trend with time-to-event.

    Parameters
    ----------
    signatures : dict
        Output from compute_leakage_signature()
    output_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
        The figure object
    """
    setup_figure_style()

    n_features = len(signatures)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (feat_name, sig) in enumerate(signatures.items()):
        ax = axes[i]

        x = range(len(sig['bin_means']))
        y = sig['bin_means']

        color = '#e74c3c' if sig['is_suspicious'] else '#2ecc71'
        ax.bar(x, y, color=color, alpha=0.7, edgecolor='black')

        ax.set_xlabel('Time-to-Event Bin (far → near)')
        ax.set_ylabel('Mean Feature Value')
        ax.set_title(f'{feat_name}\n(ρ={sig["monotonicity"]:.2f}, p={sig["p_value"]:.3f})')

        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), 'r--', alpha=0.5)

    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Leakage Signatures: Feature Means by Time-to-Event', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


def plot_inflation_heatmap(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create heatmap showing inflation across different parameters.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with condition and seed information
    output_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
        The figure object
    """
    setup_figure_style()

    # Pivot to get inflation values
    pivot = results_df.pivot_table(
        index='seed',
        columns='condition',
        values='auc',
        aggfunc='mean'
    )

    # Compute inflation relative to baseline
    if 'leak_free_grouped' in pivot.columns:
        for col in pivot.columns:
            if col != 'leak_free_grouped':
                pivot[f'{col}_inflation'] = (
                    (pivot[col] - pivot['leak_free_grouped']) /
                    pivot['leak_free_grouped'] * 100
                )

    # Select inflation columns
    inflation_cols = [c for c in pivot.columns if 'inflation' in c]

    if len(inflation_cols) == 0:
        print("No inflation data available")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        pivot[inflation_cols],
        cmap='RdYlGn_r',
        center=0,
        annot=False,
        ax=ax
    )

    ax.set_title('AUC Inflation (%) by Condition and Replicate')
    ax.set_xlabel('Condition')
    ax.set_ylabel('Replicate')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


def generate_all_figures(
    results_df: pd.DataFrame,
    output_dir: str = "results/figures"
) -> Dict[str, plt.Figure]:
    """
    Generate all figures for the paper.

    Parameters
    ----------
    results_df : pd.DataFrame
        Complete simulation results
    output_dir : str
        Directory to save figures

    Returns
    -------
    dict
        Dictionary mapping figure name to Figure object
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    # Figure 1: 4-panel simulation results
    print("Generating Figure 1: 4-panel simulation...")
    fig1 = plot_four_panel_simulation(
        results_df,
        output_path=f"{output_dir}/figure1_simulation_results.png"
    )
    figures['figure1'] = fig1

    # Figure: AUC comparison bars
    print("Generating AUC comparison bar chart...")
    fig_bars = plot_auc_comparison_bars(
        results_df,
        output_path=f"{output_dir}/auc_comparison_bars.png"
    )
    figures['auc_bars'] = fig_bars

    print(f"\nAll figures saved to {output_dir}/")

    return figures


if __name__ == "__main__":
    # Test plotting with dummy data
    print("Testing plotting functions...")

    # Create dummy results
    np.random.seed(42)
    conditions = ['leak_free_grouped', 'leak_free_random', 'norm_leak_grouped', 'explicit_leak_grouped']
    base_aucs = [0.68, 0.82, 0.78, 0.95]

    rows = []
    for seed in range(20):
        for i, cond in enumerate(conditions):
            rows.append({
                'seed': seed,
                'condition': cond,
                'auc': base_aucs[i] + np.random.normal(0, 0.03),
                'brier': 0.2 - base_aucs[i] * 0.1 + np.random.normal(0, 0.02),
                'n_eff': 38 + np.random.normal(0, 5)
            })

    df = pd.DataFrame(rows)

    # Generate figures
    figures = generate_all_figures(df, output_dir="results/figures")

    print("Done!")
    plt.show()
