"""
Diagnostics for detecting label leakage in temporal panel data.

This module provides tools to:
1. Detect temporal patterns in feature importance
2. Analyze correlation between features and time-to-event
3. Compare feature distributions across time horizons
4. Generate "leakage signatures"
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from typing import Dict, List, Tuple, Optional
import warnings


def compute_feature_time_correlation(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Compute correlation between features and time-to-event.

    High correlation indicates potential leakage (features should
    NOT know about future events).

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns 'T_e', 't', and feature columns
    feature_cols : list
        Feature column names

    Returns
    -------
    pd.DataFrame
        Correlation results for each feature
    """
    df = df[df['at_risk'] == 1].copy()

    # Compute time-to-event
    df['time_to_event'] = df['T_e'] - df['t']

    results = []
    for col in feature_cols:
        corr = np.corrcoef(df[col], df['time_to_event'])[0, 1]
        results.append({
            'feature': col,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'potential_leak': abs(corr) > 0.3  # Flag if correlation > 0.3
        })

    return pd.DataFrame(results).sort_values('abs_correlation', ascending=False)


def analyze_feature_by_horizon(
    df: pd.DataFrame,
    feature_col: str,
    horizons: List[int] = [1, 3, 7, 14, 30]
) -> pd.DataFrame:
    """
    Analyze feature distribution at different time horizons.

    If a feature is leak-free, its distribution should be similar
    regardless of time-to-event. Leaked features show systematic
    differences as we approach the event.

    Parameters
    ----------
    df : pd.DataFrame
        Data with time-to-event information
    feature_col : str
        Feature column to analyze
    horizons : list
        Time horizons to compare

    Returns
    -------
    pd.DataFrame
        Statistics by horizon
    """
    df = df[df['at_risk'] == 1].copy()
    df['time_to_event'] = df['T_e'] - df['t']

    results = []
    for h in horizons:
        # Observations where event is within h days
        mask = df['time_to_event'] <= h
        if mask.sum() > 0:
            results.append({
                'horizon': h,
                'n_obs': mask.sum(),
                'mean': df.loc[mask, feature_col].mean(),
                'std': df.loc[mask, feature_col].std(),
                'median': df.loc[mask, feature_col].median()
            })

    # Add "far from event" baseline
    baseline_mask = df['time_to_event'] > max(horizons)
    if baseline_mask.sum() > 0:
        results.append({
            'horizon': f'>{max(horizons)}',
            'n_obs': baseline_mask.sum(),
            'mean': df.loc[baseline_mask, feature_col].mean(),
            'std': df.loc[baseline_mask, feature_col].std(),
            'median': df.loc[baseline_mask, feature_col].median()
        })

    return pd.DataFrame(results)


def compute_leakage_signature(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_bins: int = 10
) -> Dict:
    """
    Compute the "leakage signature" for each feature.

    The leakage signature measures how predictive a feature is of
    time-to-event. A leak-free feature should have a flat signature;
    a leaked feature will show a strong monotonic pattern.

    Parameters
    ----------
    df : pd.DataFrame
        Data with time-to-event information
    feature_cols : list
        Feature columns to analyze
    n_bins : int
        Number of bins for time-to-event

    Returns
    -------
    dict
        Signature data for each feature
    """
    df = df[df['at_risk'] == 1].copy()
    df['time_to_event'] = df['T_e'] - df['t']

    signatures = {}

    for col in feature_cols:
        # Bin by time-to-event
        df['tte_bin'] = pd.qcut(df['time_to_event'], n_bins, duplicates='drop')

        # Compute mean feature value in each bin
        signature = df.groupby('tte_bin')[col].mean()

        # Compute monotonicity score (Spearman correlation with bin index)
        from scipy.stats import spearmanr
        bin_indices = np.arange(len(signature))
        corr, p_value = spearmanr(bin_indices, signature.values)

        signatures[col] = {
            'bin_means': signature.values.tolist(),
            'bin_labels': [str(x) for x in signature.index],
            'monotonicity': corr,
            'p_value': p_value,
            'is_suspicious': abs(corr) > 0.5 and p_value < 0.05
        }

    return signatures


def temporal_cv_diagnostic(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    t_values: np.ndarray
) -> Dict:
    """
    Diagnostic: Train on early times, test on later times.

    If features contain temporal leakage, the model will perform
    worse when forced to predict on truly future data.

    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    groups : np.ndarray
        Episode IDs
    t_values : np.ndarray
        Time values for each observation

    Returns
    -------
    dict
        Performance metrics for temporal vs random CV
    """
    from sklearn.model_selection import TimeSeriesSplit

    # Temporal split: train on first 70%, test on last 30%
    cutoff = np.percentile(t_values, 70)
    train_mask = t_values < cutoff
    test_mask = t_values >= cutoff

    if train_mask.sum() < 10 or test_mask.sum() < 10:
        return {'error': 'Not enough data for temporal split'}

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0)
    model.fit(X_train, y_train)

    # Evaluate
    if len(np.unique(y_test)) < 2:
        temporal_auc = np.nan
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
        temporal_auc = roc_auc_score(y_test, y_prob)

    # Compare to random split for reference
    from sklearn.model_selection import train_test_split
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_r = scaler.fit_transform(X_train_r)
    X_test_r = scaler.transform(X_test_r)

    model = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0)
    model.fit(X_train_r, y_train_r)
    y_prob_r = model.predict_proba(X_test_r)[:, 1]
    random_auc = roc_auc_score(y_test_r, y_prob_r)

    return {
        'temporal_auc': temporal_auc,
        'random_auc': random_auc,
        'auc_drop': random_auc - temporal_auc,
        'potential_leakage': (random_auc - temporal_auc) > 0.1
    }


def check_feature_stationarity(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Check if feature distributions are stationary over time.

    Non-stationarity in features can indicate leakage if the
    non-stationarity is correlated with the outcome.

    Parameters
    ----------
    df : pd.DataFrame
        Data with time information
    feature_cols : list
        Features to check

    Returns
    -------
    pd.DataFrame
        Stationarity test results
    """
    from scipy.stats import ks_2samp

    df = df.copy()

    # Split by time
    median_t = df['t'].median()
    early = df[df['t'] < median_t]
    late = df[df['t'] >= median_t]

    results = []
    for col in feature_cols:
        # KS test for distribution difference
        stat, p_value = ks_2samp(early[col].dropna(), late[col].dropna())

        results.append({
            'feature': col,
            'ks_statistic': stat,
            'p_value': p_value,
            'non_stationary': p_value < 0.05,
            'early_mean': early[col].mean(),
            'late_mean': late[col].mean(),
            'mean_shift': late[col].mean() - early[col].mean()
        })

    return pd.DataFrame(results)


def run_leakage_audit(
    df: pd.DataFrame,
    feature_cols: List[str],
    verbose: bool = True
) -> Dict:
    """
    Run comprehensive leakage audit on a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to audit
    feature_cols : list
        Feature columns to check
    verbose : bool
        Print results

    Returns
    -------
    dict
        Complete audit results
    """
    if verbose:
        print("=" * 60)
        print("LEAKAGE AUDIT REPORT")
        print("=" * 60)

    results = {}

    # 1. Correlation with time-to-event
    if verbose:
        print("\n1. Feature-TimeToEvent Correlations")
        print("-" * 40)

    corr_results = compute_feature_time_correlation(df, feature_cols)
    results['correlations'] = corr_results

    if verbose:
        print(corr_results.to_string(index=False))

    flagged_features = corr_results[corr_results['potential_leak']]['feature'].tolist()
    if flagged_features:
        if verbose:
            print(f"\n⚠️  WARNING: Features with high time-to-event correlation: {flagged_features}")

    # 2. Leakage signatures
    if verbose:
        print("\n2. Leakage Signatures")
        print("-" * 40)

    signatures = compute_leakage_signature(df, feature_cols)
    results['signatures'] = signatures

    suspicious = [f for f, s in signatures.items() if s['is_suspicious']]
    if verbose:
        for feat, sig in signatures.items():
            status = "⚠️  SUSPICIOUS" if sig['is_suspicious'] else "✓ OK"
            print(f"  {feat}: monotonicity={sig['monotonicity']:.3f} (p={sig['p_value']:.3f}) {status}")

    # 3. Stationarity check
    if verbose:
        print("\n3. Feature Stationarity")
        print("-" * 40)

    stationarity = check_feature_stationarity(df, feature_cols)
    results['stationarity'] = stationarity

    if verbose:
        non_stat = stationarity[stationarity['non_stationary']]
        if len(non_stat) > 0:
            print(f"  Non-stationary features: {non_stat['feature'].tolist()}")
        else:
            print("  All features appear stationary")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        n_warnings = len(flagged_features) + len(suspicious)
        if n_warnings == 0:
            print("✓ No obvious leakage detected")
        else:
            print(f"⚠️  {n_warnings} potential issues found")
            print("  - Review flagged features carefully")
            print("  - Consider using grouped CV for evaluation")

    return results


if __name__ == "__main__":
    # Test diagnostics
    from src.data_generation import generate_default_data, get_feature_columns
    from src.leakage_injection import add_explicit_leak

    print("Testing diagnostics on leak-free data...")
    df = generate_default_data(seed=42)
    feature_cols = get_feature_columns(df)

    results_clean = run_leakage_audit(df, feature_cols, verbose=True)

    print("\n" + "=" * 60)
    print("Testing diagnostics on leaked data...")
    print("=" * 60)

    df_leak = add_explicit_leak(df)
    feature_cols_leak = feature_cols + ['X_leak']

    results_leak = run_leakage_audit(df_leak, feature_cols_leak, verbose=True)
