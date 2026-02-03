"""
Inject various types of label leakage into panel data.

This module demonstrates how standard preprocessing operations
can introduce implicit leakage, and also provides explicit leakage
for comparison purposes.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


def add_explicit_leak(
    df: pd.DataFrame,
    noise_std: float = 10.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Add an explicitly leaked feature: X_leak = f(T_e - t) + noise.

    This is the "straw man" example that's easy to dismiss as a coding error.
    We include it for comparison to show that implicit leaks have similar effects.

    The leaked feature is:
        X_leak = 500  if T_e - t <= 3   (event very soon)
               = 300  if T_e - t in (3, 7]
               = 150  otherwise

    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns T_e and t
    noise_std : float
        Standard deviation of noise added to leaked feature
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Data with additional column 'X_leak'
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    # Compute time-to-event (this uses future information!)
    time_to_event = df['T_e'] - df['t']

    # Create leaked feature
    X_leak = np.where(
        time_to_event <= 3, 500,
        np.where(time_to_event <= 7, 300, 150)
    )

    # Add noise
    X_leak = X_leak.astype(float) + rng.normal(0, noise_std, len(df))

    df['X_leak'] = X_leak

    return df


def apply_global_normalization(
    df: pd.DataFrame,
    feature_cols: list
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply WRONG global normalization (uses all data including future).

    This is the common mistake: fitting StandardScaler on the entire
    dataset before train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    feature_cols : list
        Columns to normalize

    Returns
    -------
    df_normalized : pd.DataFrame
        Data with normalized features (columns renamed to X_1_norm, etc.)
    scaler : StandardScaler
        Fitted scaler (for diagnostics)
    """
    df = df.copy()

    # Fit scaler on ALL data (WRONG - includes future observations)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(df[feature_cols])

    # Add normalized columns
    for i, col in enumerate(feature_cols):
        df[f'{col}_norm'] = X_normalized[:, i]

    return df, scaler


def apply_rolling_normalization(
    df: pd.DataFrame,
    feature_cols: list,
    window: int = 10,
    min_periods: int = 3
) -> pd.DataFrame:
    """
    Apply CORRECT rolling normalization (uses only past data).

    For each observation at time t, normalize using statistics
    computed only from times s < t within the same episode.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    feature_cols : list
        Columns to normalize
    window : int
        Rolling window size
    min_periods : int
        Minimum observations required

    Returns
    -------
    pd.DataFrame
        Data with correctly normalized features
    """
    df = df.copy()

    for col in feature_cols:
        # Compute rolling statistics within each episode
        # shift(1) ensures we only use past values (not current)
        rolling_mean = df.groupby('episode_id')[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean()
        )
        rolling_std = df.groupby('episode_id')[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=min_periods).std()
        )

        # Normalize (handle edge cases)
        rolling_std = rolling_std.replace(0, 1)  # Avoid division by zero
        rolling_std = rolling_std.fillna(1)
        rolling_mean = rolling_mean.fillna(0)

        df[f'{col}_rollnorm'] = (df[col] - rolling_mean) / rolling_std

    return df


def inject_missing_and_impute_wrong(
    df: pd.DataFrame,
    feature_cols: list,
    missing_rate: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """
    Inject missing values and impute using episode means (WRONG).

    This demonstrates the imputation trap: using episode means
    includes post-event observations.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    feature_cols : list
        Columns to inject missing values
    missing_rate : float
        Fraction of values to make missing
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Data with imputed features
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    for col in feature_cols:
        # Create missing mask
        missing_mask = rng.random(len(df)) < missing_rate

        # Store original
        df[f'{col}_original'] = df[col]

        # Set missing
        df.loc[missing_mask, col] = np.nan

        # WRONG: Impute with episode mean (includes future observations)
        df[col] = df.groupby('episode_id')[col].transform(
            lambda x: x.fillna(x.mean())
        )

    return df


def inject_missing_and_impute_correct(
    df: pd.DataFrame,
    feature_cols: list,
    missing_rate: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """
    Inject missing values and impute using forward-fill (CORRECT).

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    feature_cols : list
        Columns to inject missing values
    missing_rate : float
        Fraction of values to make missing
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Data with correctly imputed features
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    for col in feature_cols:
        # Create missing mask
        missing_mask = rng.random(len(df)) < missing_rate

        # Store original
        df[f'{col}_original'] = df[col]

        # Set missing
        df.loc[missing_mask, col] = np.nan

        # CORRECT: Forward-fill within episode (uses only past)
        df[col] = df.groupby('episode_id')[col].transform(
            lambda x: x.ffill()
        )

        # Fill any remaining NaN at start of episode with 0
        df[col] = df[col].fillna(0)

    return df


def create_experimental_conditions(
    df: pd.DataFrame,
    feature_cols: list,
    seed: int = 42
) -> dict:
    """
    Create all experimental conditions for the simulation study.

    Returns a dictionary with:
    - 'leak_free': Original features (no leakage)
    - 'global_norm': Features with global normalization leak
    - 'explicit_leak': Features with explicit T_e - t leak
    - 'imputation_leak': Features with episode-mean imputation leak

    Parameters
    ----------
    df : pd.DataFrame
        Original leak-free data
    feature_cols : list
        Feature columns to use/transform
    seed : int
        Random seed

    Returns
    -------
    dict
        Dictionary mapping condition name to (X, y, groups) tuple
    """
    from src.data_generation import prepare_modeling_data

    conditions = {}

    # Condition 1: Leak-free (baseline)
    X, y, groups = prepare_modeling_data(df, feature_cols)
    conditions['leak_free'] = (X, y, groups)

    # Condition 2: Global normalization leak
    df_norm, _ = apply_global_normalization(df, feature_cols)
    norm_cols = [f'{col}_norm' for col in feature_cols]
    X_norm, y_norm, groups_norm = prepare_modeling_data(df_norm, norm_cols)
    conditions['global_norm'] = (X_norm, y_norm, groups_norm)

    # Condition 3: Explicit leak
    df_leak = add_explicit_leak(df, seed=seed)
    leak_cols = feature_cols + ['X_leak']
    X_leak, y_leak, groups_leak = prepare_modeling_data(df_leak, leak_cols)
    conditions['explicit_leak'] = (X_leak, y_leak, groups_leak)

    # Condition 4: Imputation leak
    df_impute = inject_missing_and_impute_wrong(df.copy(), feature_cols, seed=seed)
    X_impute, y_impute, groups_impute = prepare_modeling_data(df_impute, feature_cols)
    conditions['imputation_leak'] = (X_impute, y_impute, groups_impute)

    return conditions


if __name__ == "__main__":
    # Test leakage injection
    from src.data_generation import generate_default_data, get_feature_columns

    print("Testing leakage injection...")
    df = generate_default_data(seed=42)
    feature_cols = get_feature_columns(df)

    # Test explicit leak
    df_leak = add_explicit_leak(df)
    print(f"\nExplicit leak correlation with time-to-event:")
    df_atrisk = df_leak[df_leak['at_risk'] == 1]
    time_to_event = df_atrisk['T_e'] - df_atrisk['t']
    corr = np.corrcoef(df_atrisk['X_leak'], time_to_event)[0, 1]
    print(f"  Correlation: {corr:.3f}")

    # Test global normalization
    df_norm, scaler = apply_global_normalization(df, feature_cols)
    print(f"\nGlobal normalization applied to {len(feature_cols)} features")

    # Create all conditions
    conditions = create_experimental_conditions(df, feature_cols)
    print(f"\nCreated {len(conditions)} experimental conditions:")
    for name, (X, y, groups) in conditions.items():
        print(f"  {name}: X shape = {X.shape}")
