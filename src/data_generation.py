"""
Generate leak-free panel data for rare-event forecasting experiments.

The data generating process:
1. Episode effect: α_e ~ N(0, σ_α²)
2. Latent intensity: Z_{e,t} = φ·Z_{e,t-1} + ε_t, where ε_t ~ N(0, σ_ε²)
3. Hazard: h_{e,t} = sigmoid(α_e + β·Z_{e,t})
4. Event time: T_e sampled from discrete hazard
5. Features: Transformations of Z_{e,<t} (strictly causal)
6. Label: Y_{e,t} = 1{t < T_e ≤ t+H}
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy.special import expit  # sigmoid function


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return expit(x)


def generate_single_episode(
    episode_id: int,
    T_max: int,
    alpha_e: float,
    ar_coef: float,
    noise_std: float,
    hazard_coef: float,
    base_hazard: float,
    horizon: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate a single episode with AR(1) latent process.

    Parameters
    ----------
    episode_id : int
        Unique identifier for this episode
    T_max : int
        Maximum episode length (days)
    alpha_e : float
        Episode-specific random effect
    ar_coef : float
        AR(1) coefficient (0 < φ < 1 for stationarity)
    noise_std : float
        Standard deviation of AR(1) innovations
    hazard_coef : float
        Coefficient linking latent state to hazard (weak signal)
    base_hazard : float
        Baseline hazard rate (intercept in logit scale)
    horizon : int
        Forecast horizon H (days)
    rng : np.random.Generator
        Random number generator for reproducibility

    Returns
    -------
    pd.DataFrame
        Episode data with columns:
        - episode_id: Episode identifier
        - t: Time index (0, 1, ..., T_max-1 or until event)
        - Z: Latent intensity (for diagnostics, not used in modeling)
        - X_1, X_2, ...: Features (strictly causal transformations of Z)
        - T_e: Event time for this episode
        - Y: Label (1 if event within horizon, 0 otherwise)
        - at_risk: 1 if observation is before event time
    """
    # Initialize latent process
    Z = np.zeros(T_max)
    Z[0] = rng.normal(0, noise_std / np.sqrt(1 - ar_coef**2))  # Stationary initialization

    # Generate AR(1) latent intensity
    for t in range(1, T_max):
        Z[t] = ar_coef * Z[t-1] + rng.normal(0, noise_std)

    # Compute hazard at each time point
    hazard = sigmoid(base_hazard + alpha_e + hazard_coef * Z)

    # Sample event time from discrete hazard
    T_e = T_max + 1  # Default: censored (no event)
    for t in range(T_max):
        if rng.random() < hazard[t]:
            T_e = t + 1  # Event occurs at end of day t (so T_e = t+1)
            break

    # Determine actual episode length
    episode_length = min(T_e, T_max)

    # Create features (STRICTLY CAUSAL - only use Z_{<t})
    # Feature 1: Lagged latent value
    X_1 = np.zeros(episode_length)
    X_1[1:] = Z[:episode_length-1]  # X_1[t] = Z[t-1]

    # Feature 2: Rolling mean of past 3 values
    X_2 = np.zeros(episode_length)
    for t in range(episode_length):
        if t >= 3:
            X_2[t] = np.mean(Z[t-3:t])  # Mean of Z[t-3], Z[t-2], Z[t-1]
        elif t > 0:
            X_2[t] = np.mean(Z[:t])

    # Feature 3: Rolling std of past 5 values
    X_3 = np.zeros(episode_length)
    for t in range(episode_length):
        if t >= 5:
            X_3[t] = np.std(Z[t-5:t])
        elif t > 1:
            X_3[t] = np.std(Z[:t])

    # Feature 4: Trend (difference from 5 periods ago)
    X_4 = np.zeros(episode_length)
    for t in range(episode_length):
        if t >= 5:
            X_4[t] = Z[t-1] - Z[t-5]  # Use lagged values only

    # Feature 5: Episode-level static feature (known at t=0)
    X_5 = np.full(episode_length, alpha_e + rng.normal(0, 0.1))

    # Add noise to features
    noise_scale = 0.1
    X_1 += rng.normal(0, noise_scale, episode_length)
    X_2 += rng.normal(0, noise_scale, episode_length)
    X_3 += rng.normal(0, noise_scale, episode_length)
    X_4 += rng.normal(0, noise_scale, episode_length)

    # Create labels
    t_values = np.arange(episode_length)

    # Y_{e,t} = 1 if t < T_e <= t + H
    # This is 1 if the event happens within the next H days
    if T_e <= T_max:
        Y = ((t_values < T_e) & (T_e <= t_values + horizon)).astype(int)
    else:
        Y = np.zeros(episode_length, dtype=int)  # Censored: no event

    # At-risk indicator: 1 if we haven't observed the event yet
    at_risk = (t_values < T_e).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'episode_id': episode_id,
        't': t_values,
        'Z': Z[:episode_length],  # For diagnostics only
        'X_1': X_1,
        'X_2': X_2,
        'X_3': X_3,
        'X_4': X_4,
        'X_5': X_5,
        'T_e': T_e,
        'Y': Y,
        'at_risk': at_risk
    })

    return df


def generate_panel_data(
    n_episodes: int = 30,
    T_max: int = 60,
    ar_coef: float = 0.7,
    noise_std: float = 0.3,
    hazard_coef: float = 0.15,
    base_hazard: float = -3.0,
    alpha_std: float = 0.5,
    horizon: int = 14,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate complete panel dataset with multiple episodes.

    Parameters
    ----------
    n_episodes : int
        Number of episodes to generate
    T_max : int
        Maximum episode length
    ar_coef : float
        AR(1) coefficient for latent process
    noise_std : float
        Innovation standard deviation
    hazard_coef : float
        Effect of latent state on hazard (kept weak for realism)
    base_hazard : float
        Baseline log-odds of event (negative = rare events)
    alpha_std : float
        Standard deviation of episode random effects
    horizon : int
        Forecast horizon (days)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Complete panel dataset
    """
    rng = np.random.default_rng(seed)

    # Generate episode-level random effects
    alphas = rng.normal(0, alpha_std, n_episodes)

    # Generate each episode
    episodes = []
    for e in range(n_episodes):
        episode_df = generate_single_episode(
            episode_id=e,
            T_max=T_max,
            alpha_e=alphas[e],
            ar_coef=ar_coef,
            noise_std=noise_std,
            hazard_coef=hazard_coef,
            base_hazard=base_hazard,
            horizon=horizon,
            rng=rng
        )
        episodes.append(episode_df)

    # Combine all episodes
    df = pd.concat(episodes, ignore_index=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names."""
    return [col for col in df.columns if col.startswith('X_')]


def prepare_modeling_data(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for modeling (filter to at-risk observations).

    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    groups : np.ndarray
        Episode IDs (for grouped CV)
    """
    # Filter to at-risk observations only
    df_atrisk = df[df['at_risk'] == 1].copy()

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    X = df_atrisk[feature_cols].values
    y = df_atrisk['Y'].values
    groups = df_atrisk['episode_id'].values

    return X, y, groups


# Convenience function for quick data generation
def generate_default_data(seed: int = 42) -> pd.DataFrame:
    """Generate data with default parameters."""
    return generate_panel_data(
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


if __name__ == "__main__":
    # Test data generation
    print("Generating test data...")
    df = generate_default_data(seed=42)

    print(f"\nDataset shape: {df.shape}")
    print(f"Number of episodes: {df['episode_id'].nunique()}")
    print(f"Average episode length: {df.groupby('episode_id').size().mean():.1f}")
    print(f"Event rate: {df[df['at_risk']==1]['Y'].mean():.3f}")
    print(f"Number of at-risk observations: {df['at_risk'].sum()}")

    print("\nSample of data:")
    print(df.head(10))

    print("\nFeature columns:", get_feature_columns(df))

    # Prepare for modeling
    X, y, groups = prepare_modeling_data(df)
    print(f"\nModeling data shape: X={X.shape}, y={y.shape}")
    print(f"Class balance: {y.mean():.3f} positive")
