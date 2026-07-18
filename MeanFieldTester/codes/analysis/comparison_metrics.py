"""
Design choice here:
- return np.nan when the result is undefined
reasoning:
- having a list or results with np.nan can be converted in array better than having a list with None values

"""

import numpy as np

# Error metrics

def calc_mse(gt, target, **kwargs):
    """Calculates the mean squared error between ground truth and target."""
    return np.mean((target - gt)**2)

def calc_rmse(gt, target, **kwargs):
    """Calculates the root mean squared error between ground truth and target."""
    return np.sqrt(np.mean((target - gt)**2))

def calc_error_mean(gt, target, **kwargs):
    """Calculates the mean error between ground truth and target."""
    return np.mean(target - gt)

def calc_error_std(gt, target, **kwargs):
    """Calculates the standard deviation of the error between ground truth and target."""
    return np.std(target - gt)

# Correlation metrics

def calc_spearman(gt, target, **kwargs):
    """Spearman Rank Correlation"""
    from scipy.stats import spearmanr
    
    if np.std(gt) == 0 or np.std(target) == 0:
        return np.nan # Spearman correlation is undefined for constant arrays
    
    res = spearmanr(gt, target)
    return res[0]

def calc_pearson(gt, target, **kwargs):
    """Pearson Correlation"""
    from scipy.stats import pearsonr

    if np.std(gt) == 0 or np.std(target) == 0:
        return np.nan # Pearson correlation is undefined for constant arrays
    
    res = pearsonr(gt, target)
    return res[0]


# Delay metrics

def calc_lag(gt, target, dt, max_lag=100.0, **kwargs):
    """Optimal delay (lag) in ms where cross-correlation is maximized
    
    dt: time step in ms
    max_lag: maximum lag to consider in ms
    """
    if np.std(gt) == 0 or np.std(target) == 0:
        return 0.0
        
    gt_norm = (gt - np.mean(gt)) / np.std(gt)
    target_norm = (target - np.mean(target)) / np.std(target)
    
    cross_corr = np.correlate(target_norm, gt_norm, mode='full')
    
    N = len(gt)
    lags = np.arange(-N + 1, N)
    
    max_lag_samples = min(N - 1, int(max_lag / dt))
    
    valid_indices = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    lags = lags[valid_indices]
    cross_corr = cross_corr[valid_indices]
    
    if len(cross_corr) == 0:
        return 0.0
        
    best_lag_idx = np.argmax(cross_corr)
    return lags[best_lag_idx] * dt
    
def calc_max_corr(gt, target, dt, max_lag=100.0, **kwargs):
    """Maximum Correlation Coefficient at the optimal lag
    
    dt: time step in ms
    max_lag: maximum lag to consider in ms
    """
    if np.std(gt) == 0 or np.std(target) == 0:
        return np.nan
        
    gt_norm = (gt - np.mean(gt)) / np.std(gt)
    target_norm = (target - np.mean(target)) / np.std(target)
    
    cross_corr = np.correlate(target_norm, gt_norm, mode='full') / len(gt)
    
    N = len(gt)
    lags = np.arange(-N + 1, N)
    
    max_lag_samples = min(N - 1, int(max_lag / dt))
    
    valid_indices = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    cross_corr = cross_corr[valid_indices]
    
    if len(cross_corr) == 0:
        return np.nan
        
    return np.max(cross_corr)

# Power Spectral Density (PSD) similarity metrics

def calc_psd_similarity(gt, target, dt, **kwargs):
    """Cosine similarity of Power Spectral Densities (PSD)
    
    dt: time step in ms
    """
    from scipy.signal import welch
    
    fs = 1000.0 / dt  # fs in Hz
    
    nperseg = min(len(gt), 256)
    if nperseg < 8:
        return np.nan  # Not enough data for PSD estimation
        
    freqs_gt, psd_gt = welch(gt, fs=fs, nperseg=nperseg)
    freqs_target, psd_target = welch(target, fs=fs, nperseg=nperseg)
    
    norm_gt = np.linalg.norm(psd_gt)
    norm_target = np.linalg.norm(psd_target)
    
    if norm_gt == 0 or norm_target == 0:
        return 0.0
        
    return np.dot(psd_gt, psd_target) / (norm_gt * norm_target)

def calc_zero_freq_coherence(gt, target, dt, **kwargs):
    """Coherence between two signals
    
    dt: time step in ms
    """
    from scipy.signal import coherence
    
    fs = 1000.0 / dt  # fs in Hz
    
    nperseg = min(len(gt), 256)
    if nperseg < 8:
        return np.nan  # Not enough data for coherence estimation
        
    freqs, coh = coherence(gt, target, fs=fs, nperseg=nperseg)
    
    return np.sqrt(coh[0])



METRIC_REGISTRY = {
    "mse": calc_mse,
    "rmse": calc_rmse,
    "error_mean": calc_error_mean,
    "error_std": calc_error_std,
    "pearson": calc_pearson,
    "spearman": calc_spearman,
    "lag": calc_lag,
    "max_corr": calc_max_corr,
    "psd_similarity": calc_psd_similarity,
    "zero_freq_coherence": calc_zero_freq_coherence,
}
