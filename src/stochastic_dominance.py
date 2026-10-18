import numpy as np

def safe_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Safe trapezoidal integration supporting NumPy 1.x and 2.x."""
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y, x))
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))

def compute_second_order_shortfall(r: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Computes the integrated empirical shortfall function:
        D^(2)(z) = (1 / T) * sum_{t=1}^T max(0, z - R_t)
    """
    # Vectorized broadcasting: (num_grid, T)
    shortfall = np.maximum(0.0, grid[:, None] - r[None, :])
    return np.mean(shortfall, axis=1)

def test_second_order_stochastic_dominance(rx: np.ndarray, ry: np.ndarray, 
                                           name_x: str = "Strategy X", name_y: str = "Strategy Y", 
                                           num_grid: int = 250, n_boot: int = 250) -> tuple:
    """
    Conducts Davidson & Duclos (2000) and Barrett & Donald (2003) Second-Order Stochastic Dominance (SSD) tests.
    
    H0: X dominates Y under SSD <=> sup_z [ D2_X(z) - D2_Y(z) ] <= 0
    """
    T = len(rx)
    z_min, z_max = min(rx.min(), ry.min()), max(rx.max(), ry.max())
    grid = np.linspace(z_min, z_max, num_grid)
    
    shortfall_x = np.maximum(0.0, grid[:, None] - rx[None, :])
    shortfall_y = np.maximum(0.0, grid[:, None] - ry[None, :])
    
    D2_x = np.mean(shortfall_x, axis=1)
    D2_y = np.mean(shortfall_y, axis=1)
    diff = D2_x - D2_y # If diff <= 0 everywhere, X dominates Y
    
    # Davidson & Duclos (2000) asymptotic t-statistics across the grid
    d_diff = shortfall_x - shortfall_y
    var_d = np.var(d_diff, axis=1, ddof=1)
    t_stats = (D2_x - D2_y) / (np.sqrt(var_d / T) + 1e-12)
    
    # Barrett & Donald (2003) Kolmogorov-Smirnov statistics
    ks_x_dom_y = np.sqrt(T) * np.max(diff)
    ks_y_dom_x = np.sqrt(T) * np.max(-diff)
    
    pct_x_better = np.mean(diff < 0) * 100.0
    
    area_x = safe_trapezoid(D2_x, grid)
    area_y = safe_trapezoid(D2_y, grid)
    shortfall_reduction = (area_y - area_x) / area_y
    
    # Block Bootstrap for empirical KS p-value
    block_size = 10
    n_blocks = T // block_size
    boot_stats = []
    np.random.seed(42)
    for _ in range(n_boot):
        idx = np.random.randint(0, T - block_size, size=n_blocks)
        boot_idx = np.concatenate([np.arange(i, i + block_size) for i in idx])[:T]
        bx = rx[boot_idx] - np.mean(rx[boot_idx])
        by = ry[boot_idx] - np.mean(ry[boot_idx])
        b_sx = np.maximum(0.0, grid[:, None] - bx[None, :])
        b_sy = np.maximum(0.0, grid[:, None] - by[None, :])
        b_diff = np.mean(b_sx, axis=1) - np.mean(b_sy, axis=1)
        boot_stats.append(np.sqrt(T) * np.max(b_diff))
        
    p_val = np.mean(np.array(boot_stats) >= ks_x_dom_y)
    
    metrics = {
        'Comparison': f"{name_x} vs. {name_y}",
        'Mean Return (X)': rx.mean() * 252,
        'Mean Return (Y)': ry.mean() * 252,
        'Volatility (X)': rx.std() * np.sqrt(252),
        'Volatility (Y)': ry.std() * np.sqrt(252),
        'Downside Shortfall Reduction': shortfall_reduction * 100.0,
        '% Grid X Lower Risk': pct_x_better,
        'Min Davidson-Duclos t-stat': np.min(t_stats),
        'Max Davidson-Duclos t-stat': np.max(t_stats),
        'Max Viol (X dom Y)': np.max(diff),
        'Max Viol (Y dom X)': np.max(-diff),
        'KS Stat (X dom Y)': ks_x_dom_y,
        'Bootstrap p-value': p_val
    }
    
    return metrics, (grid, D2_x, D2_y, diff, t_stats)
