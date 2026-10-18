import numpy as np
from scipy import stats

def run_econometric_evaluation(rx: np.ndarray, ry: np.ndarray, rf: float = 0.02/252) -> dict:
    """
    Computes formal financial econometric and asset pricing diagnostics:
        1. Jobson-Korkie with Memmel (2003) correction for Sharpe ratio difference
        2. CAPM Jensen's Alpha, market beta, t-statistic, and p-value
        3. Asymmetric Downside Beta vs. Upside Beta (Ang, Chen, & Xing, 2006)
        4. Maximum Drawdown
        5. Historical Value-at-Risk (VaR 95%) and Expected Shortfall (CVaR 95%)
    """
    T = len(rx)
    ex_x = rx - rf
    ex_y = ry - rf
    
    sr_x = ex_x.mean() / ex_x.std() * np.sqrt(252)
    sr_y = ex_y.mean() / ex_y.std() * np.sqrt(252)
    gamma = np.corrcoef(ex_x, ex_y)[0, 1]
    
    # Memmel (2003) Sharpe difference variance
    var_diff = (1.0 / T) * (2.0 * (1.0 - gamma) + 0.5 * (sr_x**2 + sr_y**2 - 2.0 * sr_x * sr_y * gamma**2))
    z_sharpe = (sr_x - sr_y) / (np.sqrt(var_diff) + 1e-12)
    p_sharpe = 2.0 * (1.0 - stats.norm.cdf(abs(z_sharpe)))
    
    # CAPM Alpha regression: rx = alpha + beta * ry + eps
    slope, intercept, r_value, p_value_capm, std_err = stats.linregress(ry, rx)
    ann_alpha = intercept * 252
    t_alpha = intercept / (std_err + 1e-12)
    
    # Ang, Chen, Xing (2006) Downside & Upside Beta
    down_mask = ry < 0
    up_mask = ry >= 0
    down_beta = np.cov(rx[down_mask], ry[down_mask])[0, 1] / np.var(ry[down_mask])
    up_beta = np.cov(rx[up_mask], ry[up_mask])[0, 1] / np.var(ry[up_mask])
    
    # Historical VaR 95% and CVaR 95%
    var_95_x = -np.percentile(rx, 5) * np.sqrt(252)
    var_95_y = -np.percentile(ry, 5) * np.sqrt(252)
    cvar_95_x = -np.mean(rx[rx <= np.percentile(rx, 5)]) * np.sqrt(252)
    cvar_95_y = -np.mean(ry[ry <= np.percentile(ry, 5)]) * np.sqrt(252)
    
    cum_x = np.cumprod(1 + rx)
    max_dd_x = np.min(cum_x / np.maximum.accumulate(cum_x) - 1.0)
    cum_y = np.cumprod(1 + ry)
    max_dd_y = np.min(cum_y / np.maximum.accumulate(cum_y) - 1.0)
    
    return {
        'Sharpe Ratio (X)': sr_x,
        'Sharpe Ratio (Y)': sr_y,
        'Sharpe Diff z-stat': z_sharpe,
        'Sharpe Diff p-val': p_sharpe,
        'Annualized Alpha': ann_alpha,
        'Alpha t-stat': t_alpha,
        'Alpha p-val': p_value_capm,
        'Market Beta': slope,
        'Downside Beta': down_beta,
        'Upside Beta': up_beta,
        'Max Drawdown (X)': max_dd_x,
        'Max Drawdown (Y)': max_dd_y,
        'VaR 95% (X)': var_95_x,
        'VaR 95% (Y)': var_95_y,
        'CVaR 95% (X)': cvar_95_x,
        'CVaR 95% (Y)': cvar_95_y
    }
