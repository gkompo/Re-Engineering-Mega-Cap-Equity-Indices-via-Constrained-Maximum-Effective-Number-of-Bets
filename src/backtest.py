import numpy as np
import pandas as pd
from .meucci_mlt import estimate_shrinkage_covariance, compute_effective_number_of_bets
from .optimizer import solve_max_enb

def walk_forward_backtest(returns_df: pd.DataFrame, benchmark_df: pd.DataFrame, 
                           rebal_days: int = 21, max_weight: float = None, 
                           window: int = 252, t_cost_bps: float = 0.0) -> tuple:
    """
    Executes a walk-forward out-of-sample backtest with zero lookahead bias.
    
    Parameters:
        returns_df: DataFrame of daily constituent returns
        benchmark_df: DataFrame of daily benchmark returns (e.g. SPY, RSP, QQQ, QQQE)
        rebal_days: Rebalancing horizon (21 = Monthly, 42 = Bi-Monthly, 63 = Quarterly)
        max_weight: Maximum stock weight cap (e.g. 0.05, 0.02, or None for uncapped)
        window: Historical in-sample estimation window (default: 252 trading days)
        t_cost_bps: Transaction cost per one-way turnover in basis points (default: 0.0)
    
    Returns:
        (out_of_sample_df, enb_series, avg_turnover)
    """
    dates = returns_df.index
    n_assets = returns_df.shape[1]
    
    port_rets = []
    enb_history = []
    rebal_dates = []
    turnovers = []
    
    prev_w = np.ones(n_assets) / n_assets
    cost_factor = t_cost_bps / 10000.0
    
    for t in range(window, len(dates) - rebal_days, rebal_days):
        # Strict in-sample training window: [t - window, t)
        train_returns = returns_df.iloc[t - window : t].values
        
        # 1. Ledoit-Wolf Shrinkage Covariance
        cov = estimate_shrinkage_covariance(train_returns)
        
        # 2. Max-ENB Optimization via Analytical Gradient
        opt_w = solve_max_enb(cov, max_weight=max_weight)
        
        # 3. ENB Metric Tracking
        enb_val, _ = compute_effective_number_of_bets(opt_w, cov)
        enb_history.append(enb_val)
        rebal_dates.append(dates[t])
        
        # 4. Drifted Weights Turnover & Friction
        turnover = np.sum(np.abs(opt_w - prev_w))
        turnovers.append(turnover)
        cost_penalty = turnover * cost_factor
        
        # 5. Strictly Out-of-Sample Forward Realization: [t, t + rebal_days)
        fwd_returns = returns_df.iloc[t : t + rebal_days].values @ opt_w
        fwd_returns[0] -= cost_penalty
        
        port_rets.extend(fwd_returns)
        prev_w = opt_w
        
    out_dates = dates[window : window + len(port_rets)]
    df_out = pd.DataFrame(index=out_dates)
    df_out['Max-ENB'] = port_rets
    for col in benchmark_df.columns:
        df_out[col] = benchmark_df[col].loc[out_dates]
        
    enb_series = pd.Series(enb_history, index=rebal_dates, name='ENB')
    return df_out, enb_series, np.mean(turnovers)
