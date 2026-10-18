import numpy as np
from scipy.optimize import minimize

def solve_max_enb(cov_matrix: np.ndarray, max_weight: float = None) -> np.ndarray:
    """
    Solves the Constrained Maximum Effective Number of Bets Non-Linear Program (NLP):
        min_w  sum_k p_k(w) * ln(p_k(w))
        s.t.   sum_i w_i = 1
               0 <= w_i <= max_weight  (or 1.0 if uncapped)
    Uses the exact machine-precision analytical gradient for ultra-fast SLSQP convergence.
    """
    n = cov_matrix.shape[0]
    stds = np.sqrt(np.diag(cov_matrix))
    stds = np.maximum(stds, 1e-8)
    inv_stds = np.diag(1.0 / stds)
    corr = inv_stds @ cov_matrix @ inv_stds
    
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-8, None)
    c_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    M = np.diag(stds) @ c_sqrt # w_tilde = M @ w

    # Objective: Minimize negative Shannon entropy
    def objective(w):
        w_tilde = M @ w
        S = np.sum(w_tilde ** 2)
        p = (w_tilde ** 2) / (S + 1e-12)
        p = np.clip(p, 1e-12, 1.0)
        p = p / np.sum(p)
        return np.sum(p * np.log(p))
    
    # Exact analytical gradient: d(Phi)/dw
    def gradient(w):
        w_tilde = M @ w
        S = np.sum(w_tilde ** 2) + 1e-12
        p = (w_tilde ** 2) / S
        p = np.clip(p, 1e-12, 1.0)
        p = p / np.sum(p)
        log_p_1 = 1.0 + np.log(p)
        term1 = M.T @ (log_p_1 * w_tilde)
        term2 = np.sum(log_p_1 * p) * (M.T @ w_tilde)
        return (2.0 / S) * (term1 - term2)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones_like(w)}]
    
    upper_bound = max_weight if max_weight is not None else 1.0
    bounds = [(0.0, upper_bound) for _ in range(n)]
    w0 = np.ones(n) / n
    
    res = minimize(objective, w0, method='SLSQP', jac=gradient, bounds=bounds, constraints=constraints,
                   options={'maxiter': 150, 'ftol': 1e-5})
    
    if res.success:
        return res.x
    return w0
