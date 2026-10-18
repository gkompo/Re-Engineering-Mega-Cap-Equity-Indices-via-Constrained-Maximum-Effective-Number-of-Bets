import numpy as np
from sklearn.covariance import LedoitWolf

def estimate_shrinkage_covariance(returns: np.ndarray) -> np.ndarray:
    """
    Estimates a well-conditioned covariance matrix using Ledoit-Wolf optimal shrinkage.
    Essential for high dimensions (N=500, T=252).
    """
    lw = LedoitWolf()
    return lw.fit(returns).covariance_

def compute_minimum_linear_torsion(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates Attilio Meucci's Minimum Linear Torsion (MLT) transformation matrix:
        T_MT = (C^(1/2))^(-1) * D^(-1) = E * Lambda^(-1/2) * E' * D^(-1)
    where C is the correlation matrix and D is the diagonal volatility matrix.
    """
    stds = np.sqrt(np.diag(cov_matrix))
    stds = np.maximum(stds, 1e-8)
    inv_stds = np.diag(1.0 / stds)
    corr = inv_stds @ cov_matrix @ inv_stds
    
    # Riccati Root via spectral decomposition
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-8, None)
    
    # C^(-1/2) = E * Lambda^(-1/2) * E'
    inv_c_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    T_mt = inv_c_sqrt @ inv_stds
    return T_mt

def compute_effective_number_of_bets(weights: np.ndarray, cov_matrix: np.ndarray) -> tuple:
    """
    Calculates the Shannon entropy Effective Number of Bets (ENB):
        w_tilde = D * C^(1/2) * w
        p_k = (w_tilde_k^2) / sigma_p^2
        ENB = exp( - sum_k p_k * ln(p_k) )
    """
    stds = np.sqrt(np.diag(cov_matrix))
    stds = np.maximum(stds, 1e-8)
    inv_stds = np.diag(1.0 / stds)
    corr = inv_stds @ cov_matrix @ inv_stds
    
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-8, None)
    c_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    w_tilde = np.diag(stds) @ c_sqrt @ weights
    port_var = weights.T @ cov_matrix @ weights
    if port_var < 1e-12:
        return 1.0, np.ones(len(weights)) / len(weights)
        
    p = (w_tilde ** 2) / port_var
    p = np.clip(p, 1e-12, 1.0)
    p = p / np.sum(p)
    
    entropy = -np.sum(p * np.log(p))
    enb = np.exp(entropy)
    return enb, p
