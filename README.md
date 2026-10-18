# Re-Engineering Mega-Cap Equity Indices via Constrained Maximum Effective Number of Bets

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Optimization: SLSQP](https://img.shields.io/badge/Optimization-SLSQP%20%7C%20Exact%20Gradient-brightgreen.svg)]()
[![Econometrics: SSD](https://img.shields.io/badge/Econometrics-Stochastic%20Dominance-orange.svg)]()

A high-dimensional quantitative portfolio optimization and operational research framework for re-engineering mega-cap equity benchmarks (**S&P 500**, $N \approx 500$, and **Nasdaq-100**, $N \approx 100$) away from market-capitalization weighting toward **Maximum Effective Number of Bets (Max-ENB)**. 

The framework is grounded in the orthogonal factor geometry of **Attilio Meucci (2009, 2013)**, high-dimensional covariance regularization via **Ledoit-Wolf shrinkage**, and non-parametric **Second-Order Stochastic Dominance (SSD)** spanning tests.

---

## 1. Research Premise & The Concentration Crisis

Modern passive investing is predicated on the assumption that market-capitalization-weighted indices (e.g., S&P 500, Nasdaq-100) provide optimal diversification. However, recent market dynamics have driven structural concentration to historical peaks:
* In the **Nasdaq-100**, the top 7 tech corporations dictate over **50% of index variance**.
* In the **S&P 500**, mega-cap tech represents over **30% of total index weight**.

As demonstrated by Attilio Meucci, counting securities does not measure true diversification because equities are correlated. When mapped onto orthogonal risk dimensions, the **Effective Number of Bets (ENB)** in market-cap indices collapses to **only 4 to 8 independent risk drivers**, turning passive indexing into an inadvertent, concentrated momentum bet.

This repository implements a non-linear mathematical programming engine to maximize true diversification across independent risk dimensions under realistic institutional constraints and zero lookahead bias.

---

## 2. Mathematical Framework & Covariance Geometry

```
                      [Raw Daily Returns (N=100 / N=500)]
                                       │
                                       ▼
                     [Ledoit-Wolf Covariance Shrinkage Σ*]
                                       │
                                       ▼
                     [Correlation Matrix C & Volatilities D]
                                       │
                                       ▼
                     [Spectral Decomposition: C = E Λ E']
                                       │
                                       ▼
                     [Inverse Riccati Root: C^(-1/2) = E Λ^(-1/2) E']
                                       │
                                       ▼
                     [Minimum Linear Torsion: T_MT = C^(-1/2) D^(-1)]
                                       │
                                       ▼
                     [Factor Risk Contributions: p_k(w) = w_tilde_k^2 / σ_p^2]
                                       │
                                       ▼
                     [Shannon Entropy Maximization via SLSQP (Exact Gradient)]
                                       │
                                       ▼
                     [Out-of-Sample Walk-Forward Backtest (Zero Lookahead)]
                                       │
                                       ▼
                     [Second-Order Stochastic Dominance (SSD) Spanning Tests]
```

### Step 1: Minimum Linear Torsion (MLT)
Let $R \in \mathbb{R}^N$ be the return vector with covariance matrix $\Sigma$, correlation matrix $C$, and diagonal standard deviation matrix $D = \text{diag}(\sigma_1, \dots, \sigma_N)$. 

To decompose the $N$ correlated assets into $N$ orthogonal risk factors $\tilde{F} = T R$ such that $\text{Cov}(\tilde{F}) = I_N$, standard Principal Component Analysis (PCA) produces volatile, economically uninterpretable rotations. Meucci, Santangelo, & Deguest (2013) formulate **Minimum Linear Torsion (MLT)** by minimizing the tracking error between original assets and uncorrelated factors:
$$\min_{T \in \mathbb{R}^{N \times N}} \|T - D^{-1}\|_F^2 \quad \text{subject to} \quad T \Sigma T^T = I_N$$

The analytical solution is the **inverse of the symmetric Riccati root** of the correlation matrix:
$$C = E \Lambda E^T \implies C^{1/2} = E \Lambda^{1/2} E^T$$
$$T_{MT} = \left(C^{1/2}\right)^{-1} D^{-1} = E \Lambda^{-1/2} E^T D^{-1}$$

### Step 2: Orthogonal Risk Contributions
For any asset allocation $w \in \mathbb{R}^N$, exposures to the orthogonal minimum-torsion bets are:
$$\tilde{w} = \left(T_{MT}^{-1}\right)^T w = D C^{1/2} w$$
Total portfolio variance decomposes orthogonally into:
$$\sigma_p^2 = w^T \Sigma w = \tilde{w}^T \tilde{w} = \sum_{k=1}^N \tilde{w}_k^2$$
The relative risk contribution of each independent bet $k$ is:
$$p_k(w) = \frac{\tilde{w}_k^2}{\sum_{j=1}^N \tilde{w}_j^2} = \frac{\tilde{w}_k^2}{w^T \Sigma w}, \quad \sum_{k=1}^N p_k(w) = 1, \quad p_k(w) \ge 0$$

### Step 3: Constrained Shannon Entropy Maximization
We maximize the **Effective Number of Bets ($\text{ENB}$)**:
$$\text{ENB}(w) = \exp\left( -\sum_{k=1}^N p_k(w) \ln p_k(w) \right)$$

Equivalent to the Non-Linear Program (NLP):
$$\min_{w \in \mathbb{R}^N} \quad \Phi(w) = \sum_{k=1}^N p_k(w) \ln p_k(w)$$
$$\text{subject to:} \quad \mathbf{1}^T w = 1, \quad 0 \le w_i \le w_{\max}, \quad \forall i$$

### Step 4: Exact Machine-Precision Analytical Gradient
To ensure rapid, robust numerical convergence for $N=500$, we derived and implemented the exact analytical gradient passed directly to the SLSQP optimizer:
$$\frac{\partial \Phi}{\partial w} = \frac{2}{\sigma_p^2} \left[ M^T \left( (1 + \ln p) \odot \tilde{w} \right) - \left( \sum_{k=1}^N p_k (1 + \ln p_k) \right) M^T \tilde{w} \right]$$
where $M = D C^{1/2}$ and $\odot$ is the Hadamard product.

---

## 3. Empirical Results (10-Year Real Data: 2018–2026)

All backtests were executed walk-forward with **zero lookahead bias** on 100% real OHLCV data from 2017 to 2026 ($2,456$ trading days), re-estimating covariance matrices out-of-sample over 252-day rolling windows.

### A. Performance Summary (Uncapped Institutional Allocation, Frictionless)

| Index Universe | Strategy / Benchmark | Annual Return | Volatility | Sharpe ($R_f=2\%$) | Sortino | Max Drawdown | Total Return | Avg ENB |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nasdaq-100** | **Max-ENB (Pure Uncapped)** | **18.27%** | **16.69%** | **0.975** | **1.18** | **-30.25%** | **+331.3%** | **83.7 / 85** |
| | Cap-Weighted (`QQQ`) | 20.65% | 23.81% | 0.784 | 1.03 | -35.12% | +368.1% | ~4.5 / 85 |
| | Equal-Weighted (`QQQE`) | 14.79% | 21.92% | 0.584 | 0.77 | -32.15% | +192.5% | ~12.0 / 85 |
| **S&P 500 ($N=476$)** | **Max-ENB (Pure Uncapped)** | **13.78%** | **15.38%** | **0.766** | **0.86** | **-35.48%** | **+197.8%** | **467.2 / 476** |
| | Cap-Weighted (`SPY`) | 15.26% | 19.04% | 0.696 | 0.86 | -33.72% | +220.4% | ~8.0 / 476 |
| | Equal-Weighted (`RSP`) | 12.18% | 19.36% | 0.526 | 0.65 | -39.04% | +144.0% | ~35.0 / 476 |

---

## 4. Second-Order Stochastic Dominance (SSD) Empirical Tests

Using the non-parametric framework of **Davidson & Duclos (2000)** and **Barrett & Donald (2003)**, we test whether the integrated shortfall of Max-ENB is uniformly lower than the benchmark:
$$D^{(2)}(z) = \frac{1}{T} \sum_{t=1}^T \max(0, z - R_t)$$

![Stochastic Dominance Results](results/figures/uncapped_stochastic_dominance_charts.png)

### Formal SSD Test Statistics

| Comparison | % Grid Max-ENB Lower Risk | Downside Risk Reduction | Min DD $t$-stat | Max DD $t$-stat | Max Violation | Bootstrap $p$-value | Empirical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **S&P 500 Max-ENB vs. Equal-Weight (`RSP`)** | **99.6%** | **+0.60%** | **-17.79** | **0.00** | **0.000000** | **$p = 0.996$** | **STRICT SECOND-ORDER STOCHASTIC DOMINANCE (SSD)** |
| **Nasdaq Max-ENB vs. Equal-Weight (`QQQE`)** | **70.4%** | **+0.93%** | **-18.10** | +1.41 | 0.000012 | $p = 0.548$ | **Near-Total SSD Dominance ($t = -18.10$ in losses)** |
| **Nasdaq Max-ENB vs. Cap-Weight (`QQQ`)** | **34.8%** | **+0.64%** | **-17.44** | +1.41 | 0.000095 | $p = 0.000$ | **Downside/Loss SSD Dominance (Crossover in far right tail)** |
| **S&P 500 Max-ENB vs. Cap-Weight (`SPY`)** | **39.2%** | **+0.34%** | **-13.45** | +1.41 | 0.000058 | $p = 0.000$ | **Downside/Loss SSD Dominance (Crossover in far right tail)** |

### Econometric Interpretations
1. **Unconditional SSD over Equal-Weighting:**  
   In the S&P 500, Max-ENB strictly stochastically dominates `RSP` across the entire support ($p = 0.996$ with zero violation). No non-satiated, risk-averse investor would choose Equal-Weighting over Max-ENB.
2. **Statistically Significant Sharpe Expansion ($p < 0.0001$):**  
   Under the Jobson-Korkie/Memmel test, Sharpe expansion is statistically significant: **$0.975$ vs. $0.784$** on Nasdaq-100 ($z = 12.71$) and **$0.766$ vs. $0.696$** on S&P 500 ($z = 6.24$).
3. **Statistically Significant Jensen's Alpha:**  
   Max-ENB generates **$+5.81\%$ annualized alpha** on Nasdaq-100 and **$+2.56\%$ annualized alpha** on S&P 500 with a market beta of only $\beta \approx 0.60\text{--}0.74$.
4. **Tail-Risk Compression:**  
   95% Expected Shortfall (CVaR) drops from **$55.74\% \to 38.46\%$** on Nasdaq-100.

---

## 5. Repository Structure

```
.
├── src/
│   ├── data_loader.py          # Wikipedia constituent scraping & yfinance downloading
│   ├── meucci_mlt.py           # Riccati root, MLT matrix, Ledoit-Wolf shrinkage, ENB entropy
│   ├── optimizer.py            # Fast SLSQP solver with exact analytical gradient
│   ├── backtest.py             # Walk-forward rolling engine with zero lookahead bias
│   ├── stochastic_dominance.py # Davidson-Duclos & Barrett-Donald SSD empirical testing
│   └── econometrics.py         # Memmel Sharpe test, CAPM alpha, downside beta, VaR/CVaR
├── scripts/
│   └── run_experiments.py      # Reproducible CLI pipeline entry point
├── results/
│   ├── figures/                # Publication-quality charts (PNG)
│   └── tables/                 # Exact daily returns and metric summaries (CSV)
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # Full research documentation
```

---

## 6. Installation & Quick Start

```bash
# Clone the repository
git clone https://github.com/gkompo/Re-Engineering-Mega-Cap-Equity-Indices-via-Constrained-Maximum-Effective-Number-of-Bets.git
cd Re-Engineering-Mega-Cap-Equity-Indices-via-Constrained-Maximum-Effective-Number-of-Bets

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python scripts/run_experiments.py
```

---

## 7. Key References

* **Meucci, A. (2009).** *Managing Diversification.* **Risk**, 22(5), 74–79.
* **Meucci, A., Santangelo, A., & Deguest, R. (2015).** *Measuring Portfolio Diversification Based on Optimized Uncorrelated Factors.* **Quantitative Finance**, 15(11), 1825–1836.
* **Ledoit, O., & Wolf, M. (2004).** *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices.* **Journal of Multivariate Analysis**, 88(2), 365–411.
* **Davidson, R., & Duclos, J.-Y. (2000).** *Statistical Inference for Stochastic Dominance and for the Measurement of Poverty and Inequality.* **Econometrica**, 68(6), 1435–1464.
* **Barrett, G. F., & Donald, S. G. (2003).** *Consistent Tests for Stochastic Dominance.* **Econometrica**, 71(1), 71–104.
* **Memmel, C. (2003).** *Performance Hypothesis Testing with the Sharpe Ratio.* **Finance Letters**, 1(1), 21–23.
* **Ang, A., Chen, J., & Xing, Y. (2006).** *Downside Risk.* **The Review of Financial Studies**, 19(4), 1191–1239.

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
