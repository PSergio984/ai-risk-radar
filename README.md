# AI Investment Risk Radar

An end-to-end, reproducible quantitative risk forecasting system and defensive allocation dashboard built for the Stock Market Analytics Zoomcamp final project.

---

## 1. Problem Statement & Motivation

Investments in artificial intelligence and semiconductor hardware (represented by the **SOXX** semiconductor ETF and **XLK** technology ETF) have generated substantial long-term returns but suffer from acute peak-to-trough drawdowns and severe concentration volatility. For retail investors, large market corrections often trigger panic selling at market bottoms, causing permanent capital impairment.

The **AI Investment Risk Radar** provides an objective, calibrated quantitative framework to manage downside risk:
1. **Explicit Downside Risk Definition**: Forecasts the probability that an AI-sensitive reference basket (50% SOXX / 50% XLK) ends the next 20 trading days with a cumulative total return $\le -5.0\%$. This measures *end-of-horizon loss probability*, strictly distinguished from intra-horizon maximum drawdown.
2. **Delayed Allocation Policy**: When the calibrated downside probability exceeds an empirically selected threshold (selected on validation history, e.g., 50.0%), the portfolio rotates from a **Growth stance** (35% SOXX, 35% XLK, 20% SPY, 10% SHY) to a **Defensive stance** (10% SOXX, 15% XLK, 35% SPY, 40% SHY).
3. **Institutional Execution Realism**: Rebalance signals generated on the final completed trading session of a week execute at the *next* trading session's close. Daily portfolio holdings drift naturally between weekly rebalances, and realistic round-trip transaction costs (10 bps baseline, with 5 bps and 25 bps sensitivity) are deducted directly from portfolio equity using exact numerical conservation.
4. **Leakage-Free Evaluation**: Enforces strict chronological splits (5.87 years training, 12 months Platt scaling calibration, 24 months validation, and 24 months untouched holdout test) with 20-session label purging across all boundaries to prevent temporal leakage.
5. **Two-Mode Presentation Dashboard**: Interactive Streamlit dashboard providing a scannable **Compact View** (default for retail investors) and an analytical **Detailed View** (quantitative cockpit with benchmark comparisons, reliability diagnostics, and failure regimes). Both views consume identical underlying JSON reports.

---

## 2. Fixed Asset Universe

The strategy universe is restricted to four liquid US ETFs:

| Ticker | Asset Description | Role in Strategy |
| :--- | :--- | :--- |
| **SOXX** | iShares Semiconductor ETF | AI hardware & semiconductor sector proxy |
| **XLK** | Technology Select Sector SPDR Fund | Broad enterprise technology & software proxy |
| **SPY** | SPDR S&P 500 ETF Trust | Broad US equity market ballast |
| **SHY** | iShares 1-3 Year Treasury Bond ETF | Short-duration Treasury bond buffer (subject to interest rate and market risk; capital preservation buffer) |

Reference AI basket: `50% SOXX + 50% XLK` (buy-and-hold through each 20-session forecast horizon).

---

## 3. Environment Setup & Clean First Use

The project runs in Python 3.12+ as a standalone repository rooted at `/`.

### Step 1: Create Virtual Environment

```bash
# In repository root:
python -m venv .venv
.venv\Scripts\activate       # Windows PowerShell / CMD
# source .venv/bin/activate  # Linux / macOS
```

### Step 2: Install Dependencies & Editable Package

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 4. Verification & Testing

Run the full deterministic unit, financial correctness, and remediation test suite (60+ tests) and code quality linter:

```bash
# Run offline test suite
pytest -v -m "not historical_artifacts"

# Run Ruff static analysis & code quality checks
ruff check .
```

---

## 5. CLI Pipeline Commands

The system features four distinct operational workflows separating ingestion, research, holdout reproduction, and live inference:

### 1. Ingestion / Data Acquisition
Fetch live price data from yfinance aligned with the NYSE (`XNYS`) exchange calendar. New data are isolated in `data/raw/latest`, preserving frozen baseline replay snapshots in `data/raw/snapshot`:

```bash
python -m ai_risk_radar latest --data-mode refresh
```

### 2. Research & Model Training (`research`)
Performs feature engineering, train/cal/val split purging, model training, Platt scaling calibration, and validation threshold selection. Generates `artifacts/release.json` and `artifacts/research_report.json`. The holdout test partition is strictly excluded:

```bash
python -m ai_risk_radar research --config config.json --data-mode cache
# Use --data-mode refresh to fetch live prices incrementally
```

### 3. Single-Use Final Holdout Test & Bit-Exact Reproduction (`final-test`)
Evaluates the frozen release model on the 24-month holdout test partition (2024-11-18 to 2026-11-16, 500 trading sessions). Cryptographic SHA-256 release hashes and policy parameters are strictly verified before execution:

- **First-Time Evaluation:** Evaluates an untouched release manifest, records holdout consumption in `release.json`, and writes `artifacts/final_test_report.json`.
- **Reproduction Mode:** When executed on an already-evaluated release (e.g. `artifacts/release.json`), the pipeline verifies bit-exact trajectory reproduction against the historical holdout evaluation, generates `artifacts/reproduction_test_report.json`, and leaves `release.json` and `final_test_report.json` strictly immutable.

```bash
python -m ai_risk_radar final-test --release artifacts/release.json
```

### 4. Latest Risk Inference (`latest`)
Loads the frozen calibrated release model and performs forward-looking inference on the latest completed market session. Retraining and parameter tuning are strictly disabled. Writes `artifacts/latest_report.json`:

```bash
python -m ai_risk_radar latest --release artifacts/release.json --data-mode cache
```

---

## 6. Comprehensive 43-Feature Inventory

The feature pipeline in `ai_risk_radar/features.py` engineers exactly **43 causal features** derived strictly from price and volume history available at the previous session close ($t-1$):

| Feature Category | Count | Feature Names / Formula | Causal Availability | Beyond Lecture Examples |
| :--- | :---: | :--- | :---: | :--- |
| **Return Lookbacks** | 20 | `ret_{asset}_{k}` for asset $\in \{\text{SOXX}, \text{XLK}, \text{SPY}, \text{SHY}, \text{Basket}\}$, $k \in \{1, 5, 20, 63\}$ | Backward-looking close-to-close returns at $t-1$ | Multi-asset horizon spectrum extending across 1-day, 1-week, 1-month, and 1-quarter scales, including normalized composite basket tracking. |
| **Realized Volatilities** | 10 | `vol_{asset}_{k}`: Annualized rolling std dev of daily returns $\sigma_k \times \sqrt{252}$ for $k \in \{20, 63\}$ | Backward-looking rolling windows ending at $t-1$ | Realized volatility cones capturing short-term regime spikes vs quarterly structural market dispersion. |
| **Peak Drawdowns** | 4 | `dd_{asset}_63`: Current price relative to 63-session peak: $P_t / \max_{s \in [t-62, t]}(P_s) - 1$ | Backward-looking rolling maximum ending at $t-1$ | Trailing peak-to-trough drawdown ratios identifying distance from structural resistance and technical exhaustion. |
| **Relative Strength** | 6 | `rel_{asset}_spy_{k}`: Excess return spread $R_{\text{asset}, k} - R_{\text{SPY}, k}$ for $k \in \{20, 63\}$ | Backward-looking spread at $t-1$ | Cross-sectional relative performance isolating tech/semiconductor alpha vs broader market ballast. |
| **Cross-Asset Correlations** | 3 | `corr_soxx_spy_63`, `corr_xlk_spy_63`, `corr_soxx_xlk_63`: 63-session rolling correlation | Backward-looking rolling correlation at $t-1$ | Dynamic multi-asset co-movement tracking beta decoupling and diversification breakdown during stress. |
| **Total Features** | **43** | All 43 columns verified in `features.feature_columns` | All strictly $t-1$ | Comprehensive causal technical taxonomy beyond basic single-asset moving averages. |

> [!NOTE]
> **Methodological Disclosure:** The NYSE exchange calendar (`XNYS`) is strictly used as an operational alignment mechanism, not as an independent predictive dataset. Platt scaling is utilized strictly as a probability post-calibration step, not as a standalone predictive model.

---

## 7. Vectorized Baseline & Sequential Execution Reconciliation

In quantitative portfolio analysis, backtests are frequently represented either as vectorized matrix arithmetic or as sequential state engines. The system implements and formally reconciles both paradigms under matched conventions:

### Mathematical Formulations

1. **Vectorized Matrix Baseline (`vectorized_linear_backtest`):**
   $$r_{\text{gross}, t} = \sum_{j} w_{j, t-1} R_{j, t}$$
   $$r_{\text{net}, t} = r_{\text{gross}, t} - c \sum_{j} |w_{j, t} - w_{j, t-1}|$$
   $$V_t = V_0 \cdot \frac{1}{1 + c} \cdot \prod_{s=1}^{t} (1 + r_{\text{net}, s})$$
   Signals execute at the next session close ($w_{t-1}$ lag), transaction cost rate $c = 0.001$ applies to turnover, and initial cash funding incurs an initial setup friction $V_0 / (1 + c)$.

2. **Sequential Daily-Drifted Engine (`simulate`):**
   Tracks physical asset holdings $h_{j, t}$ each day. Holdings drift naturally as asset prices change:
   $$h_{j, t}^{\text{pre}} = h_{j, t-1}^{\text{post}} \times (1 + R_{j, t}), \quad V_{t}^{\text{pre}} = \sum_j h_{j, t}^{\text{pre}}$$
   At rebalance sessions, post-trade equity is solved numerically using Brent's method to conserve equity under nonlinear transaction costs:
   $$V_t^{\text{post}} = V_t^{\text{pre}} - c \sum_j |w_{j, t} V_t^{\text{post}} - h_{j, t}^{\text{pre}}|$$

### Empirical Reconciliation & Limits of Matching

The vectorized backtest serves as a **linear matrix approximation**:
- **Vectorized Linear Approximation:** Assumes target weights remain static intra-week, applying a linear turnover penalty $c \sum |\Delta w|$.
- **Sequential Execution Engine:** Tracks exact physical share holdings drifting daily with individual asset prices, and solves self-financing cash conservation via Brent's method ($V^{\text{post}} + \text{Cost} = V^{\text{pre}}$).

```
Sample Reconciliation Summary (Validation Partition, 2022-11-18 to 2024-11-18):
  total_return           | Sequential:     0.6936 | Vectorized:     0.6979 | Diff: -0.0043 (-43.4 bps)
  cagr                   | Sequential:     0.3186 | Vectorized:     0.3204 | Diff: -0.0018 (-18 bps)
  max_drawdown           | Sequential:    -0.1636 | Vectorized:    -0.1632 | Diff: -0.0005 (-5 bps)
  annualized_volatility  | Sequential:     0.2002 | Vectorized:     0.2002 | Diff:  0.0000
  sharpe                 | Sequential:     1.4794 | Vectorized:     1.4859 | Diff: -0.0064
  turnover               | Sequential:     4.8338 | Vectorized:     4.3100 | Diff: +0.5238
```

**Methodological Interpretation of Agreement:**
1. **Conventions vs Equivalence:** Close numerical proximity (~30–50 bps over multi-year horizons, e.g. ~35 bps on synthetic test fixtures, 43.4 bps on the 2-year validation partition) confirms that both implementations adhere to matched institutional conventions: 1-session delayed execution, initial capital funding deduction, and a 10 bps turnover penalty. Numerical proximity alone does *not* imply identical mathematical formulas.
2. **Sources of Divergence:** The residual divergence varies naturally across datasets and time periods depending on:
   - **Intra-Week Price Dispersion & Drift:** Asset prices move independently between rebalances, causing physical share weights to drift away from target weights. Higher return dispersion increases intra-week drift, requiring greater rebalance turnover (+0.52x) in the sequential engine.
   - **Nonlinear Cost Conservation:** Brent's method solves the exact implicit relation $V^{\text{post}} + c \sum |w_j V^{\text{post}} - h_j^{\text{pre}}| = V^{\text{pre}}$, whereas the linear vector backtest subtracts $c \sum |\Delta w|$.
3. **Exact Mathematical Equivalence Case:** When asset prices are constant (or returns are identical across all assets with zero rebalancing costs), intra-week holding drift is exactly zero and Brent's method simplifies linearly. In this case, vectorized and sequential backtests produce bit-exact identical returns ($0.000000$ difference), as verified by deterministic unit tests.

---

## 8. Prediction-Error, Calibration & Regime Failure Analysis

Quantitative evaluation across non-overlapping market regimes reveals key model strengths and honest failure boundaries:

### 1. Discriminative Performance in Trend vs Chop Regimes
- **Crash Regime Success (e.g. 2022 Semiconductor Bear Market):** The Random Forest downside model accurately flags sustained negative momentum, high realized volatility, and tech underperformance, rotating into the Defensive stance (40% SHY, 35% SPY) and reducing maximum drawdown.
- **Sideways / High-Volatility Chop Regimes:** When the reference basket oscillates near the -5.0% threshold without a sustained directional trend, the model experiences false-positive churn. This generates turnover costs without achieving downside protection, highlighting the limitation of weekly discrete rotation.

### 2. Platt Scaling Calibration & Support Disclosures
- **Calibration Support Requirements:** Platt scaling requires adequate positive downside occurrences in the 12-month calibration partition. When training on prolonged bull markets, downside event prevalence can drop below empirical support thresholds (<20 observations or <2 positive events separated by 20 sessions).
- **Graceful Fallback:** If calibration support is inadequate, probability output is marked `unavailable` and portfolio allocation safely defaults to the fixed baseline allocation (20% SOXX, 20% XLK, 20% SPY, 40% SHY) rather than emitting misleading probabilities.

### 3. Threshold Selection Sensitivity
- **Lower Thresholds (0.20 - 0.35):** Exhibit higher sensitivity, triggering frequent defensive rotations that incur transaction costs (~4.8x annual turnover) and trail buy-and-hold returns during tech rallies.
- **Conservative Threshold (0.50):** Demonstrates superior capital preservation on validation and holdout periods, rotating defensive only during severe high-conviction risk events while remaining fully invested in Growth during bull expansions.

> [!WARNING]
> **No Downside Guarantee:** Machine learning risk models do not guarantee capital protection against unexpected overnight gaps, geopolitical shocks, or flash crashes that occur between rebalance execution intervals.

---

## 9. Interactive Two-Mode Dashboard

Launch the Streamlit web dashboard:

```bash
python -m streamlit run app.py
```

### Dashboard Modes:
- **Compact View (Default):** Designed for quick daily briefings. Features a scannable Risk Level badge, downside event probability, dynamic event definition, target asset allocation weights, current trade action, and KPI summary.
- **Detailed View:** Full quantitative cockpit. Adds benchmark comparison matrix (vs Growth Buy-and-Hold, Fixed Defensive, Momentum-63), calibration reliability diagnostics (Brier scores, bin predictions vs observed frequencies), causal feature drivers, and documented model failure regimes.

Both modes read from the identical underlying artifact file. Return differences between strategies are formatted in percentage points (`pp`) rather than deceptive relative percentages.

---

## 10. Financial Performance & Backtest Results

### Holdout Test Evaluation Summary (2024-11-18 to 2026-11-16, 500 trading sessions)

Both original (evaluating capital growth relative to post-funding equity $9,990.01) and corrected arithmetic (evaluating net capital return relative to deposited $10,000.00 capital, accounting for initial funding friction) are reported transparently:

#### Corrected Net Capital Performance (Relative to $10,000.00 Initial Capital — `artifacts/corrected_holdout_evaluation.json`)

| Strategy | Total Return | Return Diff (pp) | CAGR | Max Drawdown | Sharpe ($r_f=0$) | Turnover | Defensive Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Selected Strategy (Threshold 0.50)** | **55.14%** | **Base** | **24.83%** | **-24.82%** | **0.93** | **2.15x** | **0.8%** |
| Growth Buy-and-Hold | 57.12% | +1.98 pp | 25.63% | -25.56% | 0.91 | 0.50x | 0.0% |
| Fixed Defensive (Weekly) | 36.81% | -18.33 pp | 17.15% | -15.79% | 1.01 | 1.35x | 100.0% |
| Momentum-63 (Weekly) | 42.63% | -12.50 pp | 19.64% | -21.96% | 0.90 | 8.84x | 28.0% |

#### Original Saved Artifact Baseline (`artifacts/final_test_report.json`)
- Selected Strategy: Total Return 55.29%, CAGR 24.89%, Max Drawdown -24.82%, Sharpe 0.93
- Growth Buy-and-Hold: Total Return 57.28%, CAGR 25.70%, Max Drawdown -25.56%, Sharpe 0.91
- Discrepancy explanation: ~15 bps difference arises from calculating returns from post-trade equity $9,990.01 vs initial deposited capital $10,000.00.

---

## 11. Course Rubric Evidence & Alignment Matrix

This matrix maps project deliverables directly to the **Stock Markets Analytics Zoomcamp** evaluation criteria:

| Rubric Area | Required Capability | Implementation Deliverable | Verifiable Evidence Paths |
| :--- | :--- | :--- | :--- |
| **Problem Description** | Clear investment context, downside definition, and trade delay | Plain-language retail motivation, exact 20-day downside loss event ($\le -5.0\%$), weekly signal and next-day close execution | `README.md` (§1), `GLOSSARY.md`, `ai_risk_radar/simulation.py:L40-85` |
| **Data Sources** | Clean historical ingestion, exchange alignment, adjustment handling | 4 liquid ETF universe (`SOXX`, `XLK`, `SPY`, `SHY`) aligned with NYSE (`XNYS`), split/dividend adjustments, snapshot integrity verification | `ai_risk_radar/data.py`, `tests/test_data.py`, `tests/test_blind_audit_remediation.py` |
| **Transformations & EDA** | Causal feature engineering, distributions, lookahead avoidance | 43 causal price-derived features (returns, realized volatilities, drawdowns, relative strength, correlations), non-overlapping event distribution | `ai_risk_radar/features.py`, `notebooks/research.ipynb`, `tests/test_features.py` |
| **Modeling & Calibration** | Temporal cross-validation, baseline comparisons, calibrated probabilities | Constant baseline, Decision Tree, Random Forest with 3 expanding purged folds, Platt scaling calibrator with support guards | `ai_risk_radar/models.py`, `tests/test_models.py` |
| **Trading Simulation** | Execution realism, transaction costs, baseline comparison, sensitivity | Delayed execution, daily drifted holding weights, exact transaction cost deduction (10 bps baseline, 5/25 bps sensitivity), comparison vs 3 benchmarks | `ai_risk_radar/simulation.py`, `ai_risk_radar/metrics.py`, `tests/test_simulation.py` |
| **Vectorized Reconciliation** | Distinct vectorized linear approximation and sequential reconciliation | Vectorized linear matrix approximation reconciled with sequential daily-drift engine under matched conventions (~30–50 bps divergence depending on market drift; exact in zero-drift tests) | `ai_risk_radar/simulation.py:L260-390`, `notebooks/end_to_end.ipynb`, `tests/test_blind_audit_remediation.py` |
| **Automation & Interface** | Interactive interface, scheduled automation, reproducible execution | Production-grade two-mode Streamlit dashboard (`app.py`), modular CLI commands (`latest`, `research`, `final-test`) suitable for standard OS schedulers (cron / Task Scheduler) with incremental overlapping cache ingestion (`fetch_prices`), and interactive end-to-end notebook (`notebooks/end_to_end.ipynb`) supporting clean first-use data acquisition and actionable trade schedule generation | `app.py`, `notebooks/end_to_end.ipynb`, `ai_risk_radar/data.py`, `tests/test_app.py`, `tests/test_views.py` |

---

## 12. Project Submission & Peer Review Instructions

1. **Submission Link:** [SMA Zoomcamp 2026 Project Submission Attempt 1](https://courses.datatalks.club/sma-zoomcamp-2026/project/project1) (Open until 2026-11-20 23:59:59 GMT).
2. **Repository Contents:** Ensure that no raw price datasets (`data/raw/`), model binaries (`artifacts/models/`), or temporary virtual environments (`.venv/`) are checked into git.
3. **Peer Reviews:** To receive course credit, students must evaluate 3 peer projects between the submission deadline and the evaluation cutoff date on the DatatalksClub platform.
