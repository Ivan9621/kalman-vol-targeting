# Kalman Filters for Volatility Targeting and Risk-Adjusted Returns

A Python library implementing Kalman filter-based volatility estimation for quantitative trading strategies, with a focus on **volatility targeting** to optimize risk-adjusted returns.

## Overview

This library demonstrates how real-time volatility estimation using Kalman filters can dramatically improve portfolio performance through dynamic position sizing. By scaling positions inversely to estimated volatility, we achieve higher Sharpe ratios and more consistent returns.

### Key Results from Monte Carlo Analysis (100 simulations 0.5 sigma, 0.04 theta, 1.5 kappa, 0.08 mu)

| Metric | Buy & Hold | Volatility Targeting | Improvement |
|--------|------------|---------------------|-------------|
| **Sharpe Ratio** | 0.72 | 1.00 | **+39%** |
| **Annualized Return** | 8.0% | 26.9% | **+235%** |

**Volatility Targeting results under different Volatility of Volatility & Mean Reversion speed**

![image alt](https://github.com/Ivan9621/kalman-vol-targeting/blob/main/vol_targeting_surface_analysis.png)


**Single Path Example**

![image alt](https://github.com/Ivan9621/kalman-vol-targeting/blob/main/volatility_targeting_single_path.png)



**Monte Carlo 100 Simulations**

![image alt](https://github.com/Ivan9621/kalman-vol-targeting/blob/main/volatility_targeting_monte_carlo.png)



## The Volatility Targeting Strategy

### Mathematical Framework

**Position Sizing:**
```
Position_t = σ_target / σ_t^estimated
```

**Scaled Returns:**
```
R_t^scaled = Position_t × R_t
           = (σ_target / σ_t^estimated) × R_t
```

**Why It Works:**

1. **High Volatility Periods**: Position size ↓ → Reduced exposure → Avoid large losses
2. **Low Volatility Periods**: Position size ↑ → Increased exposure → Capture more gains
3. **Result**: More consistent volatility + Asymmetric capture of returns = Higher Sharpe Ratio

### Empirical Evidence

From our Monte Carlo analysis with Heston stochastic volatility (μ=8%, σ_vol=0.5):

- **Average Sharpe Improvement**: +0.29 (39% relative improvement)
- **Volatility Targeting wins**: 68% of the time
- **Realized Volatility**: Much more stable (29.3% ± 2.6% vs 18.2% ± 7.0%)
- **Annualized Return**: 235% higher on average (26.9% vs 8.0%)

## Quick Start

### Basic Volatility Targeting Strategy

```python
import numpy as np
from kalman_estimators import create_std_estimator
from heston_model import HestonModel

# Generate market data
model = HestonModel(mu=0.08, sigma=0.5)  # 8% drift, high vol-of-vol
t, S, V, returns = model.simulate(T=2.0, n_steps=2000)
dt = t[1] - t[0]

# Initialize Kalman filter
estimator = create_std_estimator(
    initial_std=0.2 * np.sqrt(dt),
    observation_noise=0.2,
    process_noise=0.01
)

# Run volatility targeting
target_vol = 0.15
positions = []
scaled_returns = []

for i, ret in enumerate(returns):
    # Estimate current volatility
    vol_est = estimator.step(ret) / np.sqrt(dt)
    
    # Calculate position size
    position = target_vol / vol_est
    positions.append(position)
    
    # Apply to next return
    if i < len(returns) - 1:
        scaled_returns.append(position * returns[i + 1])

# Calculate Sharpe ratios
unscaled_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(1/dt)
scaled_sharpe = np.mean(scaled_returns) / np.std(scaled_returns) * np.sqrt(1/dt)

print(f"Buy & Hold Sharpe: {unscaled_sharpe:.3f}")
print(f"Vol Targeting Sharpe: {scaled_sharpe:.3f}")
print(f"Improvement: {scaled_sharpe - unscaled_sharpe:.3f}")
```

## Installation

```bash
pip install numpy matplotlib
```

Copy the library files to your project:
- `kalman_estimators.py` - Kalman filter implementations
- `heston_model.py` - Stochastic volatility model
- `example_volatility_targeting.py` - Complete demonstration

## Running the Examples

### Volatility Targeting Example (Recommended)

```bash
python example_volatility_targeting.py
```

This generates:
1. **Single path analysis**: Detailed breakdown showing volatility estimation, position sizing, cumulative returns, and rolling Sharpe ratios
2. **Monte Carlo analysis**: 100 simulations comparing buy-and-hold vs volatility targeting across various metrics

### Simple Volatility Estimation Example

```bash
python example_heston_volatility.py
```

Basic demonstration of Kalman filter for volatility estimation without portfolio application.

## Library Components

### 1. Kalman Filter Estimators (`kalman_estimators.py`)

**KalmanStdEstimator**: Estimates standard deviation from streaming data
- Uses log-space transformation for numerical stability
- Online learning: processes one observation at a time
- Configurable observation and process noise

**KalmanMeanEstimator**: Estimates mean from streaming data
- Direct state-space tracking
- Useful for trend estimation

### 2. Heston Model (`heston_model.py`)

Implements the Heston stochastic volatility model:
```
dS_t = μ S_t dt + √V_t S_t dW1_t
dV_t = κ(θ - V_t) dt + σ √V_t dW2_t
```

Perfect for testing volatility-based strategies because:
- Time-varying volatility (the key to vol targeting benefits)
- Volatility clustering (realistic market dynamics)
- Configurable volatility of volatility (σ parameter)

### 3. Volatility Targeting Example (`example_volatility_targeting.py`)

Complete implementation showing:
- Real-time volatility estimation
- Dynamic position sizing
- Performance comparison (buy-and-hold vs vol targeting)
- Monte Carlo validation
- Comprehensive visualizations

## Understanding the Results

### Single Path Analysis

The single path example shows four key panels:

1. **Volatility Estimation**: Kalman filter (orange) tracking true volatility (blue)
   - Demonstrates real-time estimation quality
   - Shows responsiveness to volatility changes

2. **Position Sizing**: Dynamic leverage based on inverse volatility
   - Position > 1: Increasing exposure in low-vol periods
   - Position < 1: Reducing exposure in high-vol periods
   - Capped at 5x for risk management

3. **Cumulative Returns**: Performance comparison
   - Volatility targeting (green) vs Buy & Hold (blue)
   - Typically shows 2-3x better final wealth

4. **Rolling Sharpe Ratio**: Time-varying risk-adjusted performance
   - Demonstrates consistency of vol targeting advantage
   - Shows resilience during volatile periods

### Monte Carlo Analysis

Four panels showing distributional properties across 100 simulations:

1. **Sharpe Ratio Distribution**: Vol targeting achieves higher Sharpe ~68% of the time
2. **Risk-Return Scatter**: Vol targeting dominates the efficient frontier
3. **Volatility Distribution**: Vol targeting achieves more consistent realized volatility
4. **Sharpe Improvement**: Distribution typically centered around +0.2 to +0.4

## Parameter Tuning for Volatility Targeting

### Observation Noise (`r`)
- **Lower (0.1-0.3)**: More responsive to recent data, noisier estimates
- **Higher (0.5-2.0)**: Smoother estimates, slower to adapt
- **Recommended for vol targeting**: 0.2

### Process Noise (`q`)
- **Lower (0.0001-0.001)**: Assumes volatility changes slowly
- **Higher (0.01-0.1)**: Allows rapid volatility adaptation
- **Recommended for vol targeting**: 0.01

### Target Volatility (`σ_target`)
- Determines overall risk exposure
- **Conservative**: 0.10 (10% annualized)
- **Moderate**: 0.15 (15% annualized)
- **Aggressive**: 0.20-0.25 (20-25% annualized)

### Position Bounds
- Always use position limits to control maximum leverage
- **Recommended**: Clip between 0.1x and 5.0x
- Prevents extreme leverage in low-volatility regimes

## Mathematical Intuition

### Why Volatility Targeting Improves Sharpe Ratios

**Sharpe Ratio Definition:**
```
SR = E[R] / σ[R]
```

**For Volatility Targeting:**
```
R_t^VT = (σ_target / σ_t) × R_t

E[R^VT] ≈ σ_target × E[R_t / σ_t]
σ[R^VT] ≈ σ_target  (more stable!)
```

**Key Insight**: 
- When volatility is high and expected returns are negative (downturns), we reduce position
- When volatility is low and expected returns are positive (calm periods), we increase position
- This creates **asymmetric exposure** that improves the return/risk ratio

**Formal Result** (under certain assumptions):
```
SR^VT > SR^BH  when volatility is time-varying and mean-reverting
```

### The Role of Volatility Clustering

Real markets exhibit **volatility clustering** (high volatility follows high volatility). This makes volatility forecastable, which is crucial for vol targeting:

1. **Kalman filter** provides good short-term volatility forecasts
2. **Position sizing** exploits these forecasts
3. **Result**: Better risk management during volatile periods

## Real-World Considerations

### Transaction Costs
- Frequent rebalancing incurs costs
- Use position change thresholds to reduce turnover
- Example: Only rebalance if position changes by >10%

### Estimation Risk
- Kalman filter estimates are uncertain, especially initially
- Use conservative bounds on leverage
- Consider ensemble methods (multiple estimators)

### Regime Changes
- Extreme market conditions may require parameter adaptation
- Consider adaptive Kalman filters
- Monitor estimation error and adjust if needed

### Slippage and Market Impact
- Large position changes may have market impact
- Implement gradual rebalancing
- Consider position limits based on liquidity

## Performance Metrics Explained

### Sharpe Ratio
- **Definition**: Excess return per unit of volatility
- **Interpretation**: Higher is better; >1.0 is good, >2.0 is excellent
- **Why it matters**: Industry standard for risk-adjusted performance

### Sortino Ratio
- **Definition**: Excess return per unit of downside volatility
- **Interpretation**: Like Sharpe but only penalizes downside risk
- **Typical result**: Vol targeting shows even stronger Sortino improvement

### Maximum Drawdown
- **Definition**: Largest peak-to-trough decline
- **Interpretation**: Worst-case loss experienced
- **Vol targeting impact**: Sometimes higher due to leverage, but compensated by higher returns

## Extensions and Advanced Topics

### Multi-Asset Portfolios
- Apply vol targeting to portfolio-level volatility
- Estimate correlation matrix using Kalman filters
- Dynamic risk parity strategies

### Adaptive Target Volatility
- Adjust target based on market regime
- Lower target in crisis periods
- Higher target in stable periods

### Transaction Cost Optimization
- Minimize rebalancing frequency
- Use portfolio optimization with transaction costs
- Implement buffer bands for position changes

### Ensemble Methods
- Combine multiple volatility estimators
- Weighted average based on recent performance
- Increases robustness to estimation error

## Comparison to Other Approaches

### vs GARCH Models
- **Kalman Filter**: Online, fast, simple
- **GARCH**: Batch processing, more parameters
- **Winner**: Kalman for real-time applications

### vs Historical/Rolling Volatility
- **Kalman Filter**: Adaptive, optimal weighting
- **Rolling Vol**: Equal weighting, fixed window
- **Winner**: Kalman for time-varying volatility

### vs Implied Volatility
- **Kalman Filter**: Realized vol forecast, no options needed
- **Implied Vol**: Market expectations, requires options data
- **Winner**: Both have merit; can be combined

## Academic References

1. **Kalman Filtering**: Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems"

2. **Heston Model**: Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"

3. **Volatility Targeting**: Moreira, A., & Muir, T. (2017). "Volatility‐Managed Portfolios", *Journal of Finance*

4. **Risk Parity**: Asness, C., Frazzini, A., & Pedersen, L. H. (2012). "Leverage Aversion and Risk Parity", *Financial Analysts Journal*

## Citation

If you use this library in your research or trading:

```bibtex
@software{kalman_vol_targeting,
  title={Kalman Filters for Volatility Targeting and Risk-Adjusted Returns},
  author={Ivan Orsolic},
  year={2026},
  url={https://github.com/Ivan9621/kalman-vol-targeting}
}
```

## License

This library is provided for educational and research purposes.

## Disclaimer

**This is for educational purposes only.** Past performance does not guarantee future results. The examples use synthetic data from stochastic models. Real market conditions may differ significantly. Always conduct thorough backtesting and risk assessment before deploying any trading strategy.

---

## Summary

This library demonstrates that **Kalman filter-based volatility targeting can significantly improve risk-adjusted returns**:

✅ **+39% average Sharpe ratio improvement** across 100 Monte Carlo simulations  
✅ **More consistent volatility** (lower variance of realized volatility)  
✅ **Real-time implementation** suitable for live trading  
✅ **Simple and interpretable** methodology  

The key insight: Time-varying volatility creates opportunities for dynamic risk management. By scaling positions inversely to estimated volatility, we achieve superior risk-adjusted returns while maintaining more consistent volatility exposure.

**Happy Trading! 📈**
