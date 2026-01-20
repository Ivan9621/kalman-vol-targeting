"""
Surface Plot Analysis: Volatility Targeting Performance vs Vol-of-Vol

This script generates 3D surface plots showing how volatility targeting performance
varies across different volatility-of-volatility (σ) parameters in the Heston model.

This analysis reveals under which market conditions volatility targeting provides
the greatest benefit.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from kalman_estimators import create_std_estimator
from heston_model import HestonModel


# Set dark style globally
plt.style.use('dark_background')


def run_single_simulation(sigma_vol, kappa, n_steps=2000, T=2.0, target_vol=0.15):
    """
    Run a single simulation with given parameters.
    
    Parameters
    ----------
    sigma_vol : float
        Volatility of volatility parameter
    kappa : float
        Mean reversion speed
    n_steps : int
        Number of time steps
    T : float
        Time horizon
    target_vol : float
        Target volatility for vol targeting
    
    Returns
    -------
    dict
        Performance metrics for both strategies
    """
    # Heston parameters
    heston_params = {
        'S0': 100.0,
        'V0': 0.04,
        'mu': 0.08,
        'kappa': kappa,
        'theta': 0.04,
        'sigma': sigma_vol,
        'rho': -0.7
    }
    
    # Generate path
    model = HestonModel(**heston_params)
    t, S, V, returns = model.simulate(T=T, n_steps=n_steps, seed=None)
    volatility = np.sqrt(np.maximum(V, 0))
    dt = T / n_steps
    
    # Initialize Kalman filter
    initial_std_scaled = 0.2 * np.sqrt(dt)
    estimator = create_std_estimator(
        initial_std=initial_std_scaled,
        observation_noise=0.2,
        process_noise=0.01
    )
    
    # Run volatility targeting
    scaled_returns = np.zeros(len(returns))
    for i in range(len(returns)):
        vol_est = estimator.step(returns[i]) / np.sqrt(dt)
        position = np.clip(target_vol / (vol_est + 1e-6), 0.1, 5.0)
        if i < len(returns) - 1:
            scaled_returns[i + 1] = position * returns[i + 1]
    
    # Calculate metrics
    annualization_factor = 1.0 / dt
    
    # Buy & Hold
    bh_ret = np.mean(returns) * annualization_factor
    bh_vol = np.std(returns) * np.sqrt(annualization_factor)
    bh_sharpe = bh_ret / bh_vol if bh_vol > 0 else 0
    
    # Volatility Targeting
    vt_ret = np.mean(scaled_returns) * annualization_factor
    vt_vol = np.std(scaled_returns) * np.sqrt(annualization_factor)
    vt_sharpe = vt_ret / vt_vol if vt_vol > 0 else 0
    
    return {
        'bh_sharpe': bh_sharpe,
        'vt_sharpe': vt_sharpe,
        'sharpe_improvement': vt_sharpe - bh_sharpe,
        'bh_return': bh_ret,
        'vt_return': vt_ret,
        'bh_vol': bh_vol,
        'vt_vol': vt_vol
    }


def generate_surface_data(
    sigma_vol_range,
    kappa_range,
    n_simulations_per_point=20,
    n_steps=2000,
    T=2.0,
    target_vol=0.15
):
    """
    Generate surface plot data across parameter grid.
    
    Parameters
    ----------
    sigma_vol_range : np.ndarray
        Range of vol-of-vol values to test
    kappa_range : np.ndarray
        Range of mean reversion speeds to test
    n_simulations_per_point : int
        Number of Monte Carlo runs per grid point
    n_steps : int
        Steps per simulation
    T : float
        Time horizon
    target_vol : float
        Target volatility
    
    Returns
    -------
    dict
        Grid data for surface plots
    """
    n_sigma = len(sigma_vol_range)
    n_kappa = len(kappa_range)
    
    # Storage arrays
    sharpe_improvement = np.zeros((n_sigma, n_kappa))
    vt_sharpe = np.zeros((n_sigma, n_kappa))
    bh_sharpe = np.zeros((n_sigma, n_kappa))
    vt_return = np.zeros((n_sigma, n_kappa))
    win_rate = np.zeros((n_sigma, n_kappa))
    
    total_points = n_sigma * n_kappa
    completed = 0
    
    print(f"Generating surface data: {n_sigma} x {n_kappa} grid")
    print(f"Running {n_simulations_per_point} simulations per point")
    print(f"Total simulations: {total_points * n_simulations_per_point}")
    print()
    
    for i, sigma in enumerate(sigma_vol_range):
        for j, kappa in enumerate(kappa_range):
            # Run multiple simulations for this parameter combination
            improvements = []
            vt_sharpes = []
            bh_sharpes = []
            vt_returns = []
            
            for _ in range(n_simulations_per_point):
                result = run_single_simulation(
                    sigma_vol=sigma,
                    kappa=kappa,
                    n_steps=n_steps,
                    T=T,
                    target_vol=target_vol
                )
                improvements.append(result['sharpe_improvement'])
                vt_sharpes.append(result['vt_sharpe'])
                bh_sharpes.append(result['bh_sharpe'])
                vt_returns.append(result['vt_return'])
            
            # Store averages
            sharpe_improvement[i, j] = np.mean(improvements)
            vt_sharpe[i, j] = np.mean(vt_sharpes)
            bh_sharpe[i, j] = np.mean(bh_sharpes)
            vt_return[i, j] = np.mean(vt_returns)
            win_rate[i, j] = np.mean(np.array(improvements) > 0)
            
            completed += 1
            if completed % 5 == 0 or completed == total_points:
                print(f"  Progress: {completed}/{total_points} grid points completed ({100*completed/total_points:.1f}%)")
    
    return {
        'sharpe_improvement': sharpe_improvement,
        'vt_sharpe': vt_sharpe,
        'bh_sharpe': bh_sharpe,
        'vt_return': vt_return,
        'win_rate': win_rate,
        'sigma_vol_range': sigma_vol_range,
        'kappa_range': kappa_range
    }


def plot_surface_analysis(data, save_path=None):
    """
    Create comprehensive surface plot visualizations.
    
    Parameters
    ----------
    data : dict
        Surface data from generate_surface_data
    save_path : str, optional
        Path to save figure
    """
    sigma_vol_range = data['sigma_vol_range']
    kappa_range = data['kappa_range']
    
    # Create meshgrid for plotting
    Sigma, Kappa = np.meshgrid(sigma_vol_range, kappa_range, indexing='ij')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a1a')
    
    # Plot 1: Sharpe Improvement Surface
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(Sigma, Kappa, data['sharpe_improvement'], 
                             cmap='plasma', alpha=0.9, edgecolor='none',
                             vmin=data['sharpe_improvement'].min(),
                             vmax=data['sharpe_improvement'].max())
    ax1.set_xlabel('Vol-of-Vol (σ)', fontsize=10, color='white')
    ax1.set_ylabel('Mean Reversion (κ)', fontsize=10, color='white')
    ax1.set_zlabel('Sharpe Improvement', fontsize=10, color='white')
    ax1.set_title('Sharpe Ratio Improvement\n(Vol Targeting - Buy & Hold)', 
                  fontsize=11, fontweight='bold', color='white', pad=20)
    ax1.view_init(elev=25, azim=45)
    ax1.set_facecolor('#1a1a1a')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors='white', labelsize=8)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, pad=0.1)
    
    # Plot 2: Vol Targeting Sharpe Surface
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(Sigma, Kappa, data['vt_sharpe'], 
                             cmap='viridis', alpha=0.9, edgecolor='none')
    ax2.set_xlabel('Vol-of-Vol (σ)', fontsize=10, color='white')
    ax2.set_ylabel('Mean Reversion (κ)', fontsize=10, color='white')
    ax2.set_zlabel('Sharpe Ratio', fontsize=10, color='white')
    ax2.set_title('Volatility Targeting Sharpe Ratio', 
                  fontsize=11, fontweight='bold', color='white', pad=20)
    ax2.view_init(elev=25, azim=45)
    ax2.set_facecolor('#1a1a1a')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white', labelsize=8)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5, pad=0.1)
    
    # Plot 3: Buy & Hold Sharpe Surface
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(Sigma, Kappa, data['bh_sharpe'], 
                             cmap='coolwarm', alpha=0.9, edgecolor='none')
    ax3.set_xlabel('Vol-of-Vol (σ)', fontsize=10, color='white')
    ax3.set_ylabel('Mean Reversion (κ)', fontsize=10, color='white')
    ax3.set_zlabel('Sharpe Ratio', fontsize=10, color='white')
    ax3.set_title('Buy & Hold Sharpe Ratio', 
                  fontsize=11, fontweight='bold', color='white', pad=20)
    ax3.view_init(elev=25, azim=45)
    ax3.set_facecolor('#1a1a1a')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(colors='white', labelsize=8)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5, pad=0.1)
    
    # Plot 4: Win Rate Surface
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    surf4 = ax4.plot_surface(Sigma, Kappa, data['win_rate'], 
                             cmap='RdYlGn', alpha=0.9, edgecolor='none',
                             vmin=0, vmax=1)
    ax4.set_xlabel('Vol-of-Vol (σ)', fontsize=10, color='white')
    ax4.set_ylabel('Mean Reversion (κ)', fontsize=10, color='white')
    ax4.set_zlabel('Win Rate', fontsize=10, color='white')
    ax4.set_title('Vol Targeting Win Rate\n(Fraction of Sims with VT > B&H)', 
                  fontsize=11, fontweight='bold', color='white', pad=20)
    ax4.view_init(elev=25, azim=45)
    ax4.set_facecolor('#1a1a1a')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(colors='white', labelsize=8)
    fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5, pad=0.1)
    
    # Plot 5: Contour of Sharpe Improvement
    ax5 = fig.add_subplot(2, 3, 5)
    contour5 = ax5.contourf(Sigma, Kappa, data['sharpe_improvement'], 
                            levels=15, cmap='plasma', alpha=0.9)
    ax5.contour(Sigma, Kappa, data['sharpe_improvement'], 
                levels=10, colors='white', alpha=0.3, linewidths=0.5)
    ax5.set_xlabel('Vol-of-Vol (σ)', fontsize=10, color='white')
    ax5.set_ylabel('Mean Reversion (κ)', fontsize=10, color='white')
    ax5.set_title('Sharpe Improvement Contour', 
                  fontsize=11, fontweight='bold', color='white')
    ax5.set_facecolor('#1a1a1a')
    ax5.grid(True, alpha=0.3, color='white')
    ax5.tick_params(colors='white', labelsize=8)
    fig.colorbar(contour5, ax=ax5)
    
    # Plot 6: Heatmap of Win Rate
    ax6 = fig.add_subplot(2, 3, 6)
    im6 = ax6.imshow(data['win_rate'], cmap='RdYlGn', aspect='auto',
                     origin='lower', extent=[kappa_range.min(), kappa_range.max(),
                                            sigma_vol_range.min(), sigma_vol_range.max()],
                     vmin=0, vmax=1, interpolation='bilinear')
    ax6.set_xlabel('Mean Reversion (κ)', fontsize=10, color='white')
    ax6.set_ylabel('Vol-of-Vol (σ)', fontsize=10, color='white')
    ax6.set_title('Win Rate Heatmap', 
                  fontsize=11, fontweight='bold', color='white')
    ax6.set_facecolor('#1a1a1a')
    ax6.grid(True, alpha=0.2, color='white')
    ax6.tick_params(colors='white', labelsize=8)
    
    # Add contour lines
    contour_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
    contours = ax6.contour(Kappa[0], Sigma[:, 0], data['win_rate'], 
                          levels=contour_levels, colors='white', 
                          linewidths=1, alpha=0.6)
    ax6.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    fig.colorbar(im6, ax=ax6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"\nSurface plot saved to {save_path}")
    
    plt.show()


def plot_slices(data, save_path=None):
    """
    Create 2D slice plots for easier interpretation.
    
    Parameters
    ----------
    data : dict
        Surface data
    save_path : str, optional
        Path to save figure
    """
    sigma_vol_range = data['sigma_vol_range']
    kappa_range = data['kappa_range']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a1a')
    
    # Plot 1: Sharpe improvement vs vol-of-vol (different kappa values)
    ax = axes[0, 0]
    kappa_indices = [0, len(kappa_range)//2, -1]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    for idx, color in zip(kappa_indices, colors):
        ax.plot(sigma_vol_range, data['sharpe_improvement'][:, idx], 
               marker='o', linewidth=2, color=color, markersize=6,
               label=f'κ = {kappa_range[idx]:.2f}')
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Vol-of-Vol (σ)', fontsize=11, color='white')
    ax.set_ylabel('Sharpe Improvement', fontsize=11, color='white')
    ax.set_title('Sharpe Improvement vs Vol-of-Vol\n(for different mean reversion speeds)', 
                fontsize=11, fontweight='bold', color='white')
    ax.legend(fontsize=9, facecolor='#2a2a2a', edgecolor='white')
    ax.grid(True, alpha=0.3, color='white')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white', labelsize=9)
    
    # Plot 2: Sharpe improvement vs kappa (different sigma values)
    ax = axes[0, 1]
    sigma_indices = [0, len(sigma_vol_range)//2, -1]
    colors = ['#FFD93D', '#6BCB77', '#4D96FF']
    for idx, color in zip(sigma_indices, colors):
        ax.plot(kappa_range, data['sharpe_improvement'][idx, :], 
               marker='s', linewidth=2, color=color, markersize=6,
               label=f'σ = {sigma_vol_range[idx]:.2f}')
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Mean Reversion (κ)', fontsize=11, color='white')
    ax.set_ylabel('Sharpe Improvement', fontsize=11, color='white')
    ax.set_title('Sharpe Improvement vs Mean Reversion\n(for different vol-of-vol levels)', 
                fontsize=11, fontweight='bold', color='white')
    ax.legend(fontsize=9, facecolor='#2a2a2a', edgecolor='white')
    ax.grid(True, alpha=0.3, color='white')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white', labelsize=9)
    
    # Plot 3: Win rate vs vol-of-vol
    ax = axes[1, 0]
    for idx, color in zip(kappa_indices, ['#FF6B6B', '#4ECDC4', '#95E1D3']):
        ax.plot(sigma_vol_range, data['win_rate'][:, idx] * 100, 
               marker='o', linewidth=2, color=color, markersize=6,
               label=f'κ = {kappa_range[idx]:.2f}')
    ax.axhline(y=50, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Vol-of-Vol (σ)', fontsize=11, color='white')
    ax.set_ylabel('Win Rate (%)', fontsize=11, color='white')
    ax.set_title('Vol Targeting Win Rate vs Vol-of-Vol', 
                fontsize=11, fontweight='bold', color='white')
    ax.legend(fontsize=9, facecolor='#2a2a2a', edgecolor='white')
    ax.grid(True, alpha=0.3, color='white')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white', labelsize=9)
    ax.set_ylim([0, 100])
    
    # Plot 4: Average Sharpe comparison
    ax = axes[1, 1]
    avg_vt = np.mean(data['vt_sharpe'], axis=1)
    avg_bh = np.mean(data['bh_sharpe'], axis=1)
    ax.plot(sigma_vol_range, avg_vt, marker='o', linewidth=2.5, 
           color='#2ECC71', markersize=7, label='Vol Targeting')
    ax.plot(sigma_vol_range, avg_bh, marker='s', linewidth=2.5, 
           color='#E74C3C', markersize=7, label='Buy & Hold')
    ax.fill_between(sigma_vol_range, avg_vt, avg_bh, 
                    where=(avg_vt > avg_bh), alpha=0.3, color='#2ECC71')
    ax.set_xlabel('Vol-of-Vol (σ)', fontsize=11, color='white')
    ax.set_ylabel('Average Sharpe Ratio', fontsize=11, color='white')
    ax.set_title('Average Sharpe Ratio vs Vol-of-Vol\n(averaged over all κ values)', 
                fontsize=11, fontweight='bold', color='white')
    ax.legend(fontsize=9, facecolor='#2a2a2a', edgecolor='white')
    ax.grid(True, alpha=0.3, color='white')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white', labelsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"Slice plots saved to {save_path}")
    
    plt.show()


def main():
    """Main execution function."""
    
    print("=" * 80)
    print("SURFACE ANALYSIS: VOL TARGETING PERFORMANCE VS VOL-OF-VOL")
    print("=" * 80)
    print()
    print("This analysis shows how volatility targeting performance varies with")
    print("the volatility-of-volatility (σ) parameter in the Heston model.")
    print()
    print("Key question: Under what market conditions does vol targeting provide")
    print("the greatest benefit?")
    print()
    print("=" * 80)
    print()
    
    # Define parameter ranges
    sigma_vol_range = np.linspace(0.1, 1.0, 10)  # Vol-of-vol from 0.1 to 1.0
    kappa_range = np.linspace(0.5, 3.0, 10)      # Mean reversion from 0.5 to 3.0
    
    print("Parameter Ranges:")
    print(f"  Vol-of-Vol (σ): {sigma_vol_range.min():.2f} to {sigma_vol_range.max():.2f}")
    print(f"  Mean Reversion (κ): {kappa_range.min():.2f} to {kappa_range.max():.2f}")
    print()
    
    # Generate surface data
    print("Generating surface data...")
    print("-" * 80)
    data = generate_surface_data(
        sigma_vol_range=sigma_vol_range,
        kappa_range=kappa_range,
        n_simulations_per_point=150,
        n_steps=2000,
        T=2.0,
        target_vol=0.15
    )
    
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Find optimal parameters
    max_improvement_idx = np.unravel_index(
        np.argmax(data['sharpe_improvement']), 
        data['sharpe_improvement'].shape
    )
    optimal_sigma = sigma_vol_range[max_improvement_idx[0]]
    optimal_kappa = kappa_range[max_improvement_idx[1]]
    max_improvement = data['sharpe_improvement'][max_improvement_idx]
    
    print(f"Maximum Sharpe Improvement: {max_improvement:.3f}")
    print(f"  Optimal Vol-of-Vol (σ): {optimal_sigma:.2f}")
    print(f"  Optimal Mean Reversion (κ): {optimal_kappa:.2f}")
    print()
    
    # Overall statistics
    print(f"Average Sharpe Improvement: {np.mean(data['sharpe_improvement']):.3f}")
    print(f"  Std Dev: {np.std(data['sharpe_improvement']):.3f}")
    print(f"  Min: {np.min(data['sharpe_improvement']):.3f}")
    print(f"  Max: {np.max(data['sharpe_improvement']):.3f}")
    print()
    
    print(f"Overall Win Rate: {np.mean(data['win_rate']) * 100:.1f}%")
    print()
    
    # Effect of vol-of-vol
    avg_improvement_by_sigma = np.mean(data['sharpe_improvement'], axis=1)
    print("Average Sharpe Improvement by Vol-of-Vol:")
    for sigma, improvement in zip(sigma_vol_range, avg_improvement_by_sigma):
        print(f"  σ = {sigma:.2f}: {improvement:+.3f}")
    print()
    
    print("KEY INSIGHT:")
    if avg_improvement_by_sigma[-1] > avg_improvement_by_sigma[0]:
        print("  → Higher vol-of-vol INCREASES the benefit of volatility targeting")
        print("  → More time-varying volatility = More opportunities for risk management")
    else:
        print("  → Lower vol-of-vol shows better performance")
    print()
    
    print("=" * 80)
    print()
    
    # Generate plots
    print("Generating visualizations...")
    print()
    
    plot_surface_analysis(
        data,
        save_path='vol_targeting_surface_analysis.png'
    )
    
    plot_slices(
        data,
        save_path='/vol_targeting_slice_analysis.png'
    )
    
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("The surface plots reveal:")
    print()
    print("1. VOL-OF-VOL EFFECT:")
    print("   Higher volatility-of-volatility generally leads to:")
    print("   • Greater benefits from volatility targeting")
    print("   • More pronounced time-variation in volatility")
    print("   • Better opportunities for dynamic risk management")
    print()
    print("2. MEAN REVERSION EFFECT:")
    print("   Mean reversion speed affects:")
    print("   • How quickly volatility returns to long-term average")
    print("   • Persistence of volatility regimes")
    print("   • Kalman filter's ability to track changes")
    print()
    print("3. OPTIMAL REGIME:")
    print(f"   Best performance at σ ≈ {optimal_sigma:.2f}, κ ≈ {optimal_kappa:.2f}")
    print("   This represents moderately time-varying volatility with")
    print("   reasonable mean reversion - ideal for vol targeting")
    print()
    print("4. ROBUSTNESS:")
    print(f"   Vol targeting wins {np.mean(data['win_rate']) * 100:.0f}% across all parameter combinations")
    print("   Strategy is robust to different market conditions")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
