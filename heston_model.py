"""
Heston Stochastic Volatility Model

This module implements the Heston model for simulating asset prices with
stochastic volatility. Used for generating synthetic data to test the
Kalman filter estimators.
"""

import numpy as np
from typing import Tuple


class HestonModel:
    """
    Heston stochastic volatility model simulator.
    
    The Heston model describes asset price S and variance V dynamics:
        dS_t = μ S_t dt + √V_t S_t dW1_t
        dV_t = κ(θ - V_t) dt + σ √V_t dW2_t
    
    where dW1 and dW2 are correlated Brownian motions with correlation ρ.
    
    Parameters
    ----------
    S0 : float
        Initial asset price
    V0 : float
        Initial variance
    mu : float
        Drift of the asset price
    kappa : float
        Mean reversion speed of variance
    theta : float
        Long-term mean of variance
    sigma : float
        Volatility of variance (vol-of-vol)
    rho : float
        Correlation between asset and variance Brownian motions
    """
    
    def __init__(
        self,
        S0: float = 100.0,
        V0: float = 0.04,
        mu: float = 0.0,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        rho: float = -0.7
    ):
        self.S0 = S0
        self.V0 = V0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
    
    def simulate(
        self,
        T: float = 1.0,
        n_steps: int = 1000,
        seed: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate paths using the Euler-Maruyama scheme.
        
        Parameters
        ----------
        T : float
            Total time horizon
        n_steps : int
            Number of time steps
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        t : np.ndarray
            Time points
        S : np.ndarray
            Asset price path
        V : np.ndarray
            Variance path
        returns : np.ndarray
            Log returns
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        
        # Initialize arrays
        S = np.zeros(n_steps + 1)
        V = np.zeros(n_steps + 1)
        S[0] = self.S0
        V[0] = self.V0
        
        # Generate correlated Brownian motions
        for i in range(n_steps):
            # Generate correlated random variables
            Z1 = np.random.standard_normal()
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.standard_normal()
            
            # Ensure variance stays positive (full truncation scheme)
            V_current = max(V[i], 0)
            
            # Update variance
            dV = self.kappa * (self.theta - V_current) * dt + \
                 self.sigma * np.sqrt(V_current * dt) * Z2
            V[i + 1] = V[i] + dV
            
            # Update asset price
            dS = self.mu * S[i] * dt + np.sqrt(V_current * dt) * S[i] * Z1
            S[i + 1] = S[i] + dS
        
        # Calculate log returns
        returns = np.diff(np.log(S))
        
        return t, S, V, returns
    
    def get_realized_volatility(self, V: np.ndarray) -> np.ndarray:
        """
        Convert variance to volatility (standard deviation).
        
        Parameters
        ----------
        V : np.ndarray
            Variance path
        
        Returns
        -------
        np.ndarray
            Volatility (standard deviation) path
        """
        return np.sqrt(np.maximum(V, 0))


def generate_heston_data(
    n_steps: int = 1000,
    T: float = 1.0,
    seed: int = 42,
    **heston_params
) -> dict:
    """
    Convenience function to generate Heston model data.
    
    Parameters
    ----------
    n_steps : int
        Number of time steps
    T : float
        Total time horizon
    seed : int
        Random seed
    **heston_params
        Additional parameters for HestonModel
    
    Returns
    -------
    dict
        Dictionary with keys: 'time', 'prices', 'variance', 'volatility', 'returns'
    """
    model = HestonModel(**heston_params)
    t, S, V, returns = model.simulate(T=T, n_steps=n_steps, seed=seed)
    volatility = model.get_realized_volatility(V)
    
    return {
        'time': t,
        'prices': S,
        'variance': V,
        'volatility': volatility,
        'returns': returns
    }
