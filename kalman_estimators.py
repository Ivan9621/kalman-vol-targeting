"""
Kalman Filter Estimators for Latent Feature Estimation

This library provides Kalman filter-based estimators for streaming data analysis,
specifically designed to estimate latent features such as mean and standard deviation
from time series data.
"""

import numpy as np
from typing import Optional


class KalmanStdEstimator:
    """
    Kalman filter for estimating the standard deviation of streaming data.
    
    Uses a log-transform to work in transformed space for better numerical stability.
    The observation model transforms x -> log(x² + epsilon) to estimate volatility.
    
    Parameters
    ----------
    r : float
        Observation noise variance (measurement uncertainty)
    q : float
        Process noise variance (how much the std can change between steps)
    m0 : float
        Initial estimate of log-transformed standard deviation
    
    Attributes
    ----------
    m : float
        Current state estimate (in log-transformed space)
    s : float
        Current state variance
    pred : float
        Current prediction of standard deviation (in original space)
    last_error : float
        Most recent observation error (innovation)
    r_mult : float
        Multiplier for observation noise (default 1.0, can be adjusted dynamically)
    """
    
    def __init__(self, r: float, q: float, m0: float):
        self.r = r
        self.q = q
        self.m = m0
        self.s = 1.0
        self.pred = 0.0
        self.r_mult = 1.0
        self.last_error = 0.0
    
    def std_transform(self, x: float) -> float:
        """Transform observation to log space: log(x² + epsilon)"""
        return np.log(x**2 + 1e-10)
    
    def std_inverse_transform(self, x: float) -> float:
        """Transform from log space back to std: sqrt(exp(x))"""
        return np.sqrt(np.exp(x))
    
    def step(self, x: float) -> float:
        """
        Process one observation and update the estimate.
        
        Parameters
        ----------
        x : float
            New observation (e.g., return or residual)
        
        Returns
        -------
        float
            Updated standard deviation estimate
        """
        observation = self.std_transform(x)
        
        # Prediction step
        m_pred = self.m
        s_pred = self.s + self.q
        
        # Update step
        K = s_pred / (s_pred + (self.r * self.r_mult))
        error = observation - m_pred
        
        self.m = m_pred + K * error
        self.s = (1 - K) * s_pred
        self.pred = self.std_inverse_transform(self.m)
        
        self.r_mult = 1.0
        self.last_error = error
        
        return self.pred
    
    def get_state(self) -> dict:
        """Return the current state of the estimator"""
        return {
            'estimate': self.pred,
            'state_mean': self.m,
            'state_variance': self.s,
            'last_error': self.last_error
        }
    
    def reset(self, m0: Optional[float] = None):
        """Reset the estimator to initial conditions"""
        if m0 is not None:
            self.m = m0
        self.s = 1.0
        self.pred = 0.0
        self.r_mult = 1.0
        self.last_error = 0.0


class KalmanMeanEstimator:
    """
    Kalman filter for estimating the mean of streaming data.
    
    Directly tracks the expected value of observations over time,
    useful for trend estimation or de-meaning data.
    
    Parameters
    ----------
    r : float
        Observation noise variance (measurement uncertainty)
    q : float
        Process noise variance (how much the mean can change between steps)
    m0 : float
        Initial estimate of the mean
    
    Attributes
    ----------
    m : float
        Current state estimate (mean)
    s : float
        Current state variance
    pred : float
        Current prediction of mean (same as m)
    last_error : float
        Most recent observation error (innovation)
    r_mult : float
        Multiplier for observation noise (default 1.0, can be adjusted dynamically)
    """
    
    def __init__(self, r: float, q: float, m0: float):
        self.r = r
        self.q = q
        self.s = 1.0
        self.m = m0
        self.pred = 0.0
        self.r_mult = 1.0
        self.last_error = 0.0
    
    def step(self, x: float) -> float:
        """
        Process one observation and update the estimate.
        
        Parameters
        ----------
        x : float
            New observation
        
        Returns
        -------
        float
            Updated mean estimate
        """
        observation = x
        
        # Prediction step
        m_pred = self.m
        s_pred = self.s + self.q
        
        # Update step
        K = s_pred / (s_pred + (self.r * self.r_mult))
        error = observation - m_pred
        
        self.m = m_pred + K * error
        self.s = (1 - K) * s_pred
        self.pred = self.m
        
        self.r_mult = 1.0
        self.last_error = error
        
        return self.pred
    
    def get_state(self) -> dict:
        """Return the current state of the estimator"""
        return {
            'estimate': self.pred,
            'state_mean': self.m,
            'state_variance': self.s,
            'last_error': self.last_error
        }
    
    def reset(self, m0: Optional[float] = None):
        """Reset the estimator to initial conditions"""
        if m0 is not None:
            self.m = m0
        self.s = 1.0
        self.pred = 0.0
        self.r_mult = 1.0
        self.last_error = 0.0


def create_std_estimator(
    initial_std: float = 0.01,
    observation_noise: float = 1.0,
    process_noise: float = 0.0001
) -> KalmanStdEstimator:
    """
    Convenience function to create a KalmanStdEstimator with sensible defaults.
    
    Parameters
    ----------
    initial_std : float
        Initial guess for standard deviation
    observation_noise : float
        Measurement uncertainty (larger = more smoothing)
    process_noise : float
        How quickly std can change (larger = more responsive)
    
    Returns
    -------
    KalmanStdEstimator
        Configured estimator instance
    """
    m0 = np.log(initial_std**2 + 1e-10)
    return KalmanStdEstimator(r=observation_noise, q=process_noise, m0=m0)


def create_mean_estimator(
    initial_mean: float = 0.0,
    observation_noise: float = 1.0,
    process_noise: float = 0.0001
) -> KalmanMeanEstimator:
    """
    Convenience function to create a KalmanMeanEstimator with sensible defaults.
    
    Parameters
    ----------
    initial_mean : float
        Initial guess for mean
    observation_noise : float
        Measurement uncertainty (larger = more smoothing)
    process_noise : float
        How quickly mean can change (larger = more responsive)
    
    Returns
    -------
    KalmanMeanEstimator
        Configured estimator instance
    """
    return KalmanMeanEstimator(r=observation_noise, q=process_noise, m0=initial_mean)
