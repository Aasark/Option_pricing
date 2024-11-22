"""
Option Pricing Framework
This module implements classes for option pricing and Greek calculation using Monte Carlo simulations.

Classes:
- Asset: Base class for assets.
- BlackScholesAsset: Implements Black-Scholes model for asset price simulation.
- Option: Base class for options, including Monte Carlo delta calculation.
- AsianOption: Specializes in Asian options with arithmetic averaging.
"""

import numpy as np
import math
from statistics import NormalDist


class Asset:
    """Base class representing an asset."""
    def __init__(self, name, current_price, *, dividend_yield=0):
        if not isinstance(name, str) or not name:
            raise ValueError("Name must be a non-empty string.")
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            raise ValueError("Current price must be a positive number.")
        if not isinstance(dividend_yield, (int, float)) or dividend_yield < 0:
            raise ValueError("Dividend yield must be a non-negative number.")

        self.name = name
        self.current_price = current_price
        self.dividend_yield = dividend_yield


class BlackScholesAsset(Asset):
    """Implements Black-Scholes model for asset price simulation."""
    def __init__(self, name, current_price, *, dividend_yield=0, volatility):
        super().__init__(name, current_price, dividend_yield=dividend_yield)
        if not isinstance(volatility, (int, float)) or volatility <= 0:
            raise ValueError("Volatility must be a positive number.")
        self.volatility = volatility

    def simulate_path(self, simulated_times, interest_rate, *, current_price=None):
        if current_price is None:
            current_price = self.current_price

        dt = np.diff([0] + simulated_times)
        prices = [current_price]

        for i, t in enumerate(simulated_times):
            drift = (interest_rate - self.dividend_yield - 0.5 * self.volatility**2) * dt[i]
            diffusion = self.volatility * np.random.normal(0, math.sqrt(dt[i]))
            new_price = prices[-1] * math.exp(drift + diffusion)
            prices.append(new_price)

        return {t: price for t, price in zip([0] + simulated_times, prices)}

    def _dst_ds0(self, path, time):
        initial_price = path[0]
        return path[time] / initial_price


class Option:
    """Base class for options."""
    def __init__(self, name, underlying):
        if not isinstance(name, str) or not name:
            raise ValueError("Name must be a non-empty string.")
        if not isinstance(underlying, Asset):
            raise ValueError("Underlying must be an Asset instance.")

        self.name = name
        self.underlying = underlying

    def monte_carlo_delta(self, simulations, *, confidence_level=0.95, interest_rate):
        monte_deltas = []

        for _ in range(simulations):
            path = self.underlying.simulate_path(self.monitoring_times, interest_rate)
            delta = self._path_delta(path, interest_rate)
            monte_deltas.append(delta)

        mean_delta = np.mean(monte_deltas)
        std_delta = np.std(monte_deltas, ddof=1)
        z = NormalDist().inv_cdf((1 + confidence_level) / 2)
        margin = z * std_delta / math.sqrt(simulations)

        return mean_delta - margin, mean_delta, mean_delta + margin


class AsianOption(Option):
    """Specialized class for Asian options."""
    def __init__(self, name, underlying, *, option_type, monitoring_times, strike_factor):
        super().__init__(name, underlying)
        if option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'.")
        self.option_type = option_type
        self.monitoring_times = monitoring_times
        self.strike_factor = strike_factor

    def payoff(self, path):
        avg_price = np.mean([path[t] for t in self.monitoring_times])
        final_price = path[self.monitoring_times[-1]]
        if self.option_type == 'call':
            return max(final_price - self.strike_factor * avg_price, 0)
        elif self.option_type == 'put':
            return max(self.strike_factor * avg_price - final_price, 0)

    def _path_delta(self, path, interest_rate):
        deltas = [self.underlying._dst_ds0(path, t) for t in self.monitoring_times]
        payoff = self.payoff(path)
        discounted_payoff = payoff * math.exp(-interest_rate * self.monitoring_times[-1])
        return deltas[-1] * discounted_payoff


if __name__ == "__main__":
    # Example usage
    asset = BlackScholesAsset(name="Stock", current_price=100, dividend_yield=0.02, volatility=0.3)
    option = AsianOption(
        name="AsianCall",
        underlying=asset,
        option_type="call",
        monitoring_times=[1, 2, 3],
        strike_factor=1.1
    )

    try:
        delta_bounds = option.monte_carlo_delta(1000, confidence_level=0.95, interest_rate=0.05)
        print(f"Delta Confidence Interval: {delta_bounds}")
    except ValueError as e:
        print(f"Error: {e}")
