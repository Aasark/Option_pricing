import streamlit as st
import numpy as np
import math
from statistics import NormalDist
import matplotlib.pyplot as plt

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


import streamlit as st
import numpy as np
import math
from statistics import NormalDist
import matplotlib.pyplot as plt

# Define Asset, BlackScholesAsset, Option, and AsianOption classes (no changes from your code)

st.title("Floating Price Asian Option Calculator")

st.sidebar.header("Option Parameters")
name = st.sidebar.text_input("Stock Name", "AZN")
current_price = st.sidebar.number_input("Current Stock Price (S₀)", value=100.0, step=1.0)
if current_price <= 0:
    st.error("Current Stock Price must be positive.")

initial_purchase_price = st.sidebar.number_input("Initial Purchase Price of Option", value=10.0, step=1.0)
if initial_purchase_price <= 0:
    st.error("Initial Purchase Price of Option must be positive.")

dividend_yield = st.sidebar.number_input("Dividend Yield", value=0.02, step=0.01)
volatility = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
option_type = st.sidebar.selectbox("Option Type", options=["call", "put"])
strike_factor = st.sidebar.number_input("Strike Factor", value=1.2, step=0.1)

monitoring_times_input = st.sidebar.text_input("Monitoring Times (comma-separated)", "1,3,7,10")
try:
    monitoring_times = list(map(float, monitoring_times_input.split(',')))
    if any(t <= 0 for t in monitoring_times):
        st.error("All monitoring times must be positive.")
except ValueError:
    monitoring_times = []
    st.error("Monitoring Times must be valid numbers separated by commas.")

r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
n_simulations = st.sidebar.number_input("Number of Simulations", value=10000, step=1000)

if st.sidebar.button("Calculate Confidence Interval"):
    try:
        asset = BlackScholesAsset(
            name=name,
            current_price=current_price,
            dividend_yield=dividend_yield,
            volatility=volatility,
        )

        option = AsianOption(
            name="AsianOption",
            underlying=asset,
            option_type=option_type,
            monitoring_times=monitoring_times,
            strike_factor=strike_factor,
        )

        confidence_interval = option.monte_carlo_delta(
            simulations=int(n_simulations),
            confidence_level=0.95,
            interest_rate=r,
        )
        st.write(f"Monte Carlo Delta Confidence Interval: {confidence_interval}")
    except Exception as e:
        st.error(f"Error during calculation: {e}")

if st.sidebar.button("Plot Payoff"):
    try:
        asset = BlackScholesAsset(
            name=name,
            current_price=current_price,
            dividend_yield=dividend_yield,
            volatility=volatility,
        )

        option = AsianOption(
            name="AsianOption",
            underlying=asset,
            option_type=option_type,
            monitoring_times=monitoring_times,
            strike_factor=strike_factor,
        )

        simulated_paths = [
            asset.simulate_path(monitoring_times, r) for _ in range(int(n_simulations))
        ]
        payoffs = [option.payoff(path) for path in simulated_paths]

        avg_prices = [
            np.mean([path[time] for path in simulated_paths]) for time in monitoring_times
        ]
        discounted_payoffs = [
            np.mean(payoffs) * math.exp(-r * time) for time in monitoring_times
        ]

        fig, ax = plt.subplots()
        ax.plot(monitoring_times, discounted_payoffs, label="Discounted Payoff")
        ax.axhline(y=initial_purchase_price, color='r', linestyle='--', label="Initial Purchase Price")
        ax.set_xlabel("Monitoring Time")
        ax.set_ylabel("Payoff")
        ax.set_title("Payoff vs. Monitoring Time")
        ax.legend()

        st.pyplot(fig)

        st.write(f"Initial Purchase Price: £{initial_purchase_price:.2f}")
        st.write(f"Simulated Average Discounted Payoff: £{np.mean(discounted_payoffs):.2f}")

    except Exception as e:
        st.error(f"Error during calculation: {e}")
