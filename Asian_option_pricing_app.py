# Installing necessary libraries
import streamlit as st
import numpy as np
import math
from statistics import NormalDist
import matplotlib.pyplot as plt

# Parent Class
class Asset:
    """
    Represents a financial asset (we assume it is a stock).

    Attributes:
        name (str): The name of the asset.
        current_price (float): The current market price of the asset.
        dividend_yield (float): The annual dividend yield of the asset as a decimal (default is 0).
    """

    def __init__(self, name, current_price, *, dividend_yield=0):
        """
        Initializes an Asset instance.

        Args:
            name (str): The name of the asset. Must be a non-empty string.
            current_price (float): The current market price of the asset. Must be a positive number.
            dividend_yield (float, optional): The dividend yield as a decimal (e.g., 0.02 for 2%). Must be non-negative. Defaults to 0.

        Raises:
            ValueError: If the `name` is not a non-empty string.
            ValueError: If `current_price` is not a positive number.
            ValueError: If `dividend_yield` is negative.
        """
        # name of the asset must be non-empty string
        if not isinstance(name, str) or not name:
            raise ValueError("Name must be a non-empty string.")
        
        # current price must be positive
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            raise ValueError("Current price must be a positive number.")
        
        # dividend yield must be non-negative
        if not isinstance(dividend_yield, (int, float)) or dividend_yield < 0:
            raise ValueError("Dividend yield must be a non-negative number.")

        # Set the instance attributes
        self.name = name  
        self.current_price = current_price  
        self.dividend_yield = dividend_yield  

class BlackScholesAsset(Asset):
    """
    Uses Black Scholes equation to price an asset.

    Inherits from the parent class Asset and adds functionality for simulating asset price paths
    based on the Black-Scholes model.

    Attributes:
        volatility (float): The volatility of the asset as a decimal.
    """

    def __init__(self, name, current_price, *, dividend_yield=0, volatility):
        """
        Initializes a BlackScholesAsset class instance.

        Args:
            name (str): The name of the asset. Must be a non-empty string.
            current_price (float): The current market price of the asset. Must be a positive number.
            dividend_yield (float, optional): The dividend yield as a decimal. Must be non-negative. Defaults to 0.
            volatility (float): The volatility of the asset. Must be a positive number.

        Raises:
            ValueError: If the `volatility` is not a positive number.
        """
        # Initializing the parent Asset class
        super().__init__(name, current_price, dividend_yield=dividend_yield)

        if not isinstance(volatility, (int, float)) or volatility <= 0:
            raise ValueError("Volatility must be a positive number.")
        
        # Set the volatility attribute
        self.volatility = volatility

    def simulate_path(self, simulated_times, interest_rate, *, current_price=None):
        """
        Simulates the price path of the asset over specified times using the Black-Scholes model.

        Args:
            simulated_times (list of float): A list of times (in years) at which to simulate prices.
            interest_rate (float): The risk-free interest rate as a decimal.
            current_price (float, optional): The starting price for the simulation. Defaults to the current_price of the asset.

        Returns:
            dict: A dictionary where keys are times and values are simulated prices.
        """
        # Use the asset's current price if no custom starting price is provided
        if current_price is None:
            current_price = self.current_price

        # Calculate time intervals (dt) between consecutive simulation times
        dt = np.diff([0] + simulated_times)

        # Initialize the list of simulated prices, starting with the initial price
        prices = [current_price]

        # Loop through each time step to simulate the price using the Black-Scholes formula
        for i, t in enumerate(simulated_times):
            # Calculate the drift term based on interest rate, dividend yield, and volatility
            drift = (interest_rate - self.dividend_yield - 0.5 * self.volatility**2) * dt[i]

            # Calculate the diffusion term using a random normal variable
            diffusion = self.volatility * np.random.normal(0, math.sqrt(dt[i]))

            # Calculate the new price
            new_price = prices[-1] * math.exp(drift + diffusion)

            # Append the new price to the list of simulated prices
            prices.append(new_price)

        # Return a dictionary of times and their corresponding simulated prices
        return {t: price for t, price in zip([0] + simulated_times, prices)}

    def _dst_ds0(self, path, time):
        """
        Calculates the sensitivity of the price at a specific time to the initial price.

        Args:
            path (dict): A dictionary of simulated prices where keys are times and values are prices.
            time (float): The specific time at which to calculate the sensitivity.

        Returns:
            float: The sensitivity (price at `time` / initial price).
        """
        # Retrieve the initial price from the path
        initial_price = path[0]

        # Calculate and return the sensitivity
        return path[time] / initial_price

class Option:
    """
    Represents a standard option contract that derives its value from an underlying asset.
    Subclasses are used to implement specific option types, such as European or Asian options.

    Attributes:
        name (str): The name of the option.("Call" or "Put").
        underlying (Asset): The asset that underlies the option (Will be an instance of another class).
    """

    def __init__(self, name, underlying):
        """
        Args:
            name (str): The name of the option. Must be a non-empty string.
            underlying (Asset): The underlying asset of the option. Must be an instance of the Asset class.

        Raises:
            ValueError: If `name` is not a non-empty string.
            ValueError: If `underlying` is not an instance of the Asset class.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Name must be a non-empty string.")

        if not isinstance(underlying, Asset):
            raise ValueError("Underlying must be an Asset instance.")

        self.name = name  
        self.underlying = underlying 

    def monte_carlo_delta(self, simulations, *, confidence_level=0.95, interest_rate):
        """
        Estimates the delta of the option using Monte Carlo simulation.
        This function generates simulated price paths for the underlying asset and calculates the delta for each path.

        Args:
            simulations (int): The number of Monte Carlo simulations to run.
            confidence_level (float, optional): The confidence level for the delta estimate (e.g., 0.95 for 95%). Defaults to 0.95.
            interest_rate (float): The risk-free interest rate as a decimal (e.g., 0.05 for 5% annual rate).

        Returns:
            tuple: A 3-tuple containing the lower bound, mean, and upper bound of the delta confidence interval.

        Raises:
            AttributeError: If `monitoring_times` or `_path_delta` is not implemented in a subclass.
        """
        # List to store calculated deltas from each simulation
        monte_deltas = []

        # Perform Monte Carlo simulations
        for _ in range(simulations):
            # Simulate a price path for the underlying asset
            path = self.underlying.simulate_path(self.monitoring_times, interest_rate)

            # Calculate the delta for the simulated path
            delta = self._path_delta(path, interest_rate)

            # Append the delta to the list
            monte_deltas.append(delta)

        # Calculate the mean and standard deviation of the deltas
        mean_delta = np.mean(monte_deltas)
        std_delta = np.std(monte_deltas, ddof=1)

        # Compute the confidence interval
        z = NormalDist().inv_cdf((1 + confidence_level) / 2)  # Z-score for the confidence level
        margin = z * std_delta / math.sqrt(simulations)  # Margin of error

        # Return the confidence interval as a tuple
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
