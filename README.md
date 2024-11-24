# Asian Option Pricing App
This repository contains a Python application that calculates the price of Asian options using Monte Carlo simulations. 
The app is built using Streamlit (https://streamlit.io/) for an interactive web interface and uses Black-Scholes model to simulate underlying asset price paths.

## Features
Interactive Input Panel:

Users can input option parameters such as current stock price, dividend yield, volatility, strike factor, and risk-free rate.
A sidebar allows dynamic adjustments to parameters.

## Asian Option Pricing:
Supports floating strike Asian options (Call and Put).
Calculates the discounted payoff using Monte Carlo simulations.

## Visualizations:
Displays the simulated payoff vs monitoring time chart.
Highlights the profit/loss zones with green and red areas.

Installation and Usage
Prerequisites
Python 3.8 or higher
Required Python libraries:
streamlit
numpy
matplotlib

## To run it locally
1. Clone this github repo
2. Install all dependencies listed in requirements.txt
3. run the following command on your terminal
   streamlit run Asian_option_pricing_app.py
