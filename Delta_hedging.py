#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:11:33 2023

@author: aureliedubost
"""
from Functions import *
import pandas as pd
import numpy as np

####### USER INPUTS #######
strike_price = 100
premium = 5
underlying_price_range = np.arange(80, 121, 1) # Define the underlying asset price range
underlying_price = 100
S = np.array([100])  # Spot price as an array
S = 100  # Spot price as an array

K = 105  # Strike price
t = 1  # Time to maturity (in years)

r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
option_type = 'call'  # Option type (choose 'call' or 'put')

# Delta hedging simulation inputs
n_simulations = 100  # Number of simulations
n_steps = 252  # Number of time steps (daily for one year)

####### Calculate the Black-Scholes Greeks
price, delta, gamma, theta, vega, rho = black_scholes(S, K, t, r, sigma, 'call')
print("Price:", price)
print("Delta:", delta)
print("Gamma:", gamma)
print("Theta:", theta)
print("Vega:", vega)
print("Rho:", rho)

# Calculate the Black-Scholes Greeks
black_scholes = black_scholes(S, K, t, r, sigma, option_type)
black_scholes_delta = black_scholes_delta(S, K, t, r, sigma, option_type)
black_scholes_gamma = black_scholes_gamma(S, K, t, r, sigma)
black_scholes_theta = black_scholes_theta(S, K, t, r, sigma)
black_scholes_vega = black_scholes_vega(S, K, t, r, sigma)
black_scholes_rho = black_scholes_rho(S, K, t, r, sigma)

######## Plot ########
#### Payoff ####
# Calculate the payoff for each option strategy
long_call_payoff = calculate_long_call_payoff(underlying_price, strike_price, premium)
long_put_payoff = calculate_long_put_payoff(underlying_price, strike_price, premium)
short_call_payoff = calculate_short_call_payoff(underlying_price, strike_price, premium)
short_put_payoff = calculate_short_put_payoff(underlying_price, strike_price, premium)

# Plot the option payoff diagrams
plt.figure(figsize=(8, 6))
plt.plot(underlying_price, long_call_payoff, label='Long Call')
plt.plot(underlying_price, long_put_payoff, label='Long Put')
plt.plot(underlying_price, short_call_payoff, label='Short Call')
plt.plot(underlying_price, short_put_payoff, label='Short Put')
plt.xlabel('Underlying Price')
plt.ylabel('Payoff')
plt.title('Option Payoff Diagrams')
plt.legend()
plt.grid(True)
plt.show()
plot_option_payoff(underlying_price, long_call_payoff, long_put_payoff, short_call_payoff, short_put_payoff)


#### Delta hedging ####
# Calculate the initial delta
delta = black_scholes_delta(S, K, t, r, sigma, option_type)
# Simulate the delta hedge strategy
n_simulations = 100  # Number of simulations
n_steps = 252  # Number of time steps (daily for one year)
# Create arrays to store P&L values for the hedged and non-hedged strategies
non_hedged_pnl = np.zeros(n_simulations)
hedged_pnl = np.zeros(n_simulations)
np.random.seed(0)
# Generate random returns
returns = np.random.normal(0, 1, (n_simulations, n_steps))
simulated_prices = S * np.exp((r - 0.5 * sigma**2) * (t / n_steps) + sigma * np.sqrt(t / n_steps) * returns.cumsum(axis=1))
simulated_deltas = black_scholes_delta(simulated_prices, K, t, r, sigma, option_type)

# Calculate the hedging quantities and portfolio values
hedging_quantities = initial_delta - simulated_deltas

# Check the shapes of the arrays
print('initial_delta shape:', initial_delta.shape)
print('simulated_prices shape:', simulated_prices.shape)
print('hedging_quantities shape:', hedging_quantities.shape)

# Calculate the portfolio values with proper shapes
portfolio_value_hedged = initial_delta * S[:, None] - hedging_quantities * simulated_prices
portfolio_value_nonhedged = initial_delta * S[:, None] - simulated_prices

# Calculate the P&L for both strategies
hedged_pnl = portfolio_value_hedged[:, -1] - portfolio_value_hedged[:, 0]
non_hedged_pnl = portfolio_value_nonhedged[:, -1] - portfolio_value_nonhedged[:, 0]

# Create a DataFrame to display the P&L values for both strategies
pnl_df = pd.DataFrame({'Simulation': range(1, n_simulations+1),
                       'Hedged P&L': hedged_pnl,
                       'Non-hedged P&L': non_hedged_pnl})

# Print the DataFrame
print(pnl_df)