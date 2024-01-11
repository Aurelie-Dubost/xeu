#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 22:49:28 2023

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

plot_option_payoff(underlying_price, long_call_payoff, long_put_payoff, short_call_payoff, short_put_payoff)



### Gamma for a risk reversal ####
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define Black-Scholes Greeks
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def gamma(S, K, T, r, sigma):
    d1_val = d1(S, K, T, r, sigma)
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))

# Parameters
S = np.linspace(50, 150, 100)  # Range of stock prices
K_call = 110  # Strike price of the call
K_put = 90    # Strike price of the put
T = 1         # Time to maturity
r = 0.01      # Risk-free interest rate
sigma = 0.2   # Volatility

# Calculate gamma for each option
gamma_call = gamma(S, K_call, T, r, sigma)
gamma_put = -gamma(S, K_put, T, r, sigma)  # negative for the short put position

# Total gamma for the risk reversal strategy
gamma_total = gamma_call + gamma_put

# Plot
plt.figure(figsize=(10, 6))
plt.plot(S, gamma_total, label='Gamma of Risk Reversal')
plt.xlabel('Underlying Price (S)')
plt.ylabel('Gamma')
plt.title('Gamma for a Risk Reversal Strategy')
plt.legend()
plt.grid(True)
plt.show()
