#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 03:35:29 2023

@author: aureliedubost
"""
import numpy as np
import matplotlib.pyplot as plt
from Chatgpt_functions import *

# DNT Option Parameters
L = 0.92  # Lower barrier
U = 0.96  # Upper barrier
K = 1e6   # Payout
T = 0.25  # Time to maturity in years
vol = 0.06  # Volatility
v = vol
r = 0.025  # Risk-free rate
b = -0.025  # Cost of carry

# Define a range for the underlying spot price
spot_range = np.linspace(0.9, 0.98, 400)  # Wider range to show the barriers

# Calculate the delta for the DNT option
delta = np.gradient([dnt_price(s, K, U, L, vol, T, r, b) for s in spot_range], spot_range)

# Calculate the payoff for the DNT option
payoff = [dnt_payoff(s, K, L, U) for s in spot_range]

# Plot the delta and payoff
plt.figure(figsize=(12, 6))

# Plot Delta
plt.subplot(1, 2, 1)
plt.plot(spot_range, delta, label='DNT Delta')
plt.axvline(x=L, color='red', linestyle='--', label='Lower Barrier')
plt.axvline(x=U, color='green', linestyle='--', label='Upper Barrier')
plt.title('Delta of a 1Y Double-No-Touch Option')
plt.xlabel('Spot Price')
plt.ylabel('Delta')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# DNT Option Parameters
L = 0.92  # Lower barrier
U = 0.96  # Upper barrier
K = 1e6   # Payout
T = 0.25  # Time to maturity in years
vol = 0.06  # Volatility
r = 0.025  # Risk-free rate
b = -0.025  # Cost of carry

# Define the DNT pricing function
def dnt_price(S, K, U, L, sigma, T, r, b, N=20):
    # ... (same as the previous function)
    return np.sum(v)

# Define a range for the underlying spot price
spot_range = np.linspace(L, U, 200)

# Calculate the price for a range of spots
prices = np.array([dnt_price(s, K, U, L, vol, T, r, b) for s in spot_range])

# Calculate Greeks using finite differences
delta = np.gradient(prices, spot_range)
gamma = np.gradient(delta, spot_range)
vega = np.gradient(prices, vol)
theta = np.gradient(prices, T)

# Plot the Greeks
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Delta
axs[0, 0].plot(spot_range, delta, label='Delta')
axs[0, 0].set_title('Delta')
axs[0, 0].set_xlabel('Spot Price')
axs[0, 0].set_ylabel('Delta')
axs[0, 0].grid(True)

# Gamma
axs[0, 1].plot(spot_range, gamma, label='Gamma', color='orange')
axs[0, 1].set_title('Gamma')
axs[0, 1].set_xlabel('Spot Price')
axs[0, 1].set_ylabel('Gamma')
axs[0, 1].grid(True)

# Vega
axs[1, 0].plot(spot_range, vega, label='Vega', color='green')
axs[1, 0].set_title('Vega')
axs[1, 0].set_xlabel('Spot Price')
axs[1, 0].set_ylabel('Vega')
axs[1, 0].grid(True)

# Theta
axs[1, 1].plot(spot_range, theta, label='Theta', color='red')
axs[1, 1].set_title('Theta')
axs[1, 1].set_xlabel('Spot Price')
axs[1, 1].set_ylabel('Theta')
axs[1, 1].grid(True)

for ax in axs.flat:
    ax.legend()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

# Black-Scholes Delta for a Call Option
def call_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# Parameters for plotting
S = np.linspace(50, 150, 100)  # Range of underlying asset prices
T = np.linspace(0.01, 0.5, 100)  # Time to expiry in years (from now to 6 months)
K = 100  # Strike price
r = 0.01  # Risk-free rate
sigma = 0.15  # Volatility

# Create meshgrid
S_grid, T_grid = np.meshgrid(S, T)
Delta_grid = np.zeros_like(S_grid)

# Calculate Delta for each pair of (S, T)
for i in range(len(T)):
    for j in range(len(S)):
        Delta_grid[i, j] = call_delta(S[j], K, T[i], r, sigma)

# Create plot
fig = plt.figure(figsize=(14, 8))

# 3D plot
ax1 = fig.add_subplot(1, 2, 2, projection='3d')
ax1.plot_surface(S_grid, T_grid, Delta_grid, cmap='viridis')
ax1.set_xlabel('Underlying Price')
ax1.set_ylabel('Time to Expiry')
ax1.set_zlabel('Delta')
ax1.set_title('Call Option Delta Surface')

# 2D plot for 6 months expiry
ax2 = fig.add_subplot(1, 2, 1)
ax2.plot(S, call_delta(S, K, 0.1, r, sigma), label='Delta 1m call', color='blue')
ax2.plot(S, call_delta(S, K, 1, r, sigma), label='Delta 1y call', color='red')
ax2.set_xlabel('Underlying Price')
ax2.set_ylabel('Delta')
ax2.set_title('Delta vs Underlying Price for 6m Call')
ax2.legend()

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a date range from 2021 to 2024
dates = pd.date_range(start='2021-01-01', end='2024-01-01', freq='M')

# Generate random data to simulate the sentiment indicator
np.random.seed(0)  # for reproducibility
data = np.random.randn(len(dates))  # standard normal distribution

# Create a DataFrame
df = pd.DataFrame(data, index=dates, columns=['Sentiment'])

# Calculate rolling mean and standard deviation
rolling_mean = df['Sentiment'].rolling(window=12).mean()
rolling_std = df['Sentiment'].rolling(window=12).std()

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(df.index, df['Sentiment'], width=15, color='blue', edgecolor='black')

# Add lines for +1 and -1 standard deviations
plt.axhline(y=1, color='black', linestyle='--', label='+1 stdev ("stretched")')
plt.axhline(y=-1, color='black', linestyle='--', label='-1 stdev ("light")')

# Annotations for the last bar to indicate value
plt.text(df.index[-1], df['Sentiment'].iloc[-1], f"{df['Sentiment'].iloc[-1]:.1f}", 
         va='center', ha='center')

# Set the y-axis limit
plt.ylim(-3, 3)

# Set x-axis to yearly ticks
plt.xticks(pd.date_range(start='2021-01-01', end='2024-01-01', freq='Y'))

# Adding labels and title
plt.title('US Equity Sentiment Indicator')
plt.xlabel('Year')
plt.ylabel('Sentiment Indicator')

# Show the legend
plt.legend()

# Show grid
plt.grid(axis='y')

# Display the plot
plt.tight_layout()
plt.show()


