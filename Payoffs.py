#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:01:19 2023

@author: aureliedubost
"""
import numpy as np
import matplotlib.pyplot as plt

####### USER INPUTS #######
strike_price = 100
premium = 5
# Define the underlying asset price range
underlying_price = np.arange(80, 121, 1)


####### Functions #######
### Function to calculate the payoff of a option strategies
# Classic
def calculate_long_call_payoff(underlying_price, strike_price, premium):
    return np.maximum(underlying_price - strike_price, 0) - premium

def calculate_long_put_payoff(underlying_price, strike_price, premium):
    return np.maximum(strike_price - underlying_price, 0) - premium

def calculate_short_call_payoff(underlying_price, strike_price, premium):
    return -calculate_long_call_payoff(underlying_price, strike_price, premium)

def calculate_short_put_payoff(underlying_price, strike_price, premium):
    return -calculate_long_put_payoff(underlying_price, strike_price, premium)
# Structrued product
#

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