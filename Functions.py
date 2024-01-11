#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:01:19 2023

@author: aureliedubost
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

####### Payoff of a option strategies #######
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

####### BSM and Greeks #######
def black_scholes(S, K, t, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(t))
    theta = (-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(t)) - r * K * np.exp(-r * t) * norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(t)
    rho = K * t * np.exp(-r * t) * norm.cdf(d2)

    return price, delta, gamma, theta, vega, rho

# All in recedent function
def black_scholes_delta(S, K, t, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    if option_type == 'call':
        delta = np.exp(-r * t) * norm.cdf(d1)
    elif option_type == 'put':
        delta = np.exp(-r * t) * (norm.cdf(d1) - 1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    return delta

def black_scholes_gamma(S, K, t, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    gamma = np.exp(-r * t) * norm.pdf(d1) / (S * sigma * np.sqrt(t))
    return gamma

def black_scholes_theta(S, K, t, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    theta = (-S * np.exp(-r * t) * norm.pdf(d1) * sigma) / (2 * np.sqrt(t)) - r * K * np.exp(-r * t) * norm.cdf(d2)
    return theta

def black_scholes_vega(S, K, t, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    vega = S * np.exp(-r * t) * norm.pdf(d1) * np.sqrt(t)
    return vega

def black_scholes_rho(S, K, t, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    rho = K * t * np.exp(-r * t) * norm.cdf(d2)
    return rho

###### PLOT
def plot_option_payoff(underlying_price, long_call_payoff, long_put_payoff, short_call_payoff, short_put_payoff):
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