#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 03:54:02 2023

@author: aureliedubost
"""

##### 
# DNT
import numpy as np

# Define the DNT pricing function based on the approach from the resource
def dnt_price(S, K, U, L, sigma, T, r, b, N=20):
    if L > S or S > U:
        return 0
    Z = np.log(U/L)
    alpha = -0.5 * (2 * b / sigma**2 - 1)
    beta = -0.25 * (2 * b / sigma**2 - 1)**2 - 2 * r / sigma**2
    v = np.zeros(N)
    for i in range(1, N + 1):
        v[i-1] = (2 * np.pi * i * K / Z**2) * (((S/L)**alpha - (-1)**i * (S/U)**alpha) /
                  (alpha**2 + (i * np.pi/Z)**2)) * np.sin(i * np.pi/Z * np.log(S/L)) * \
                  np.exp(-0.5 * ((i * np.pi/Z)**2 - beta) * sigma**2 * T)
    return np.sum(v)

# Define the payoff function for the DNT option
def dnt_payoff(S, K, L, U):
    return K if L < S < U else 0