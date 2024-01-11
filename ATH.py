#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 20:56:45 2023

@author: aureliedubost
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Creating a longer DataFrame with stock prices
dates = pd.date_range(start='2021-01-01', periods=30)
prices = np.random.randint(95, 130, size=30).astype(float)  # Random stock prices between 95 and 130
prices[5] = 135  # Ensuring at least one ATH
df_long = pd.DataFrame({'Date': dates, 'Stock_Price': prices})

# Calculating All-Time High (ATH)
df_long['ATH'] = df_long['Stock_Price'].cummax()

# Function to plot the time series with a triangle at ATH
def plot_stock_with_ath(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Stock_Price'], label='Stock Price')

    # Marking the ATHs with triangles
    ath_days = df[df['Stock_Price'] == df['ATH']]
    plt.scatter(ath_days['Date'], ath_days['Stock_Price'], color='red', marker='^', label='ATH')

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Time Series with All-Time Highs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting the graph
plot_stock_with_ath(df_long)

