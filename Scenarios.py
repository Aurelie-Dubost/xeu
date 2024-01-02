#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 1 19:00:00 2024

@author: aureliedubost
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import log, sqrt, exp
import matplotlib.pyplot as plt
from datetime import timedelta

##########################################################
######## 1. Simulate Custom Trade  ########
##########################################################
def simulate_custom_trade_structure(option_chain, trade_structure, current_price):
    """
    Simulate a custom options trade structure based on user-defined legs of long/short calls/puts.
    
    :param option_chain: DataFrame containing the options data.
    :param trade_structure: List of dictionaries defining the trade legs.
    :param current_price: The current price of the underlying asset.
    :return: Dictionary with the structure details and the total cost/premium.
    """
    total_cost = 0
    trade_details = []
    
    for leg in trade_structure:
        option_type = leg['Type']
        position = leg['Position']
        strike = leg.get('Strike')
        delta = leg.get('Delta')

        # If a specific strike is given, use it; otherwise find the closest strike based on delta
        if strike is None and delta is not None:
            filtered_options = option_chain[option_chain['Type'] == option_type]
            if option_type == 'call':
                # Select call option with the closest delta
                filtered_options = filtered_options[filtered_options['Delta'] >= delta]
            else:
                # Select put option with the closest delta
                filtered_options = filtered_options[filtered_options['Delta'] <= delta]
            
            if not filtered_options.empty:
                # Select the option with delta closest to the target
                closest_option = filtered_options.iloc[(filtered_options['Delta'] - delta).abs().argsort()[:1]]
                strike = closest_option['Strike'].values[0]
                premium = closest_option['Premium'].values[0]
            else:
                # Skip to next leg if no option matches the delta criteria
                continue
        else:
            # Get the premium for the specified strike
            premium = option_chain[(option_chain['Type'] == option_type) & (option_chain['Strike'] == strike)]['Premium'].iloc[0]
        
        # Calculate cost/premium for the leg (negative for shorts, positive for longs)
        cost = -premium if position == 'short' else premium
        total_cost += cost
        
        trade_details.append({
            'Type': option_type,
            'Position': position,
            'Strike': strike,
            'Premium': premium,
            'Cost': cost
        })

    return {
        'Trade Details': trade_details,
        'Total Cost/Premium': total_cost
    }

##########################################################
######## 2. Greeks  ########
##########################################################
# Calculate Black-Scholes Greeks for an option.
# This function will return the delta, gamma, theta, and vega for a given option based on current market conditions.
def black_scholes_greeks(option_type, S, K, T, r, sigma):
    epsilon = 1e-8  # Small number to prevent division by zero

    # Check and correct for near-zero or negative values of T and sigma
    T = max(T, epsilon)
    sigma = max(sigma, epsilon)

    # Calculating d1 and d2 using the Black-Scholes formula components
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    # Initializing Greeks dictionary
    greeks = {}

    # Calculate the Greeks based on whether it's a call or put option
    if option_type == 'call':
        greeks['delta'] = norm.cdf(d1)
        greeks['gamma'] = norm.pdf(d1) / (S * sigma * sqrt(T))
        greeks['theta'] = (-S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)
        greeks['vega'] = S * sqrt(T) * norm.pdf(d1)
    elif option_type == 'put':
        greeks['delta'] = -norm.cdf(-d1)
        greeks['gamma'] = norm.pdf(d1) / (S * sigma * sqrt(T))
        greeks['theta'] = (-S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-d2)
        greeks['vega'] = S * sqrt(T) * norm.pdf(d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return greeks

# Function to calculate Greeks for each leg of the trade
import numpy as np
import pandas as pd
from scipy.stats import norm
from math import exp, sqrt, log

# Make sure that black_scholes_greeks is defined somewhere in your code.

def calculate_greeks_for_each_leg_with_total(trade_structure, market_data_df, r, sigma):
    epsilon = 1e-8  # Define epsilon within the function scope

    # Create a list to store the greeks data for each leg over time
    greeks_list = []

    # Assign a 'Leg Number' to each leg in the trade structure
    for leg_number, leg in enumerate(trade_structure, start=1):
        option_type = leg['Type']
        position = leg['Position']
        strike = leg['Strike']

        for index, row in market_data_df.iterrows():
            S = row['Spot']
            # Ensure T is not zero or negative
            T = max((leg['Expiration'] - index).days / 365.25, epsilon)

            # Calculate the Greeks
            leg_greeks = black_scholes_greeks(option_type, S, strike, T, r, sigma)
            leg_greeks.update({
                'Date': index, 
                'Type': option_type, 
                'Position': position, 
                'Strike': strike,
                'Leg Number': leg_number  # Add 'Leg Number' here
            })
            greeks_list.append(leg_greeks)

    # Convert the list of dictionaries to a DataFrame
    greeks_df = pd.DataFrame(greeks_list)
    # Set the index to be a combination of 'Date' and 'Leg Number'
    greeks_df.set_index(['Date', 'Leg Number'], inplace=True)

    return greeks_df


# Function to calculate Greeks at initiation for each leg of the trade and return as a DataFrame
def calculate_greeks_at_initiation_df(trade_structure, S, T, r, sigma):
    """
    Calculate the Greeks for each leg of a trade at initiation and return as a DataFrame.

    Parameters:
    - trade_structure: List of dictionaries with the trade structure details.
    - S: Current spot price of the underlying.
    - T: Time to expiration in years.
    - r: Risk-free interest rate.
    - sigma: Volatility of the underlying asset.

    Returns:
    A pandas DataFrame with Greeks for each leg of the trade.
    """
    greeks_data = []
    for leg in trade_structure:
        leg_greeks = black_scholes_greeks(leg['Type'], S, leg['Strike'], T, r, sigma)
        greeks_data.append({
            'Type': leg['Type'],
            'Position': leg['Position'],
            'Strike': leg['Strike'],
            **leg_greeks  # Unpack the Greeks dictionary into the trade details
        })
    
    # Convert the list of dictionaries into a DataFrame
    greeks_df = pd.DataFrame(greeks_data)
    return greeks_df
# Function to compute the evolution of Greeks over time for a given trade structure
def compute_greeks_evolution(trade_structure, market_data_df, r, sigma):
    greeks_evolution_list = []

    for date, market_row in market_data_df.iterrows():
        # Placeholder for actual time to expiration calculation
        # Replace this with your specific logic for calculating T
        T = (expiration_date - date).days / 365.25
        
        for leg_number, leg in enumerate(trade_structure, start=1):
            leg_greeks = black_scholes_greeks(
                option_type=leg['Type'],
                S=market_row['Spot'],
                K=leg['Strike'],
                T=T,
                r=r,
                sigma=sigma
            )
def compute_greeks_evolution(trade_structure, market_data_df, r, sigma):
    """
    Compute the evolution of Greeks over time for a given trade structure, indexed by time and with legs in rows.
    """
    # Ensure 'Date' is the index and of datetime type
    market_data_df.set_index('Date', inplace=True)
    market_data_df.index = pd.to_datetime(market_data_df.index)
    
    # Create a list to store the greeks data for each leg over time
    greeks_time_indexed = []

    # Iterate through the market data to calculate Greeks for each date
    for date, market_row in market_data_df.iterrows():
        # Calculate time to expiration
        T = (market_data_df.index.max() - date).days / 365
        
        # Calculate Greeks for each leg and append to list
        for leg in trade_structure:
            leg_greeks = black_scholes_greeks(leg['Type'], market_row['Spot'], leg['Strike'], T, r, sigma)
            leg_greeks.update({
                'Date': date, 
                'Type': leg['Type'], 
                'Position': leg['Position'], 
                'Strike': leg['Strike']
            })
            greeks_time_indexed.append(leg_greeks)

    # Convert the list of dictionaries to a DataFrame
    greeks_df = pd.DataFrame(greeks_time_indexed)
    
    # Set the index to be a combination of 'Date' and a new 'Leg' level
    greeks_df.set_index(['Date', greeks_df.groupby('Date').cumcount() + 1], inplace=True)
    greeks_df.index.names = ['Date', 'Leg']
    
    return greeks_df

import pandas as pd

def compute_greeks_evolution_2(trade_structure, market_data_df, r, sigma, expiration_date):
    """
    Compute the evolution of Greeks over time for a given trade structure, indexed by time and with legs in rows.

    Parameters:
    - trade_structure: List of dictionaries defining the trade structure (legs).
    - market_data_df: DataFrame containing the market data.
    - r: Risk-free interest rate.
    - sigma: Volatility of the underlying asset.
    - expiration_date: Expiration date of the options.

    Returns:
    - DataFrame with the Greeks evolution over time.
    """
    # Initialize a list to store the Greeks for each leg over time
    greeks_evolution_list = []

    # Iterate over the market data rows
    for date, market_row in market_data_df.iterrows():
        # Calculate time to expiration for each date
        T = (pd.to_datetime(expiration_date) - date).days / 365.25

        # Calculate Greeks for each leg
        for leg_number, leg in enumerate(trade_structure, start=1):
            leg_greeks = black_scholes_greeks(
                option_type=leg['Type'],
                S=market_row['Spot'],
                K=leg['Strike'],
                T=T,
                r=r,
                sigma=sigma
            )
            # Add additional information to the leg_greeks dictionary
            leg_greeks.update({
                'Date': date,
                'Leg': leg_number,
                'Type': leg['Type'],
                'Position': leg['Position'],
                'Strike': leg['Strike']
            })
            greeks_evolution_list.append(leg_greeks)

    # Convert the list of dictionaries to a DataFrame
    greeks_evolution_df = pd.DataFrame(greeks_evolution_list)

    # Set 'Date' and 'Leg' as multi-level index
    greeks_evolution_df.set_index(['Date', 'Leg'], inplace=True)

    return greeks_evolution_df


##########################################################
######## 3. PnL  ########
##########################################################
def calculate_pnl(greeks_df, market_data_df, trade_structure):
    """
    Calculate the PnL for each leg of the trade and for the total trade given the market data.

    Parameters:
    - greeks_df: DataFrame containing the Greeks for each leg of the trade.
    - market_data_df: DataFrame containing the market data.
    - trade_structure: List of dictionaries defining the trade legs.

    Returns:
    A pandas DataFrame with the PnL for each leg and the total trade.
    """
    pnl_df = pd.DataFrame(index=market_data_df.index)
    
    for index, row in market_data_df.iterrows():
        total_pnl = 0
        for leg in trade_structure:
            if 'Type' in leg and 'Strike' in leg and 'Position' in leg and 'Quantity' in leg:
                leg_greeks = black_scholes_greeks(leg['Type'], row['Spot'], leg['Strike'], T, r, sigma)
                delta_pnl = leg_greeks['delta'] * (row['Spot'] - leg['Strike'])
                position_multiplier = 1 if leg['Position'] == 'long' else -1
                leg_pnl = delta_pnl * leg['Quantity'] * position_multiplier
                pnl_df.loc[index, f"Leg {leg['Type']} {leg['Strike']} PnL"] = leg_pnl
                total_pnl += leg_pnl
            else:
                raise KeyError("Trade structure dictionary missing required keys.")
        
        pnl_df.loc[index, 'Total PnL'] = total_pnl

    return pnl_df

# Basic PnL Calculation
def calculate_pnl_basic(greeks_evolution_df, market_data_df, trade_structure):
    pnl_df = pd.DataFrame(index=market_data_df.index)

    for index, row in market_data_df.iterrows():
        total_pnl = 0
        for leg in trade_structure:
            leg_number = leg['Leg Number']
            leg_greeks = greeks_evolution_df.loc[(index, leg_number)]

            change_in_spot = row['Spot'] - market_data_df.iloc[0]['Spot']
            delta_pnl = leg_greeks['delta'] * change_in_spot
            position_multiplier = 1 if leg['Position'] == 'long' else -1
            leg_pnl = delta_pnl * position_multiplier

            pnl_label = f"Leg {leg_number} {leg['Type']} {leg['Strike']} PnL"
            pnl_df.loc[index, pnl_label] = leg_pnl
            total_pnl += leg_pnl

        pnl_df.loc[index, 'Total PnL'] = total_pnl

    return pnl_df




# Advanced PnL Calculation
def calculate_pnl_advanced(greeks_evolution_df, market_data_df, trade_structure):
    pnl_df = pd.DataFrame(index=market_data_df.index)

    # Check if 'Implied Volatility' column exists
    has_iv = 'Implied Volatility' in market_data_df.columns

    for index, row in market_data_df.iterrows():
        total_pnl = 0
        for leg_index, leg in enumerate(trade_structure, start=1):
            # Access the Greeks for the specific leg and date
            # Check if greeks_evolution_df has a MultiIndex
            if isinstance(greeks_evolution_df.index, pd.MultiIndex):
                leg_greeks = greeks_evolution_df.loc[(index, leg_index)]
            else:
                # If there's no MultiIndex, use a column to filter the leg
                # This requires a column in greeks_evolution_df that identifies the leg, such as 'Leg Number'
                leg_greeks = greeks_evolution_df[(greeks_evolution_df['Date'] == index) & 
                                                 (greeks_evolution_df['Leg Number'] == leg_index)]

            # Calculate the PnL components
            change_in_spot = row['Spot'] - market_data_df.iloc[0]['Spot']
            delta_pnl = leg_greeks['delta'] * change_in_spot
            gamma_pnl = 0.5 * leg_greeks['gamma'] * change_in_spot ** 2
            theta_pnl = leg_greeks['theta'] / 365

            vega_pnl = 0
            if has_iv:
                vega_pnl = leg_greeks['vega'] * (row['Implied Volatility'] - market_data_df.iloc[0]['Implied Volatility'])

            position_multiplier = 1 if leg['Position'] == 'long' else -1
            leg_total_pnl = (delta_pnl + gamma_pnl + theta_pnl + vega_pnl) * position_multiplier

            # Update the PnL DataFrame
            pnl_label = f"Leg {leg['Type']} {leg['Strike']} PnL"
            pnl_df.loc[index, pnl_label] = leg_total_pnl if pnl_label in pnl_df.columns else leg_total_pnl
            total_pnl += leg_total_pnl

        pnl_df.loc[index, 'Total PnL'] = total_pnl

    return pnl_df

##########################################################
######## FAKE MARKET DATA  ########
##########################################################
option_chain = pd.DataFrame({
    'Strike': [90, 95, 100, 105, 110],
    'Type': ['put', 'put', 'call', 'call', 'call'],
    'Delta': [-0.2, -0.15, 0.15, 0.2, 0.25],
    'Premium': [1.2, 0.8, 0.7, 1.1, 1.5]
})