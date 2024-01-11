#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 21:13:17 2023

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
######## Basic functions  ########
##########################################################

# Function to calculate Black-Scholes Greeks
def black_scholes_greeks(option_type, S, K, T, r, sigma):
    # S: spot price
    # K: strike price
    # T: time to expiry (in years)
    # r: risk-free rate
    # sigma: volatility of the underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -((S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        delta = -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -((S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

def calculate_simple_skew(data):
    """
    Calculate the skew for each period.
    :param data: DataFrame with columns for dates, and implied volatilities for OTM puts, ATM options, and OTM calls.
    :return: DataFrame with calculated skew for each period.
    """
    # Calculate skew as the difference between the IV of OTM puts and calls relative to ATM options
    data['Skew'] = (data['IV_OTM_Put'] - data['IV_OTM_Call']) / data['IV_ATM']
    return data

def filter_for_low_skew_periods(data, skew_threshold):
    """
    Return periods when the skew was below the specified threshold.
    :param data: DataFrame with a 'Skew' column.
    :param skew_threshold: The level below which we consider the skew to be low.
    :return: DataFrame with periods when the skew was below the threshold.
    """
    # Filter periods where skew is below the threshold
    low_skew_periods = data[data['Skew'] < skew_threshold]
    return low_skew_periods


# This function calculates the skewness levels based on percentile and standard deviation threshold
def calculate_skewness_levels(data, iv_put_col, iv_call_col, iv_atm_col, percentile=10, std_devs_below_mean=1):
    """
    Calculate the skewness levels based on the percentile and standard deviation threshold.
    
    :param data: DataFrame with columns for the implied volatilities for OTM puts, ATM options, and OTM calls.
    :param iv_put_col: Column name for implied volatilities of OTM puts.
    :param iv_call_col: Column name for implied volatilities of OTM calls.
    :param iv_atm_col: Column name for implied volatilities of ATM options.
    :param percentile: Percentile to determine the low skew level.
    :param std_devs_below_mean: Number of standard deviations below the mean to determine low skew level.
    :return: A dictionary with 'percentile_level' and 'std_dev_level' for skewness.
    """
    data['Skew'] = (data[iv_put_col] + data[iv_call_col]) / 2 - data[iv_atm_col]
    percentile_level = np.percentile(data['Skew'], percentile)
    mean_skew = data['Skew'].mean()
    std_dev_skew = data['Skew'].std()
    std_dev_level = mean_skew - std_devs_below_mean * std_dev_skew
    return {
        'percentile_level': percentile_level,
        'mean_skew': mean_skew,
        'std_dev_skew': std_dev_skew,
        'std_dev_level': std_dev_level
    }

def filter_low_skew_periods(data, skew_threshold):
    """
    Filter the dataset to identify the periods when the skew is below a predefined threshold.
    :param data: DataFrame with a 'Skew' column.s
    :param skew_threshold: Threshold below which the skew is considered low.
    :return: DataFrame with periods where skew is below the threshold.
    """
    low_skew_periods = data[data['Skew'] < skew_threshold]
    return low_skew_periods



#############################
####### Skew analysis #######
#############################

def calculate_skew(data, iv_put_col, iv_call_col, iv_atm_col):
    """
    Calculate the skew as the difference between the IV of OTM puts and calls relative to ATM options.
    """
    data['UpSkew'] = data[iv_call_col] - data[iv_atm_col]
    data['DownSkew'] = data[iv_atm_col] - data[iv_put_col]
    data['Skew'] = (data['IV_OTM_Put'] - data['IV_OTM_Call']) / data['IV_ATM']
    return data

def calculate_spreads_and_ratios(data):
    """
    Calculate the spread between downskew and upskew, and the ratio of upskew to skew.
    """
    data['Spread'] = data['DownSkew'] - data['UpSkew']
    data['Ratio'] = data['UpSkew'] / data['Skew'].replace(0, np.nan)  # Avoid division by zero
    return data

def find_threshold_periods(data, spread_threshold, ratio_threshold):
    """
    Find the periods where the spread and ratio meet certain thresholds.
    """
    condition = (data['Spread'] > spread_threshold) | (data['Ratio'] > ratio_threshold)
    return data[condition]

def calculate_vol_shifts(data, threshold_periods, shifts):
    """
    Calculate the average and median vol shifts after certain thresholds have been reached.
    """
    results = {}
    for shift in shifts:
        shift_days = shift * 30
        vol_shifts = []
        for date in threshold_periods['Date']:
            start_vol = data.loc[data['Date'] == date, 'IV_ATM'].values
            end_date = date + timedelta(days=shift_days)
            end_vol = data.loc[data['Date'] == end_date, 'IV_ATM'].values
            if start_vol.size > 0 and end_vol.size > 0:
                vol_shifts.append(end_vol[0] - start_vol[0])

        results[shift] = {'average': np.nanmean(vol_shifts), 'median': np.nanmedian(vol_shifts)}
    return pd.DataFrame(results)

##########################################################
####### 2. Identify Qualifying Periods #######
##########################################################
def calculate_skewness_levels(data, iv_put_col, iv_call_col, iv_atm_col, percentile=10, std_devs_below_mean=1):
    data['Skew'] = (data[iv_put_col] + data[iv_call_col]) / 2 - data[iv_atm_col]
    percentile_level = np.percentile(data['Skew'], percentile)
    mean_skew = data['Skew'].mean()
    std_dev_skew = data['Skew'].std()
    std_dev_level = mean_skew - std_devs_below_mean * std_dev_skew
    return {
        'percentile_level': percentile_level,
        'mean_skew': mean_skew,
        'std_dev_level': std_dev_level
    }

def categorize_skew_periods(data, skew_levels):
    """
    Categorize periods as high, average, or low skew based on skew levels.
    """
    high_skew_threshold = skew_levels['mean_skew'] + skew_levels['std_dev_level']
    low_skew_threshold = skew_levels['percentile_level']

    conditions = [
        data['Skew'] > high_skew_threshold, 
        data['Skew'] < low_skew_threshold,
        (data['Skew'] <= high_skew_threshold) & (data['Skew'] >= low_skew_threshold)
    ]
    choices = ['High', 'Low', 'Average']
    data['Skew_Category'] = np.select(conditions, choices, default='Average')

    return data

##########################################################
######## 3. Selecting strikes based on delta and moneyness 
##########################################################
def select_delta_based_strikes(option_chain, target_delta):
    """
    Select strike prices for call and put options based on target deltas.

    :param option_chain: DataFrame containing option chain data with deltas.
    :param target_delta: The target delta value for the strategy.
    :return: Dictionary with selected put and call strike prices.
    """
    # Find the call and put options with deltas closest to the target delta
    call_strike = option_chain[(option_chain['Type'] == 'call') & 
                               (option_chain['Delta'] >= target_delta)].iloc[0]['Strike']
    put_strike = option_chain[(option_chain['Type'] == 'put') & 
                              (option_chain['Delta'] <= -target_delta)].iloc[-1]['Strike']

    return {'Call Strike': call_strike, 'Put Strike': put_strike}

def select_moneyness_based_strikes(current_price, percentage_above, percentage_below):
    """
    Select strike prices for call and put options based on moneyness.

    :param current_price: The current price of the underlying asset.
    :param percentage_above: The percentage above the current price for the call option.
    :param percentage_below: The percentage below the current price for the put option.
    :return: Dictionary with selected put and call strike prices.
    """
    call_strike = current_price * (1 + percentage_above / 100)
    put_strike = current_price * (1 - percentage_below / 100)

    return {'Call Strike': call_strike, 'Put Strike': put_strike}

##########################################################
######## 4. Simulate Custom Trade  ########
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
######## 5. Track and Manage Positions ########
#########################################################
def track_and_manage_positions(strangle_positions, underlying_prices):
    # Track the daily price movements of both options
    # In practice, you would use an options pricing model to calculate daily prices
    # Here we simulate price movements for the call and put options
    call_prices = underlying_prices * 0.05  # Placeholder for call option prices
    put_prices = (100 - underlying_prices) * 0.05  # Placeholder for put option prices
    return call_prices, put_prices

##########################################################
######## # 6. Calculate PnL ########
#########################################################
def calculate_pnl(strangle_positions, call_prices, put_prices):
    # Calculate the PnL for the strangle at expiration
    # In practice, you would calculate the final prices based on market data
    final_call_price = call_prices.iloc[-1]
    final_put_price = put_prices.iloc[-1]
    pnl = (final_call_price + final_put_price) - strangle_positions['Total Premiums Paid']
    return pnl - transaction_costs  # Subtract transaction costs

##########################################################
######## # 7. Backtesting ########
#########################################################
def backtest_strategy(underlying_data, option_chain):
    # Filter periods based on skew
    low_skew_periods = underlying_data[underlying_data['Skew'] < skew_threshold]
    pnls = []
    for date in low_skew_periods['Date']:
        # Simulate strangle positions for the date
        strangle_positions = simulate_strangle_positions(option_chain)
        # Track and manage the positions
        call_prices, put_prices = track_and_manage_positions(strangle_positions, underlying_data['Price'])
        # Calculate PnL at expiration
        pnl = calculate_pnl(strangle_positions, call_prices, put_prices)
        pnls.append(pnl)
    
    # Calculate average performance over all periods
    average_pnl = np.mean(pnls)
    return average_pnl