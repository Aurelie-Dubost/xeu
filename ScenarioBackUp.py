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

