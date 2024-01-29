#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:01:19 2023

@author: aureliedubost
"""
# get_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to find the most recent past Friday
def last_friday():
    today = datetime.now()
    offset = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=offset)
    return last_friday.date()

# Generates fake time series data for given securities efficiently using vectorization
def generate_data(start, end, freq, parameters):
    dates = pd.date_range(start=start, end=end, freq=freq)
    return pd.DataFrame(np.random.randn(len(dates), len(parameters)), index=dates, columns=parameters)

# Generate data for CAC and SPX
parameters = ['spot', 'atmfs', 'convexity', 'skew', 'down skew', 'up skew', 'rv', 'vrp', 'term structure']
extended_data = {
    'CAC': generate_data('2020-01-01', '2024-12-31', 'D', parameters),
    'SPX': generate_data('2020-01-01', '2024-12-31', 'D', parameters)
}
