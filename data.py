import pandas as pd
import numpy as np

def generate_fake_data():
    """
    Generate fake data for compo (weekly data) and vol (daily data) DataFrames.

    :return: compo (DataFrame), vol (DataFrame)
    """
    # Generate fake data for compo DataFrame (weekly data)
    dates_compo = pd.date_range(start='2023-01-01', end='2023-12-31', freq='W')
    data_compo = {
        'Asset_A': np.random.random(size=len(dates_compo)),
        'Asset_B': np.random.random(size=len(dates_compo)),
        'Asset_C': np.random.random(size=len(dates_compo)),
        'Asset_D': np.random.random(size=len(dates_compo))
    }
    compo = pd.DataFrame(data_compo, index=dates_compo)

    # Generate fake data for vol DataFrame (daily data)
    dates_vol = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    data_vol = {
        'Asset_A': np.random.random(size=len(dates_vol)),
        'Asset_B': np.random.random(size=len(dates_vol)),
        'Asset_C': np.random.random(size=len(dates_vol)),
        'Asset_D': np.random.random(size=len(dates_vol))
    }
    vol = pd.DataFrame(data_vol, index=dates_vol)
    
    return compo, vol