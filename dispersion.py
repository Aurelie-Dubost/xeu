import pandas as pd
import numpy as np

def downcast_df(df):
    """
    Downcast numeric columns in a DataFrame to more memory-efficient types.
    
    :param df: DataFrame to be downcasted.
    :return: Downcasted DataFrame.
    """
    float_cols = df.select_dtypes(include=['float64']).columns
    int_cols = df.select_dtypes(include=['int64']).columns

    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int32')

    return df

def compute_weighted_statistics(compo, vol, col_name, start_date=None, end_date=None):
    """
    This function computes the weighted average, means, and median of the vol DataFrame columns,
    excluding the specified col_name from the calculations but including it in the results.
    The weights are derived from the compo DataFrame. Calculations are performed within the specified date range.

    :param compo: DataFrame containing weekly allocation time series for different assets.
    :param vol: DataFrame containing daily volatility data for different assets.
    :param col_name: Column name to be excluded from the calculations but included in the results.
    :param start_date: Start date for the calculation (inclusive).
    :param end_date: End date for the calculation (inclusive).
    :return: A dictionary with weighted average, means, and median for each column including col_name.
    """

    # Efficiently filter data based on the specified date range
    if start_date:
        compo = compo.loc[start_date:]
        vol = vol.loc[start_date:]
    if end_date:
        compo = compo.loc[:end_date]
        vol = vol.loc[:end_date]

    # Downcast dataframes to save memory
    compo = downcast_df(compo)
    vol = downcast_df(vol)

    # Align compo with vol by forward filling missing weekly values to daily frequency
    compo_aligned = compo.reindex(vol.index, method='ffill')

    # Exclude the specified column from calculations
    compo_excluded = compo_aligned.drop(columns=[col_name])
    vol_excluded = vol.drop(columns=[col_name])

    # Calculate the weighted statistics excluding the specified column
    weighted_avg = (vol_excluded * compo_excluded).sum() / compo_excluded.sum()
    means = vol_excluded.mean()
    medians = vol_excluded.median()

    # Add the excluded column back to the results
    weighted_avg[col_name] = vol[col_name].mean()
    means[col_name] = vol[col_name].mean()
    medians[col_name] = vol[col_name].median()

    # Compile results into a dictionary
    results = {
        'weighted_avg': weighted_avg,
        'means': means,
        'medians': medians
    }

    return results