import numpy as np
import pandas as pd
from scipy.stats import zscore

def calculate_zscore_surface(nested_dict, udl, date, moneyness_levels, delta_levels, strike_type='Moneyness', transpose=False):
    """
    Calculates the Z-score surface for a given option data structure.

    Args:
        nested_dict (dict): Nested dictionary containing option data.
        udl (str): Underlying asset identifier.
        date (str/datetime): Date for which to calculate the Z-score surface.
        moneyness_levels (list): List of moneyness levels.
        delta_levels (list): List of delta levels.
        strike_type (str): Type of strike measure ('Moneyness' or 'Delta').
        transpose (bool): Whether to transpose the output DataFrame.

    Returns:
        pd.DataFrame: Z-score surface with moneyness/delta levels as index and maturities as columns.
    """
    # Define translation dictionary for strike types
    translation_dict = {"Moneyness": "IV", "Delta": "TVTD"}

    # Get translated strike type
    translated_strike_type = translation_dict.get(strike_type, strike_type)

    try:
        # Ensure date is in string format
        date_str = date if isinstance(date, str) else date.strftime("%Y-%m-%d")

        # Determine parameter levels based on strike type
        if translated_strike_type == "IV":
            param_levels = moneyness_levels
        elif translated_strike_type == "TVTD":
            param_levels = delta_levels
        else:
            print("Invalid vol type selected.")
            return None

        # Backfill missing data
        filled_data = backfill_missing_data(nested_dict, udl, translated_strike_type, direction=backfill_direction)
        if filled_data is None or pd.to_datetime(date_str) not in filled_data.index:
            print(f"No available data for {date_str} after backfilling.")
            return pd.DataFrame()

        # Create empty DataFrame for storing Z-scores
        zscore_surface = pd.DataFrame(index=param_levels, columns=filled_data.columns.levels[1])

        # Calculate Z-score for each maturity
        for param in param_levels:
            for matu in zscore_surface.columns:
                if (param, matu) not in filled_data.columns:
                    zscore_surface.at[param, matu] = np.nan
                    continue

                values = filled_data[(param, matu)].dropna().values  # Drop NaNs before Z-score calculation
                current_value = filled_data.at[pd.to_datetime(date_str), (param, matu)]

                if len(values) > 1:
                    z_scores = zscore(values, nan_policy="omit")
                    sorted_values = np.argsort(values)
                    sorted_zscores = z_scores[sorted_values]
                    zscore_surface.at[param, matu] = sorted_zscores[np.searchsorted(values, current_value)]
                else:
                    zscore_surface.at[param, matu] = np.nan

        max_maturity = 12
        zscore_surface = zscore_surface.loc[:, zscore_surface.columns.map(lambda x: x <= max_maturity)]

        # Rename columns/index for better formatting
        if translated_strike_type == "IV":
            zscore_surface.columns = [f'{int(mon)}m' if mon % 1 == 0 else f'{mon:.1f}m' for mon in zscore_surface.columns]
            zscore_surface.index = [f'{int(mon)}' if mon % 1 == 0 else f'{mon:.1f}' for mon in zscore_surface.index]
        else:
            zscore_surface.columns = [f'{int(mon)}m' if mon % 1 == 0 else f'{mon:.1f}m' for mon in zscore_surface.columns]
            zscore_surface.index = [f'Δ{int(delta)}' for delta in transform_delta_list(zscore_surface.index)]

        return zscore_surface

    except Exception as e:
        print(f"An error occurred in calculate_zscore_surface: {e}")
        return pd.DataFrame()