#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:01:19 2023

@author: aureliedubost
"""
# data_output.py

# Import the necessary libraries
import base64
import io
import pandas as pd
from app import extended_data  # Import extended_data from app.py

def create_download_link(underlying):
    # Create a Pandas Excel writer using xlsxwriter as the engine
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        extended_data[underlying].to_excel(writer, sheet_name='Data')

    output.seek(0)

    # Define the download link
    download_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{base64.b64encode(output.read()).decode()}" download="{underlying}_data.xlsx">Download Data</a>'

    return download_link

