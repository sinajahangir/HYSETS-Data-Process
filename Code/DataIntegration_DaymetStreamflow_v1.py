# -*- coding: utf-8 -*-
"""
Code for integrating water level, and streamflow with Daymet v4 (Thornton et al., 2022) data for 02GA014 gauge station in CA

Streamflow and water-level Data was processed using:
Gauge_StreamflowWaterlevel_Retrieval_v.py

Daymet data was obtained via:
    https://github.com/sinajahangir/Cload-Data-Retrieval/blob/main/DaymetRetrieval_v1.ipynb

Reference:
Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S-C. Kao, and B.E. Wilson. 2022.
Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2129
"""
#%%
#import necessary libraries
import pandas as pd
import os
#%%
def find_overlapping_dates(df1, df2, date_col1='date', date_col2='date'):
    """
    From df1, select a slice that starts from the first date in df2
    and has the same length as df2.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        date_col1 (str): Date column name in df1.
        date_col2 (str): Date column name in df2.

    Returns:
        pd.DataFrame: A slice of df1 aligned to df2 by start date and length.
    """
    import pandas as pd

    # Ensure date columns are datetime
    df1 = df1.copy()
    df2 = df2.copy()
    df1[date_col1] = pd.to_datetime(df1[date_col1]).dt.date
    df2[date_col2] = pd.to_datetime(df2[date_col2]).dt.date

    # Sort df1 just in case
    df1 = df1.sort_values(by=date_col1).reset_index(drop=True)

    # Get the starting date from df2
    start_date = pd.to_datetime(df2[date_col2].iloc[0], errors='coerce')

    # Filter df1 to start from start_date
    df1_slice = df1[df1[date_col1] >= start_date]

    # Get the slice of same length as df2

    return df1_slice.iloc[:len(df2),:]
#%%
def add_missing_leap_dec31(df, date_col='date', value_cols=None):
    """
    Adds missing Dec 31 for leap years in df by averaging Dec 30 and Dec 31
    from the previous year and inserting it.

    Args:
        df (pd.DataFrame): Input DataFrame with a datetime column.
        date_col (str): Name of the datetime column.
        value_cols (list): List of columns to average (if None, all non-date columns used).

    Returns:
        pd.DataFrame: DataFrame with missing Dec 31 filled in.
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    if value_cols is None:
        value_cols = [col for col in df.columns if col != date_col]

    leap_years = [y for y in df[date_col].dt.year.unique() if y % 4 == 0 and 
                  (y % 100 != 0 or y % 400 == 0)]

    rows_to_add = []

    for year in leap_years:
        dec31 = pd.Timestamp(f"{year}-12-31")
        if dec31 not in df[date_col].values:
            prev_dec31 = pd.Timestamp(f"{year - 1}-12-31")
            prev_dec30 = pd.Timestamp(f"{year}-12-30")

            if prev_dec31 in df[date_col].values and prev_dec30 in df[date_col].values:
                val1 = df.loc[df[date_col] == prev_dec30, value_cols].values[0]
                val2 = df.loc[df[date_col] == prev_dec31, value_cols].values[0]
                avg_vals = (val1 + val2) / 2

                new_row = {date_col: dec31}
                new_row.update({col: avg_vals[i] for i, col in enumerate(value_cols)})
                rows_to_add.append(new_row)

    if rows_to_add:
        df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)
        df = df.sort_values(by=date_col)

    return df
#%%
def drop_trailing_nans(df, column):
    """
    Removes rows from the point where all remaining values in `column` are NaN.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): Name of the column to check for trailing NaNs.

    Returns:
        pd.DataFrame: The trimmed DataFrame.
    """
    # Reverse boolean mask: True where values are not NaN
    not_nan_mask = df[column].notna()[::-1]
    
    # Find index of the last non-NaN value
    last_valid_index = not_nan_mask.idxmax()
    
    # Keep rows up to and including the last valid index
    return df.loc[:last_valid_index].copy()
#%%
# main script
if __name__ == "__main__":
  os.chdir('F:\msjahangir\CVAE-HYSETS\Data\Flood')
sites=['02GA014']
for ii in range(0,len(sites)):
    #read Daymet data
    df_camels=pd.read_csv('%s_Daymet.csv'%(sites[ii]))
    #read waterlevel data
    df_w=pd.read_csv('%s_waterlevel.csv'%(sites[ii]))
    df_w=df_w.rename(columns={'Value':'wl'})
    
    #read streamflow data
    df_s=pd.read_csv('%s_streamflow.csv'%(sites[ii]))
    df_s=df_s.rename(columns={'Value':'q'})
    
    # Add december 31st for leap years
    df_camels_mod= add_missing_leap_dec31(df_camels)
    
    df_camels_mod_slice_w=find_overlapping_dates(df_camels_mod,df_w)
    
    df_data=pd.concat((df_camels_mod_slice_w.reset_index(drop=True),df_w.loc[:,'wl'].reset_index(drop=True),df_s.loc[:,'q'].reset_index(drop=True)),axis=1)
    
    df_data_nanrm= drop_trailing_nans(df_data,'prcp')
    df_data_nanrm=df_data_nanrm.fillna(df_data_nanrm.median(numeric_only=True))
    #save to csv
    df_data_nanrm.to_csv('%s_DI.csv'%(sites[ii]),index=None)
    
    
    
    
    
    
    
    

