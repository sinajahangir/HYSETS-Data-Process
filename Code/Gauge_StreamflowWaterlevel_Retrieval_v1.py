# -*- coding: utf-8 -*-
"""
Code for saving water level and streamflow data for 02GA014 gauge station in CA
The data was originally retrieved from:
https://wateroffice.ec.gc.ca/report/historical_e.html?stn=02GA014&dataType=Daily&parameterType=Level&first_year=2023&last_year=2023&mode=Graph
"""
#%%
#import necessary libraries
import pandas as pd
import os
#%%
#change directory
os.chdir('F:\msjahangir\CVAE-HYSETS\Data\Flood')
#%%
#read the original file containing the water level and streamflow data
file_name='daily_20250707T1942.csv'
df_all=pd.read_csv(file_name,header=1)
#%%
# make sure that the date format is correct

#first renaming the Date column
df_all=df_all.rename(columns={'Date': 'date'})

#chnage format
df_all['date']=pd.to_datetime(df_all['date']).dt.date

#%%
#filtering based on date and feature

# Set the start and end dates properly
start_date = pd.to_datetime('2002-01-01').date()
end_date = pd.to_datetime('2023-12-31').date()

# Filter for PARAM == 2 (water level)
df_w = df_all[df_all['PARAM'] == 2]

# Filter within date range
df_w = df_w[(df_w['date'] >= start_date) & (df_w['date'] <= end_date)]


# Filter for PARAM == 1 (Streamflow data)
df_s = df_all[df_all['PARAM'] == 1]

# Filter within date range
df_s = df_s[(df_s['date'] >= start_date) & (df_s['date'] <= end_date)]
#%%
#save to csv files
savename='02GA014_'

df_w.to_csv(savename+'waterlevel.csv',columns=['date','Value'],index=None)
df_s.to_csv(savename+'streamflow.csv',columns=['date','Value'],index=None)