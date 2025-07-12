# -*- coding: utf-8 -*-
"""
Code for combining the static features of HYSETS (Arsenault et al., 2024)  and Caravan (Kratzert, 2022) for 02GA014 gauge station in CA
Reference:
    
Arsenault, R., Brissette, F., Martel, J.-L., Troin, M., Lévesque, G., Davidson-Chaput, J., … Poulin, A. (2024, October 1).
         HYSETS - A 14425 watershed Hydrometeorological Sandbox over North America. https://doi.org/10.17605/OSF.IO/RPC3W
         
Kratzert, F. (2022). Caravan ‐ A global community dataset for large‐sample hydrology. Zenodo. https://doi.org/10.5281/zenodo.7540792

"""
#%% Import the libraries
import os
import pandas as pd
#%% Change directory
os.chdir('F:\msjahangir\CVAE-HYSETS\Data')
#%%
#421 camles catchments
df_camels=pd.read_csv('BasinList_AllDELM_v1.csv')
#All original HYSETS info
df_hysets=pd.read_csv('HYSETS Info file.csv')
#static features for HYSETS but obtained from CARAVAN
df_static=pd.read_csv('attributes_caravan_hysets.csv')  
#%%
basins_camels=[]
for ii in range(0,len(df_camels)):
    basin_id=ii
    id_=str(df_camels.iloc[basin_id,0])
    if len(id_)==7:
      id_='0'+id_
    basins_camels.append(id_)
#%%
basin_hysets=df_hysets[['Official_ID']]
#catchments both in CAMELS and HYSETS
matching_catchments=df_hysets[basin_hysets['Official_ID'].isin(basins_camels)]
#HYSETS in US and the catchment(s) in Canada
#The id was manually selected. Check df_hysets. 02GA005 was selected instead of 02GA014 as the latter is not available in Caravan
total_catchments=pd.concat((matching_catchments,df_hysets.iloc[[268],:]),axis=0)
#%%
basins_hysets_list=[]
for ii in range(0,len(total_catchments)):
    id_='hysets_'+total_catchments['Official_ID'].iloc[ii]
    basins_hysets_list.append(id_)
#%%
basin_hysets_caravan=df_static[['gauge_id']]
# These are catchments that are in all three: CAMELS, CARAVAN, and HYSETS
matching_catchments_ = basin_hysets_caravan['gauge_id'].isin(basins_hysets_list)
count = matching_catchments_.sum()

matching_catchments=df_static[basin_hysets_caravan['gauge_id'].isin(basins_hysets_list)]
#%%
for ii in range(0,len(basins_hysets_list)):
    if basins_hysets_list[ii] in matching_catchments['gauge_id'].values:
        match=matching_catchments[matching_catchments['gauge_id'] == basins_hysets_list[ii]]
        df_temp=pd.concat((match.reset_index(drop=True),total_catchments.iloc[[ii]].reset_index(drop=True)),axis=1)
        if ii==0:
            df_all=df_temp
        else:
            df_all=pd.concat((df_temp,df_all),axis=0)
#%%
#save data
df_all.to_csv('catchments_stats_selected_all.csv',index=None)       

