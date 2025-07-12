# -*- coding: utf-8 -*-
'''
Code for input (csv) file generation for HYSETS (Arsenault et al., 2024) catchment(s).
Example below is for 02GA014 gauge station in CA.

The data comproses of dynamic features (obtained from Daymet v4; Thornton et al., 2022) and static features
obtained from Caravan (Kratzert, 2022)  and original HYSETS dataset

The used datasets are obtained from (pre-processed):
    SaveStaticfeatures_v (static features)
    DataIntegration_DaymetStreamflow_v (dynamic features)

References:

    Arsenault, R., Brissette, F., Martel, J.-L., Troin, M., Lévesque, G., Davidson-Chaput, J., … Poulin, A. (2024, October 1).
             HYSETS - A 14425 watershed Hydrometeorological Sandbox over North America. https://doi.org/10.17605/OSF.IO/RPC3W
             
    Kratzert, F. (2022). Caravan ‐ A global community dataset for large‐sample hydrology. Zenodo. https://doi.org/10.5281/zenodo.7540792
    
    Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S-C. Kao, and B.E. Wilson. 2022.
    Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2129
'''
#%%
#importing necessary libraries
import numpy as np
import pandas as pd
import os
#%% Change directory
os.chdir('F:\msjahangir\CVAE-HYSETS\Data') #change this
#%%
#read static data
static_df=pd.read_csv('catchments_stats_model_v1.csv') #change this
static_df=static_df[static_df['Source']=='HYDAT']
static_df=static_df.drop(columns=['Source'])
id_static=static_df[['Official_ID']]
static_df_s=static_df.drop(columns=['Official_ID'])
static_col=static_df_s.columns.to_list()
#%%
#path to save independent catchments' data
save_path_results='Generaldata_Train_Test_Flood'
if not os.path.isdir(save_path_results):
    os.mkdir(save_path_results)
#%%
# save all the data to a df
kk=0
list_id=[]
filenames = ['02GA014'] #list of catchments
for ii in range(0,len(filenames)):
    basin_id=ii
    id_=id_static.iloc[ii,0]
    filename=r'%s_DI.csv'%(filenames[ii])
    df_data=pd.read_csv(filename,header=0)
    df_data['basin_id']=kk
    
    
    
    
    
    df_data_temp=df_data
    for jj in range(0,len(static_col)):
        df_data_temp[static_col[jj]]=static_df_s.iloc[np.where(np.asarray(id_static.iloc[:,0])==id_)[0][0],jj]
        
    
    df_data_temp_train=df_data_temp.iloc[:int(0.85*len(df_data_temp))]
    df_data_temp_test=df_data_temp.iloc[int(0.85*len(df_data_temp)):]
    
    df_data_temp_train.to_csv(r'%s/Regionaltrain_%d_%s.csv'%(save_path_results,kk,id_),index=None)
    df_data_temp_test.to_csv(r'%s/Regionaltest_%d_%s.csv'%(save_path_results,kk,id_),index=None)
    
    
    if ii==0:
        df_total_train=df_data_temp_train
        df_total_test=df_data_temp_test
    else:
        df_total_train=pd.concat([df_total_train,df_data_temp_train])
        df_total_test=pd.concat([df_total_test,df_data_temp_test])
    kk=kk+1  
        
#%%
# save the df to csv
df_total_train.to_csv('Alldata_Train_st_%s.csv'%(filenames[0]),index=None)
df_total_test.to_csv('Alldata_Test_st_%s.csv'%(filenames[0]),index=None)