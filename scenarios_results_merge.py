# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 22:21:07 2022

@author: loren
"""

#%%
import pandas as pd

paths = {
    "all": {
        "path": r"C:\Users\loren\Documents\GitHub\SESAM\ESM-Italy\ESM_Italy_10\case_studies\Italy24\scenarios\Risultati_Tesi\0results_ALL",
        "scenarios": {
            "STEPS_newcap":"a.1_STEPS_res",
            "STEPS_invcap":"a.2.0_STEPS_inv",
            "STEPS_NUC":"b.0_NUC",
            "NetZero2050":"c.NZE",
            "NetZero2050_NUC":"d.0_NZE-NUC",
            }
        },
    "CT": {
        "path": r"C:\Users\loren\Documents\GitHub\SESAM\ESM-Italy\ESM_Italy_10\case_studies\Italy24\scenarios\Risultati_Tesi\0results_ALL_increasingCT",
        "scenarios": {
            "STEPS_invcap":"a.2.3_STEPS_inv_CTgrow",
            }
        },
    "cd_sens_7": {
        "path": r"C:\Users\loren\Documents\GitHub\SESAM\ESM-Italy\ESM_Italy_10\case_studies\Italy24\scenarios\Risultati_Tesi\0results_cd_sens\7years",
        "scenarios": {
            "STEPS_NUC":"b.1_NUC_7"
            }
        },                   
    "cd_sens_13": {
        "path": r"C:\Users\loren\Documents\GitHub\SESAM\ESM-Italy\ESM_Italy_10\case_studies\Italy24\scenarios\Risultati_Tesi\0results_cd_sens\13years",
        "scenarios": {
            "STEPS_NUC":"b.2_NUC_13"
            }
        },
    "cd_sens_17": {
        "path": r"C:\Users\loren\Documents\GitHub\SESAM\ESM-Italy\ESM_Italy_10\case_studies\Italy24\scenarios\Risultati_Tesi\0results_cd_sens\17years",
        "scenarios": {
            "STEPS_NUC":"b.3_NUC_17"
            }
        },
}
    
save = r"C:\Users\loren\Documents\GitHub\SESAM\ESM-Italy\Final results"

#%%
files = {
          "BV_E\\f.eCO2": [0,1,2,3,4],
          "BV_U\\f.eCO2": [0,1,2,3,4,5],
          "BP": [0,1,2,3,4],
          "BU": [0,1,2,3,4],
          "BV": [0,1,2,3,4],
          "cap_d": [0,1,2,3],
          "cap_o": [0,1,2,3],
          "cap_n": [0,1,2,3],
          "CU": [0,1,2,3,4],
          "CU_mr": [0,1,2,3,4],
          "E": [0,1,2,3,4],
          "qh": [0,1,2,3,4],
          "qy": [0,1,2,3],
          "soc": [0,1,2,3,4],
          "U": [0,1,2,3,4,5],
          "V": [0,1,2,3,4,5],
          "xh": [0,1,2,3,4],
          "xy": [0,1,2,3],
        }

#%%
for file,ic in files.items():
    merged = pd.DataFrame()
    for path in paths:
        
        data = pd.read_csv(f"{paths[path]['path']}\{file}.csv", index_col=ic)
       
        scenarios = list(paths[path]["scenarios"].keys())
        data = data.loc[scenarios,:]

        data.index.names = ic
        data.reset_index(inplace=True)

        data[0] = data[0].map(paths[path]["scenarios"])
        data.set_index(list(data.columns[:-1]), inplace=True)
        data.columns = ['value']
        data.index.names = [None] * len(data.index.names)

            
        merged = pd.concat([
            merged,
            data], axis=0)
    
    merged.to_csv(f"{save}\{file}.csv")
        
        
    
    
    

























# %%
