# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 22:21:07 2022

@author: loren
"""

#%%
import pandas as pd

paths = {
    "10y": {
        "path": r"C:\Users\loren\Documents\GitHub\SESAM\ESM-Italy\ESM_Italy_10\case_studies\Italy24\scenarios\results",
        "scenarios": {
            "a.1_STEPS_res":"a.1_STEPS_res",
            "a.2.0_STEPS_inv":"a.2.0_STEPS_inv",
            "a.2.1_STEPS_inv_RESconst":"a.2.1_STEPS_inv_RESconst",
            "a.2.2_STEPS_inv_RES50":"a.2.2_STEPS_inv_RES50",
            "a.2.3_STEPS_inv_CTgrow":"a.2.3_STEPS_inv_CTgrow",
            "a.2.4_STEPS_inv_ELZ50":"a.2.4_STEPS_inv_ELZ50",
            "b.0_NUC":"b.0_NUC",
            "c.NZE":"c.NZE",
            "d.0_NZE-NUC":"d.0_NZE-NUC",
            }
        },
    "7y": {
        "path": r"C:\Users\loren\Documents\GitHub\SESAM\ESM-Italy\ESM_Italy_7\case_studies\Italy24\scenarios\results",
        "scenarios": {
            "b.1_NUC_7":"b.1_NUC_7",
            "d.1_NZE-NUC_7":"d.1_NZE-NUC_7",
            }
        },
    "13y": {
        "path": r"C:\Users\loren\Documents\GitHub\SESAM\ESM-Italy\ESM_Italy_13\case_studies\Italy24\scenarios\results",
        "scenarios": {
            "b.2_NUC_13":"b.2_NUC_13",
            "d.2_NZE-NUC_13":"d.2_NZE-NUC_13",
            }
        },
    "17y": {
        "path": r"C:\Users\loren\Documents\GitHub\SESAM\ESM-Italy\ESM_Italy_17\case_studies\Italy24\scenarios\results",
        "scenarios": {
            "b.3_NUC_17":"b.3_NUC_17",
            "d.3_NZE-NUC_17":"d.3_NZE-NUC_17",
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
        
#%%