#%%
import pandas as pd
import os

folders = ['ESM_Italy_10/case_studies/Italy24','ESM_Italy_7/case_studies/Italy24','ESM_Italy_13/case_studies/Italy24','ESM_Italy_17/case_studies/Italy24',]

matrix = 'cu'
row = 'c_in'
data_file = "TechnologyData.xlsx"

new_data = "new_params.xlsx"
header = ('s.hydrogen','t.electrolyzer','electrolyzer')
          
#%%
for folder in folders:
    clusters = pd.read_excel(f"{folder}/Time_clusters.xlsx",index_col=[0])[matrix].to_frame()
    case_studies = os.listdir(folder + '/scenarios')

    for case_study in case_studies:
        if case_study != 'results':
            tech_data = pd.read_excel(f"{folder}/scenarios/{case_study}/inputs/{data_file}",index_col=[0,1,2],header=[0,1,2], sheet_name=None)

            for sheet,data in tech_data.items():
                cluster =sheet.split('.')[-1]
                year = clusters.query(f"{matrix}==@cluster").index[0]
                
                data.loc[(matrix,row,slice(None)),header] = pd.read_excel(new_data,index_col=[0]).loc[year,'new value']
    
            # Save new data
            with pd.ExcelWriter(f"{folder}/scenarios/{case_study}/inputs/new_data.xlsx") as writer:
                for sheet,data in tech_data.items():
                    data.to_excel(writer,sheet_name=sheet)

# %%
