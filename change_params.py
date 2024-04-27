#%%
import pandas as pd
import os

folders = ['ESM_Italy_10/case_studies/Italy24']#,'ESM_Italy_7/case_studies/Italy24','ESM_Italy_13/case_studies/Italy24','ESM_Italy_17/case_studies/Italy24',]

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
            with pd.ExcelWriter(f"{folder}/scenarios/{case_study}/inputs/{data_file}") as writer:
                for sheet,data in tech_data.items():
                    data.to_excel(writer,sheet_name=sheet)

#%%
for folder in folders:
    clusters = pd.read_excel(f"{folder}/Time_clusters.xlsx",index_col=[0])[matrix].to_frame()
    case_studies = os.listdir(folder + '/scenarios')

    data_dict = {}
    for case_study in case_studies:
        if case_study != 'results':
            tech_data = pd.read_excel(f"{folder}/scenarios/{case_study}/inputs/{data_file}",index_col=[0,1,2],header=[0,1,2], sheet_name=None)

            data_dict[case_study] = pd.DataFrame()
            
            for sheet,data in tech_data.items():
                cluster =sheet.split('.')[-1]
                years = clusters.query(f"{matrix}==@cluster").index
                
                for tech in data.columns.get_level_values(-1):
                    df = pd.DataFrame()
                    for y in years:
                        df = pd.concat([df, data.loc[(matrix,row,slice(None)),(slice(None),slice(None),tech)]],axis=0)

                    df.index = years
                    df.columns = [tech]

                    data_dict[case_study] = pd.concat([data_dict[case_study],df],axis=1)
                    
            data_dict[case_study] = data_dict[case_study].fillna(0)
            data_dict[case_study] = data_dict[case_study].groupby(level=[0],axis=1).sum()

            # Save new data
            with pd.ExcelWriter("inv_costs.xlsx") as writer:
                for sheet,data in data_dict.items():
                    data.to_excel(writer,sheet_name=sheet)


# %%
import plotly.express as px 
inv_costs = pd.read_excel("inv_costs.xlsx",index_col=[0],header=[0,1,2], sheet_name='a.1_STEPS_res').stack().stack().stack().to_frame()
inv_costs.columns = ['value']
inv_costs/=1000
inv_costs.index.names = ['year','name','tech','sector']
inv_costs.reset_index(inplace=True) 

years = range(2020,2051,10)
techs = ['Coal','Natural gas','Nuclear','Off-shore wind','On-shore wind','PV','Electrolyzers','Steam reformers']

tech_colors_map = {
    'Coal':'#F44336',
    'Natural gas':'#FF9800',
    'Nuclear':'#AD1457',
    'Off-shore wind':'#00b4d8',
    'On-shore wind':"#0077b6",
    'PV':'#FFC107',
    'Electrolyzers':'#83C5BE',
    'Steam reformers':'#0081A7'
    }

inv_costs = inv_costs.query('year in @years & tech in @techs')

# inv_costs = inv_costs.sort_values('tech', key=lambda x: x.map({tech: i for i, tech in enumerate(techs)}))


fig = px.line(inv_costs, x='year', y='value', color='tech', facet_col='sector', color_discrete_map=tech_colors_map,facet_col_spacing=0.1)

fig.update_layout(
    {
        'template': 'seaborn',
        'font': {'family':'Helvetica','size': 14},
        'xaxis': {'title': '','tickangle':-45,},
        'xaxis2': {'title': '','tickangle':-45,},
        'yaxis': {'title': '€/kW','range':[0,4000],'dtick': 1000},
        'legend': {'title': None},
    },
)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.write_html('Plots/inv_costs.html',auto_open=True)

# %%
