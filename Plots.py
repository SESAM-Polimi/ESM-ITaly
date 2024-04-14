#%%
import pandas as pd
import plotly.express as px

results_folder = 'Final results'
aggregation = pd.read_excel('PowerBi\sets_aggregation.xlsx',sheet_name=None)

#%% parse E
E = pd.read_csv(f'{results_folder}/E.csv')
E.columns = ['SCENARIOS','esm_SECTORS','YEARS','REGIONS','esm_FLOWS','VALUE']
E = E.drop('REGIONS',axis=1)

#%% parse U
U = pd.read_csv(f'{results_folder}/U.csv')
U.columns = ['SCENARIOS','REGIONS from','esm_SECTORS','YEARS','REGIONS to','esm_FLOWS','VALUE']
U = U.drop(['REGIONS from','REGIONS to'],axis=1)

#%%
def df_cols_to_dict(
        df,
        key,
        value,
    ):

    dict_from_df_cols = dict(zip(df[key], df[value]))
    return dict_from_df_cols

def remap_col(
        df,
        set_map,
        new_aggr,
        df_col,
    ):

    if not new_aggr:
        return df
    else:
        dict_cols = df_cols_to_dict(set_map, set_map.columns[0], new_aggr)
        df[df_col] = df[df_col].map(dict_cols)
        return df

def UE(
        U,
        E,
    ):

    
    


#%% Figures
def figure_1(
        path,
        df,
        set_map,
        scenario,
        flow=None,
        year=None,
        auto_open=True,
    ):

    df = remap_col(df, set_map['esm_FLOWS'], flow, 'esm_FLOWS')
    df = remap_col(df, set_map['YEARS'], year, 'YEARS')
    df = df.query("SCENARIOS==@scenario")
    df.set_index(["esm_FLOWS",'YEARS'], inplace=True)
    df.drop(['SCENARIOS','esm_SECTORS'], axis=1, inplace=True)
    df = df.groupby(["esm_FLOWS",'YEARS']).sum().reset_index()

    fig = px.bar(df, x='YEARS', y='VALUE', color='esm_FLOWS')
    fig.write_html(path,auto_open=auto_open)


#%%
figure_1(
    path='Plots/Figure_1.html',
    df=E,
    set_map=aggregation,
    scenario='a.1_STEPS_res',
    flow='Figure 1',
    year='Figure 1',
    )

# %%
