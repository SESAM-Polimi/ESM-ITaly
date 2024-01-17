
#%%

if __name__ == '__main__':

    from esm import set_log_verbosity
    set_log_verbosity('critical')

    from esm import Model
    from esm.utils import cvxpyModified as cm
    from esm.utils import constants
    from esm import Plots

    import pandas as pd
    import numpy as np
    import cvxpy as cp


    MOD = Model(f'case_studies/1_tests/sets.xlsx',integrated_model = False,)

    cluster_gen = False
    if cluster_gen:
        MOD.generate_clusters_excel(f'case_studies/1_tests/input_clusters_raw/clusters.xlsx')
    else:
        pass

    file_gen = False
    if file_gen:
        MOD.create_input_excels(f'case_studies/1_tests/input_excels_raw')
        MOD.to_excel(path=f'case_studies/1_tests/sets_code.xlsx',item='sets')
    else:
        pass

    # model generation and run
    MOD.read_clusters(f'case_studies/1_tests/clusters.xlsx')
    MOD.read_input_excels(f'case_studies/1_tests/input_excels')
    MOD.Base._model_generation()
    MOD.Base._data_assigment()
    MOD.Base._model_completion()
    MOD.Base._model_run(solver=cp.GUROBI)

    results = Plots()
    results.save_path = f'case_studies/1_tests/plots'

#%%

    if hasattr(MOD.Base, "results"):
        results.upload_from_model(model=MOD,scenario='s1',color='black')

    results.plot_hourly_techs(
        path=f"{results.save_path}/Power_hourly.html",
        regions=results.Regions,
        hourly_techs=results._ids.sectors_techs["s.elect"] + results._ids.sectors_techs["s.elect_storage"],
        scenarios=results.scenarios,
        year=2020,
        kind="bar",
    )

    results.plot_total_techs_production(
        path=f"{results.save_path}/Power_Yearly_Production.html",
        regions=results.Regions,
        techs=results._ids.sectors_techs["s.elect"] + results._ids.sectors_techs["s.elect_storage"],
        scenarios=results.scenarios,
        period="all",
        kind="bar",
    )

    results.plot_sector_capacity(
        path=f"{results.save_path}/Power_Capacity_new-dis.html",
        regions=results.Regions,
        sectors=["s.elect"],
        scenarios=results.scenarios,
        kind="bar",
        period="all",
    )

    results.plot_sector_capacity(
        path=f"{results.save_path}/Power_Capacity_ope.html",
        regions=results.Regions,
        sectors=["s.elect"],
        scenarios=results.scenarios,
        kind="bar",
        period="all",
        to_show="cap_o",
    )



# %%
from esm.utils import cvxpyModified as cm
self=MOD.Base
p=self.p
y=2025
# %%
