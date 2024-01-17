"""
A test for checking the feasiblity of the world model with demand_exeff change
"""


#%%
import sys
import os
import logging

sys.path.append(
    os.path.abspath('.')
)

regions_map = {
   "r.A": "Europe28",
   "r.B": "United States",
   "r.C": "China",
   "r.D": "Africa",
   "r.E": "Middle East",
   "r.F": "India",
   "r.G": "Australia",
   "r.H": "SouthEast Asia",
}
#%%
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


from esm import Model
import pandas as pd
from copy import deepcopy
import json
class DemandTest:
    def __init__(self,scenario,obj,exclude_regions=None):

        self.model = Model(f'case_studies/World/sets.xlsx')
        self.model.read_clusters(f'case_studies/World/clusters.xlsx')
        print('IMPORT SETS: PASSED')


        self.scenario = scenario
        self.obj = obj
        self.log = {}

        regions = deepcopy(self.model.Base.Regions)

        if exclude_regions is not None:
            regions = list(set(regions).difference(set(exclude_regions)))

        print(f'REGIONS TO TEST: {regions}')

        list(map(self.create_region_demand,regions))

        with open("DemandTest.json","w") as fp:
            json.dump(self.log,fp)



    def create_region_demand(self,region):
        Demand = pd.ExcelFile(
            f'case_studies/{regions_map[region]}/{self.scenario}/Inputs/Demand.xlsx'
            )

        Demand_exeff = pd.ExcelFile(
            f'case_studies/World/{self.scenario}/Inputs/Demand_exeff.xlsx'
        )

        self.file = f'Demand_exeff_{region}.xlsx'

        with pd.ExcelWriter(f'case_studies/World/{self.scenario}/Inputs/{self.file}') as file:
            for sheet in Demand.sheet_names:
                exeff = Demand_exeff.parse(sheet,index_col=[0,1,2],header=0)
                demand = Demand.parse(sheet,index_col=[0,1,2],header=0)

                exeff.loc[demand.index,demand.columns] = demand.values

                exeff.to_excel(file,sheet_name=sheet)

        print(f'FILE CREATATION: {region} PASSED.')

        self.run_test()

    def run_test(self):
        self.model.Base._model_generation(self.obj)
        print('MODEL GENERATION: PASSED')
        self.model.read_input_excels(f'case_studies/World/{self.scenario}/Inputs',Demand=self.file)
        self.model.Base._data_assigment()
        self.model.Base._model_completion()
        self.model.Base._model_run('GUROBI')

        if hasattr(self.model.Base,'results'):
            print(f"MODEL RUN: {self.file} CASE PASSED.")
            delattr(self.model.Base,'results')
            self.log[self.file] = 'passed'
        else:
            print("MODEL RUN: {self.file} CASE FAILED.")
            self.log[self.file] = 'failed'



if __name__ == "__main__":
    test = DemandTest('FLS','cost_discount',['r.A','r.C','r.H','r.D'])





# %%
