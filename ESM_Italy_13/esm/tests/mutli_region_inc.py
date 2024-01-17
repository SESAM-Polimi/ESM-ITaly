#%%
import os
import sys
import json

import pandas as pd
sys.path.append(
    os.path.abspath('.')
)

from esm import Model

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

all_regions = [*regions_map]

base_regions = all_regions[0:3]
class MultiRegion:
    def __init__(self,scenario,regions,Demand=None,FlowData=None):

        self.counter = 0
        self.scenario = scenario
        self.Demand = Demand
        self.FlowData = FlowData
        self.regions = regions

        print(regions)
        print("***********************")
    def create_model(self,):
        regions = self.regions
        set_file = pd.ExcelFile(
            'case_studies/World/sets.xlsx'
            )

        with pd.ExcelWriter(f'case_studies/World/sets_{self.counter}.xlsx') as file:
            for sheet in set_file.sheet_names:
                if sheet == 'Regions':

                    data = set_file.parse(
                                    sheet,
                                    index_col=0
                                ).loc[regions,:]
                else:
                    data = set_file.parse(
                        sheet,
                        index_col=0
                    )

                data.to_excel(file,sheet)

        model = Model(f'case_studies/World/sets_{self.counter}.xlsx')
        model.read_clusters('case_studies/World/clusters.xlsx')
        model.create_input_excels(
            'case_studies/World/raw'
        )

        self._integrate_files([regions_map[rr] for rr in regions])
        model.Base._model_generation(obj_fun='cost_discount')
        model.read_input_excels(
            f'case_studies/World/{self.scenario}/Inputs',
            Demand= self.Demand,
            FlowData= self.FlowData,
        )
        model.Base._data_assigment()
        model.Base._model_completion()
        print("SEND MODEL TO SOLVER.")
        model.Base._model_run(
            solver = "GUROBI",
            verbose = True,
            )
        self.model = model

        msg = dict(
                region = regions,
                Demand = self.Demand,
                FlowData = self.FlowData,
                )
        if hasattr(model.Base,'results'):
            msg["Status"] = 'passed'
        else:
            msg["Status"] = "failed"

        print(msg)
        return msg







    def _integrate_files(self,regions):

        region_by_sheet = {
            "Availability": dict(index_col=[0],header=[0,1,2,3]),
            "NewCapacityMax": dict(index_col=[0],header=[0,1,]),
            "NewCapacityMin": dict(index_col=[0],header=[0,1,]),
            "OperativeCapacityMax": dict(index_col=[0],header=[0,1,]),
            "OperativeCapacityMin": dict(index_col=[0],header=[0,1,]),
            "TechnologyData": dict(index_col=[0,1,2],header=[0,1,2,]),
            "TechProductionMix": dict(index_col=[0],header=[0,1,]),
            "TechProductionMixMin": dict(index_col=[0],header=[0,1,]),
            "TechProductionMixMax": dict(index_col=[0],header=[0,1,]),
            "TotalEmission": dict(index_col=[0],header=[0,]),
        }

        append = {
            "Demand":dict(index_col=[0,1,2],header=[0,]),
            "Demand_exeff":dict(index_col=[0,1,2],header=[0,]),
            "DemandProfiles": dict(index_col=[0,],header=[0,1,2]),
            "MoneyRates": dict(index_col=[0,],header=[0,1,]),
            }


        scenario = self.scenario
        world = {}
        for file,config in region_by_sheet.items():
            world[file] = {}
            for reg in regions:
                file_path = f'case_studies/{reg}/{scenario}/Inputs/{file}.xlsx'
                for sheet in pd.ExcelFile(f'case_studies/{reg}/{scenario}/Inputs/{file}.xlsx').sheet_names:
                    world[file][sheet] = pd.read_excel(
                        io =file_path,
                        sheet_name = sheet,
                        **config
                    )




        for file,config in append.items():
            world[file] = {}

            try:
                excel = pd.ExcelFile(
                    f'case_studies/World/raw/{file}.xlsx'
                    )
            except:
                continue

            for sheet in excel.sheet_names:
                df = excel.parse(sheet_name=sheet,**config)

                for reg in regions:
                    path = f'case_studies/{reg}/{scenario}/Inputs/{file}.xlsx'
                    to_assign = pd.read_excel(
                        path,
                        sheet_name = sheet,
                        **config,
                    )

                    df.loc[to_assign.index,to_assign.columns
                        ] = to_assign.loc[
                            to_assign.index,
                            to_assign.columns
                            ].values

                world[file][sheet] = df





        for file,sheets in world.items():

            write_xl = pd.ExcelWriter(f'case_studies/World/{scenario}/Inputs/{file}.xlsx')

            for sheet,df  in sheets.items():
                df.to_excel(write_xl, sheet_name=sheet)

            write_xl.save()


if __name__ == "__main__":
    log = {}
    regions = ['r.A', 'r.F', 'r.D', 'r.C']#base_regions
    model = MultiRegion(scenario='FLS',regions=regions,FlowData='FlowData_Trades.xlsx')
    log[0] = model.create_model()


# %%
