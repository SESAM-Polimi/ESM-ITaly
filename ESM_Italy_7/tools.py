#%%
import pandas as pd
import numpy as np

#%%
regions = ["Middle East - Copia"]
scenarios = ["FLS","BLS"]

path = r"C:\Users\loren\Documents\GitHub\ESM\case_studies"

filenames = {
    "cap_o": ["OperativeCapacityMax","OperativeCapacityMin"],
    "cap_n": ["NewCapacityMax","NewCapacityMin"],
    }



#%%
def plot(results, total_line=True, y_scope=range(1990,2051,1)):

    # Plotting
    if results:
        print("\noutput plot...", end='')


        for result in results.values():
            regions = result.Regions
            scenarios = result.scenarios


            # costs
            result.plot_costs_by_items(
                path = f"{result.save_path}/Cost_by_items.html",
                regions=regions,
                scenarios=scenarios,
                unit= "M $",
                period = y_scope,
            )

            result.plot_costs_by_techs(
                path = f"{result.save_path}/Cost_by_techs.html",
                regions=regions,
                scenarios=scenarios,
                unit= "M $",
                period = y_scope,
                techs=result.Technologies
            )
            # Power Sector
            techs = (
                result._ids.sectors_techs["s.elect"]
                + result._ids.sectors_techs["s.elect_storage"]
                + ["t.bev_storage_grid"]
            )

            result.plot_mix(
                path = f"{result.save_path}/Power_production_mix.html",
                regions = regions,
                scenarios=scenarios,
                sectors='s.elect',
                period = y_scope,
            )

            result.plot_mix(
                path = f"{result.save_path}/Power_operative_capacity_mix.html",
                regions = regions,
                scenarios=scenarios,
                sectors='s.elect',
                period = y_scope,
                what='cap_o'
            )

            result.plot_hourly_techs(
                path=f"{result.save_path}/Power_hourly_2018.html",
                regions=regions,
                hourly_techs=techs,
                scenarios=scenarios,
                year=2018,
                kind="area",
                total_line = total_line,
            )

            result.plot_hourly_techs(
                path=f"{result.save_path}/Power_hourly_2040.html",
                regions=regions,
                hourly_techs=techs,
                scenarios=scenarios,
                year=2040,
                kind="area",
                total_line = total_line,
            )

            result.plot_sector_capacity(
                path=f"{result.save_path}/Power_Capacity_new-dis.html",
                regions=regions,
                sectors=["s.elect"],
                scenarios=scenarios,
                kind="bar",
                period=y_scope,
                total_line = total_line,
            )

            result.plot_sector_capacity(
                path=f"{result.save_path}/Power_Capacity_ope.html",
                regions=regions,
                sectors=["s.elect"],
                scenarios=scenarios,
                kind="area",
                period=y_scope,
                to_show="cap_o",
                total_line = total_line,
            )
            _techs = (
                result._ids.sectors_techs["s.elect"]
                + result._ids.sectors_techs["s.elect_storage"]
            )
            result.plot_total_techs_production(
                path=f"{result.save_path}/Power_Yearly_Production.html",
                regions=regions,
                techs=_techs,
                scenarios=scenarios,
                period=y_scope,
                kind="area",
                total_line = total_line,
                tot_abs=True,
            )

            # storages
            result.plot_sector_capacity(
                path=f"{result.save_path}/Power_Capacity_Batteries_ope.html",
                regions=regions,
                sectors=["s.elect_storage","s.elect_storage_transp"],
                scenarios=scenarios,
                kind="area",
                period=y_scope,
                to_show="cap_o",
                total_line = total_line,
            )

            result.plot_sector_capacity(
                path=f"{result.save_path}/Power_Capacity_Batteries_new-dis.html",
                regions=regions,
                sectors=["s.elect_storage","s.elect_storage_transp"],
                scenarios=scenarios,
                kind="bar",
                period="all",
                total_line = total_line,
            )

            result.plot_hourly_techs(
                path=f"{result.save_path}/Power_hourly_storage_bev_2040.html",
                regions=regions,
                hourly_techs=["t.bev_storage_grid","t.bev_storage_bev"],
                scenarios=scenarios,
                year=2040,
                kind="area",
                total_line = total_line,
            )

            result.plot_hourly_techs(
                path=f"{result.save_path}/Power_hourly_storage_2040.html",
                regions=regions,
                hourly_techs=["t.elect_storage",],
                scenarios=scenarios,
                year=2040,
                kind="area",
                total_line = total_line,
            )

            # Transport
            result.plot_sector_capacity(
                path=f"{result.save_path}/Transport_Capacity_new-dis.html",
                regions=regions,
                sectors=["s.transport"],
                scenarios=scenarios,
                kind="bar",
                period=y_scope,
                total_line = total_line,
            )

            result.plot_mix(
                path = f"{result.save_path}/Transport_service_mix.html",
                regions = regions,
                scenarios=scenarios,
                sectors='s.transport',
                period = y_scope,
            )


            result.plot_sector_capacity(
                path=f"{result.save_path}/Transport_Capacity_ope.html",
                regions=regions,
                sectors=["s.transport"],
                scenarios=scenarios,
                to_show=["cap_o"],
                kind="area",
                period=y_scope,
                total_line = total_line,
            )

            result.plot_total_techs_production(
                path=f"{result.save_path}/Transport_Yearly_Production.html",
                regions=regions,
                techs=result._ids.sectors_techs["s.transport"],
                scenarios=scenarios,
                kind="area",
                period=y_scope,
                total_line = total_line,
            )

            # Hydrogen
            techs = result._ids.sectors_techs["s.hydrogen"] + ["t.fcv_storage_grid"]
            result.plot_hourly_techs(
                path=f"{result.save_path}/Hydrogen_hourly.html",
                regions=regions,
                hourly_techs=techs,
                scenarios=scenarios,
                year=2040,
                kind="area",
                total_line = total_line,
            )

            result.plot_mix(
                path = f"{result.save_path}/Blender_mix.html",
                regions = regions,
                scenarios=scenarios,
                sectors='s.blender',
                period = y_scope,
            )

            result.plot_sector_capacity(
                path=f"{result.save_path}/Hydrogen_Capacity_new-dis.html",
                regions=regions,
                sectors=["s.hydrogen"],  # "s.hydrogen_storage_transp"],
                scenarios=scenarios,
                kind="bar",
                period=y_scope,
                total_line = total_line,
            )

            result.plot_sector_capacity(
                path=f"{result.save_path}/Hydrogen_Capacity_operative.html",
                regions=regions,
                sectors=["s.hydrogen"],
                to_show=["cap_o"],
                scenarios=scenarios,
                kind="area",
                period=y_scope,
                total_line = total_line,
            )

            result.plot_total_techs_production(
                path=f"{result.save_path}/Hydrogen_Yearly_Production.html",
                regions=regions,
                techs=['t.electrolyzer','t.steam_reform'],
                scenarios=scenarios,
                period=y_scope,
                kind="area",
                total_line = total_line,
            )

            # Housing
            result.plot_sector_capacity(
                path=f"{result.save_path}/Housing_Capacity_operative.html",
                regions=regions,
                sectors=["s.house"],
                to_show=["cap_o"],
                scenarios=scenarios,
                kind="area",
                period=y_scope,
                total_line = total_line,
            )

            result.plot_sector_capacity(
                path=f"{result.save_path}/Housing_Capacity_new-dis.html",
                regions=regions,
                sectors=["s.house"],
                scenarios=scenarios,
                kind="bar",
                period=y_scope,
                total_line = total_line,
            )

            # Fuel Productions
            result.plot_flow_production(
                path=f"{result.save_path}/Fuels_Production.html",
                regions=regions,
                products=["f.coal", "f.oil", "f.natgas", "f.elect", "f.uranium", "f.hydrogen"],
                scenarios=scenarios,
                kind="area",
                period=y_scope,
                total_line = total_line,
            )

            # Fuel consumption
            for fuel in ['f.blended_gas','f.natgas','f.coal','f.oil','f.hydrogen']:
                result.plot_use(
                    path=f"{result.save_path}/{fuel}_consumption.html",
                    regions=regions,
                    scenarios=scenarios,
                    flow=fuel,
                    period = y_scope,
                    aggregation = consumption_aggregation,
                )

            # Emissions
            result.emissions_by_flow(
                path=f"{result.save_path}/Emission_techs.html",
                regions=regions,
                scenarios=scenarios,
                emission=["f.eCO2"],
                kind="area",
                period=y_scope,
                total_line = total_line,
            )

            result.emissions_by_flow(
                path=f"{result.save_path}/Emissions_sectors.html",
                regions=regions,
                scenarios=scenarios,
                emission=["f.eCO2"],
                kind="area",
                period=y_scope,
                by='sectors',
                aggregation=sector_aggregation,
                total_line = total_line,
            )

            result.emissions_by_flow(
                path=f"{result.save_path}/Emissions_fuels.html",
                regions=regions,
                scenarios=scenarios,
                emission=["f.eCO2"],
                kind="area",
                period=y_scope,
                by='fuels',
                aggregation=fuel_aggregation,
                total_line = total_line,
            )


        print(" DONE")

    else:
        print("problem infeasible...")


#%%
fuel_aggregation = {
    "Natural Gas": {
        "items": [
            "f.natgas_extract",
            "f.natgas",
        ],
        "color": "rgb( 86,108,140 )",
    },

    "Coal": {
        "items": [
            "f.coal_extract",
            "f.coal",
        ],
        "color": "rgb( 0,0,0 )",
    },
    "Oil": {
        "items": [
            "f.oil_extract",
            "f.oil",
        ],
        "color": "rgb( 121,43,41 )",
    },
    "Uranium": {
        "items": [
            "f.uranium_extract",
            "f.uranium",
        ],
        "color": "rgb( 192,80,150 )",
    },
    "Biomass": {
        "items": [
            "f.biomass",
        ],
        "color": "rgb( 123,96,83 )",
    },
    "Blended Gas": {
        'items':[
            'f.blended_gas'
        ],
        'color': "#196ce6"
    },

    "Rest": {
        "items": [
            "f.elect",
            "f.elect_bev",
            "f.house",
            "f.transport",
            "f.hydrogen",
            "f.hydrogen_fcv",
            "s.lucf",
        ],
        "color": "rgb( 193,112,3 )",
    }


}

sector_aggregation = {
    "Fuel Extraction": {
        "items": [
            "s.coal_extract",
            "s.oil_extract",
            "s.natgas_extract",
            "s.uranium_extract",
            "s.forestry",
        ],
        "color": "rgb( 86,108,140 )",
    },
    "Refinary": {
        "items": ["s.coal", "s.oil", "s.natgas", "s.uranium",],
        "color": "rgb( 129,149,177 )",
    },
    "Electricity": {"items": ["s.elect"], "color": "rgb( 242,98,0 )",},
    "Transmission": {"items": ["s.elect_trans"], "color": "blue",},
    "Transport": {"items": ["s.transport"], "color": "rgb( 193,112,3 )",},
    "Housing": {"items": ["s.house"], "color": "rgb( 209,58,54 )",},
    "Land Use": {"items": ["s.lucf"], "color": "rgb( 123,96,83 )"},
    "Rest": {
        "items": [
            "s.food",
            "s.elec_equipment",
            "s.manufacturing",
            "s.chemical",
            "s.metal",
            "s.water_transport",
            "s.air_transport",
            "s.construction",
            "s.agriculture",
            "s.mining",
            "s.cement",
            "s.steel",
            "s.icev_manufacturing",
            "s.hev_manufacturing",
            "s.bev_manufacturing",
            "s.fcev_manufacturing",
            "s.other_services",
            "s.rail_transport",
            "s.pipeline",
            "s.waste_treat",
            "s.manufac_batt",
            "s.manufac_electrolyzer",
            "s.households",
            "s.government",
            "s.elect_storage",
            "s.elect_storage_transp",
            "s.hydrogen_storage_transp",
            "s.hydrogen_storage",
            "s.hydrogen",
            "s.blender",
        ],
        "color": "rgb( 86,108,140 )",
    },
}

techs_aggregation = {
    "Residential Buildings" : {
        'items': ['t.house','t.house_new'],
        'color': "rgb( 234,67,0 )",
            },

    }


consumption_aggregation = {
    "Fuel Extraction": {
        "items": [
            "t.coal_extract",
            "t.oil_extract",
            "t.natgas_extract",
            "t.uranium_extract",
        ],
        "color": 'rgb( 121,43,41 )'
    },
    "Fuel Refinary": {
        "items": [
            "t.coal",
            "t.oil",
            "t.natgas",
            "t.uranium",
            "t.biomass",
        ],
        "color": 'rgb( 192,80,150 )',
    },
    "Power Production": {
        "items": [
            "t.elect_coal",
            "t.elect_coal_ccs",
            "t.elect_oil",
            "t.elect_oil_ccs",
            "t.elect_natgas",
            "t.elect_natgas_ccs",
            "t.elect_uranium",
            "t.elect_pv",
            "t.elect_wind",
            "t.elect_geothermal",
            "t.elect_hydro",
            "t.elect_waste_biomass",
            "t.elect_waste_biomass_ccs",
        ],
        "color": "rgb( 242,98,0 )",
    },
    "Road Transport": {
        "items": [
            "t.icev",
            "t.hev",
            "t.bev",
            "t.fcev",
        ],
        "color": "rgb( 193,112,3 )",
    },
    "Residential Buidlings":{
        "items": [
            "t.house",
            "t.house_new",
        ],
        "color": 'rgb( 234,67,0 )',
    },
    "Hydrogen Production":{
        "items":[
            "t.steam_reform",
            "t.electrolyzer",
        ],
        "color": 'rgb( 255,204,204 )',
    },
    "Gas Blending":{
        "items": ["t.gas_blender",],
        "color": "rgb( 86,108,140 )"
    },
     "H2 blender":{
        "items": ["t.hydrogen_blender",],
        "color": "rgb( 86,108,140 )"
    },
    "Rest" : {
        "items": [
            "s.food",
            "s.elec_equipment",
            "s.manufacturing",
            "s.chemical",
            "s.metal",
            "s.water_transport",
            "s.air_transport",
            "s.construction",
            "s.agriculture",
            "s.mining",
            "s.cement",
            "s.steel",
            "s.icev_manufacturing",
            "s.hev_manufacturing",
            "s.bev_manufacturing",
            "s.fcev_manufacturing",
            "s.other_services",
            "s.rail_transport",
            "s.pipeline",
            "s.waste_treat",
            "s.manufac_batt",
            "s.manufac_electrolyzer",
            "s.households",
            "s.government",
            "t.elect_storage",
            "t.bev_storage_grid",
            "t.bev_storage_bev",
            "t.fcv_storage_grid",
            "t.fcv_storage_fcv",
            "t.hydrogen_storage",
            # "t.hydrogen_blender",
            "t.lucf",
        ],
        "color": "rgb( 138,171,71 )",
    }

}


#%%
def scenario_warm_up_consistency(result, scenario_to, variable, path, warm_up_years, tolerance):

    if isinstance(result,dict):
        result_variable = result[variable]
    else:
        result_variable = pd.read_csv(fr"{path}\results_fls\{variable}.csv", index_col=[0,1,2,3], header=[0]).loc["FLS","value"].to_frame()
        result_variable = result_variable.unstack(level=[0,1]).droplevel(0,axis=1)

    regions = result_variable.columns.unique(level=0)

    for file in filenames[variable]:

        output = {}

        for region in regions:
            data = pd.read_excel(f"{path}/{scenario_to}/Inputs/{file}.xlsx",
                                 sheet_name=region,
                                 index_col=[0],
                                 header=[0,1])

            if data.index.name:
                    row = data.index.name
                    data.index.name = None
                    data.loc[row,:] = np.nan
                    data = data.sort_index()

            # data.columns = [data.iloc[0,:], data.iloc[1,:]]
            # data = data.loc[data.index.dropna(),:]
            # data.index = [int(i) for i in data.index]

            cols = result_variable.columns.get_level_values(-1)
            if "Max" in file:
                data.loc[warm_up_years,(slice(None),cols)] = (result_variable.loc[warm_up_years,(region,cols)]*(1+tolerance)).values
            elif "Min" in file:
                data.loc[warm_up_years,(slice(None),cols)] = (result_variable.loc[warm_up_years,(region,cols)]*(1-tolerance)).values

            output[region] = data

        with pd.ExcelWriter(f"{path}/{scenario_to}/Inputs/{file}.xlsx") as excel:
            for k,v in output.items():
                v.to_excel(excel, k)


#%%
def aggregated_emission(region,scenario,save_path = 'total_emissions.xlsx'):

    bv_e = pd.read_csv(
        f'case_studies/{region}/results_raw/BV_E/f.eCO2.csv',
        index_col = [0,1,2,3,4],
        header = 0
        ).loc[scenario,:].groupby(level=[1,2]).sum().unstack(-1)

    bv_u = pd.read_csv(
        f'case_studies/{region}/results_raw/BV_U/f.eCO2.csv',
        index_col = [0,1,2,3,4],
        header = 0
    ).loc[scenario,:].groupby(level=[2,3]).sum().unstack(-1)

    file = pd.read_excel(save_path,index_col=0)

    tot_emission = (bv_e + bv_u).droplevel(level=0,axis=1)

    #return file,tot_emission
   # print(file.loc[tot_emission.index,region])
    #print(tot_emission[tot_emission.index,'value'].values)
    file.loc[tot_emission.index,region] = tot_emission.loc[tot_emission.index,'value'].values

    file.to_excel(save_path)

#%%
iea_data_folder = "/Users/mohammadamintahavori/Library/CloudStorage/OneDrive-SharedLibraries-PolitecnicodiMilano/Matteo Vincenzo Rocco - 2021_EY_Energy transition/model/data/iea emissions"
#iea_data_folder = r"C:\Users\loren\Politecnico di Milano\Matteo Vincenzo Rocco - 2021_EY_Energy transition\model\data\iea emissions"
esm_results_folder = "case_studies"

# key = esm_regs, value = region/regions from IEA
regs = {
    "Europe28": "European Union - 28",
    "United States": "United States",
    "Australia":"Australia",
    "India":"India",
    "Middle East":"Middle East",
    "Africa":"Africa",
    "China": "China (People's Republic of China and Hong Kong China)",
    "SouthEast Asia": [
        "Japan",
        "Korea",
        "Malaysia",
        "Philippines",
        "Singapore",
        "Thailand",
        "Indonesia",
    ]

}

# ESM to IEA flows map
flows_to_take = {
        "f.coal_extract": "Coal",
        "f.oil_extract": "Oil",
        "f.natgas_extract": "Natural gas",
        "f.coal": "Coal",
        "f.oil": "Oil",
        "f.natgas": "Natural gas",
        "f.blended_gas": "Natural gas"
    }

allias_regions = {
    "Europe28" : "r.A",
    "United States" : "r.B",
    "China" : "r.C",
    "Africa" : "r.D",
    "Middle East" : "r.E",
    "India" : "r.F",
    "SouthEast Asia" : "r.H",
    "Australia" : "r.G",
}
def get_file(reg,what):
    """Returns the name of the file

    Parameters
    ----------
    reg : str
        region
    what : str
        esm or iea for defining the path

    Returns
    -------
    str
        name of the file
    """
    if what == "iea":
        return f"{iea_data_folder}/CO2 emissions by energy source - {reg}.csv"

    elif what == "esm":
        return f"{esm_results_folder}/{reg}/results_raw"


def read_file(file,what):
    """Reads and returns the files

    Parameters
    ----------
    file : str
        the file as the output of the get_file function
    what : str
        esm or iea

    Returns
    -------
    pd.DataFrame
        postprocessed data
    """
    if what == 'iea':
        try:
            return pd.read_csv(file,index_col=0,header=3,sep=',').drop(["Units","Other"],axis=1)
        except KeyError:
            return pd.read_csv(file,index_col=0,header=3,sep=',').drop(["Units",],axis=1)

    elif what == 'esm':
        # return pd.read_csv(file,index_col=[0,1,2,3],header=0).loc[
        #     (slice(None),slice(None),[*flows_to_take],slice(None))
        # ].rename(flows_to_take,level=2).groupby(level=[2,3]).sum().unstack(level=0).droplevel(level=0,axis=1)

        bv_e = pd.read_csv(
        f'{file}/BV_E/f.eCO2.csv',
        index_col = [0,1,2,3,4],
        header = 0
        ).groupby(level=[2,3,4]).sum()

        bv_u = pd.read_csv(
            f'{file}/BV_U/f.eCO2.csv',
            index_col = [0,1,2,3,4,5],
            header = 0
        ).groupby(level=[3,4,5]).sum()


        bv = (bv_u + bv_e).unstack(-1).droplevel(0,axis=1).rename(flows_to_take,axis=1).groupby(level=0,axis=1).sum()

        return bv


def get_regional_data(world_data,region):
    """gets the regional data from world data

    Parameters
    ----------
    world_data : pd.DataFrame
        The World data
    region : str
        Specific esm region to be sliced from the world data

    Returns
    -------
    pd.DataFrame
        sepcific region data
    """

    allias = allias_regions[region]

    return world_data.loc[(slice(None),allias),:]

#%%
def recalculate_coefficients():

    to_print = {}
    world_file = get_file("World","esm")
    world_data = read_file(world_file,'esm')


    #%%

    for esm_reg,iea_reg in regs.items():
        # Single Region
        if isinstance(iea_reg,str):
            to_read = [iea_reg]

        # Multi Region
        else:
            to_read = iea_reg

        files    = [get_file(ii,"iea") for ii in to_read]
        iea_data = sum([read_file(ii,"iea") for ii in files])

        esm_data = get_regional_data(world_data,esm_reg).loc[iea_data.index,iea_data.columns].droplevel(1)

        difference = iea_data - esm_data

        to_print[esm_reg] = pd.concat(
            {i:eval(i) for i in ['iea_data','esm_data','difference']},
            axis=1
            )

    with pd.ExcelWriter('iea_vs_esm_emissions.xlsx') as file:
        for reg,df in to_print.items():
            df.to_excel(file,reg)

    #%%
    '''Recalculating the emission coefficients'''

    emission_coefficients = {
        "Coal":0.380,
        "Oil":0.260,
        "Natural gas":0.202,
    }

    reference_year = 2015
    corrected_coefficients = {}

    for reg , data in to_print.items():
        corrected_coefficients[reg] = {}
        for fuel,original_coefficient in emission_coefficients.items():
            iea_data = data.loc[reference_year,('iea_data',fuel)]
            esm_data = data.loc[reference_year,('esm_data',fuel)]

            corrected_coefficients[reg][fuel] = original_coefficient * iea_data / esm_data


    with pd.ExcelWriter('corrected_coefficients.xlsx') as file:
        for reg,vals in corrected_coefficients.items():
            df = pd.DataFrame.from_dict(vals,orient='index')
            df.columns = [reference_year]

            df.to_excel(file,reg)


#%%

def print_out_emissions():
    world_file = get_file("World","esm")
    world_data = read_file(world_file,'esm')[list(set(flows_to_take.values()))]
    #%%
    with pd.ExcelWriter("new_emission.xlsx") as file:

        for esm_reg,reg in allias_regions.items():
            df = get_regional_data(world_data,esm_reg).loc[list(range(1990,2021)),:].droplevel(-1)
            df.to_excel(file,esm_reg)
# %%
import pandas as pd
# %%
sectors = [
        ('s.transport',	["t.icev",	"t.hev",	"t.bev",	"t.fcev",]),
        ("s.house",["t.house","t.house_new"]),
        ("s.elect",["t.elect_coal","t.elect_coal_ccs","t.elect_oil","t.elect_oil_ccs","t.elect_natgas","t.elect_natgas_ccs","t.elect_uranium","t.elect_pv","t.elect_wind","t.elect_geothermal","t.elect_hydro","t.elect_waste_biomass","t.elect_waste_biomass_ccs",]),
        #("s.elect_storage_transp",["t.bev_storage_grid","t.bev_storage_bev"]),
        #("s.hydrogen_storage_transp",["t.fcv_storage_grid","t.fcv_storage_fcv"]),
        ("s.hydrogen",["t.steam_reform","t.electrolyzer"]),
        ("s.blender",["t.gas_blender","t.hydrogen_blender"]),
    ]
def extract_mix():

    xy = pd.read_csv('case_studies/World/BAS/xy.csv',index_col=[0,1,2,3],header=0).droplevel(0)

    year = 2022
    tolerance = 0.05

    max_data = {}
    min_data = {}
    for region in xy.index.unique(level=0):
        region_data = xy.loc[(region,slice(None),year)].droplevel(level=[0,-1])
        mix_min = pd.read_excel('case_studies/World/BAS/Inputs/TechProductionMixMin.xlsx',sheet_name=region,index_col=0,header=[0,1])
        mix_max = pd.read_excel('case_studies/World/BAS/Inputs/TechProductionMixMax.xlsx',sheet_name=region,index_col=0,header=[0,1])
        for sector,techs in sectors:
            production = region_data.loc[techs]
            mix = production/production.sum()

            mix_min.loc[list(range(year,2071)),(sector,techs)] = (mix.loc[techs,'value'].values)*(1-tolerance)
            mix_max.loc[list(range(year,2071)),(sector,techs)] = (mix.loc[techs,'value'].values)*(1+tolerance)

        max_data[region] = mix_max
        min_data[region] = mix_min

    with pd.ExcelWriter('TechProductionMixMin.xlsx') as file:
        for reg,df in min_data.items():
            df.to_excel(file,reg)

    with pd.ExcelWriter('TechProductionMixMax.xlsx') as file:
        for reg,df in max_data.items():
            df.to_excel(file,reg)

#%%
def extract_tot_cap():

    cap_o = pd.read_csv('case_studies/World/BAS/cap_o.csv',index_col=[0,1,2,3],header=0).droplevel(0)

    year = 2020
    tolerance = 0.05

    max_data = {}
    min_data = {}
    for region in cap_o.index.unique(level=0):
        region_data = cap_o.loc[(region,slice(None),year)].droplevel(level=[0,-1])
        cap_min = pd.read_excel('case_studies/World/BAS/Inputs/OperativeCapacityMin.xlsx',sheet_name=region,index_col=0,header=[0,1])
        cap_max = pd.read_excel('case_studies/World/BAS/Inputs/OperativeCapacityMax.xlsx',sheet_name=region,index_col=0,header=[0,1])
        for sector,techs in sectors:
            production = region_data.loc[techs]
            mix = production

            cap_min.loc[year,(sector,techs)] = (mix.loc[techs,'value'].values)*(1-tolerance)
            cap_max.loc[year,(sector,techs)] = (mix.loc[techs,'value'].values)*(1+tolerance)

        max_data[region] = cap_max
        min_data[region] = cap_min

    with pd.ExcelWriter('OperativeCapacityMin.xlsx') as file:
        for reg,df in min_data.items():
            df.to_excel(file,reg)

    with pd.ExcelWriter('OperativeCapacityMax.xlsx') as file:
        for reg,df in max_data.items():
            df.to_excel(file,reg)


# %%
