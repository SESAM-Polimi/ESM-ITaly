# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:38:58 2021

@author: Amin
"""

from esm.utils.constants import (_MI,
                                 _CLUSTER_YEARLY_SHEETS,
                                 _COST_ITEMS,
                                 _TP_ITEMS,
                                 _ST_ITEMS,
                                 _MONEY_RATES,
                                 )

from esm.core.FreezeDict import Dict
from esm.utils.tools import cluster_frames
from esm.log_exc.exceptions import WrongInput

import numpy as np
import pandas as pd

def sectors_techs(instance) -> dict:
    '''Returns a dict mapping sectors and techs

    Returns
    -------
    A dict in which sectors are the keys and corrsepoding techs are the values
    '''
    sectors      = instance.__sets_frames__[_MI['s']]
    prod_sectors = list(sectors.loc[sectors['TYPE'] == 'production'].index)

    techs = instance.__sets_frames__[_MI['t']]


    _map = {}
    for sector in prod_sectors:
        _techs = techs[techs['SECTOR']==sector]['SECTOR'].index.tolist()

        if not(_techs):
            raise WrongInput(f'No technology for productive sector {sector} introduced')
        else:
            _map[sector] = _techs

    return _map

def techs_sectors(instance) -> dict:
    '''Returns a dict mapping the techs to sectors
    '''
    techs = instance.__sets_frames__[_MI['t']]

    _map = {tech: values['SECTOR'] for tech,values in techs.iterrows()}

    return _map





def period_length(instance) -> int:
    '''
    Return
    =========
    An integer representing the period length based on the definition of time slices

    Raise
    ==========
    WrongInput: if:
                    1. given data on correspondance and period length are int
                    2. period length is greater than correspondance
    '''

    correspondance = instance.__sets_frames__[_MI['h']]['CORRESPONDANCE'].values[0]
    period_length_ = instance.__sets_frames__[_MI['h']]['PERIOD LENGTH'].values[0]

    if any(not isinstance(item,(np.int64,np.int32,np.int16,np.int8)) for item in [correspondance,period_length_]):
        raise WrongInput('columns \'CORRESPONDANCE\' and \'PERIOD LENGTH\' in sheet {}'
                         ' can be only int'.format(_MI['h']))

    if period_length_ > correspondance:
        raise WrongInput('timeslice period length cannot be greater than correspondace ')

    return int(period_length_)


def time_slices(instance) -> list:
    '''
    Returns
    ==========
    A list of the range of time_slices
    '''
    period_length_ = instance.__sets_frames__[_MI['h']]['PERIOD LENGTH'].values[0]

    return list(range(1,period_length_+1))


def run_years(instance) ->list:
    '''
    Return
    =========
    A list of the main running years of the model
    '''
    years = instance.__sets_frames__[_MI['y']]
    return list(years.loc[years['TYPE'] == 'run'].index)


def warm_up_years(instance) ->list:
    '''
    Return
    =========
    A list of the main warm up years of the model
    '''
    years = instance.__sets_frames__[_MI['y']]
    return list(years.loc[years['TYPE'] == 'warm up'].index)


def cool_down_years(instance) ->list:
    '''
    Return
    =========
    A list of the main down up years of the model
    '''
    years = instance.__sets_frames__[_MI['y']]
    return list(years.loc[years['TYPE'] == 'cool down'].index)


def all_years(instance) ->list:
    '''
    Return
    =========
    A list with all years of the time horizon
    '''
    return list(instance.__sets_frames__[_MI['y']].index)


def capacity_techs(instance) -> list:
    '''
    RETURN
    -------
    A list of the technologies that can have capacity expansion
    '''

    techs = instance.__sets_frames__[_MI['t']]
    return list(techs.loc[techs['CAPACITY'] == True].index)


def hourly_capacity_techs(instance) -> list:
    '''
    RETURN
    -------
    A list of the technologies that can have capacity expansion and hourly dispatch resolution
    '''

    sectors = instance.__sets_frames__[_MI['s']]
    techs   = instance.__sets_frames__[_MI['t']]

    target_sectors = list(sectors[(sectors['TYPE']=='production') & (sectors['DISPATCH RESOLUTION']==_MI['h'])].index)
    target_techs = list(techs.loc[(techs['CAPACITY']==True) & (techs['SECTOR'].isin(target_sectors))].index)

    return target_techs


def yearly_capacity_techs(instance) -> list:
    '''
    RETURN
    -------
    A list of the technologies that can have capacity expansion and yearly dispatch resolution
    '''

    sectors = instance.__sets_frames__[_MI['s']]
    techs   = instance.__sets_frames__[_MI['t']]

    target_sectors = list(sectors[(sectors['TYPE']=='production') & (sectors['DISPATCH RESOLUTION']==_MI['y'])].index)
    target_techs = list(techs.loc[(techs['CAPACITY']==True) & (techs['SECTOR'].isin(target_sectors))].index)

    return target_techs


def capacity_techs_equality(instance) -> list:
    '''
    RETURN
    -------
    A list of the technologies that can have capacity expansion
    '''

    techs = instance.__sets_frames__[_MI['t']]
    return list(techs.loc[techs['AVAILABILITY'] == 'equality'].index)


def hourly_capacity_techs_equality(instance) -> list:
    '''
    RETURN
    -------
    A list of the technologies that can have capacity expansion
    '''

    sectors = instance.__sets_frames__[_MI['s']]
    techs = instance.__sets_frames__[_MI['t']]

    target_sectors = list(sectors[(sectors['TYPE']=='production') & (sectors['DISPATCH RESOLUTION']==_MI['h'])].index)

    return list(techs.loc[((techs['AVAILABILITY'] == 'equality') & (techs['SECTOR'].isin(target_sectors)))].index)


def yearly_capacity_techs_equality(instance) -> list:
    '''
    RETURN
    -------
    A list of the technologies that can have capacity expansion
    '''

    sectors = instance.__sets_frames__[_MI['s']]
    techs = instance.__sets_frames__[_MI['t']]

    target_sectors = list(sectors[(sectors['TYPE']=='production') & (sectors['DISPATCH RESOLUTION']==_MI['y'])].index)

    return list(techs.loc[((techs['AVAILABILITY'] == 'equality') & (techs['SECTOR'].isin(target_sectors)))].index)


def capacity_techs_range(instance) -> list:
    '''
    RETURN
    -------
    A list of the technologies that can have capacity expansion
    '''

    techs = instance.__sets_frames__[_MI['t']]
    return list(techs.loc[techs['AVAILABILITY'] == 'range'].index)


def hourly_capacity_techs_range(instance) -> list:
    '''
    RETURN
    -------
    A list of the technologies that can have capacity expansion
    '''

    sectors = instance.__sets_frames__[_MI['s']]
    techs = instance.__sets_frames__[_MI['t']]

    target_sectors = list(sectors[(sectors['TYPE']=='production') & (sectors['DISPATCH RESOLUTION']==_MI['h'])].index)

    return list(techs.loc[((techs['AVAILABILITY'] == 'range') & (techs['SECTOR'].isin(target_sectors)))].index)


def yearly_capacity_techs_range(instance) -> list:
    '''
    RETURN
    -------
    A list of the technologies that can have capacity expansion
    '''

    sectors = instance.__sets_frames__[_MI['s']]
    techs = instance.__sets_frames__[_MI['t']]

    target_sectors = list(sectors[(sectors['TYPE']=='production') & (sectors['DISPATCH RESOLUTION']==_MI['y'])].index)

    return list(techs.loc[((techs['AVAILABILITY'] == 'range') & (techs['SECTOR'].isin(target_sectors)))].index)


def capacity_techs_demand(instance) -> list:
    '''
    Returns
    ==========
    A list of technologies whose capacities have availabilities constrained within ranges
    '''
    techs = instance.__sets_frames__[_MI['t']]
    return list(techs.loc[(techs['AVAILABILITY'] == 'demand') & (techs['CAPACITY'] == True)].index)


def hourly_capacity_techs_demand(instance) -> list:
    '''
    RETURN
    -------
    A list of the technologies that can have capacity expansion
    '''

    sectors = instance.__sets_frames__[_MI['s']]
    techs = instance.__sets_frames__[_MI['t']]

    target_sectors = list(sectors[(sectors['TYPE']=='production') & (sectors['DISPATCH RESOLUTION']==_MI['h'])].index)

    return list(techs.loc[((techs['AVAILABILITY'] == 'demand') & (techs['SECTOR'].isin(target_sectors)))].index)


def yearly_capacity_techs_demand(instance) -> list:
    '''
    RETURN
    -------
    A list of the technologies that can have capacity expansion
    '''

    sectors = instance.__sets_frames__[_MI['s']]
    techs = instance.__sets_frames__[_MI['t']]

    target_sectors = list(sectors[(sectors['TYPE']=='production') & (sectors['DISPATCH RESOLUTION']==_MI['y'])].index)

    return list(techs.loc[((techs['AVAILABILITY'] == 'demand') & (techs['SECTOR'].isin(target_sectors)))].index)


def products(instance) -> list:
    '''
    RETURNS
    =========
    A list of products flows

    '''
    flows = instance.__sets_frames__[_MI['f']]
    return list(flows.loc[flows['TYPE'] == 'product'].index)

def hourly_products(instance) -> list:

    '''
    Returns
    ========
    A list of the hourly products
    '''
    flows = instance.__sets_frames__[_MI['f']]
    return list(flows.loc[(flows['TYPE'] == 'product') & (flows['DISPATCH RESOLUTION'] == _MI['h'])].index)


def yearly_products(instance) -> list:

    '''
    Returns
    ========
    A list of the hourly products
    '''
    flows = instance.__sets_frames__[_MI['f']]
    return list(flows.loc[(flows['TYPE'] == 'product') & (flows['DISPATCH RESOLUTION'] == _MI['y'])].index)


def hourly_techs(instance) -> list:

    '''
    Returns
    ========
    A list of the technologies which their parent sector is hourly and
    is production sector
    '''
    sectors = instance.__sets_frames__[_MI['s']]
    techs   = instance.__sets_frames__[_MI['t']]

    target_sectors = list(sectors[(sectors['TYPE']=='production') & (sectors['DISPATCH RESOLUTION']==_MI['h'])].index)

    output = []

    nested_lists = [list(techs.loc[techs['SECTOR'] == sector].index) for sector in target_sectors]
    for inner_list in nested_lists:
        output.extend(inner_list)

    return list(output)

def yearly_techs(instance) -> list:

    '''
    Returns
    ========
    A list of the technologies which their parent sector is hourly and
    is production sector
    '''
    sectors = instance.__sets_frames__[_MI['s']]
    techs   = instance.__sets_frames__[_MI['t']]

    target_sectors = list(sectors[(sectors['TYPE']=='production') & (sectors['DISPATCH RESOLUTION']==_MI['y'])].index)

    output = []

    nested_lists = [list(techs.loc[techs['SECTOR'] == sector].index) for sector in target_sectors]
    for inner_list in nested_lists:
        output.extend(inner_list)

    return list(output)

def hourly_sectors(instance) -> list:
    '''
    RETURNS
    =========
    A list of hourly sectors
    '''
    sectors = instance.__sets_frames__[_MI['s']]
    return list(sectors.loc[(sectors['DISPATCH RESOLUTION'] == _MI['h']) & (sectors['TYPE'] == 'production') ].index)

def yearly_sectors(instance) -> list:
    '''
    RETURNS
    =========
    A list of yearly sectors
    '''
    sectors = instance.__sets_frames__[_MI['s']]
    return list(sectors.loc[(sectors['DISPATCH RESOLUTION'] == _MI['y']) & (sectors['TYPE'] == 'production')].index)



def consumption_sectors(instance) -> list:
    '''
    RETURNS
    =========
    A list of consumption sectors
    '''
    sectors = instance.__sets_frames__[_MI['s']]
    return list(sectors.loc[sectors['TYPE'] == 'consumption'].index)


def production_sectors(instance) -> list:
    '''
    RETURNS
    =========
    A list of production sectors
    '''
    sectors = instance.__sets_frames__[_MI['s']]
    return list(sectors.loc[sectors['TYPE'] == 'production'].index)


def primary_resources(instance) ->list:
    '''
    Returns
    ==========
    A list of primary resources from the flows set
    '''
    flows = instance.__sets_frames__[_MI['f']]
    return list(flows.loc[flows['TYPE'] == _MI['pr']].index)


def energy_wastes(instance) ->list:
    '''
    Returns
    =========
    A list of energy related wastes
    '''
    flows = instance.__sets_frames__[_MI['f']]
    return list(flows.loc[flows['TYPE'] == _MI['ew']].index)


def emission_by_flows(instance) ->list:
    '''
    Returns
    =========
    A list of emissions by flows
    '''
    flows = instance.__sets_frames__[_MI['f']]
    return list(flows.loc[flows['TYPE'] == _MI['ep']].index)

def emission_by_techs(instance) ->list:
    '''
    Returns
    =========
    A list of emissions by techs
    '''
    flows = instance.__sets_frames__[_MI['f']]
    return list(flows.loc[flows['TYPE'] == _MI['et']].index)

def storages(instance) ->list:
    '''
    Returns
    =========
    A list of storage technologies
    '''
    techs = instance.__sets_frames__[_MI['t']]
    return list(techs.loc[techs['TYPE'].isin(['storage','storage+'])].index)

def storages_plus(instance) ->list:
    '''
    Returns
    =========
    A list of storage technologies
    '''
    techs = instance.__sets_frames__[_MI['t']]
    return list(techs.loc[techs['TYPE'] == 'storage+'].index)

def storages_non_plus(instance) ->list:
    '''
    Returns
    =========
    A list of storage technologies (non +)
    '''
    techs = instance.__sets_frames__[_MI['t']]
    return list(techs.loc[techs['TYPE'] == 'storage'].index)


def availability_eq(instance) -> list:
    '''
    Returns
    ==========
    A list of technologies whose capacities have availabilities constant throughout the hourly time steps
    '''

    techs = instance.__sets_frames__[_MI['t']]
    return list(techs.loc[(techs['AVAILABILITY'] == 'equality') & (techs['CAPACITY'] == True)].index)

def availability_range(instance) -> list:
    '''
    Returns
    ==========
    A list of technologies whose capacities have availabilities constrained within ranges
    '''
    techs = instance.__sets_frames__[_MI['t']]
    return list(techs.loc[(techs['AVAILABILITY'] == 'range') & (techs['CAPACITY'] == True)].index)

def availability_demand(instance) -> list:
    '''
    Returns
    ==========
    A list of technologies whose capacities have availabilities constrained within ranges
    '''
    techs = instance.__sets_frames__[_MI['t']]
    return list(techs.loc[(techs['AVAILABILITY'] == 'demand') & (techs['CAPACITY'] == True)].index)

#def tech_costs(instance) -> list:
#    '''
#    Returns
#    =========
#    A list of costs by techs
#    '''
#    techs = instance.__sets_frames__[_MI['t']]
#    return list(techs.loc[techs['TYPE'] == _MI['cu']].index)
    
    

class Identifiers:
    '''
    Description
    ============
    This class contains a couple of methods and attributes that help the
    development of the optimization problem
    '''

    @property
    def mode(self):
        return self.__mode__

    @mode.setter
    def mode(self,mode):
        assert mode in ['sfc-integrated','stand-alone'],'acceptable modes are {}'.format(['sfc-integrated','stand-alone'])
        self.__mode__ = mode

        if mode == 'stand-alone':

            self.exogenous = ['v','u','bp','wu','bu','ef',
                              'af_eq','af_min','af_max','cu','mr',
                              'tp','st','bv',
                              'E','E_tld','E_tld_diag','e','dp',
                              'xy_mix','xy_mix_min','xy_mix_max',
                              'cap_o_min','cap_o_max','cap_n_min','cap_n_max','te', 'tin']

            self.endogenous = ['xy','xh','qy','qh',
                               'BV','BU','BV_U','BV_E','BP',
                               'CU','CU_mr',
                               'cap_o','cap_n','cap_d',
                               'soc',]


    def __set_properties__(self,):

        '''
        Description
        ==============
        This function, first check if id property already exists or not.
        if not, it will create it.

        self.ids ==> a Dict object that contains key attributes of the model
        for example

        self.ids.run_years -> returns the main modeling time-horizon
        self.ids.hourly_techs -> returns all the hourly technologies
        '''
        if hasattr(self,'ids'):
            raise ValueError('ids cannot be overwritten.')

        attributes = ['run_years',
                      'period_length',
                      'time_slices',
                      'warm_up_years',
                      'cool_down_years',
                      'all_years',
                      'capacity_techs',
                      'hourly_capacity_techs',
                      'yearly_capacity_techs',
                      'capacity_techs_equality',
                      'hourly_capacity_techs_equality',
                      'yearly_capacity_techs_equality',
                      'capacity_techs_range',
                      'hourly_capacity_techs_range',
                      'yearly_capacity_techs_range',
                      'capacity_techs_demand',
                      'hourly_capacity_techs_demand',
                      'yearly_capacity_techs_demand',
                      'products',
                      'hourly_products',
                      'yearly_products',
                      'hourly_techs',
                      'yearly_techs',
                      'storages',
                      'storages_plus',
                      'storages_non_plus',
                      'hourly_sectors',
                      'yearly_sectors',
                      'consumption_sectors',
                      'production_sectors',
                      'primary_resources',
                      'energy_wastes',
                      'emission_by_flows',
                      'emission_by_techs',
                      'availability_range',
                      'availability_eq',
                      'availability_demand',
                      'sectors_techs',
                      'techs_sectors',
                      #'tech_costs'
                      ]


        ids = {}
        for attribute in attributes:
            ids[attribute] = eval(attribute)(self)

        self.ids = Dict(_type = (list,int,dict,pd.DataFrame),
                        **ids)

        self._check_storage_plus()

    def _check_storage_plus(self):
        to_add= []
        for tech in self.ids.storages_plus:
            sector_of_tech = self.ids.techs_sectors[tech]
            all_techs = self.ids.sectors_techs[sector_of_tech]

            coupled_tech= set(all_techs).difference(set([tech]))

            if len(coupled_tech)!=1:
                raise WrongInput('Storage+ techs should have only one coupled technology within its sector.'
                                 f' tech {tech} in secotr {sector_of_tech} has not one coupled technology'
                                 f' {coupled_tech}')

            to_add.append(list(coupled_tech)[0])

        self.ids['storage_plus_couple'] = to_add




    def _year_cluster(self,
                        period: str
                        ) -> iter:

        '''
        DESCRIPTION
        =============
        The function is a generator to return the time cluster.
        example:

            for year, cluster in self.__yearCluster__():
                year --> the real modeling time horizon such as 2020

                cluster --> will be a dictionary like below
                {'use'  : T1,
                 'make' : T2}
                which shows for 2020, use matrix should read the data for T1
                & make matrix should read the data for T2.


        RETURN
        ============
        In a for loop, it returns the main year and its corresponding clusters
        for all the items

        '''
        assert period in ['run','warm_up','cool_down','all'], 'valid periods are {}'.format(['run','warm_up','cool_down','all'])

        for year,cluster in self.time_cluster.clusters.items():
            if year in getattr(self.ids,f'{period}_years'):
                yield year,cluster


    def __Frames__(self,):

         '''
         Description
         ===============
         The function is in charge of preparing the frames to be filled.
         The shape of the frame follows specific rules, as described below
         '''


         matrix   = {}
         techs    = self.ids.yearly_techs + self.ids.hourly_techs
         regions  = getattr(self,_MI['r'])
         products = self.ids.yearly_products + self.ids.hourly_products
         sectors  = self.ids.yearly_sectors  + self.ids.hourly_sectors

         '''u'''
         u_index   = pd.MultiIndex.from_product([regions,products])
         u_columns = pd.MultiIndex.from_product([regions,techs])

         '''v'''
         v_columns = u_index
         v_index   = pd.MultiIndex.from_product([regions,sectors])

         '''bp'''
         bp_columns = u_columns
         bp_index   = pd.Index(self.ids.primary_resources)

         '''wu'''
         wu_columns = u_columns
         wu_index   = pd.Index(self.ids.energy_wastes)

         '''bu'''
         bu_columns = u_columns
         bu_index   = pd.Index(self.ids.emission_by_techs)

         '''BU'''
         BU_columns = bu_columns
         BU_index   = bu_index

         '''bv'''
         bv_columns = v_columns
         bv_index   = pd.Index(self.ids.emission_by_flows)

         '''BV'''
         BV_columns = bv_columns
         BV_index   = bv_index

         '''BP'''
         BP_columns = u_columns
         BP_index   = pd.Index(self.ids.primary_resources)

         '''af_eq'''
         af_eq_columns = pd.MultiIndex.from_product([regions,self.ids.availability_eq])
         af_eq_index   = pd.Index(self.ids.time_slices)

         '''af_min'''
         af_min_columns = pd.MultiIndex.from_product([regions,self.ids.availability_range])
         af_min_index   = af_eq_index

         '''af_max'''
         af_max_columns = af_min_columns
         af_max_index   = af_min_index

         '''cu'''
         cu_columns = u_columns
         cu_index   = pd.Index(list(_COST_ITEMS.keys()) + self.ids.emission_by_flows + self.ids.emission_by_techs)

         '''CU'''
         CU_columns = u_columns
         CU_index   = pd.Index(list(_COST_ITEMS.keys()) + self.ids.emission_by_flows + self.ids.emission_by_techs)
         CU_index   = pd.Index(list(_COST_ITEMS.keys()) + self.ids.emission_by_flows + self.ids.emission_by_techs + ['c_in_d'])
         
         '''tin'''
         tin_index   = pd.Index(self.ids.all_years)
         tin_columns = pd.MultiIndex.from_product([self.Regions,list(_COST_ITEMS.keys())]) 
         #tin_columns = pd.Index(_COST_ITEMS.keys())
         
        
         '''ce'''
         # define if sfc is needing total cost of emissions

         '''CE'''
         # define if sfc is needing total cost of emissions

         '''CU_mr'''
         CU_mr_columns = u_columns
         CU_mr_index   = pd.Index(list(_COST_ITEMS.keys()) + self.ids.emission_by_flows + self.ids.emission_by_techs + ['c_in_d'])

         '''CE_mr'''
         # define if sfc is needing total cost of emissions


         '''mr'''
         mr_columns = pd.MultiIndex.from_product([regions,_MONEY_RATES.keys()])
         mr_index   = pd.Index(self.ids.all_years)

         '''tp'''
         tp_columns = u_columns
         tp_index   = pd.Index(_TP_ITEMS.keys())

         '''st'''
         st_columns = pd.MultiIndex.from_product([regions,self.ids.storages])
         st_index   = pd.Index(_ST_ITEMS.keys())

         '''dp'''
         dp_index   = self.ids.time_slices
         dp_columns = pd.MultiIndex.from_product([regions,self.ids.yearly_techs + self.ids.consumption_sectors])

         '''E'''
         E_index   = pd.MultiIndex.from_product(([regions,products]))
         E_columns = pd.MultiIndex.from_product(([self.ids.consumption_sectors,self.ids.all_years]))

         '''E_tld'''
         E_tld_index   = pd.MultiIndex.from_product(([regions,products]))
         E_tld_columns = pd.MultiIndex.from_product(([self.ids.consumption_sectors,self.ids.all_years]))

         '''E_tld_diag'''
         E_tld_diag_index   = pd.MultiIndex.from_product(([regions,products]))
         E_tld_diag_columns = pd.MultiIndex.from_product(([self.ids.consumption_sectors,regions,self.ids.all_years]))

         '''e'''
         e_index   = E_index
         e_columns = E_columns

         '''m'''
         m_index   = E_index
         m_columns = E_columns

         '''xy'''
         xy_index   = [0]
         xy_columns = pd.MultiIndex.from_product(([regions,self.ids.yearly_techs + self.ids.hourly_techs]))

         '''xh'''
         xh_index   = self.ids.time_slices
         xh_columns = pd.MultiIndex.from_product(([regions,self.ids.hourly_techs]))

         '''qy'''
         qy_index   = [0]
         qy_columns = pd.MultiIndex.from_product(([regions,self.ids.yearly_products + self.ids.hourly_products]))

         '''xy_mix'''
         xy_mix_index   = pd.Index(self.ids.all_years)

         df = pd.DataFrame()
         df = pd.concat([pd.DataFrame(index=self.indeces['xy_mix']['index'],columns=self.indeces['xy_mix']['columns'])
                        for region in self.Regions],
                        axis=1,
                        sort=False,
                        keys=self.Regions
                        )

         xy_mix_columns = df.columns

         '''xy_mix_min'''
         xy_mix_min_index   = pd.Index(self.ids.all_years)

         df = pd.DataFrame()
         df = pd.concat([pd.DataFrame(index=self.indeces['xy_mix_min']['index'],columns=self.indeces['xy_mix_min']['columns'])
                        for region in self.Regions],
                        axis=1,
                        sort=False,
                        keys=self.Regions
                        )

         xy_mix_min_columns = df.columns

         '''xy_mix_max'''
         xy_mix_max_index   = pd.Index(self.ids.all_years)

         df = pd.DataFrame()
         df = pd.concat([pd.DataFrame(index=self.indeces['xy_mix_max']['index'],columns=self.indeces['xy_mix_max']['columns'])
                        for region in self.Regions],
                        axis=1,
                        sort=False,
                        keys=self.Regions
                        )

         xy_mix_max_columns = df.columns

         '''cap_o_min'''
         cap_o_min_index   = pd.Index(self.ids.all_years)

         df = pd.DataFrame()
         df = pd.concat([pd.DataFrame(index=self.indeces['cap_o_min']['index'],columns=self.indeces['cap_o_min']['columns'])
                        for region in self.Regions],
                        axis=1,
                        sort=False,
                        keys=self.Regions
                        )

         cap_o_min_columns = df.columns

         '''cap_o_max'''
         cap_o_max_index   = pd.Index(self.ids.all_years)

         df = pd.DataFrame()
         df = pd.concat([pd.DataFrame(index=self.indeces['cap_o_max']['index'],columns=self.indeces['cap_o_max']['columns'])
                        for region in self.Regions],
                        axis=1,
                        sort=False,
                        keys=self.Regions
                        )

         cap_o_max_columns = df.columns

         '''cap_n_min'''
         cap_n_min_index   = pd.Index(self.ids.all_years)

         df = pd.DataFrame()
         df = pd.concat([pd.DataFrame(index=self.indeces['cap_n_min']['index'],columns=self.indeces['cap_n_min']['columns'])
                        for region in self.Regions],
                        axis=1,
                        sort=False,
                        keys=self.Regions
                        )

         cap_n_min_columns = df.columns

         '''cap_n_max'''
         cap_n_max_index   = pd.Index(self.ids.all_years)

         df = pd.DataFrame()
         df = pd.concat([pd.DataFrame(index=self.indeces['cap_n_max']['index'],columns=self.indeces['cap_n_max']['columns'])
                        for region in self.Regions],
                        axis=1,
                        sort=False,
                        keys=self.Regions
                        )

         cap_n_max_columns = df.columns

         '''qh'''
         qh_index   = self.ids.time_slices
         qh_columns = pd.MultiIndex.from_product(([regions,self.ids.hourly_products]))

         '''cap_o'''
         cap_o_index = list(self.__sets_frames__[_MI['y']].index)
         cap_o_columns = pd.MultiIndex.from_product(([regions,self.ids.capacity_techs]))

         '''cap_n'''
         cap_n_index = list(self.__sets_frames__[_MI['y']].index)
         cap_n_columns = pd.MultiIndex.from_product(([regions,self.ids.capacity_techs]))

         '''cap_d'''
         cap_d_index = list(self.__sets_frames__[_MI['y']].index)
         cap_d_columns = pd.MultiIndex.from_product(([regions,self.ids.capacity_techs]))

         '''soc'''
         soc_index = self.ids.time_slices
         soc_columns = pd.MultiIndex.from_product(([regions,self.ids.storages]))

         '''ef'''
         ef_index = pd.MultiIndex.from_product([regions,products])
         ef_columns = pd.MultiIndex.from_product([regions,techs])

         '''te'''
         te_index = pd.Index(self.ids.all_years)
         te_columns = pd.MultiIndex.from_product([self.Regions,self.ids.emission_by_flows])

         '''BV_U'''
         BV_U_index = pd.MultiIndex.from_product([regions,products])
         BV_U_columns = pd.MultiIndex.from_product([regions,techs])

         '''BV_E'''
         BV_E_index = pd.MultiIndex.from_product([regions,products])
         BV_E_columns = pd.MultiIndex.from_product(([self.ids.consumption_sectors,self.ids.all_years]))


         matrix['dp'] = {}
         for cluster in self.time_cluster.parameter_clusters('dp'):
             matrix['dp'][cluster] = {}
             for flow in self.ids.hourly_products:
                 matrix['dp'][cluster][flow] = pd.DataFrame(0,index=dp_index,columns=dp_columns)

         matrix['ef'] = {}
         for cluster in self.time_cluster.parameter_clusters('ef'):
             matrix['ef'][cluster] = {}
             for emis_flow in self.ids.emission_by_flows:
                 matrix['ef'][cluster][emis_flow] = pd.DataFrame(0,index=ef_index,columns=ef_columns)

         matrix['BV_U'] = {}
         for emis_flow in self.ids.emission_by_flows:
             matrix['BV_U'][emis_flow] = pd.DataFrame(0,index=BV_U_index,columns=BV_U_columns)

         matrix['BV_E'] = {}
         for emis_flow in self.ids.emission_by_flows:
             matrix['BV_E'][emis_flow] = pd.DataFrame(0,index=BV_E_index,columns=BV_E_columns)


         for variable in self.exogenous:
             if variable in ['E','E_tld','E_tld_diag','e','m','dp','xy_mix','xy_mix_min','xy_mix_max','mr','ef','cap_o_min','cap_o_max','cap_n_min','cap_n_max','te','tin']: # this exception must be generalized for the modes
                 continue

             matrix[variable] = cluster_frames(self,
                                                eval(f'{variable}_index'),
                                                eval(f'{variable}_columns'),
                                                0,
                                                variable,
                                                )

         for variable in set(self.endogenous + ['E','E_tld','E_tld_diag','e','m','xy_mix','xy_mix_min','xy_mix_max','mr','cap_o_min','cap_o_max','cap_n_min','cap_n_max','te','tin']) - set(['BV_U','BV_E']):
             matrix[variable] = pd.DataFrame(0,
                                             index = eval(f'{variable}_index'),
                                             columns = eval(f'{variable}_columns'),
                                             )

         self.__matrices__ = Dict((dict,pd.DataFrame),**matrix)



    def __Indexer__(self,):

        '''
        Description
        ==============
        This function is in charge of generating indices for the dataframes for
        the data input. all the indices (index and columns) will be stored in a
        Dict object as follow:

            self.indeces -> keys: name of the matrix

                            values: a Dict -> key: 'index' or 'columns'
                                              value: the MultiIndex Object

        example:
            self.indeces['bu']['index'] returns the index for bu matrix while
            self.indeces['bu']['columns'] returns the columns for bu matrix
        '''
        def return_index(names,*levels):
            if len(levels) == 1:
                return pd.Index(levels[0],name=names)

            return pd.MultiIndex.from_arrays([i for i in levels],
                                              names= names,
                                             )

        if hasattr(self,'indeces'):
            raise ValueError('Index object already exist')

        indeces = {}

        index_names = ['variable_name',
                       'flow',
                       'label',
                       ]

        columns_names = ['sector',
                         'technology',
                         'label',
                         ]

        '''
        The columns will be in the following orders
        1. yearly sectors (and their technologies)
        2. hourly sectors (and their tehcnologies)

        '''

        column_1 = list(self.ids.yearly_techs) + list(self.ids.hourly_techs)
        column_0 = [self.__sets_frames__[_MI['t']].loc[tech,'SECTOR']
                    for tech in column_1]
        column_2 = list(self.__sets_frames__[_MI['t']].loc[column_1,'NAME'] )

        Columns = return_index(columns_names,column_0,column_1,column_2)

        '''
        u index,columns
        '''

        index_1 = list(self.ids.products)
        index_2 = self.__sets_frames__[_MI['f']].loc[index_1,'NAME'].values
        index_0 = [_MI['u']] * len(index_1)

        Index = return_index(index_names,index_0,index_1,index_2)

        indeces['u'] = dict(index=Index,columns = Columns)

        '''
        bp index,columns
        '''

        index_1 = list(self.ids.primary_resources)
        index_0 = [_MI['bp']] * len(index_1)
        index_2 = self.__sets_frames__[_MI['f']].loc[index_1,'NAME'].values

        Index = return_index(index_names,index_0,index_1,index_2)

        indeces['bp'] = dict(index=Index,columns = Columns)


        '''
        wu index,columns
        '''
        index_1 = list(self.ids.energy_wastes)
        index_0 = [_MI['wu']] * len(index_1)
        index_2 = self.__sets_frames__[_MI['f']].loc[index_1,'NAME'].values

        Index = return_index(index_names,index_0,index_1,index_2)

        indeces['wu'] = dict(index=Index,columns = Columns)

        '''
        st index,columns
        '''


        index_2 = list(_ST_ITEMS.keys())

        index_0 = [_MI['st']] * len(index_2)
        index_1 = list(_ST_ITEMS.keys())


        Index = return_index(index_names,index_0,index_1,index_2)

        indeces['st'] = dict(index=Index,columns = Columns)


        '''
        bu index,columns
        '''

        index_1 = list(self.ids.emission_by_techs)
        index_0 = [_MI['bu']] * len(index_1)
        index_2 = self.__sets_frames__[_MI['f']].loc[index_1,'NAME'].values

        Index = return_index(index_names,index_0,index_1,index_2)

        indeces['bu'] = dict(index=Index,columns = Columns)

        '''
        cu index,columns
        '''

        index_2 = list(_COST_ITEMS.values()) + self.__sets_frames__[_MI['f']].loc[self.ids.emission_by_flows + self.ids.emission_by_techs,'NAME'].tolist()

        index_0 = [_MI['cu']] * len(index_2)
        index_1 = list(_COST_ITEMS.keys()) + self.ids.emission_by_flows + self.ids.emission_by_techs

        Index = return_index(index_names,index_0,index_1,index_2)

        indeces['cu'] = dict(index=Index,columns = Columns)

        '''
        tp index,columns
        '''

        index_2 = list(_TP_ITEMS.values())

        index_0 = [_MI['tp']] * len(index_2)
        index_1 = list(_TP_ITEMS.keys())

        Index = return_index(index_names,index_0,index_1,index_2)

        indeces['tp'] = dict(index=Index,columns = Columns)


        '''
        em index, columns

        Emission Filter, This is so tricky

        emission filters should be defined for every specific emission by flow
        on every flow. For example, if user defines 2 emissions by flow and the
        model has two flows, it means that 4 emissions by flow should be defined.
        '''

        emissions = self.ids.emission_by_flows
        flows     = self.ids.products

        ''' single_flow_index '''
        index_single_1 = flows
        index_single_2 = list(self.__sets_frames__[_MI['f']].loc[flows,'NAME'].values)

        index_0 = []
        index_1 = []
        index_2 = []

        for emission in emissions:
            index_single_0 = ['{}_{}'.format(_MI['ef'],emission)] * len(index_single_1)

            index_0.extend(index_single_0)
            index_1.extend(index_single_1)
            index_2.extend(index_single_2)

        Index = return_index(index_names,index_0,index_1,index_2)

        indeces['ef'] = dict(index=Index,columns = Columns)


        '''
        af_eq,af_min,af_max index,columns
        '''
        index = self.ids.time_slices
        Index = return_index('time_slice',index)

        'eq'

        column_1 = self.ids.availability_eq
        _id      = ['equality'] * len(column_1)
        column_0 = list(self.__sets_frames__[_MI['t']].loc[column_1,'SECTOR'].values)
        column_2 = list(self.__sets_frames__[_MI['t']].loc[column_1,'NAME'].values)

        Columns = return_index(['type']+columns_names,_id,column_0,column_1,column_2)
        indeces['af_eq'] = dict(index=Index,columns = Columns)

        'max,min'
        column_1 = self.ids.availability_range
        column_0 = list(self.__sets_frames__[_MI['t']].loc[column_1,'SECTOR'].values)
        column_2 = list(self.__sets_frames__[_MI['t']].loc[column_1,'NAME'].values)

        _id      = ['minimum'] * len(column_1)
        Columns = return_index(['type']+columns_names,_id,column_0,column_1,column_2)
        indeces['af_min'] = dict(index=Index,columns = Columns)

        _id      = ['maximum'] * len(column_1)
        Columns = return_index(['type']+columns_names,_id,column_0,column_1,column_2)
        indeces['af_max'] = dict(index=Index,columns = Columns)

        '''
        dp index,columns
        '''
        columns_names = ['country',
                         'technology/sector',
                         'label',
                         ]

        index = self.ids.time_slices

        regions = getattr(self,_MI['r'])

        yearly_techs_ids   = self.ids.yearly_techs
        yearly_techs_names = list(self.__sets_frames__[_MI['t']].loc[yearly_techs_ids,'NAME'].values)

        consumption_sectors_ids   = self.ids.consumption_sectors
        consumption_sectors_names = list(self.__sets_frames__[_MI['s']].loc[consumption_sectors_ids,'NAME'].values)

        column_0 = sorted(regions * len(yearly_techs_ids+consumption_sectors_ids))
        column_1 = (yearly_techs_ids + consumption_sectors_ids) * len(regions)
        column_2 = (yearly_techs_names + consumption_sectors_names) * len(regions)


        Index   = return_index('time_slice',index)
        Columns = return_index(columns_names,column_0,column_1,column_2)

        indeces['dp'] = dict(index=Index,columns = Columns)

        '''
        E,e index,columns
        '''
        regions    = getattr(self,_MI['r'])
        products   = self.ids.yearly_products + self.ids.hourly_products
        prod_names = list(self.__sets_frames__[_MI['f']].loc[products,'NAME'].values)

        index_0 = sorted(regions*len(products))
        index_1 = products * len(regions)
        index_2 = prod_names * len(regions)

        Index = return_index(['region','flow','label'],index_0,index_1,index_2)

        column_0 = self.ids.all_years

        Columns = return_index('year',column_0,)

        indeces['E'] = dict(index=Index,columns = Columns)
        indeces['e'] = dict(index=Index,columns = Columns)

        '''
        xy_mix
        '''
        index_0  = pd.Index(self.ids.all_years)
        column_0 = []
        column_1 = []


        for tech in self.ids.yearly_techs + self.ids.hourly_techs:
            column_0.append(self.ids.techs_sectors[tech])
            column_1.append(tech)

        Columns = return_index(['sector','tech'],column_0,column_1)
        Index   = return_index('year',index_0)
        indeces['xy_mix'] = dict(index= Index,columns= Columns)

        '''
        xy_mix_min
        '''
        index_0  = pd.Index(self.ids.all_years)
        column_0 = []
        column_1 = []

        for tech in self.ids.yearly_techs + self.ids.hourly_techs:
            column_0.append(self.ids.techs_sectors[tech])
            column_1.append(tech)

        Columns = return_index(['sector','tech'],column_0,column_1)
        Index   = return_index('year',index_0)
        indeces['xy_mix_min'] = dict(index= Index,columns= Columns)

        '''
        xy_mix_max
        '''
        index_0  = pd.Index(self.ids.all_years)
        column_0 = []
        column_1 = []

        for tech in self.ids.yearly_techs + self.ids.hourly_techs:
            column_0.append(self.ids.techs_sectors[tech])
            column_1.append(tech)

        Columns = return_index(['sector','tech'],column_0,column_1)
        Index   = return_index('year',index_0)
        indeces['xy_mix_max'] = dict(index= Index,columns= Columns)

        '''
        cap_o_min
        '''
        index_0  = pd.Index(self.ids.all_years)
        column_0 = []
        column_1 = []

        for tech in self.ids.capacity_techs:
            column_0.append(self.ids.techs_sectors[tech])
            column_1.append(tech)

        Columns = return_index(['sector','tech'],column_0,column_1)
        Index   = return_index('year',index_0)
        indeces['cap_o_min'] = dict(index= Index,columns= Columns)

        '''
        cap_o_max
        '''
        index_0  = pd.Index(self.ids.all_years)
        column_0 = []
        column_1 = []

        for tech in self.ids.capacity_techs:
            column_0.append(self.ids.techs_sectors[tech])
            column_1.append(tech)

        Columns = return_index(['sector','tech'],column_0,column_1)
        Index   = return_index('year',index_0)
        indeces['cap_o_max'] = dict(index= Index,columns= Columns)

        '''
        cap_n_min
        '''
        index_0  = pd.Index(self.ids.all_years)
        column_0 = []
        column_1 = []

        for tech in self.ids.capacity_techs:
            column_0.append(self.ids.techs_sectors[tech])
            column_1.append(tech)

        Columns = return_index(['sector','tech'],column_0,column_1)
        Index   = return_index('year',index_0)
        indeces['cap_n_min'] = dict(index= Index,columns= Columns)

        '''
        cap_n_max
        '''
        index_0  = pd.Index(self.ids.all_years)
        column_0 = []
        column_1 = []

        for tech in self.ids.capacity_techs:
            column_0.append(self.ids.techs_sectors[tech])
            column_1.append(tech)

        Columns = return_index(['sector','tech'],column_0,column_1)
        Index   = return_index('year',index_0)
        indeces['cap_n_max'] = dict(index= Index,columns= Columns)

        '''
        V index,columns
        '''
        column_0    = self.ids.yearly_products + self.ids.hourly_products
        column_1   = list(self.__sets_frames__[_MI['f']].loc[column_0,'NAME'].values)

        Columns = return_index(['flow','label'],column_0,column_1)

        _id = [_MI['v']]
        regions = self.Regions
        sectors_id   = self.ids.yearly_sectors + self.ids.hourly_sectors
        sectors_name = list(self.__sets_frames__[_MI['s']].loc[sectors_id,'NAME'].values)

        index_0 = sorted(regions * len(sectors_id))
        index_1 = sectors_id * len(regions)
        index_2 = sectors_name * len(regions)

        names = ['variable_name','region','sector','label']
        Index = return_index(names,_id*len(index_0),index_0,index_1,index_2)

        indeces['v'] = dict(index=Index,columns = Columns)

        '''
        bv index,columns
        '''
        index_1 = self.ids.emission_by_flows
        index_0 = ['-'] * len(index_1)
        index_2 = list(self.__sets_frames__[_MI['f']].loc[index_1,'NAME'].values)

        _id = [_MI['bv']] * len(index_1)

        Index = return_index(names,_id,index_0,index_1,index_2)

        indeces['bv'] = dict(index=Index,columns = Columns)
        # assign the indeces into the indeces dict

        '''
        mr index,columns
        '''
        Columns = pd.MultiIndex.from_product([regions,_MONEY_RATES.keys()])
        Index   = pd.Index(self.ids.all_years)
        indeces['mr'] = dict(index=Index,columns=Columns)


        self.indeces = Dict(dict,**indeces)

        """
        te index,columns: total emission
        """
        Index = pd.Index(self.ids.all_years)
        Columns = pd.Index(self.ids.emission_by_flows)
        self.indeces['te'] = dict(index=Index,columns=Columns)
        
        """
        tin index,columns: total investment costs
        """
        Index = pd.Index(self.ids.all_years)
        #Columns = pd.Index(_COST_ITEMS.keys())
        Columns = pd.MultiIndex.from_product([regions,_COST_ITEMS.keys()])
        self.indeces['tin'] = dict(index=Index,columns=Columns)
     
      


