#-*- coding: utf-8 -*-
"""
Created on Fri Jul 30 17:36:56 2021

@author: Amin
"""
from ESM.utils.constants import (_SETS_READ,
                                 _MI,
                                 _CLUSTER_YEARLY_SHEETS,
                                 _CLUSTER_SLICE_SHEETS,
                                 _AV_MAP,
                                 _COST_ITEMS
                                 )

from ESM.utils.errorCheck import  (check_excel_col_row,
                                   validate_data,
                                   check_file_extension,
                                   nans_exist,
                                   )

from ESM.utils.tools import (delete_duplicates,
                             remove_items,
                             generate_random_colors,
                             dataframe_to_xlsx,
                             excel_read,
                             )

from ESM.log_exc.exceptions import (WrongInput,
                                    WrongExcelFormat,
                                    )

from ESM.utils import cvxpyModified as cm
from ESM.log_exc.logging import log_time
from ESM.core.Properties import Identifiers
from ESM.core.Clusters import NonTimeCluster,TimeCluster
from ESM.core.FreezeDict import Dict

import cvxpy as cp
import numpy as np
import pandas as pd
import logging
from copy import deepcopy as dc

logger = logging.getLogger(__name__)

class Base(Identifiers):
    '''
    DESCRIPTION
    ============
    Base class, provides the basic methods and properties of the energy model.
    The user will not instanciate this class.
    This class is created as a parent class for the Model class
    '''
    
    def __readsets__(self,path):
        
        '''
        DESCRIPTION
        ============
        This function is in charge of reading all the sets based on the predifend structure
        by _SETS_READ and sets the attrubutres.
        
        '''
        if isinstance(path,str):
            # Read the sets from excel file
            self._read_sets_from_excel(path)
            
        self.__set_properties__()
        self.__Indexer__()
        self.time_cluster = TimeCluster(self)
        self.__Frames__()    
        self.data = dc(dict(**self.__matrices__))
            
    
    def _item_indexer(self,item):
        '''
        Function returns some infromation regarding the index levels 
        when reading excel files.
        '''
        
        if item == 'TechnologyData':
            index    = self.indeces['u']['index']
            columns  = self.indeces['u']['columns']
            matrices = ['u','bp','wu','bu','st','cu','tp',]
            
        elif item == 'DemandProfiles':
            index    = self.indeces['dp']['index']
            columns  = self.indeces['dp']['columns'] 
            matrices = ['dp'] 
        
        elif item == 'FlowData':
            index    = self.indeces['v']['index']
            columns  = self.indeces['v']['columns']            
            matrices = ['v','bv',]
            
        elif item == 'Demand':
            index    = self.indeces['E']['index'] 
            columns  = self.indeces['E']['columns'] 
            matrices = self.ids.consumption_sectors
            
        elif item == 'DemandCoefficients':
            index    = self.indeces['e']['index'] 
            columns  = self.indeces['e']['columns'] 
            matrices = ['e',]            
            
        elif item == 'Availability':
            all_avaliabilities = pd.concat([pd.DataFrame(index   = ['dummy'],
                                                         columns = self.indeces[item]['columns']
                                                         )
                                            for item in ['af_eq','af_min','af_max']
                                            ],
                                           axis = 1,
                                           )
            
            # the three levels of eq,min and max have the same index
            index    = self.indeces['af_eq']['index']
            columns  = all_avaliabilities.columns  
            matrices = ['af_eq','af_min','af_max']
        
        elif item == 'TechProductionMix':
            index   = self.indeces['xy_mix']['index']
            columns = self.indeces['xy_mix']['columns']
            matrices= ['xy_mix'] 
            
        if isinstance(columns,pd.MultiIndex):
            col_range = len(columns[0])
        else:
            col_range = 1
            
        if isinstance(index,pd.MultiIndex):
            ind_range = len(index[0])
        else:
            ind_range = 1
            
        return dict(index     = index,
                    columns   = columns,
                    header    = [i for i in range(col_range)],
                    index_col = [i for i in range(ind_range)],
                    matrices  = matrices,
                    )
                    

    def _read_paramaters_from_excel(self,
                                    file : str,
                                    item : str,
                                    ) -> None:
        '''
        Reading the inputs from excel file
        '''
        all_clusters = self.time_cluster.parameter_clusters('all')
        __indexer__ = self._item_indexer(item)
        
        if item in ['Demand','DemandCoefficients']:
            to_fill = self.data['E']
            for sheet in __indexer__['matrices']:
                data = excel_read(io= file,
                                  sheet_name= sheet,
                                  header= __indexer__['header'],
                                  index_col= __indexer__['index_col']
                                  )
                
                check_excel_col_row(given= data.index,
                                    correct= __indexer__['index'],
                                    file= file,
                                    sheet= sheet,
                                    level= 'index',
                                    check= 'equality',
                                    )
                
                check_excel_col_row(given= data.columns,
                                    correct= __indexer__['columns'],
                                    file= file,
                                    sheet= sheet,
                                    level= 'columns',
                                    check= 'equality',
                                    )    
                

                index  = to_fill.index
                columns= self.ids.all_years
               
               
                to_fill.loc[index,(sheet,columns)] = data.loc[(index.get_level_values(0),
                                                               index.get_level_values(1),
                                                               slice(None)
                                                               ),
                                                              columns
                                                              ].values
                
                                
        elif item == 'DemandProfiles':
            for product in self.ids.hourly_products:
                for cluster in self.time_cluster.parameter_clusters('dp'):
                    
                    sheet_name = f'{product}.{cluster}'
                    data = excel_read(io= file,
                                      sheet_name= sheet_name,
                                      header= __indexer__['header'],
                                      index_col= __indexer__['index_col']
                                      )
            
                    # checking if the columsn are correct for every sheet to read
                    check_excel_col_row(given= data.columns,
                                        correct= __indexer__['columns'],
                                        file= file,
                                        sheet= sheet_name,
                                        level= 'columns',
                                        check= 'equality',
                                        )           
            
                    data.columns = [data.columns.get_level_values(i) for i in [0,1]]
                    
                    to_fill = self.data['dp'][cluster][product]
                    index   = to_fill.index
                    columns = to_fill.columns
                    
                    try:
                        self.data['dp'][cluster][product].loc[index,columns] =\
                        data.loc[index,columns].values
                    except KeyError:
                        raise WrongExcelFormat('Possible issues found in the indexing in {}, sheet {}. '
                                                'To avoid problems it is suggested to use the built-in functions '
                                                'to print the input files.'.format(file,sheet_name))    
                        
        elif item == 'TechProductionMix':
            frame = pd.DataFrame()
            to_fill = self.data['xy_mix']
            for region in self.Regions:
                data = excel_read(io= file,
                                  sheet_name= region,
                                  header= __indexer__['header'],
                                  index_col= __indexer__['index_col']
                                  ) 
                
                
                check_excel_col_row(given= data.columns,
                                    correct= __indexer__['columns'],
                                    file= file,
                                    sheet= region,
                                    level= 'columns',
                                    check= 'equality',
                                    )
                
                index     = to_fill.index
                
                columns_0 = data.columns.get_level_values(0).tolist()
                columns_1 = data.columns.get_level_values(1).tolist()
                try:
                    to_fill.loc[index,(region,columns_0,columns_1)] = data.loc[index,(columns_0,columns_1)] .values
                
                except KeyError as e:
                    raise WrongExcelFormat('Model can not find {} in {}, sheet {}'
                                           '. This can be due to nan values in the excel file.'.format(e.args,
                                                                                                       file,
                                                                                                       sheet_name))
        else:
            for region in getattr(self,_MI['r']):
                for cluster in all_clusters:
                    sheet_name =f'{region}.{cluster}'
                    
                    data = excel_read(io= file,
                                      sheet_name= sheet_name,
                                      header= __indexer__['header'],
                                      index_col= __indexer__['index_col']
                                      )
                        
                    for matrix in __indexer__['matrices']:
                        # search for the data of the matrix if it exist in the cluster
                        if cluster in self.time_cluster.parameter_clusters(matrix):
                            # checking if the columns are correct for every sheet to read
                            check_excel_col_row(given= data.columns,
                                                correct= __indexer__['columns'],
                                                file= file,
                                                sheet= sheet_name,
                                                level= 'columns',
                                                check= 'equality',
                                                )                             
                            if item == 'Availability':
                                try:
                                    take    = data[_AV_MAP[matrix]]
                                except KeyError as e:
                                    raise WrongExcelFormat('Model can not find {} in {}, sheet {}'
                                                           '. This can be due to nan values in the excel file.'.format(e.args,
                                                                                                                       file,
                                                                                                                       sheet_name))
                                take.columns = take.columns.get_level_values(1)
                                
                                to_fill = self.data[matrix][cluster]
                                
                                index   = to_fill.index
                                columns = delete_duplicates(to_fill.columns.get_level_values(-1))
                                
                                try:
                                    self.data[matrix][cluster].loc[index,(region,columns)] =\
                                        take.loc[index,columns].values
    
                                except KeyError:
                                    raise WrongExcelFormat('Possible issues found in the indexing in {}, sheet {} for matrix {}. '
                                                           'To avoid problems it is suggested to use the built-in functions '
                                                           'to print the input files.'.format(file,sheet_name,matrix))                           
                                
                                
                            elif item in ['TechnologyData','FlowData']:
                
                                try:
                                    take    = data.loc[matrix,:]
                                except KeyError as e:
                                    raise WrongExcelFormat('Model can not find {} in {}, sheet {}'
                                                           '. This can be due to nan values in the excel file.'.format(e.args,
                                                                                                                       file,
                                                                                                                       sheet_name))
                                
                                # we need to change the index format to be inline
                                # with what we have in __matrices__
                                if item == 'TechnologyData':
                                    index_level = 0
                                    take.index   = take.index.get_level_values(index_level)
                                    take.columns = take.columns.get_level_values(1) 
                                    
                                    to_fill = self.data[matrix][cluster]
                                    
                                    index   = delete_duplicates(to_fill.index.get_level_values(-1))
                                    columns = delete_duplicates(to_fill.columns.get_level_values(-1))
                                    
                                    if matrix in ['bp','wu','bu','st','cu','tp']:
                                        indexer = index
                                    else:
                                        indexer = (region,index)
                                        
                                else:            
                                    if matrix == 'v':
                                        take.index = [take.index.get_level_values(i)
                                                      for i in [0,1]]
                                    else:
                                        take.index = take.index.get_level_values(1)
                                
                                    take.columns = take.columns.get_level_values(0)
                                    
                                    to_fill = self.data[matrix][cluster]
                                    
                                    index   = to_fill.index
                                    indexer = index
                                    columns = delete_duplicates(to_fill.columns.get_level_values(-1))
                                    
                                try:
                                    self.data[matrix][cluster].loc[indexer,(region,columns)] =\
                                        take.loc[index,columns].values
    
                                except KeyError:
                                    raise WrongExcelFormat('Possible issues found in the indexing in {}, sheet {} for matrix {}. '
                                                           'To avoid problems it is suggested to use the built-in functions '
                                                           'to print the input files.'.format(file,sheet_name,matrix))            

        
    def _read_sets_from_excel(self,
                              path : str) -> None:
        
        '''
        This function will be used in read __readsets__ function if the given 
        path is a str.
        
        '''
        
        self.warnings    = []
        self.non_time_clusters = NonTimeCluster()
        
        
        sets_frames = {}
        for set_name,info in _SETS_READ.items():
            
            data = pd.read_excel(path, **info['read'])

            log_time(logger,f'Sets: {set_name} sheet imported successfully.')
            
            check_excel_col_row(given   = list(data.columns),
                                correct = info['columns'],
                                file    = path,
                                sheet   = info['read']['sheet_name'],
                                level   = 'columns',
                                check   = 'contain'
                                )
            
            for validation_item,acceptable_values in info['validation'].items():
                
                if isinstance(acceptable_values,str):
                    acceptable_values = eval(acceptable_values)

                validate_data(list(data[validation_item]),
                              acceptable_values,
                              f'{set_name}, column:{validation_item}')             
                
            for non_acceptable_nan_columns in info['stop_nans']:
                nans_exist(data   = data[non_acceptable_nan_columns],
                           action = 'raise error',
                           info   = f'{set_name}: {non_acceptable_nan_columns}')
                
            sets_without_nans = remove_items(data[info['set']], nans= True)
            sets_unique       = delete_duplicates(sets_without_nans,
                                                  warning=True,
                                                  comment=f'{set_name} has duplciate values'
                                                          ' in the rows. only the first row of'
                                                          ' duplicate values will be kept.',
                                                  level = 'critical',
                                                  )
            
            sets_sorted       = sorted(sets_unique) if info['sort'] else sets_unique
            
            # Filling the default values
            data = self.__default_values__( data     = data,
                                            category = 'sets',
                                            info     = info,
                                            name     = set_name,
                                            )
            
            
            
            # Setting the index of dataframe based on unique sets_unique
            data = data.drop_duplicates(subset = info['set'])
            data = data.set_index([info['set']])

            data = data.loc[sets_sorted,:]
            
            if set_name == _MI['h'] and data.shape[0] != 1:
                raise WrongInput('for {}, only one item (row) can be defined.'.format(_MI['h']))
                
            if set_name !=_MI['h'] :
                self.non_time_clusters.check_cluster_exists(dataframe = data,
                                                            set_name  = set_name)
                
                data = self.non_time_clusters.re_organize_main_dataframes(set_name)
            
            sets_frames[set_name] = data
            setattr(self,set_name,sets_sorted)

            log_time(logger,f'Sets: {set_name} creted successfully')
            
        self.__sets_frames__ = Dict(**sets_frames)


    def __generate_excel__(self,
                           path,
                           what : str):
        
        '''
        This function generates formatted excel input files 
        
        '''
                
        write = True
        sheets = {}
        all_clusters = self.time_cluster.parameter_clusters('all')
        
        if what == 'TechnologyData':
             
            for region in self.Regions:
                for cluster in all_clusters:
                    
                    frame = pd.DataFrame()
                    sheet_name = f'{region}.{cluster}'

                    
                    for item in _CLUSTER_YEARLY_SHEETS:

                        if item in ['v','bv','E','m','e']:
                            continue
                        if cluster in self.time_cluster.parameter_clusters(item):

                            new_frame = pd.DataFrame(data    = 0,
                                                     index   = self.indeces[item]['index'],
                                                     columns = self.indeces[item]['columns']
                                                     ) 
                            
                            frame = pd.concat([frame,new_frame])
                    
                    sheets[sheet_name] = frame

        elif what == 'TechProductionMix':
            
            for region in self.Regions:
                
                sheets[region] = pd.DataFrame(data    = 0,
                                              index   = self.indeces['xy_mix']['index'],
                                              columns = self.indeces['xy_mix']['columns'],
                                              )
                

                
                
                
        elif what == 'FlowData':
             
            for region in self.Regions:
                for cluster in all_clusters:
                    
                    frame = pd.DataFrame()
                    sheet_name = f'{region}.{cluster}'
                    
                    for item in ['v','bv']:

                        if cluster in self.time_cluster.parameter_clusters(item):
                            
                            new_frame = pd.DataFrame(data    = 0,
                                                     index   = self.indeces[item]['index'],
                                                     columns = self.indeces[item]['columns']
                                                     )                            
                            frame = pd.concat([frame,new_frame])
                    
                    sheets[sheet_name] = frame
                    
        elif what == 'DemandProfiles':
            
            for product in self.ids.hourly_products:
                for cluster in all_clusters:
                    sheet_name = f'{product}.{cluster}'
                    if cluster in self.time_cluster.parameter_clusters('dp'):
                        frame = pd.DataFrame(data    = 0,
                                             index   = self.indeces['dp']['index'],
                                             columns = self.indeces['dp']['columns'],
                                             )

                    sheets[sheet_name] = frame
                
        
        elif what == 'Availability':
            
            for region in self.Regions:
                for cluster in all_clusters:
                    
                    frame = pd.DataFrame()
                    sheet_name = f'{region}.{cluster}'
                    
                    for item in _CLUSTER_SLICE_SHEETS:
                        if cluster in self.time_cluster.parameter_clusters(item) and item != 'dp' :
                            new_frame = pd.DataFrame(data    = 0,
                                                     index   = self.indeces[item]['index'],
                                                     columns = self.indeces[item]['columns']
                                                     )                   
                            
                            frame = pd.concat([frame,new_frame],axis=1)
                    
                    sheets[sheet_name] = frame
                    
        elif what == 'Demand':
            if self.mode == 'stand-alone':
                
                for sector in self.ids.consumption_sectors:
                    sheets[sector] = pd.DataFrame(data    = 0,
                                                  index   = self.indeces['E']['index'],
                                                  columns = self.indeces['E']['columns']
                                                  )     

            else:
                write = False
            
        elif what == 'DemandCoefficients':
            if self.mode == 'sfc-integrated':
                sheets['e'] = pd.DataFrame(data    = 0,
                                           index   = self.indeces['e']['index'],
                                           columns = self.indeces['e']['columns']
                                           )     
            else:
                write = False                

        if write:      
            dataframe_to_xlsx(path,**sheets)
            log_time(logger, f'ExcelWriter: file {path} created successfully.')
            
            
            
    def __default_values__(self,
                           data : [pd.DataFrame],
                           category : str,
                           **kwargs) -> [pd.DataFrame]:
        
        '''
        DESCRIPTION
        =============
        The function is in charge of finding the default values.
        
        PARAMETERS
        =============
        data     : the data to fill the missing values
        category : defines which kind of information are suppused to give to 
                   the function
                   
        kwargs   : info -> in case of sets, info should be passed.
        '''
        
        if category == 'sets':
            
            info = kwargs.get('info')
            
            assert info is not None, 'For sets, we need the info dictionary to be given to the function.'
            
            for item,default in info['defaults'].items():
                                
                missing_items = data.loc[data[item].isna(),item]
                
                # if any missing item exists
                if len(missing_items):
                    
                    data.loc[data[item].isna(),item] = eval(default)
                    
                    set_name = kwargs.get('name')
                    self.warnings.append('{} for {}, {} is missed and filled by default values.'.format(item,
                                                                                                        set_name,
                                                                                                        data.loc[missing_items.index,info['set']].values.tolist()
                                                                                                        )
                                         )
                    
                
            return data
        
        elif category == 'inputs':
            parameter = kwargs.get('parameter')
            
            assert parameter is not None, 'For inputs, we need to specify the parameter'
            
            '''fill the parameters default values'''


    def _model_generation(self):
        
        '''
        DESCRIPTION
        =============
        The function generates endogenous/exogenous variables, sets and solve
        the optimization problem and generates numerical results.
        
        '''
        
        # DEFINITION OF SECTOR-TECHNOLOGY IDENTITY MATRIX
        I_st = pd.DataFrame(np.zeros((list(self.__matrices__['v'].values())[0].shape[0],list(self.__matrices__['u'].values())[0].shape[1])),
                            index = list(self.__matrices__['v'].values())[0].index,
                            columns = list(self.__matrices__['u'].values())[0].columns)
        
        techs_frame = self.__sets_frames__[_MI['t']]
                       
        for region in self.Regions:
             for sector in self.ids.production_sectors:
                 techs = techs_frame.loc[(techs_frame['SECTOR']==sector) & (techs_frame['TYPE']!='storage')].index.tolist()
                 I_st.loc[(region,sector),(region,techs)] = 1
 
    
        # DEFINITION OF ENDOGENOUS/EXOGENOUS PARAMETERS 
        var = {}
        par = {}
        
        for item in self.endogenous: 
            var[item] = {}
            
            if item not in ['cap_o','cap_n','cap_d']:
            
                for y in self.ids.run_years:
                    var[item][y] = cm.Variable(shape = self.__matrices__[item].shape,
                                               nonneg = False,
                                               index = self.__matrices__[item].index,
                                               columns = self.__matrices__[item].columns)
                
                for y in self.ids.warm_up_years + self.ids.cool_down_years:
                    if item in ['xh','qh','soc']:
                        pass
                    else: 
                        var[item][y] = cm.Variable(shape = self.__matrices__[item].shape,
                                                   nonneg = False,
                                                   index = self.__matrices__[item].index,
                                                   columns = self.__matrices__[item].columns)
            else:
                var[item] = cm.Variable(shape = self.__matrices__[item].shape,
                                        nonneg = False,
                                        index = self.__matrices__[item].index,
                                        columns = self.__matrices__[item].columns)
        
        for item in self.exogenous:
            par[item] = {}
            
            if item in ['xy_mix','E','E_tld','E_tld_diag','e',]:
                par[item] = cm.Parameter(shape = self.__matrices__[item].shape,
                                         index = self.__matrices__[item].index,
                                         columns = self.__matrices__[item].columns,)
            
            elif item == 'dp':
                for y in self.ids.run_years:
                    par[item][y] = {}
                    
                    for flow in list(self.__matrices__[item].values())[0].keys():
                            par[item][y][flow] = cm.Parameter(shape = list(self.__matrices__[item].values())[0][flow].shape,
                                                              index = list(self.__matrices__[item].values())[0][flow].index,
                                                              columns = list(self.__matrices__[item].values())[0][flow].columns,)
            
            else:
                for y in self.ids.all_years:
                    try:  
                        par[item][y] = cm.Parameter(shape = list(self.__matrices__[item].values())[0].shape,
                                                    index = list(self.__matrices__[item].values())[0].index,
                                                    columns = list(self.__matrices__[item].values())[0].columns,)
                    except ValueError: 
                        if item in ['wu','bu','st','bv']:
                            pass
                        else: 
                            raise

        
        # ===============================================================================================================================        
        # MASK FUNCTION FOR SLICING VARIABLES/PARAMETERS
        def p(name : str,
              year : int = None,
              lv = slice(None), # eventual sub-level (requested for demand profiles only, sub-indexed by flow)
              r1 = slice(None), # row level 1
              r2 = slice(None), # row level 2
              c1 = slice(None), # col level 1
              c2 = slice(None), # col level 2
              c3 = slice(None), # col level 3
              ):
            
            # parameter definition
            if name in self.exogenous:
                if isinstance(par[name],dict):
                    if year not in self.ids.all_years:
                        raise AssertionError('a year within time horizon must be passed as argument')
                    parameter = par[name][year]
                else:
                    if year not in self.ids.all_years + [None]:
                        raise AssertionError('if a year is passed, it must be within the time horizon')
                    parameter = par[name]

            elif name in self.endogenous:
                if isinstance(var[name],dict):
                    if year not in self.ids.all_years:
                        raise AssertionError('a year within time horizon must be passed as argument')
                    parameter = var[name][year]
                else:
                    if year not in self.ids.all_years + [None]:
                        raise AssertionError('if a year is passed, it must be within the time horizon')
                    parameter = var[name]            
            
            elif name == 'I_st':
                if year != None:
                    raise AssertionError('year must not be passed as argument')
                parameter = I_st                
            
            else:
                raise AssertionError('name of parameter is not valid or not defined in mask function')
            
            # slicing keys
            slc = { 'slice(None, None, None)' : slice(None),
                    'hh' : self.ids.time_slices,
                    'fh' : self.ids.hourly_products,
                    'fy' : self.ids.yearly_products,
                    'sh' : self.ids.hourly_sectors,
                    'sy' : self.ids.yearly_sectors,
                    'th' : self.ids.hourly_techs,
                    'ty' : self.ids.yearly_techs,
                    'sc' : self.ids.consumption_sectors,
                    'ts' : self.ids.storages,
                    'nts': list(set(self.Technologies)-set(self.ids.storages)),
                    'tc' : self.ids.capacity_techs,
                    'tce': self.ids.capacity_techs_equality,
                    'tcr': self.ids.capacity_techs_range,
                    'tcd': self.ids.capacity_techs_demand,
                    }            
            
            for item in ['t','s','r','f']:
                for k in list(self.__sets_frames__[_MI[item]].index):
                    slc[k] = []
                    slc[k].append(k)
            
            # slicing
            if isinstance(parameter,dict):
                if lv == slice(None):
                    raise AssertionError('a sub-level must be specified before slicing')
                else:
                    parameter = parameter[lv]            
            
            if name in ['v','u']:
                sliced_parameter = parameter.cloc[(slc[str(r1)],slc[str(r2)]),:].cloc[:,(slc[str(c1)],slc[str(c2)])]
            
            elif name in ['wu','bp','bu','cu','tp','st','bv','xy','xh','qy','qh','CU','af_eq','af_min','af_max','soc','dp',]:
                if str(r1) in slc.keys():
                    sliced_parameter = parameter.cloc[slc[str(r1)],:].cloc[:,(slc[str(c1)],slc[str(c2)])]
                else: 
                    sliced_parameter = parameter.cloc[[r1],:].cloc[:,(slc[str(c1)],slc[str(c2)])]  
            
            elif name in ['E','E_tld','e','m',]:
                if year == None:
                    sliced_parameter = parameter.cloc[(slc[str(r1)],slc[str(r2)]), :].cloc[:, (slc[str(c1)],slice(None))]
                else:
                    sliced_parameter = parameter.cloc[(slc[str(r1)],slc[str(r2)]), :].cloc[:, (slc[str(c1)],[year])]            
            
            elif name in ['E_tld_diag']:
                if year == None:
                    sliced_parameter = parameter.cloc[(slc[str(r1)],slc[str(r2)]), :].cloc[:, (slc[str(c1)],slc[str(c2)],slice(None))]
                else:
                    sliced_parameter = parameter.cloc[(slc[str(r1)],slc[str(r2)]), :].cloc[:, (slc[str(c1)],slc[str(c2)],[year])]
            
            elif name in ['cap_o','cap_n','cap_d']:
                if year == None:
                    sliced_parameter = parameter.cloc[:,(slc[str(c1)],slc[str(c2)])]
                else: 
                    sliced_parameter = parameter.cloc[[year],:].cloc[:,(slc[str(c1)],slc[str(c2)])]   

            elif name in ['xy_mix']:
                if year == None:
                    sliced_parameter = parameter.cloc[:,(slc[str(c1)],slc[str(c2)],slc[str(c3)])]
                else: 
                    sliced_parameter = parameter.cloc[[year],:].cloc[:,(slc[str(c1)],slc[str(c2)],slc[str(c3)])] 
                    
            elif name in ['I_st']:
                sliced_parameter = parameter.loc[(slc[str(r1)],slc[str(r2)]),:].loc[:,(slc[str(c1)],slc[str(c2)])]
                    
            else: 
                raise AssertionError('parameter not defined within the slicer function')
            
            return sliced_parameter


        # ===============================================================================================================================        
        # WARM-UP PROBLEM - OBJECTIVE FUNCTION
        esm_warmup_obj = cp.Minimize( sum([cm.rcsum(cm.rcsum(p('CU',y),0),1) for y in self.ids.warm_up_years]) )
        
        # WARM-UP PROBLEM - CONSTRAINTS
        esm_warmup_eqs = []

        # new, disposed, operative capacities by year always positive or zero
        esm_warmup_eqs.append( p('cap_o') >= 0 )
        esm_warmup_eqs.append( p('cap_n') >= 0 )
        esm_warmup_eqs.append( p('cap_d') >= 0 )  
        
        for y in self.ids.warm_up_years:

            # total yearly production by technology always positive
            esm_warmup_eqs.append( p('qy',y) >= 0 )
            esm_warmup_eqs.append( p('xy',y) >= 0 )
            esm_warmup_eqs.append( p('CU',y) >= 0 )

            # production of storage technology zero (avoids free energy generation)
            esm_warmup_eqs.append( p('xy',y,c2='ts') == 0 )
    
            # production balance, yearly flows
            esm_warmup_eqs.append( cm.trsp(p('qy',y)) - 
                                   cm.matmul(p('u',y), cm.trsp(p('xy',y))) - 
                                   p('E',y) == 0 )
            
            # production balance, yearly sectors
            esm_warmup_eqs.append( cm.matmul(p('v',y), cm.trsp(p('qy',y))) -  
                                   cp.matmul(p('I_st').values, cm.trsp(p('xy',y))) == 0 )
            
            # capacity constraints (averaged yearly availability)
            esm_warmup_eqs.append( p('xy',y,c2='tce') == cp.multiply(cm.rcsum(p('af_eq',y,c2='tce'),0)*(1/self.ids.period_length), p('cap_o',y,c2='tce')*8760) )
            esm_warmup_eqs.append( p('xy',y,c2='tcr') >= cp.multiply(cm.rcsum(p('af_min',y,c2='tcr'),0)*(1/self.ids.period_length), p('cap_o',y,c2='tcr')*8760) )
            esm_warmup_eqs.append( p('xy',y,c2='tcr') <= cp.multiply(cm.rcsum(p('af_max',y,c2='tcr'),0)*(1/self.ids.period_length), p('cap_o',y,c2='tcr')*8760) )   

            # capacity balances 
            if y == self.ids.warm_up_years[0]:
                esm_warmup_eqs.append( p('cap_o',y) == p('cap_n',y) ) # - p('cap_d',y) 
            else:
                esm_warmup_eqs.append( p('cap_o',y) == p('cap_o',y-1) + p('cap_n',y) ) # - p('cap_d',y) 
                
            # for technologies like housing/transport, the operative capacity must be used
            esm_warmup_eqs.append( p('xy',y,c2='tcd') == cm.multiply(p('tp',y,r1='t_ca',c2='tcd'), p('cap_o',y,c2='tcd')) )
            
            # disposed capacity
            #todo
            
            # cost items (add new cost items)
            esm_warmup_eqs.append( p('CU',y,r1='c_fu') == cm.matmul(p('cu',y,r1='c_fu'), cm.diag(p('xy',y))) )                        
            esm_warmup_eqs.append( p('CU',y,r1='c_in',c2='tc') == cm.matmul(p('cu',y,r1='c_in',c2='tc'), cm.diag(p('cap_n',y))) )                        

            # emissions by flow, by technology, by sector
            #todo
            
            # constraints on generation mix xy_mix
            esm_warmup_eqs.append( p('xy',y) == cm.multiply(p('xy_mix',y), cm.matmul(cm.matmul(p('xy',y), cm.trsp(I_st)), I_st)) )

         
        # ===============================================================================================================================                                 
        # RUN and COOL-DOWN PROBLEM - OBJECTIVE FUNCTION
        esm_run_obj = cp.Minimize( sum([cm.rcsum(cm.rcsum(p('CU',y),0),1) for y in self.ids.run_years + self.ids.cool_down_years]) )
        
        # RUN and COOL-DOWN PROBLEM - CONSTRAINTS
        esm_run_eqs = []

        # constraints to both run and cool-down periods
        for y in self.ids.run_years + self.ids.cool_down_years:

            # new, disposed, operative capacities by year always positive or zero
            esm_run_eqs.append( p('cap_o',y) >= 0 )
            esm_run_eqs.append( p('cap_n',y) >= 0 )
            esm_run_eqs.append( p('cap_d',y) >= 0 )  
            
            # total yearly production by technology always positive
            esm_run_eqs.append( p('qy',y) >= 0 )
            esm_run_eqs.append( p('xy',y) >= 0 )
            esm_run_eqs.append( p('CU',y) >= 0 )
                        
            # capacity balance
            # if y == self.ids.run_years[0]:
            #     esm_run_eqs.append( p('cap_o',y) == 0 + p('cap_n',y) ) # - p('cap_d',y) 
            # else:
            #     esm_run_eqs.append( p('cap_o',y) == p('cap_o',y-1) + p('cap_n',y) ) # - p('cap_d',y)             
            
            
            esm_run_eqs.append( p('cap_o',y) == p('cap_o',y-1) + p('cap_n',y) ) # - p('cap_d',y) 
            


                
            # limits in annual capacity expansion by technology (tricky, carefully check data!)
            esm_run_eqs.append( p('cap_n',y) <= cm.multiply(p('cap_o',y-1), p('tp',y,r1='t_mi',c2='tc')) )
            
            # for technologies like housing/transport, the operative capacity must be used
            esm_run_eqs.append( p('xy',y,c2='tcd') == cm.multiply(p('tp',y,r1='t_ca',c2='tcd'), p('cap_o',y,c2='tcd')) )
            
            # disposed capacity
            #todo
            
            # cost items (add new cost items)
            esm_run_eqs.append( p('CU',y,r1='c_fu') == cm.matmul(p('cu',y,r1='c_fu'), cm.diag(p('xy',y))) )                        
            esm_run_eqs.append( p('CU',y,r1='c_in',c2='tc') == cm.matmul(p('cu',y,r1='c_in',c2='tc'), cm.diag(p('cap_n',y))) )                        

            # emissions by flow, by technology, by sector
            #todo

        
        # constraints applied to cool-down period only (yearly resolution)
        for y in self.ids.cool_down_years:
            
            # production of storage technology zero (avoids free energy generation)
            esm_run_eqs.append( p('xy',y,c2='ts') == 0 )

            # production balance, yearly flows
            esm_run_eqs.append( cm.trsp(p('qy',y)) - 
                                cm.matmul(p('u',y), cm.trsp(p('xy',y))) - 
                                p('E',y) == 0 )
            
            # production balance, yearly sectors
            esm_run_eqs.append( cm.matmul(p('v',y), cm.trsp(p('qy',y))) -  
                                cp.matmul(p('I_st').values, cm.trsp(p('xy',y))) == 0 )
            
            # capacity constraints (averaged yearly availability)
            esm_run_eqs.append( p('xy',y,c2='tce') == cp.multiply(cm.rcsum(p('af_eq',y,c2='tce'),0)*(1/self.ids.period_length), p('cap_o',y,c2='tce')*8760) )
            esm_run_eqs.append( p('xy',y,c2='tcr') >= cp.multiply(cm.rcsum(p('af_min',y,c2='tcr'),0)*(1/self.ids.period_length), p('cap_o',y,c2='tcr')*8760) )
            esm_run_eqs.append( p('xy',y,c2='tcr') <= cp.multiply(cm.rcsum(p('af_max',y,c2='tcr'),0)*(1/self.ids.period_length), p('cap_o',y,c2='tcr')*8760) )   
        
        
        # constraints applied to run period (hourly resolution)
        for y in self.ids.run_years:
            
            # total hourly production always positive (except for storages)
            esm_run_eqs.append( p('qh',y) >= 0 )
            esm_run_eqs.append( p('xh',y,c2='nts') >= 0 ) 
            
            # for x,q: summing production by hours and nesting into production by year
            esm_run_eqs.append( p('xy',y,c2='th') == cm.multiply(cm.rcsum(p('xh',y,c2='th'),0),(8760/self.ids.period_length)) )
            esm_run_eqs.append( p('qy',y,c2='fh') == cm.multiply(cm.rcsum(p('qh',y,c2='fh'),0),(8760/self.ids.period_length)) )
            
            # production balance, yearly flows/sectors
            esm_run_eqs.append( cm.trsp(p('qy',y,c2='fy')) - 
                                cm.matmul(p('u',y,r2='fy'), cm.trsp(p('xy',y))) - 
                                cm.rcsum(p('E_tld',y,r2='fy'),1) == 0 )
            
            esm_run_eqs.append( cm.matmul(p('v',y,r2='sy'), cm.trsp(p('qy',y))) -  
                                cp.matmul(p('I_st',r2='sy').values, cm.trsp(p('xy',y))) == 0 ) 
            
            # production balance, hourly flows/sectors
            for flow in self.ids.hourly_products:
                esm_run_eqs.append( cm.trsp(p('qh',y,c2=flow)) -
                                    cp.matmul(cm.matmul(p('u',y,r2=flow,c2='ty'), cm.diag(p('xy',y,c2='ty')))*(self.ids.period_length/8760), cm.trsp(p('dp',y,lv=flow,c2='ty'))) -
                                    cm.matmul(p('u',y,r2=flow,c2='th'), cm.trsp(p('xh',y,c2='th'))) -
                                    cm.matmul(p('E_tld_diag',y,r2=flow), cm.trsp(p('dp',y,lv=flow,c2='sc'))) == 0 )
            
            esm_run_eqs.append( cm.matmul(p('v',y,r2='sh',c2='fh'), cm.trsp(p('qh',y,c2='fh'))) - 
                                cp.matmul(p('I_st',r2='sh',c2='th').values, cm.trsp(p('xh',y,c2='th'))) == 0 )
            
            # capacity constraints by hour
            esm_run_eqs.append( p('xh',y,c2='tce') == cm.multiply(p('af_eq',y,c2='tce'), p('cap_o',y,c2='tce')) )
            esm_run_eqs.append( p('xh',y,c2='tcr') >= cm.multiply(p('af_min',y,c2='tcr'), p('cap_o',y,c2='tcr')) )
            esm_run_eqs.append( p('xh',y,c2='tcr') <= cm.multiply(p('af_max',y,c2='tcr'), p('cap_o',y,c2='tcr')) )
            
            # storage 1: state of charge by hour
            esm_run_eqs.append( p('soc',y) == cm.multiply(p('st',y,r1='st_soc_start'), p('cap_o',y,c2='ts')) +
                                              cp.matmul(np.tril(np.ones([self.ids.period_length,self.ids.period_length])), -p('xh',y,c2='ts')) )
            
            # storage 2: all soc periods greater than minimum
            esm_run_eqs.append( p('soc',y) >= cm.multiply(p('st',y,r1='st_soc_min'), p('cap_o',y,c2='ts')) ) # check dimensions 
            
            # storage 3: last soc period constrained
            if y == self.ids.run_years[0]:
                esm_run_eqs.append( p('soc',y)[[0],:] == cm.multiply(p('st',y,r1='st_soc_start'), p('cap_o',y,c2='ts')) )
            else:
                esm_run_eqs.append( p('soc',y)[[0],:] == p('soc',y-1)[[-1],:] )
            
            # alternatively, constrain the last and the first hours to be equal (all not working)
            # esm_eqs.append( p('soc',y,r2=self.ids.period_length) == cm.multiply(p('st',y,r2='state of charge, y.h=-1 [%]'), p('cap_o',y,c2='ts')) )
            # esm_eqs.append( cm.rcsum(p('xh',y,c2='ts'),0) == 0 )
            # esm_eqs.append( cm.multiply(p('st',y,r2='state of charge, y.h=0 [%]'), p('cap_o',y,c2='ts')) +
            #                 cm.rcsum(p('xh',y,c2='ts'),0) == 
            #                 cm.multiply(p('st',y,r2='state of charge, y.h=-1 [%]'), p('cap_o',y,c2='ts')) ) 
                
            # storage 4: soc periods cannot exceed operative capacity
            esm_run_eqs.append( p('soc',y) <= p('cap_o',y,c2='ts') ) # check dimensions 

            # storage 5: charge/discharge rates cannot exceed a given rate
            # absolute value of production split to avoid non-linearity
            esm_run_eqs.append( p('xh',y,c2='ts') <= cm.multiply(p('st',y,r1='st_cd_rate'), p('cap_o',y,c2='ts')) )
            esm_run_eqs.append( -p('xh',y,c2='ts') <= cm.multiply(p('st',y,r1='st_cd_rate'), p('cap_o',y,c2='ts')) )    
                


        self.par_ex = par
        self.par_en = var
        
        self.problem_warmup = cp.Problem(esm_warmup_obj,esm_warmup_eqs)
        self.problem_run = cp.Problem(esm_run_obj,esm_run_eqs)



    def _data_assigment(self):
        
        '''
        DESCRIPTION
        =============
        text
        
        '''

        # PARAMETERS VALUES ASSIGNEMENT

        for item in self.exogenous:
            
            # non-clustered parameters
            if item in ['E','e','xy_mix']:
                self.par_ex[item].value = self.data[item].values.copy()
            
            elif item in ['E_tld','E_tld_diag']:
                E_tld = pd.DataFrame(self.par_ex['E'].value.copy(), self.par_ex['E'].index, self.par_ex['E'].columns)
                E_tld.loc[(slice(None),self.ids.hourly_products),:] = \
                    E_tld.loc[(slice(None),self.ids.hourly_products),:] * self.ids.period_length / 8760
                self.par_ex['E_tld'].value = E_tld.values.copy()
                
                # new reshaped final demand tilde for writing hourly flow balances
                E_tld_diag = pd.DataFrame(0,self.par_ex['E_tld_diag'].index,self.par_ex['E_tld_diag'].columns)
                
                for region in self.Regions:
                    E_tld_diag.loc[(region,slice(None)),(slice(None),region,slice(None))] = \
                        E_tld.loc[(region,slice(None)),:].values
                self.par_ex['E_tld_diag'].value = E_tld_diag.values.copy()
            
            # clustered parameters
            else: 
                for y,cluster in self._year_cluster('run'):                    
                    if item != 'dp':
                        self.par_ex[item][y].value = self.data[item][cluster[item]].values.copy()
                    elif item == 'dp':
                        for flow in self.ids.hourly_products:
                            self.par_ex[item][y][flow].value = self.data[item][cluster[item]][flow].values.copy()
                
                for y,cluster in self._year_cluster('warm_up'):
                    if item == 'dp':
                        pass
                    else:
                        self.par_ex[item][y].value = self.data[item][cluster[item]].values.copy()        
                
                for y,cluster in self._year_cluster('cool_down'):
                    if item == 'dp':
                        pass
                    else:
                        self.par_ex[item][y].value = self.data[item][cluster[item]].values.copy()  
             
   
    
    def _model_run(self):
        
        '''
        DESCRIPTION
        =============
        text
        
        '''

        # PROBLEM SOLVING
        # self.problem_warmup.solve(solver=cp.GUROBI,verbose=True)
        # self.problem_run.solve(solver=cp.GUROBI,verbose=True)
        
        self.problem = self.problem_warmup + self.problem_run
        self.problem.solve(solver=cp.GUROBI,verbose=True)
        
        
        # RESULTS DATAFRAMES
        results = {}
        # if self.problem_run.status == 'optimal' and self.problem_warmup.status == 'optimal':
        for var_key,var in self.par_en.items():
            results[var_key] = {}
            
            if var_key not in ['cap_o','cap_n','cap_d']:
                for year,value in var.items():
                    results[var_key][year] = cm.cDataFrame(value)
            else:
                results[var_key] = cm.cDataFrame(var)
   
        self.results = results    
       
          

#%%

if __name__ == '__main__':

    from ESM import set_log_verbosity
    set_log_verbosity('critical')    

    from ESM import Model
    from ESM.utils import cvxpyModified as cm
    from ESM.utils import constants
    import pandas as pd
    import numpy as np
    import cvxpy as cp

    check = Model(r'ESM\unit_test\set_reader\sets_sample.xlsx',integrated_model = False,)
    
    check.to_excel(path=r'ESM\unit_test\set_reader\sets_sample_code.xlsx',item='sets')
    # check.create_input_excels(r'ESM\unit_test\input_excels')
    check.read_input_excels(r'ESM\unit_test\input_excels')
    check.Base._model_generation()
    check.Base._data_assigment()
    check.Base._model_run()

    # check.generate_clusters_excel(r'ESM\unit_test\time_clusters\clusters.xlsx',  )
    # check.read_clusters(r'ESM\unit_test\time_clusters\clusters.xlsx',  )
    # check.to_excel('sets', r'ESM\unit_test\set_reader\sets_sample_code.xlsx')


#%% plots

    import matplotlib.pyplot as plt
    dpi = 300
    x_axis_y = check.Base.ids.all_years 
    x_axis_h = np.arange(1,check.Base.ids.period_length+1)


    # FINAL DEMAND by FLOW (NON-PRODUCTIVE SECTORS ONLY)
    fig, ax = plt.subplots()    
    
    bot = np.zeros([len(x_axis_y)])
    for flow in check.Base.ids.products:
        ax.bar( x_axis_y,
                np.round(check.Base.par_ex['E'].cloc[(slice(None),flow),(slice(None),x_axis_y)].value.ravel(),0),
                linestyle='-',
                bottom = bot,
                # color='black',
                label=flow,
                )
        bot += np.round(check.Base.par_ex['E'].cloc[(slice(None),flow),(slice(None),x_axis_y)].value.ravel(),0)
    
    ax.set_xlabel('years')
    ax.set_title('final demand by flow')
    ax.legend()
    ax.grid(axis='both',linestyle='--')
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = dpi

    
    # TOTAL PRODUCTION BY FLOW (only energy flows)
    fig, ax = plt.subplots()    
    
    bot = np.zeros([len(x_axis_y)])
    for flow in set(check.Base.ids.products)-{'f.house','f.trans'}:
        ax.bar( x_axis_y, 
                [np.round(check.Base.results['qy'][y].loc[:,(slice(None),flow)].values.ravel()[0],0) for y in x_axis_y],
                linestyle='-',
                bottom = bot,
                # color='black',
                label=flow,
                )
        bot += [np.round(check.Base.results['qy'][y].loc[:,(slice(None),flow)].values.ravel()[0],0) for y in x_axis_y]
    
    ax.set_xlabel('years')
    ax.set_title('total production by flow')
    ax.legend()
    ax.grid(axis='both',linestyle='--')
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = dpi
    

    # TOTAL PRODUCTION BY HOURLY TECHNOLOGIES (power sector only)
    fig, ax = plt.subplots()    
    
    bot = np.zeros([len(x_axis_y)])
    for techs in ['t.elect_coal', 't.elect_natgas', 't.elect_pv', 't.elect_hydro', 't.elect_storage']:
        ax.bar( x_axis_y, 
                [np.round(check.Base.results['xy'][y].loc[:,(slice(None),techs)].values.ravel()[0],0) for y in x_axis_y],
                linestyle='-',
                bottom = bot,
                # color='black',
                label=techs,
                )
        bot += [np.round(check.Base.results['xy'][y].loc[:,(slice(None),techs)].values.ravel()[0],0) for y in x_axis_y]
    
    ax.set_xlabel('years')
    ax.set_title('total production by hourly technology')
    ax.legend()
    ax.grid(axis='both',linestyle='--')    
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = dpi
    
    
    # HOURLY PRODUCTION BY FLOW
    fig, ax = plt.subplots()    
    y = check.Base.ids.run_years[0]
    
    bot = np.zeros([len(x_axis_h)])
    for flow in ['f.elect']:
        ax.bar( x_axis_h, 
                check.Base.results['qh'][y].loc[:,(slice(None),flow)].values.ravel(),
                linestyle='-',
                bottom = bot,
                # color='black',
                label=flow,
                )
        bot += check.Base.results['qh'][y].loc[:,(slice(None),flow)].values.ravel()
    
    ax.set_xlabel('hours')
    ax.set_title('total production by flow')
    ax.legend()
    ax.grid(axis='both',linestyle='--')
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = dpi
    
    
    # storage SOC
    for year in [0,1,5,10]:
        fig, ax = plt.subplots()    
        y = check.Base.ids.run_years[year]
        
        ax.plot( x_axis_h, 
                check.Base.results['soc'][y].values.ravel(),
                linestyle='-',
                color='black',
                label='state of charge',
                )
        
        ax.bar( x_axis_h, 
                check.Base.results['xh'][y].loc[:,(slice(None),'t.elect_storage')].values.ravel(),
                linestyle='-',
                # color='black',
                label='hourly production',
                )
        
        ax.set_xlabel('hours')
        ax.set_title('electric battery storage')
        ax.legend()
        ax.grid(axis='both',linestyle='--')
        plt.tight_layout()
        plt.rcParams['figure.dpi'] = dpi
    

    # hourly FINAL DEMAND by FLOW (NON-PRODUCTIVE SECTORS ONLY)
    fig, ax = plt.subplots()    
    y = check.Base.ids.run_years[0]
    
    bot = np.zeros([len(x_axis_h)])
    for flow in ['f.elect']:
        ax.bar( x_axis_h, 
                cm.matmul(check.Base.par_ex['E_tld_diag'].cloc[(slice(None),flow),(slice(None),slice(None),y)], 
                          cm.trsp(check.Base.par_ex['dp'][y][flow].cloc[:,(slice(None),check.Base.ids.consumption_sectors)])).value.ravel(),
                linestyle='-',
                bottom = bot,
                # color='black',
                label=flow,
                )
        bot += cm.matmul(check.Base.par_ex['E_tld_diag'].cloc[(slice(None),flow),(slice(None),slice(None),y)], 
                         cm.trsp(check.Base.par_ex['dp'][y][flow].cloc[:,(slice(None),check.Base.ids.consumption_sectors)])).value.ravel()
    
    ax.set_xlabel('hours')
    ax.set_title('total demand by flow (non-productive sectors)')
    ax.legend()
    ax.grid(axis='both',linestyle='--')
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = dpi
    

    # CAPACITY PLOTS     
    cap = 'cap_o'
    
    # INSTALLED CAPACITY BY TECH (electricity)
    fig, ax = plt.subplots()    
    bot = np.zeros([len(x_axis_y)])
    for tech in ['t.elect_coal', 't.elect_natgas', 't.elect_pv', 't.elect_hydro', 't.elect_storage']: 
        ax.bar( x_axis_y, 
                check.Base.results[cap].loc[x_axis_y,:].loc[:,(slice(None),tech)].values.ravel(),
                linestyle='-',
                bottom = bot,
                # color='black',
                label=tech,
                )
        bot += check.Base.results[cap].loc[x_axis_y,:].loc[:,(slice(None),tech)].values.ravel()
    
    ax.set_xlabel('years')
    ax.set_ylabel(cap)
    ax.set_title('capacity by technology (electricity)')
    ax.legend()
    ax.grid(axis='both',linestyle='--')
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = dpi
    
    # INSTALLED CAPACITY BY TECH (housing)
    fig, ax = plt.subplots()    
    bot = np.zeros([len(x_axis_y)])
    for tech in ['t.house']: 
        ax.bar( x_axis_y, 
                check.Base.results[cap].loc[x_axis_y,:].loc[:,(slice(None),tech)].values.ravel(),
                linestyle='-',
                bottom = bot,
                # color='black',
                label=tech,
                )
        bot += check.Base.results[cap].loc[x_axis_y,:].loc[:,(slice(None),tech)].values.ravel()
    
    ax.set_xlabel('years')
    ax.set_ylabel(cap)
    ax.set_title('opearative capacity by technology (housing)')
    ax.legend()
    ax.grid(axis='both',linestyle='--')
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = dpi
    
    # INSTALLED CAPACITY BY TECH (transport)
    fig, ax = plt.subplots()    
    bot = np.zeros([len(x_axis_y)])
    for tech in ['t.trans_road']: 
        ax.bar( x_axis_y, 
                check.Base.results[cap].loc[x_axis_y,:].loc[:,(slice(None),tech)].values.ravel(),
                linestyle='-',
                bottom = bot,
                # color='black',
                label=tech,
                )
        bot += check.Base.results[cap].loc[x_axis_y,:].loc[:,(slice(None),tech)].values.ravel()
    
    ax.set_xlabel('years')
    ax.set_ylabel(cap)
    ax.set_title('opearative capacity by technology (transport)')
    ax.legend()
    ax.grid(axis='both',linestyle='--')
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = dpi






    # y = 2034
    # cm.cDataFrame(check.Base.par_ex['E'][y])
    # check.Base.results['xy'][y]
    # check.Base.results['qy'][y]


    # y_axis = [sum(sum(check.Base.par_ex[varName][y].value)) for y in x_axis]

    
    # check.Base.par_ex['af_eq'][2035].value

    # testEX = check.Base.par_ex['E'][2035].value
    # testEN = check.Base.results['CU'][2040].values
    
    # test1 = check.Base.par_ex['u'][2021].value @ np.diagflat(check.Base.results['xy'][2021].values)
    # test2 = check.Base.results['xy'][2021].values
    # test3 = check.Base.results['qy'][2021].values
    # test4 = check.Base.par_ex['E'][2021].value
    
    
    
    
    
    
    
