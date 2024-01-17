# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 21:35:47 2021

@author: Amin
"""

from esm.utils.constants import (_CLUSTER_YEARLY_SHEETS,
                                 _CLUSTER_SLICE_SHEETS,
                                 _MI,
                                 _SETS_READ,
                                 )

from esm.utils.errorCheck import (check_file_extension,
                                  check_excel_col_row,
                                  nans_exist,
                                  )

from esm.utils.tools import (generate_random_colors,
                             delete_duplicates,
                             )

from esm.log_exc.logging import log_time
from esm.log_exc.exceptions import (HeterogeneousCluster,
                                    AlreadyExist,
                                    )

import esm.utils.cvxpyModified as cm

from copy import deepcopy as dc
import pandas as pd
import numpy as np
import logging


logger = logging.getLogger(__name__)

class TimeCluster:

    '''
    DESCRIPTION
    ==============
    Time cluster class is an accessory function to optimizie the data handling of the model
    As the model is dynamic, all the inputs of the model can change for evey year of  modeling
    time-horizon.

    In case that the user does not need to provide different data for all the users, for every
    single parameter a cluster of time can be defined. Data then will be given accordingly to
    the clusters.

    EXAMPLE
    ================
    1. If the user needs to give a single sets of data for the Use matrix, there will be a single
       time cluster, that is correspond to all the time horizon.
    2. If the user needs to give for every year, different Use matrix, the number of clusters will
        be the same of time-horizon
    '''

    def __init__(self,
                 instance:list
                 ) -> None:

        self.instance = instance
        self.years = instance.Years

        '''
        By default, we need to create the clusters. Then if the user needs
        to change, it, will provide the information on the clusters
        '''
        self.clusters_frame = pd.DataFrame('T1',index=self.years,columns=_CLUSTER_YEARLY_SHEETS+_CLUSTER_SLICE_SHEETS)
        self.clusters = self.clusters_frame.to_dict('index')

        # A parameter to avoid double reading
        self.already_exist = False


    def get_clusters_for_file(self,file):
        """Returns the clusters exists for a given file
        """
        mapper = {
            "TechnologyData":['u','bp','wu','bu','st','cu','tp','ef'],
            "FlowData": ['v','bv'],
            "Availability": ['af_max','af_min','af_eq'],
            "DemandProfiles": ['dp'],
        }

        parameters = self.clusters_frame[mapper[file]].values

        return np.unique(parameters)

    def parameter_clusters(self,
                          parameter: str
                          ) -> list :
        '''
        Description
        =============
        Function returns a list of all the clusters defined for a specific
        parameter

        Parameters
        ============
        parameter: parameters listed in _CLUSTER_SHHETS or all

                if parameter is all, it returns all the clusters defined


        Returns
        ============
        A list of all the defined clusters for a given parameter
        '''
        if parameter == 'all':
            clusters = self.clusters_frame.values.tolist()

            clusters = [j[i]
                        for j in clusters
                        for i,h in enumerate(j)]

        else:
            if parameter[0:2] == 'ef':
                parameter = 'ef'

            clusters = self.clusters_frame[parameter].values.tolist()

        return delete_duplicates(sorted(clusters))

    def generate_excel(self,
                       path:str
                       ) -> None:

        '''
        DESCRIPTION
        =============
        This function generates an excel files to define the clusters for different
        parameters in different sheets

        PARAMETERS
        =============
        path: the path of the excel file with .xlsx format
        '''

        check_file_extension(path,['xlsx'])

        with pd.ExcelWriter(path) as file:
            pd.DataFrame(index=self.years,columns=_CLUSTER_YEARLY_SHEETS+_CLUSTER_SLICE_SHEETS).to_excel(file)

    def read_clusters(self,
                      path: str) -> None:

        '''
        DESCRIPTION
        ============
        This function is in charge of reading the clusters.

        PARAMETERS
        ============
        path: path of the ecel file containing the definition of the clusters.
        '''
        if self.already_exist:
            raise AlreadyExist('Clusters are already defined or input parameters are'
                               ' already parsed. Cluster can be chenged only before'
                               ' parsing the inputs.')


        check_file_extension(path,['xlsx'])
        acceptable_clusters = ['T{}'.format(y) for y in range(1,len(self.years)+1)]
        frames = pd.DataFrame(index   = self.years,
                              columns = _CLUSTER_YEARLY_SHEETS+_CLUSTER_SLICE_SHEETS)

        data = pd.read_excel(path,
                             index_col  = [0],
                             header     = [0],
                             dtype      = str,
                            )

        check_excel_col_row(list(data.index),
                            self.years,
                            path,
                            'main sheet',
                            'index',
                            check = 'equality')


        for column in _CLUSTER_YEARLY_SHEETS+_CLUSTER_SLICE_SHEETS:
            try:

                data_col = nans_exist(data   = data[column],
                                      action = 'raise error',
                                      info   = column+' Time Cluster. To use the default values, you can delete the column ',
                                      )

                given_clusters  = data_col[column]

                for year,value in given_clusters.iteritems():
                    if value not in acceptable_clusters:
                        raise Exception(f'{value} is not an acceptable time cluster'
                                         f' for time. (column = {column}). \n'
                                         f'Acceptbale values are {acceptable_clusters}')



                frames[column] = given_clusters

            except KeyError:
                log_time(logger,
                         f'column {column} not found in the {path}. Default values (single cluster)'
                         ' will be considered.',
                         'critical',
                         )

                frames[column] = self.clusters_frame[column]


        self.clusters_frame = frames
        self.clusters = frames.to_dict('index')
        self.instance.__Frames__()
        self.instance.data = dc(dict(**self.instance.__matrices__))

# if exists in a column, it will be a cluster
cluster_str_check = 'CLS'
set_cluster_columns   = ['ID','NAME','COLOR']
set_cluster_accaptable = [_MI['r'],
                          _MI['t'],
                          _MI['f'],
                          _MI['s'],
                         ]
class NonTimeCluster:

    def __init__(self):

        self.results = {}
        self.clusters_columns = {}


    def check_cluster_exists(self,dataframe,set_name):

        dataframe = dc(dataframe)
        cluster_counter = 1
        columns = list(dataframe.columns)

        while  f'{cluster_str_check}{cluster_counter}.ID' in columns :

            nans_exist(data   = dataframe[f'{cluster_str_check}{cluster_counter}.ID'],
                       action = 'raise error',
                       info   = f'{cluster_str_check}{cluster_counter}.ID for set: {set_name}',
                       )

            self.control_clusters_aggregation_errors(dataframe,
                                                    set_name,
                                                    dataframe[f'{cluster_str_check}{cluster_counter}.ID'].values,
                                                    cluster_counter)

            for extra_items in ['NAME','COLOR']:
               if f'{cluster_str_check}{cluster_counter}.{extra_items}' not in columns:
                   dataframe[f'{cluster_str_check}{cluster_counter}.{extra_items}'] = np.array([np.nan]*len(dataframe))

               missing_items = dataframe.loc[dataframe[f'{cluster_str_check}{cluster_counter}.{extra_items}'].isna(),
                                             f'{cluster_str_check}{cluster_counter}.{extra_items}']

               # if any missing item exists
               if len(missing_items):

                   if extra_items == 'NAME':
                       dataframe.loc[dataframe[f'{cluster_str_check}{cluster_counter}.{extra_items}'].isna(),
                                             f'{cluster_str_check}{cluster_counter}.{extra_items}'] =  \
                       dataframe.loc[dataframe[f'{cluster_str_check}{cluster_counter}.{extra_items}'].isna(),
                                     f'{cluster_str_check}{cluster_counter}.ID'].index

                   else:
                       dataframe.loc[dataframe[f'{cluster_str_check}{cluster_counter}.{extra_items}'].isna(),
                                             f'{cluster_str_check}{cluster_counter}.{extra_items}'] =  \
                       generate_random_colors(len(missing_items))

            cluster_counter += 1

        log_time(logger,f'Sets Clusters: {cluster_counter-1} cluster found for {set_name}')

        self.results[set_name] = dataframe
        new_columns = _SETS_READ[set_name]['columns'] + [f'{cluster_str_check}{i}.{j}'
                                                                  for i in range(1,cluster_counter)
                                                                  for j in ['ID','NAME','COLOR']]
        new_columns.remove(_SETS_READ[set_name]['set'])
        self.clusters_columns[set_name] = new_columns

    def re_organize_main_dataframes(self,set_name):
        data = self.results[set_name]
        columns_to_drop = set(data.columns).difference(set(self.clusters_columns[set_name]))
        return data.drop(columns_to_drop,axis=1,)

    def control_clusters_aggregation_errors(self,dataframe,set_name,clusters,cluster_counter):
        '''
        # sectors
            # criterion: having the same units, dispatch resolution and type
        '''

        def check_all(columns,dataframe,):
            for column in columns:
                for cluster in set(clusters):
                    if len(set(dataframe.loc[dataframe[f'{cluster_str_check}{cluster_counter}.ID']== cluster,column]))>1:
                        raise HeterogeneousCluster(f'cluster {cluster}, {set_name} has non homogeneous {column}')

        columns_to_check = {_MI['s']: ['CAPACITY UNIT','DISPATCH RESOLUTION','PRODUCTION UNIT','TYPE'],
                            _MI['f']: ['DISPATCH RESOLUTION','PRODUCTION UNIT','TYPE'],
                            _MI['r']: [],
                            }

        if set_name in columns_to_check:
            check_all(columns_to_check[set_name],dataframe)

        # for technologies, we have a different story
        else:
            check_all(['TYPE'],dataframe)
            for cluster in set(clusters):
                take_sectors = list(dataframe.loc[dataframe[f'{cluster_str_check}{cluster_counter}.ID']== cluster,'SECTOR'])

                for column in columns_to_check[_MI['s']]:
                    if len(set(self.results[_MI['s']].loc[take_sectors,column].values))>1:
                        raise HeterogeneousCluster(f'cluster {cluster}, {set_name} has non homogeneous sectors definition {take_sectors}.')




























if __name__ == '__main__':
    test = TimeCluster(years=list(range(2020,2040)))
    test.generate_excel(r'esm\unit_test\time_clusters\clusters.xlsx',)
    #test.read_clusters(r'esm\unit_test\time_clusters\clusters.xlsx')

