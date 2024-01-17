# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 17:48:40 2021

@author: Amin


This module defines the structure of the datasets of the model, to ease the coding process
and to reduce the impact of changing naming conventions
"""



_MI = { 't' : 'Technologies',
        's' : 'Sectors',
        'r' : 'Regions',
        'f' : 'Flows',
        'h' : 'Timeslice',
        'y' : 'Years',
        'E' : 'E',
        'e' : 'e',
        'v' : 'v',
        'pr': 'primary resources',
        'ew': 'energy related waste',
        'ep': 'emissions by flow',
        'et': 'emissions by technology',
        'u' : 'u',
        'bp': 'bp',
        'wu': 'wu',
        'st': 'st',
        'bu': 'bu',
        'cu': 'cu',
        'tp': 'tp',
        'ef': 'ef',
        'bv': 'bv',
        'dp': 'dp',
        'af_eq' :'af_eq',
        'af_min':'af_min',
        'af_max':'af_max',
        'mr': 'mr',
        }

_FORMAT = { 'border': 1,
            'bg_color': '#CCECFF',
            'bold': True,
           }

_CLUSTER_YEARLY_SHEETS = [_MI[item] for item in ['u','bp','wu','bu','st','cu','tp','ef','v','bv']]
_CLUSTER_SLICE_SHEETS  = [_MI[item] for item in ['af_eq','af_min','af_max','dp']]

_SETS_READ = {_MI['r']: {'read':{'sheet_name':_MI['r'],
                                  'index_col' : None,
                                  'header'    : [0],
                                  },

                          'sort': False,
                          'columns'  : ['ID','NAME','TIME-ZONE','COLOR'],
                          'set'      : 'ID',
                          'defaults' : {'NAME'      : 'data.loc[data[\'NAME\'].isna().index,\'ID\']',
                                        'TIME-ZONE' : '\'missed\'',
                                        'COLOR'     : 'generate_random_colors(data.loc[data[\'COLOR\'].isna()].shape[0],\'list\')'},

                          'stop_nans' : [],
                          'validation': {},
                          },

                _MI['y']:  {'read':{'sheet_name':_MI['y'],
                                    'index_col' : None,
                                    'header'    : [0],
                                    },

                            'sort' : True,
                            'columns': ['YEAR','TYPE'],
                            'set'    : 'YEAR',
                            'defaults': {},
                            'validation': {'TYPE':['warm up','run','cool down']},
                            'stop_nans' : [],

                            },

                _MI['s']: {'read':{'sheet_name':_MI['s'],
                                  'index_col' : None,
                                  'header'    : [0],
                                  },

                          'sort': False,
                          'columns': ['ID','NAME','COLOR','PRODUCTION UNIT','CAPACITY UNIT','DISPATCH RESOLUTION','TYPE'],
                          'set'    : 'ID',
                          'defaults' : {'NAME'     : 'data.loc[data[\'NAME\'].isna().index,\'ID\']',
                                        'COLOR'    : 'generate_random_colors(data.loc[data[\'COLOR\'].isna()].shape[0],\'list\')'},

                          'stop_nans' : ['PRODUCTION UNIT',	'CAPACITY UNIT'],
                          'validation': {'TYPE': ['production','consumption'],
                                         'DISPATCH RESOLUTION': [_MI['h'],_MI['y'],'nn']},
                          },

                _MI['t']: {'read':{'sheet_name':_MI['t'],
                                          'index_col' : None,
                                          'header'    : [0],
                                          },

                          'sort': False,
                          'columns': ['ID','NAME','COLOR','CAPACITY','SECTOR','TYPE','AVAILABILITY'],
                          'set'    : 'ID',
                          'defaults' : {'NAME'     : 'data.loc[data[\'NAME\'].isna().index,\'ID\']',
                                        'COLOR'    : 'generate_random_colors(data.loc[data[\'COLOR\'].isna()].shape[0],\'list\')'},
                          'stop_nans' : [],
                          'validation': {'CAPACITY'    : [True,False],
                                         'AVAILABILITY': ['range','equality','demand','nn'],
                                         'TYPE'        : ['conversion','transmission','storage','storage+'],
                                         'SECTOR'      : 'self.Sectors'}

                          },

                _MI['f']: {'read':{'sheet_name':_MI['f'],
                                          'index_col' : None,
                                          'header'    : [0],
                                          },

                          'sort': False,
                          'columns': ['ID','NAME','COLOR','TYPE','PRODUCTION UNIT','DISPATCH RESOLUTION',],
                          'set'    : 'ID',
                          'defaults' : {'NAME'     : 'data.loc[data[\'NAME\'].isna().index,\'ID\']',
                                        'COLOR'    : 'generate_random_colors(data.loc[data[\'COLOR\'].isna()].shape[0],\'list\')'},
                          'stop_nans' : ['PRODUCTION UNIT'],
                          'validation': {'TYPE'        : ['product',_MI['pr'],_MI['ew'],_MI['ep'],_MI['et']]}
                          },

                _MI['h']: {'read':{'sheet_name':_MI['h'],
                           'index_col' : None,
                           'header'    : [0],
                           },

                          'sort': False,
                          'columns': ['ID','CORRESPONDANCE','PERIOD LENGTH'],
                          'set'    : 'ID',
                          'defaults' : {},
                          'stop_nans' : ['ID','CORRESPONDANCE','PERIOD LENGTH'],
                          'validation': {},
                          }
                  }

#  all the items for the cost
_COST_ITEMS = {'c_fu' : 'fuel cost',
               'c_op' : 'variable costs',
               'c_in' : 'fixed costs',
               'c_ds' : 'disposal costs',
               }

# items for money rates
_MONEY_RATES = {'mr_d' : 'discount rate',
                'mr_i' : 'interest rate',
                }

# Technology parameter items
_TP_ITEMS = {'t_tl' : 'technology lifetime',
             't_el' : 'economic lifetime (amortization)',
             't_ds' : 'disposal weibull shape',
             't_ca' : 'capacity to activity',
             't_mi' : 'maximum annual capacity increase',
             't_cd' : 'construction delay'
             }

# storage items
_ST_ITEMS = {'st_soc_min'   :'state of charge, min [%]',
             'st_cd_rate'   :'rate of charge/discharge [% of capacity per h]',
             'st_soc_start' :'state of charge, y.h=0 [%]',
             'st_soc_end'   :'state of charge, y.h=-1 [%]',
            }

# just for more beautiful headers
_AV_MAP = {'af_eq'  : 'equality',
           'af_min' : 'minimum',
           'af_max' : 'maximum',
           }

_INPUT_FILES = {'stand-alone': ['NewCapacityMin',
                                'NewCapacityMax',
                                'OperativeCapacityMin',
                                'OperativeCapacityMax',
                                'DemandProfiles',
                                'TechProductionMix',
                                'TechProductionMixMin',
                                'TechProductionMixMax',
                                'TechnologyData',
                                'Availability',
                                'MoneyRates',
                                'FlowData',
                                'Demand',
                                'TotalEmission',
                                'TotalInvestment',
                               ],

                'sfc-integrated':  ['NewCapacityMin',
                                    'NewCapacityMax',
                                    'OperativeCapacityMin',
                                    'OperativeCapacityMax',
                                    'DemandCoefficients',
                                    'TechProductionMix',
                                    'TechProductionMixMin',
                                    'TechProductionMixMax',
                                    'DemandProfiles',
                                    'TechnologyData',
                                    'Availability',
                                    'MoneyRates',
                                    'FlowData',
                                   ],

               }
