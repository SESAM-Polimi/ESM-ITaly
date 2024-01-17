# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 11:35:36 2021

@author: Amin
"""
import pandas as pd
import copy

class Dict(dict):
    
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__    
    
    
    def __init__(self,_type=pd.DataFrame,**source_dict,):
        
        self._type = _type

        super().__init__()
        
        for k,v in source_dict.items():
            self._set_item(k, v)
            
    def _set_item(self, k, v):
        if not isinstance(v,self._type):
            raise TypeError('Dict object accepts only {}. not {}'.format(self._type,type(v)))
        
        if k in self:
            raise ValueError(f'{k} already exists')
                   
        super().__setitem__(k, v)
        
    def __delitem__(self, v):
        raise ValueError('not supported')
        
    def copy(self):
        return copy.deepcopy(self)
        

