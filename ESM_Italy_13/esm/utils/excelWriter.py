# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 18:18:16 2021

@author: Amin
"""

_PARAM = ['u','bp','wu','bu']

def input_parameters(instance):
    
    # Defining the orders of the matrices
    parameters =  ['u','bp','wu','bu']
    
    # adding the parameters for the emission filter
    for emission in instance.emission_by_flow:
        parameters.append(f'ef_{emission}')
        
    parameters.append('st')