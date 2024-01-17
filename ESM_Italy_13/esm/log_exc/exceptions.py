# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 10:13:37 2021

@author: Amin

In this modue, all the exceptions, valueerrors, .... are defined for better
error handling of the code
"""

class NotRectangular(Exception):
    
    '''
    raise this exception if the model is not rectangluar so,
    the optimization model can not be defined
    '''
    pass

class WrongInput(ValueError):
    
    '''
    raise the value error if wrong inputs are given by the user
    '''    
    pass

class WrongExcelFormat(Exception):
    
    '''
    raise this exception if the format of the excel file is not correct
    '''
    pass


class NotCheneryTable(Exception):
    
    '''
    raise this exception if the table is not in chenery format
    '''
    pass

class WrongFileExtension(Exception):
    '''
    raise this exception if the file extension given by user is not correct
    '''
    pass

class NanValuesNotAccepted(Exception):
    '''
    riase this exception if nan values are not acceptable
    '''    
    pass

class AlreadyExist(Exception):
    '''
    raise this exception if sth that user is asking, already exists
    '''
    pass

class HeterogeneousCluster(Exception):
    '''
    raise this exception if a cluster has non-uniform aggregation in terms of
    units, types, .....
    '''
    pass



