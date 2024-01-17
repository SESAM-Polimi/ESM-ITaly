# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 13:49:45 2021

@author: Amin
"""
from typing import List
from esm.log_exc.exceptions import (WrongExcelFormat, 
                                    WrongInput,
                                    WrongFileExtension,
                                    NanValuesNotAccepted,
                                    )

import pandas as pd

def check_excel_col_row(given,correct,file,sheet,level,check):
    
    '''
    DESCRIPTION
    =============
    A function to check the incorrect index and column in an excel file
    
    PARAMETERS
    ============
    given   : the index, columns to be checked
    correct : correct values of index and columns
    file    : the file that you are checking
    sheet   : sheet of the excel file that you are reading
    level   : which level, index or columns
    check   : can be 'contain' or 'equality'. If contain, it just check if the
              correct indices exist in given or not. if is equality, it checks
              if they are exactly the same.
              
    RAISE
    ============
    Raise WrongExcelFormat if the format is not correct
    
    '''
    
    assert check in ['equality','contain'],'valid items for check are [\'equality\',\'contain\']'
    
    if check == 'equality':
        if isinstance(given,list):
            if given != correct:
                raise WrongExcelFormat(f'The {level} in file: {file}, sheet {sheet} is not correct.'
                                       f' Correct values: \n {correct} \n Given values: \n {given}.')
                
        elif isinstance(given,(pd.MultiIndex,pd.Index)):         
            if not given.equals(correct):
                raise WrongExcelFormat(f'The {level} in file: {file}, sheet {sheet} is not correct.'
                                       f' Correct values: \n {correct} \n Given values: \n {given}.')                
    else:
        acceptable = all(element in given for element in correct)
        
        if not acceptable:
            raise WrongExcelFormat(f'The {level} in file: {file}, sheet {sheet} is not correct.'
                                   f' The presence of following items is necessary\n:{correct}.'
                                   f' Given items are {given}')


        
def validate_data(given   : list,
                  correct : list,
                  name    : str,
                  ) -> None:

    '''
    DESCRIPTION
    =============
    A function to check the if the given data are valid or not
    
    PARAMETERS
    ============
    given   : the data to be checked
    correct : acceptable values
    name    : name of the item that is going to be checked

    RAISE
    ============
    Raise WrongInput if the data is not valid
    '''    
    
    difference = set(given).difference(set(correct))
    
    if len(difference):
        raise WrongInput('{} are not valid items for {}. Valid items are {}'.format(difference,
                                                                                    name,
                                                                                    set(correct)
                                                                                    )
                         )


def check_file_extension(file:str,
                         acceptables: List[str],
                         ) -> None:
    
    '''
    DESCRIPTION
    =============
    Function checks if the given file by the user has the correct extension or not
    
    PARAMETERS
    ============
    file       : file path given by the user
    acceptable : acceptable extensions for the code
    
    RAISES
    ============
    WrongFileExtension

    '''

    extension = file.split('.')[-1]

    if extension not in acceptables:
        raise WrongFileExtension(f'{file} has not a valid extension for the operation.'
                                 f' Acceptable extensions are {acceptables}.')   
        
def nans_exist(data   : [pd.DataFrame,pd.Series],
               action : str,
               info   : str,
               fillna : [str,int,float,] = None,
               ) -> None:
    
    '''
    DESCRIPTION
    =============
    This function is in charge of checking the nan existnace and act accordingly
    
    PARAMETERS
    ============
    data   : data to check
    action : if raise_error, will stop the code (if nan values are not acceptable)
             if pass, will do nothing and pass.
             
    RAISES
    ============
    NanValuesNotAccepted if action == 'raise error'
    
    '''
    assert action in ['raise error','pass'],'action can be only \'raise error\' or \'pass\''
    
    if isinstance(data,pd.Series):
        data = data.to_frame()
        
    if data.isna().sum().sum():
        if action == 'raise error':
            raise NanValuesNotAccepted(f'nan values are not acceptable for {info}')
            
    return data
    
        

if __name__ == '__main__':
    
    pass
    
    
    
    
    