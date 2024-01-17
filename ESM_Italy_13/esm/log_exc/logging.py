# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 10:13:27 2021

@author: Amin


This module is in charge of handling the logging formats of the model.


User: Will only use the set_log_verbosity
Coder: Will use the log_time to do the logging

"""


import logging
import sys
import os

_time_format = "%Y-%m-%d %H:%M:%S"


def setup_root_logger(verbosity,capture_warnings,log_file):
    '''
    The function is in charge of formating the logging. 
    Nobody needs to use it!!!
    '''
    root_logger = logging.getLogger()
    
    # Removing all the existing handlers 
    if root_logger.hasHandlers():
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
            
    # Defining the formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s: %(message)s", datefmt=_time_format
    )   
    
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    
 
    root_logger.addHandler(console)

    if log_file is not None:
        try:
            os.remove(log_file)
        except FileNotFoundError:
            pass
        
        file = logging.FileHandler(log_file)
        file.setFormatter(formatter)
        root_logger.addHandler(file)   
        
    root_logger.setLevel(verbosity.upper())
    
    if capture_warnings:
        logging.captureWarnings(True)
        pywarning_logger = logging.getLogger("py.warnings")
        pywarning_logger.setLevel(verbosity.upper())

    return root_logger
    
def log_time(
        logger , 
        comment,
        level='info'):
    
    '''
    this function is used for logging purposes.
    
    HOW TO USE
    ============
    
    1. in the beginning of every module, you need to define the module logger as follow:
        import logging
        logger = logging.getLogger(__name__)
        
    2. when logging is needed use the function, passing the logger defined in previous step, the comment and the level of logging:
        e.g., log_time(logger,'my comment','info')
    '''

    getattr(logger, level)(comment)
    
def set_log_verbosity(verbosity       :str  = 'info',
                      capture_warnings:bool = True,
                      log_file        :str  = None,
                      ):

    '''
    DESCRIPTION
    ==============
    This function is in charge of defining the level of logging of the code.
    
    PARAMTERS
    ==============
    verbosity: str, Defines the level of verbosity:
        debug, info, warn, critical
        
    capture_warning: boolean, If true, captures all the warning even if the the level is lower than the warning
    
    log_file: defines tha path that user needs to save the log file. if it is None
              no log file will be created
    '''
    backend_logger = logging.getLogger('ESM')
    backend_logger.setLevel(verbosity.upper())
    setup_root_logger(verbosity=verbosity,
                      capture_warnings=capture_warnings,
                      log_file=log_file)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    