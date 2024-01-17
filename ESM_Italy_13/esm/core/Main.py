# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 17:41:11 2021

@author: Amin
"""

from esm.core.CoreModel import Base
from esm.core.Clusters import TimeCluster
from esm.log_exc.exceptions import (
    AlreadyExist,
    WrongInput,
)

from esm.utils.errorCheck import check_file_extension

from esm.utils.constants import (
    _CLUSTER_YEARLY_SHEETS,
    _INPUT_FILES,
)

from esm.utils.tools import dataframe_to_xlsx

from esm.log_exc.logging import log_time

from copy import deepcopy as dc
from os import path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Model:

    """
    DESCRIPTION
    ============
    esm Model:
        Creating an Running Energy System Optimization
    """

    def __init__(self, sets_file: str, integrated_model: bool = False) -> None:

        """
        DESCRIPTION
        ============
        Initializing the model

        PARAMETERS
        =============
        sets_file        : Defines the path of the sets excel file:
                           The sets should be given as an xlsx file with a predifend structure.
                           For more information, use esm.Guide.sets

        integrated_model : if True, the model can be coupled with SFC model.
                           if False, the model will be stand-alone. This will need more data
                           to be given to the model.
        """

        self.Base = Base()

        if integrated_model:
            self.Base.mode = "sfc-integrated"
            log_time(
                logger,
                "Model: Setting model at {} configuration. "
                "In this mode, final demand will be defined by SFC model".format(
                    self.Base.mode
                ),
            )
        else:
            self.Base.mode = "stand-alone"
            log_time(
                logger,
                "Model: Setting model at {} configuration. "
                "This will require the definition of final demand by user".format(
                    self.Base.mode
                ),
            )

        self.Base.__readsets__(sets_file)

        # just for testing
        self.an = self.Base.__matrices__

    def read_input_excels(
        self,
        directory: str,
        NewCapacityMin=None,
        NewCapacityMax=None,
        OperativeCapacityMin=None,
        OperativeCapacityMax=None,
        TechProductionMix=None,
        TechProductionMixMin=None,
        TechProductionMixMax=None,
        DemandCoefficients=None,
        MoneyRates=None,
        DemandProfiles=None,
        TechnologyData=None,
        Availability=None,
        FlowData=None,
        Demand=None,
        TotalEmission=None,
        TotalInvestment=None,
    ) -> None:

        for item in _INPUT_FILES[self.Base.mode]:
            if eval(item) is None:
                self.Base._read_paramaters_from_excel(
                    r"{}/{}.xlsx".format(directory, item), item
                )

            else:
                self.Base._read_paramaters_from_excel(
                    r"{}/{}".format(directory, eval(item)), item
                )

            log_time(logger, f"Parser: {item} successfully parsed.")

        self.Base.time_cluster.already_exist = True

    def create_input_excels(
        self,
        directory: str,
        NewCapacityMin=None,
        NewCapacityMax=None,
        OperativeCapacityMin=None,
        OperativeCapacityMax=None,
        TechProductionMix=None,
        TechProductionMixMin=None,
        TechProductionMixMax=None,
        DemandCoefficients=None,
        DemandProfiles=None,
        TechnologyData=None,
        MoneyRates=None,
        Availability=None,
        FlowData=None,
        Demand=None,
        TotalEmission=None,
        TotalInvestment=None,
    ) -> None:

        """
        Description
        ==============
        This function will create a set of excel files in the given directory
        that helps the user to fill the input data for the model

        Parameters
        =============
        directory:
            it should the directory (a folder).
            In the given directory, following files will be created:
                1. TechnologyData
                2. DemandProfiles
                3. TimeSliceData
        """

        if not path.exists(directory):
            raise WrongInput(f"{directory} does not exist")
        if not path.isdir(directory):
            raise WrongInput("directory should be a folder, not be a file.")

        for item in _INPUT_FILES[self.Base.mode]:
            if eval(item) is None:
                self.Base.__generate_excel__(
                    path=r"{}/{}.xlsx".format(directory, item), what=item
                )
            else:
                log_time(
                    logger,
                    "no file name for {item} is given. Default name {item}.xlsx will be used".format(
                        item=item
                    ),
                    "warn",
                )
                self.Base.__generate_excel__(
                    path=r"{}/{}".format(directory, eval(item)), what=item
                )

    def generate_clusters_excel(self, path: str, cluster_type: str = "time",) -> None:

        if cluster_type == "time":
            self.Base.time_cluster.generate_excel(path)

        else:
            raise WrongInput(f"{cluster_type} is not acceptable input.")

    def read_clusters(self, path: str, cluster_type: str = "time",) -> None:

        if cluster_type == "time":
            self.Base.time_cluster.read_clusters(path)

    def time_cluster(
        self, parameter=None,
    ):

        if parameter is None:
            return dc(self.Base.time_cluster.clusters_frame)

        else:
            try:
                return dc(self.Base.time_cluster.clusters_frame)[parameter]
            except KeyError:
                raise WrongInput(
                    f"{parameter} is not an accpetable parameter for time clusters"
                )

    def to_excel(self, item: str, path: str) -> None:

        """
        DESCRIPTION
        =============
        The function can be used to print out different sets of data to excel

        PARAMETERS
        =============
        item: defines what to pring acceptable items are ['sets','results'].
        path: defines the path to save the file.
        """

        check_file_extension(path, ["xlsx"])

        if item == "sets":
            with pd.ExcelWriter(path) as file:
                for key, value in self.Base.__sets_frames__.items():
                    if key == "_type":
                        continue
                    value.to_excel(file, sheet_name=key)


#%% test

if __name__ == "__main__":

    from esm import set_log_verbosity

    set_log_verbosity("critical")

    check = Model(r"esm\unit_test\set_reader\sets_sample.xlsx", False)
    # check.generate_clusters_excel(r'esm\unit_test\time_clusters\clusters.xlsx',  )
    # check.read_clusters(r'esm\unit_test\time_clusters\clusters.xlsx',  )
    check.create_input_excels(r"esm\unit_test\input_excels")
    # check.to_excel('sets', r'esm\unit_test\set_reader\sets_sample_code.xlsx')
    check.read_input_excels(r"esm\unit_test\input_excels")

